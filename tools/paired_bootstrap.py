#!/usr/bin/env python3
"""
VD4a — Paired bootstrap CI cho hieu so Delta = metric(mo hinh A) - metric(mo hinh B).

QUAN TRONG — dieu kien de ket qua hop le: hai file du doan PHAI cung hang voi
nhau (dong thu i trong pred_a ung voi CUNG mot mau nhu dong thu i trong
pred_b). Dieu nay dung khi ca hai duoc tao bang tools/run_inference.py tren
CUNG mot --testfile: molgrid ExampleProvider trong che do inference dung
shuffle=False + SmallEpoch (setup.py), nen thu tu mau la xac dinh va giong
het nhau giua cac lan chay tren cung file .types.

Cach chay:
    python3 tools/paired_bootstrap.py \
        --pred_a results/predictions/geoformerdock.csv \
        --pred_b results/predictions/pafnucy.csv \
        --metric mae --n_boot 2000 --seed 2026

    # Chay het cac chi so chinh cung luc:
    python3 tools/paired_bootstrap.py \
        --pred_a results/predictions/geoformerdock.csv \
        --pred_b results/predictions/pafnucy.csv \
        --metric all
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from exact_metrics import (  # noqa: E402
    read_predictions, mae, rmse, pearson_r, spearman_rho, exact_concordance_index,
)


def _regression_metric(name: str) -> Callable[[List[float], List[float]], float]:
    return {
        "mae": mae, "rmse": rmse, "pearson": pearson_r, "spearman": spearman_rho,
        "cidx": lambda p, t: exact_concordance_index(p, t)[0],
    }[name]


REGRESSION_METRICS = ("mae", "rmse", "pearson", "spearman", "cidx")
LOWER_IS_BETTER = {"mae", "rmse"}


def paired_bootstrap_delta(
    aff_true: List[float], pred_a: List[float], pred_b: List[float],
    metric_fn: Callable[[List[float], List[float]], float],
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 2026,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(aff_true)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        t = [aff_true[i] for i in idx]
        pa = [pred_a[i] for i in idx]
        pb = [pred_b[i] for i in idx]
        deltas.append(metric_fn(pa, t) - metric_fn(pb, t))
    deltas.sort()
    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = int(n_boot * (1 - alpha / 2)) - 1
    mean_delta = sum(deltas) / n_boot
    return mean_delta, deltas[max(lo_idx, 0)], deltas[min(hi_idx, n_boot - 1)]


def align_and_filter(
    labels_a, aff_true_a, pred_a, labels_b, aff_true_b, pred_b,
) -> Tuple[List[float], List[float], List[float]]:
    n = min(len(aff_true_a), len(aff_true_b))
    if len(aff_true_a) != len(aff_true_b):
        print(
            f"  [CANH BAO] So dong khac nhau (A={len(aff_true_a)}, B={len(aff_true_b)}) "
            f"— chi dung {n} dong dau, KET QUA CO THE KHONG HOP LE neu day khong phai "
            f"do cung mot testfile.",
            file=sys.stderr,
        )
    mismatches = sum(1 for i in range(n) if abs(aff_true_a[i] - aff_true_b[i]) > 1e-6)
    if mismatches:
        print(
            f"  [CANH BAO] {mismatches}/{n} dong co affinity_true khac nhau giua A va B "
            f"— hai file co ve KHONG cung hang/cung testfile. Ket qua paired bootstrap "
            f"se KHONG hop le. Dung lai va kiem tra input.",
            file=sys.stderr,
        )
    good_idx = [i for i in range(n) if aff_true_a[i] > 0]
    t = [aff_true_a[i] for i in good_idx]
    pa = [pred_a[i] for i in good_idx]
    pb = [pred_b[i] for i in good_idx]
    return t, pa, pb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_a", required=True, type=Path, help="CSV du doan mo hinh A (vd mo hinh de xuat)")
    parser.add_argument("--pred_b", required=True, type=Path, help="CSV du doan mo hinh B (baseline)")
    parser.add_argument("--metric", default="all", choices=list(REGRESSION_METRICS) + ["all"])
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    labels_a, aff_true_a, _pg_a, pred_a = read_predictions(args.pred_a)
    labels_b, aff_true_b, _pg_b, pred_b = read_predictions(args.pred_b)

    t, pa, pb = align_and_filter(labels_a, aff_true_a, pred_a, labels_b, aff_true_b, pred_b)
    print(f"So mau dung (y_aff > 0, sau khi align): {len(t)}")
    print(f"A = {args.pred_a.name}   B = {args.pred_b.name}")
    print(f"Delta = metric(A) - metric(B)   [n_boot={args.n_boot}, alpha={args.alpha}, seed={args.seed}]")
    print()

    metrics_to_run = REGRESSION_METRICS if args.metric == "all" else [args.metric]
    header = f"{'metric':<10} {'metric(A)':>10} {'metric(B)':>10} {'mean Delta':>12} {'CI low':>10} {'CI high':>10}  ket luan"
    print(header)
    print("-" * len(header))

    for name in metrics_to_run:
        fn = _regression_metric(name)
        val_a = fn(pa, t)
        val_b = fn(pb, t)
        mean_d, lo, hi = paired_bootstrap_delta(t, pa, pb, fn, args.n_boot, args.alpha, args.seed)

        crosses_zero = lo <= 0 <= hi
        lower_is_better = name in LOWER_IS_BETTER
        if crosses_zero:
            verdict = "khong ro ret (CI chua 0)"
        elif (mean_d < 0) == lower_is_better:
            verdict = "A tot hon dang ke"
        else:
            verdict = "B tot hon dang ke"

        print(f"{name:<10} {val_a:>10.4f} {val_b:>10.4f} {mean_d:>+12.4f} {lo:>+10.4f} {hi:>+10.4f}  {verdict}")


if __name__ == "__main__":
    main()
