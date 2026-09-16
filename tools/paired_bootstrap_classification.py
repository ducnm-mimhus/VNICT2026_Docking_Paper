#!/usr/bin/env python3
"""
B1 (phan bien) — Paired bootstrap CI cho Delta = metric(A) - metric(B) tren
NHANH PHAN LOAI TU THE (Balanced Accuracy, PR-AUC), tren TOAN BO tap test
(khong loc theo y_aff > 0 nhu nhanh ai luc — nhanh tu the dung ca 4618 mau).

Dung chung quy uoc voi tools/paired_bootstrap.py: cung seed=2026, n_boot=2000,
alpha=0.05, resample CO HOAN LAI tren chi so hang, hai file du doan PHAI cung
hang (cung --testfile, cung thu tu, khong shuffle).

Cach chay:
    python3 tools/paired_bootstrap_classification.py \
        --pred_a results/predictions/geoformerdock.csv \
        --pred_b results/predictions/pafnucy.csv --n_boot 2000 --seed 2026
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_pose(path: Path) -> Tuple[List[int], List[float]]:
    labels, p_good = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append(int(row["label"]))
            p_good.append(float(row["p_good"]))
    return labels, p_good


def balanced_accuracy(labels: np.ndarray, p_good: np.ndarray, thr: float = 0.5) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    recall_pos = (p_good[pos] >= thr).sum() / n_pos
    recall_neg = (p_good[neg] < thr).sum() / n_neg
    return 0.5 * (recall_pos + recall_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP = sum_i (recall_i - recall_{i-1}) * precision_i, tinh tai cac nguong
    PHAN BIET (gop cac mau cung diem so vao cung 1 nguong, giong sklearn)."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    distinct = np.where(np.diff(s) != 0)[0]
    threshold_idx = np.r_[distinct, len(s) - 1]
    tps = np.cumsum(y)[threshold_idx]
    fps = (threshold_idx + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum(np.diff(recall) * precision[1:]))


def paired_bootstrap(labels: List[int], pa: List[float], pb: List[float],
                      metric_fn, n_boot: int, alpha: float, seed: int):
    labels_arr, pa_arr, pb_arr = np.array(labels), np.array(pa), np.array(pb)
    n = len(labels)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.array([rng.randrange(n) for _ in range(n)])
        deltas.append(metric_fn(labels_arr[idx], pa_arr[idx]) - metric_fn(labels_arr[idx], pb_arr[idx]))
    deltas.sort()
    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = int(n_boot * (1 - alpha / 2)) - 1
    mean_delta = sum(deltas) / n_boot
    return mean_delta, deltas[max(lo_idx, 0)], deltas[min(hi_idx, n_boot - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_a", required=True, type=Path)
    parser.add_argument("--pred_b", required=True, type=Path)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    labels_a, pg_a = read_pose(args.pred_a)
    labels_b, pg_b = read_pose(args.pred_b)

    if len(labels_a) != len(labels_b):
        print(f"[CANH BAO] So dong khac nhau (A={len(labels_a)}, B={len(labels_b)})", file=sys.stderr)
    n = min(len(labels_a), len(labels_b))
    mismatches = sum(1 for i in range(n) if labels_a[i] != labels_b[i])
    if mismatches:
        print(f"[CANH BAO] {mismatches}/{n} dong co label khac nhau giua A va B — "
              f"hai file co ve KHONG cung hang/cung testfile.", file=sys.stderr)

    labels = labels_a[:n]
    pa, pb = pg_a[:n], pg_b[:n]

    print(f"So mau (toan bo test): {n}")
    print(f"A = {args.pred_a.name}   B = {args.pred_b.name}")
    print(f"Delta = metric(A) - metric(B)   [n_boot={args.n_boot}, alpha={args.alpha}, seed={args.seed}]")
    print()

    header = f"{'metric':<20} {'metric(A)':>10} {'metric(B)':>10} {'mean Delta':>12} {'CI low':>10} {'CI high':>10}  ket luan"
    print(header)
    print("-" * len(header))

    for name, fn in [("balanced_accuracy", balanced_accuracy), ("pr_auc", average_precision)]:
        labels_arr = np.array(labels)
        val_a = fn(labels_arr, np.array(pa))
        val_b = fn(labels_arr, np.array(pb))
        mean_d, lo, hi = paired_bootstrap(labels, pa, pb, fn, args.n_boot, args.alpha, args.seed)
        crosses_zero = lo <= 0 <= hi
        if crosses_zero:
            verdict = "khong ro ret (CI chua 0)"
        elif mean_d > 0:
            verdict = "A tot hon dang ke"
        else:
            verdict = "B tot hon dang ke"
        print(f"{name:<20} {val_a:>10.4f} {val_b:>10.4f} {mean_d:>+12.4f} {lo:>+10.4f} {hi:>+10.4f}  {verdict}")


if __name__ == "__main__":
    main()
