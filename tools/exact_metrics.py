#!/usr/bin/env python3
"""
A1 — tinh cac chi so DUNG (khong lay mau ngau nhien) tu file du doan tung mau
(output cua tools/run_inference.py).

VD4b: metrics.concordance_index() trong dockbench/metrics.py lay mau toi da
50.000 CAP khong seed — voi vai nghin mau, so cap that len toi hang trieu, nen
C-index bao cao trong training co the dao dong o chu so thap phan thu 3 giua
cac lan chay. Script nay tinh DU MOI CAP (O(N^2), chap nhan duoc voi N ~ vai
nghin mau tap test), khong con la uoc luong.

Cach chay:
    python3 tools/exact_metrics.py --predictions results/predictions/geoformerdock.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_predictions(path: Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    labels, aff_true, p_good, aff_pred = [], [], [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append(int(row["label"]))
            aff_true.append(float(row["affinity_true"]))
            p_good.append(float(row["p_good"]))
            aff_pred.append(float(row["affinity_pred"]))
    return labels, aff_true, p_good, aff_pred


def exact_concordance_index(
    pred: List[float], target: List[float], chunk_size: int = 512,
) -> Tuple[float, int]:
    """
    C-index tinh tren TOAN BO cap (i,j), target[i] != target[j].

    Vector hoa bang numpy, xu ly theo tung chunk hang de gioi han bo nho voi
    N lon (tap test day du ~4.6k mau: ban pure-Python truoc day mat ~1.6s/lan,
    tuc ~54 phut cho 2000 lan resample trong paired_bootstrap.py — qua cham
    de dung thuc te. Ban numpy nay nhanh hon ~2-3 bac do.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n = len(p)
    if n < 2:
        return 0.5, 0

    concordant = 0.0
    n_pairs = 0
    idx_all = np.arange(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        t_diff = t[start:end, None] - t[None, :]          # [chunk, N]
        p_diff = p[start:end, None] - p[None, :]          # [chunk, N]
        upper = idx_all[None, :] > idx_all[start:end, None]  # moi cap (i,j) dem 1 lan
        valid = upper & (np.abs(t_diff) > 1e-9)
        conc = valid & ((t_diff * p_diff) > 0)
        tie = valid & (np.abs(p_diff) < 1e-9)
        concordant += float(conc.sum()) + 0.5 * float(tie.sum())
        n_pairs += int(valid.sum())

    if n_pairs == 0:
        return 0.5, 0
    return concordant / n_pairs, n_pairs


def mae(pred: List[float], target: List[float]) -> float:
    return sum(abs(p - t) for p, t in zip(pred, target)) / len(pred)


def rmse(pred: List[float], target: List[float]) -> float:
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred))


def pearson_r(pred: List[float], target: List[float]) -> float:
    n = len(pred)
    mp, mt = sum(pred) / n, sum(target) / n
    num = sum((p - mp) * (t - mt) for p, t in zip(pred, target))
    dp = math.sqrt(sum((p - mp) ** 2 for p in pred))
    dt = math.sqrt(sum((t - mt) ** 2 for t in target))
    if dp * dt < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (dp * dt)))


def spearman_rho(pred: List[float], target: List[float]) -> float:
    def rank(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    return pearson_r(rank(pred), rank(target))


def pose_metrics(labels: List[int], p_good: List[float], threshold: float = 0.5) -> dict:
    tp = sum(1 for l, p in zip(labels, p_good) if l == 1 and p >= threshold)
    fn = sum(1 for l, p in zip(labels, p_good) if l == 1 and p < threshold)
    tn = sum(1 for l, p in zip(labels, p_good) if l == 0 and p < threshold)
    fp = sum(1 for l, p in zip(labels, p_good) if l == 0 and p >= threshold)
    n_pos, n_neg = tp + fn, tn + fp
    recall_pos = tp / n_pos if n_pos else float("nan")
    recall_neg = tn / n_neg if n_neg else float("nan")
    acc = (tp + tn) / len(labels) if labels else float("nan")
    bal_acc = (
        0.5 * (recall_pos + recall_neg)
        if not (math.isnan(recall_pos) or math.isnan(recall_neg))
        else float("nan")
    )
    out = {
        "Accuracy": acc, "Balanced Accuracy": bal_acc,
        "Pose Recall Pos": recall_pos, "Pose Recall Neg": recall_neg,
        "N_pos": n_pos, "N_neg": n_neg,
    }
    try:
        from sklearn.metrics import average_precision_score
        if n_pos and n_neg:
            out["PR AUC"] = float(average_precision_score(labels, p_good))
    except ImportError:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Nguong quyet dinh cho P(good) (mac dinh 0.5 — hanh vi cu). "
             "Sua VD9 phan nguong: dung --threshold_file de tu doc nguong da chon "
             "tren validation (ghi boi tools/run_inference.py) thay vi tu go tay.",
    )
    parser.add_argument(
        "--threshold_file", type=Path, default=None,
        help="Doc nguong tu file phu <predictions>.threshold do tools/run_inference.py "
             "ghi (neu checkpoint co pose_threshold). De trong = dung --threshold.",
    )
    args = parser.parse_args()

    threshold = args.threshold
    if args.threshold_file is not None:
        if args.threshold_file.exists():
            threshold = float(args.threshold_file.read_text().strip())
            print(f"[NGUONG] Doc tu {args.threshold_file}: threshold={threshold}")
        else:
            print(
                f"[CANH BAO] Khong thay {args.threshold_file}, dung --threshold={args.threshold} "
                "(vd checkpoint chua co --valfile nen chua chon nguong).",
            )

    labels, aff_true, p_good, aff_pred = read_predictions(args.predictions)
    print(f"Tong so mau: {len(labels)}")

    good_idx = [i for i, a in enumerate(aff_true) if a > 0]
    print(f"So mau y_aff > 0 (VD2 — dung cho MAE/RMSE/r/rho/C-index): {len(good_idx)}")

    if good_idx:
        pt = [aff_true[i] for i in good_idx]
        pp = [aff_pred[i] for i in good_idx]
        cidx, n_pairs = exact_concordance_index(pp, pt)
        print(f"  MAE          = {mae(pp, pt):.4f}")
        print(f"  RMSE         = {rmse(pp, pt):.4f}")
        print(f"  Pearson R    = {pearson_r(pp, pt):.4f}")
        print(f"  Spearman Rho = {spearman_rho(pp, pt):.4f}")
        print(f"  C-index      = {cidx:.4f}  (dung du {n_pairs:,} cap, khong lay mau)")
    else:
        print("  Khong co mau y_aff > 0 — bo qua cac chi so hoi quy.")

    pm = pose_metrics(labels, p_good, threshold=threshold)
    print(f"\nChi so phan loai cau hinh (toan bo mau, threshold={threshold}):")
    for k, v in pm.items():
        if isinstance(v, float):
            print(f"  {k:<20} = {v:.4f}")
        else:
            print(f"  {k:<20} = {v}")


if __name__ == "__main__":
    main()
