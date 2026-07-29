#!/usr/bin/env python3
"""
Đọc trực tiếp payload đã lưu trong ``best_model.pt`` của từng mô hình để lấy
epoch/metric TỐT NHẤT THỰC SỰ, không đi qua ``summary.json``.

Vì sao cần: ``dockbench/training.py::log_test_results`` thiếu khai báo
``nonlocal best_epoch, best_metrics`` (chỉ có ``nonlocal best_score,
bad_eval_count``), nên hai biến đó không bao giờ được cập nhật ra ngoài hàm
lồng, và ``summary.json`` cuối cùng luôn rơi về ``final_test`` — tức là
metric của EPOCH CUỐI, không phải epoch tốt nhất theo early-stopping.

Ngược lại, ``_save_weights(best_weights_path, ..., metrics_dict=test_evaluator
.state.metrics, ...)`` nhận ``metrics_dict`` làm THAM SỐ trực tiếp tại đúng
thời điểm phát hiện cải thiện, nên payload lưu trong file ``.pt`` không bị
ảnh hưởng bởi lỗi trên — đây là nguồn số liệu đáng tin cậy nhất hiện có.

Script này CHỈ đọc ``payload["epoch"]`` và ``payload["metrics"]`` (dict số
thực + int, không phải tensor) nên:
  - Không cần GPU, không cần ``data/``, không cần ``molgrid``.
  - Không cần khởi tạo lại kiến trúc mô hình (không gọi ``model.load_state_
    dict``), nên chạy được trên máy chỉ có CPU + torch.

Cách chạy
---------
    python tools/inspect_checkpoints.py
    python tools/inspect_checkpoints.py --models_dir results/models
    python tools/inspect_checkpoints.py --checkpoint final_model.pt   # so sánh cả bản cuối
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dockbench.models import BENCHMARK_MODELS, MODEL_DISPLAY_NAMES

# payload["metrics"] key  ->  summary.json "final_*" key tương ứng
METRIC_TO_SUMMARY_KEY = {
    "Balanced Accuracy": "final_bal_acc",
    "Accuracy": "final_acc",
    "Pose Recall Pos": "final_pose_recall_pos",
    "Pose Recall Neg": "final_pose_recall_neg",
    "PR AUC": "final_pr_auc",
    "MCC": "final_mcc",
    "MAE": "final_mae",
    "RMSE": "final_rmse",
    "Pearson R": "final_pearson_r",
    "Spearman Rho": "final_spearman_rho",
    "C-index": "final_c_index",
}

DISPLAY_COLUMNS = [
    "Balanced Accuracy",
    "Accuracy",
    "Pose Recall Pos",
    "Pose Recall Neg",
    "PR AUC",
    "MAE",
    "RMSE",
    "Pearson R",
    "Spearman Rho",
    "C-index",
]


def load_checkpoint_payload(path: Path) -> Optional[Dict[str, Any]]:
    """Load a training checkpoint and return its raw payload dict.

    ``weights_only=False`` is required: the payload mixes tensors
    (model/optimizer state) with plain Python objects (epoch, metrics dict,
    score, target-normalizer state) that newer torch defaults would reject.
    These are checkpoints this project produced itself, so this is safe.
    """
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def consistency_check(metrics: Dict[str, float]) -> Optional[float]:
    """Solve for pi = P(pose_label=1) from Acc = pi*Rpos + (1-pi)*Rneg.

    Returns None if the required fields aren't present or the pair
    (Rpos, Rneg) is degenerate.
    """
    acc = metrics.get("Accuracy")
    rpos = metrics.get("Pose Recall Pos")
    rneg = metrics.get("Pose Recall Neg")
    if acc is None or rpos is None or rneg is None:
        return None
    denom = rpos - rneg
    if abs(denom) < 1e-9:
        return None
    return (acc - rneg) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=Path, default=Path("results/models"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Tên file checkpoint cần đọc trong mỗi thư mục model (mặc định: best_model.pt)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/logs/best_checkpoint_audit.tsv"),
        help="Đường dẫn file TSV để ghi kết quả",
    )
    args = parser.parse_args()

    rows_for_tsv = []
    pis = []

    print("=" * 100)
    print(f"AUDIT CHECKPOINT: {args.checkpoint}  (models_dir={args.models_dir})")
    print("=" * 100)

    for model_key in BENCHMARK_MODELS:
        display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        model_dir = args.models_dir / model_key
        ckpt_path = model_dir / args.checkpoint
        summary_path = model_dir / "summary.json"

        print(f"\n--- {display} ({model_key}) ---")

        payload = load_checkpoint_payload(ckpt_path)
        if payload is None:
            print(f"  [MISSING] Không tìm thấy {ckpt_path}")
            rows_for_tsv.append({"model": model_key, "status": "missing_checkpoint"})
            continue

        true_epoch = payload.get("epoch")
        true_metrics = payload.get("metrics", {}) or {}
        score = payload.get("score")
        early_stop_metric = payload.get("early_stop_metric")

        print(f"  epoch (thật, từ checkpoint) : {true_epoch}")
        print(f"  score composite ({early_stop_metric}) : {score}")

        summary_epoch = None
        summary_metrics: Dict[str, float] = {}
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as fh:
                summ = json.load(fh)
            summary_epoch = summ.get("best_epoch")
            for metric_key, summ_key in METRIC_TO_SUMMARY_KEY.items():
                if summ_key in summ:
                    summary_metrics[metric_key] = summ[summ_key]
        else:
            print(f"  [WARN] Không tìm thấy {summary_path} để so sánh")

        header = f"  {'metric':<20} {'checkpoint (thật)':>18} {'summary.json':>15} {'chenh lech':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        row = {
            "model": model_key,
            "status": "ok",
            "true_epoch": true_epoch,
            "summary_json_best_epoch_field": summary_epoch,
        }
        for col in DISPLAY_COLUMNS:
            true_v = true_metrics.get(col)
            summ_v = summary_metrics.get(col)
            diff = (true_v - summ_v) if (true_v is not None and summ_v is not None) else None
            true_s = f"{true_v:.4f}" if true_v is not None else "N/A"
            summ_s = f"{summ_v:.4f}" if summ_v is not None else "N/A"
            diff_s = f"{diff:+.4f}" if diff is not None else "N/A"
            print(f"  {col:<20} {true_s:>18} {summ_s:>15} {diff_s:>12}")
            row[f"true_{col}"] = true_v
            row[f"summary_{col}"] = summ_v

        pi = consistency_check(true_metrics)
        if pi is not None:
            print(f"\n  Kiem tra Acc = pi*Rpos + (1-pi)*Rneg  ->  pi suy ra = {pi:.4f}")
            print("  (ky vong ~0.1349 tren tap test theo data_plots/outputs/.../summary.json)")
            pis.append((model_key, pi))
            row["pi_implied"] = pi

        rows_for_tsv.append(row)

    if pis:
        print("\n" + "=" * 100)
        print("TONG HOP pi suy ra tu tung mo hinh (phai gan bang nhau neu cung 1 tap test nhat quan):")
        for model_key, pi in pis:
            print(f"  {model_key:<20} pi = {pi:.4f}")
        print("=" * 100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({k for r in rows_for_tsv for k in r.keys()})
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(all_keys) + "\n")
        for r in rows_for_tsv:
            fh.write("\t".join(str(r.get(k, "")) for k in all_keys) + "\n")
    print(f"\nDa ghi: {args.out}")


if __name__ == "__main__":
    main()
