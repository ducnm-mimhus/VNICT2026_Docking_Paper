#!/usr/bin/env python3
"""
A2 — chon epoch bang chi so tren TAP TRAIN, doi chung voi cach chon "best-on-test"
hien co trong summary.json.

Boi canh: pipeline train (dockbench/training.py) chon best_model.pt bang early
stopping DUA TREN CHINH TAP TEST (0.5*C-index + 0.5*Balanced Accuracy do tren
test_loader). Day la VD9 trong issues_and_fixes.md — can khai bao trung thuc.

Script nay dua ra MOT phep chon thay the: epoch co composite CAO NHAT tren TAP
TRAIN (khong nhin test), roi tra ve gia tri test TAI DUNG epoch do. Neu thu
hang giua cac mo hinh khong doi giua hai cach chon, ket luan so sanh tuong doi
duoc cung co.

KHONG can data/, KHONG can GPU — chi doc cac file da co san:
    results/models/<model>/training_metrics_train.csv
    results/models/<model>/training_metrics_test.csv

Cach chay:
    python3 tools/select_epoch_by_train.py
    python3 tools/select_epoch_by_train.py --models_dir results/models
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dockbench.models import BENCHMARK_MODELS, MODEL_DISPLAY_NAMES
except ImportError:
    # Cho phep chay ngay ca khi khong import duoc dockbench (vd chua cai torch)
    BENCHMARK_MODELS = (
        "gnina_dense", "gnina_default2018", "pafnucy",
        "potentialnet", "equibind", "tankbind", "geoformerdock",
    )
    MODEL_DISPLAY_NAMES = {}

DISPLAY_COLUMNS = [
    "Balanced Accuracy", "Accuracy", "Pose Recall Pos", "Pose Recall Neg",
    "PR AUC", "MAE", "RMSE", "Pearson R", "Spearman Rho", "C-index",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def composite(row: Dict[str, str], w_cidx: float = 0.5) -> Optional[float]:
    try:
        ci = float(row["C-index"])
        ba = float(row["Balanced Accuracy"])
    except (KeyError, ValueError):
        return None
    return w_cidx * ci + (1.0 - w_cidx) * ba


def select_best_epoch_by_train(train_rows: List[Dict[str, str]], w_cidx: float = 0.5) -> Optional[str]:
    best_epoch, best_score = None, None
    for row in train_rows:
        score = composite(row, w_cidx)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score, best_epoch = score, row["Epoch"]
    return best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=Path, default=Path("results/models"))
    parser.add_argument("--w_cidx", type=float, default=0.5, help="Trong so C-index trong composite (mac dinh 0.5, khop early_stop_composite_w_cidx)")
    parser.add_argument("--out", type=Path, default=Path("results/logs/epoch_selection_audit.tsv"))
    args = parser.parse_args()

    print("=" * 100)
    print("A2 — CHON EPOCH BANG TRAIN COMPOSITE, DOI CHUNG VOI best-on-test HIEN CO")
    print(f"  composite = {args.w_cidx:.2f}*C-index + {1.0 - args.w_cidx:.2f}*Balanced Accuracy")
    print("=" * 100)

    tsv_rows = []

    for model_key in BENCHMARK_MODELS:
        display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        model_dir = args.models_dir / model_key
        train_csv = model_dir / "training_metrics_train.csv"
        test_csv = model_dir / "training_metrics_test.csv"

        print(f"\n--- {display} ({model_key}) ---")

        if not train_csv.is_file() or not test_csv.is_file():
            print(f"  [MISSING] Khong tim thay {train_csv} hoac {test_csv}")
            tsv_rows.append({"model": model_key, "status": "missing_csv"})
            continue

        train_rows = read_csv_rows(train_csv)
        test_rows = read_csv_rows(test_csv)
        test_by_epoch = {r["Epoch"]: r for r in test_rows}

        best_epoch_by_train = select_best_epoch_by_train(train_rows, args.w_cidx)
        if best_epoch_by_train is None or best_epoch_by_train not in test_by_epoch:
            print(f"  [WARN] Khong xac dinh duoc epoch hop le tu train composite")
            tsv_rows.append({"model": model_key, "status": "no_valid_epoch"})
            continue

        test_at_train_best = test_by_epoch[best_epoch_by_train]
        # "best-on-test" hien co = dong CUOI trong CSV (epoch cuoi da train,
        # vi summary.json bi loi khong ghi dung best_epoch — xem VD9/VD11)
        test_final = test_rows[-1]

        print(f"  Epoch chon boi TRAIN composite : {best_epoch_by_train}")
        print(f"  Epoch cuoi (final, trong CSV)  : {test_final['Epoch']}")
        header = f"  {'metric':<20} {'test @ train-best':>18} {'test @ final epoch':>20}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        row = {
            "model": model_key,
            "status": "ok",
            "epoch_train_best": best_epoch_by_train,
            "epoch_final": test_final["Epoch"],
        }
        for col in DISPLAY_COLUMNS:
            v_trainbest = test_at_train_best.get(col)
            v_final = test_final.get(col)
            v1 = float(v_trainbest) if v_trainbest not in (None, "") else None
            v2 = float(v_final) if v_final not in (None, "") else None
            s1 = f"{v1:.4f}" if v1 is not None else "N/A"
            s2 = f"{v2:.4f}" if v2 is not None else "N/A"
            print(f"  {col:<20} {s1:>18} {s2:>20}")
            row[f"train_best_{col}"] = v1
            row[f"final_{col}"] = v2
        tsv_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({k for r in tsv_rows for k in r.keys()})
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(all_keys) + "\n")
        for r in tsv_rows:
            fh.write("\t".join(str(r.get(k, "")) for k in all_keys) + "\n")
    print(f"\nDa ghi: {args.out}")


if __name__ == "__main__":
    main()
