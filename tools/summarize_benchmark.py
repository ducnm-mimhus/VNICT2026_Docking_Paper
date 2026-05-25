#!/usr/bin/env python3
"""Print benchmark comparison table and write results/logs/benchmark_summary.tsv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from code_docking root (PYTHONPATH set by scripts/run_training.sh)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dockbench.models import BENCHMARK_MODELS, LEGACY_ALIASES, MODEL_DISPLAY_NAMES

BENCHMARK = [
    (name, MODEL_DISPLAY_NAMES.get(name, name))
    for name in BENCHMARK_MODELS
]

ALIASES = LEGACY_ALIASES

COLUMNS = [
    ("best_epoch", "best_epoch", None),
    ("params", "params", None),
    ("Rpos", "final_pose_recall_pos", "{:.4f}"),
    ("Rneg", "final_pose_recall_neg", "{:.4f}"),
    ("BalAcc", "final_bal_acc", "{:.4f}"),
    ("PR-AUC", "final_pr_auc", "{:.4f}"),
    ("MCC", "final_mcc", "{:.4f}"),
    ("MAE", "final_mae", "{:.4f}"),
    ("RMSE", "final_rmse", "{:.4f}"),
    ("Pearson", "final_pearson_r", "{:.4f}"),
    ("Spearm.", "final_spearman_rho", "{:.4f}"),
    ("C-idx", "final_c_index", "{:.4f}"),
    ("EF1%", "final_ef_1pct", "{:.4f}"),
    ("EF5%", "final_ef_5pct", "{:.4f}"),
    ("S@5", "final_success_at_5", "{:.4f}"),
]


def fmt_value(key: str, value) -> str:
    if value is None:
        return "N/A"
    if key == "best_epoch":
        return str(int(value))
    if key == "params":
        return f"{int(value):,}"
    for _, json_key, spec in COLUMNS:
        if json_key == key and spec:
            return spec.format(float(value))
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=Path, default=Path("results/models"))
    parser.add_argument("--logs_dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--only_model", default="")
    args = parser.parse_args()

    only = ALIASES.get(args.only_model.strip(), args.only_model.strip())
    rows = BENCHMARK if not only else [e for e in BENCHMARK if e[0] == only]

    header = f"{'Model':<32} | " + " | ".join(f"{h:<7}" for h, _, _ in COLUMNS)
    sep = "=" * len(header)
    lines = ["\t".join(["model"] + [h for h, _, _ in COLUMNS])]

    print(sep)
    print(header)
    print(sep)

    for model_key, display in rows:
        sp = args.models_dir / model_key / "summary.json"
        if not sp.is_file():
            cells = [display] + ["N/A"] * len(COLUMNS)
            print(f"{display:<32} | " + " | ".join(f"{'N/A':<7}" for _ in COLUMNS))
            lines.append("\t".join(cells))
            continue
        with sp.open(encoding="utf-8") as fh:
            data = json.load(fh)
        cells = [display]
        print_parts = []
        for _, jkey, _ in COLUMNS:
            val = data.get(jkey)
            s = fmt_value(jkey, val)
            cells.append(s)
            print_parts.append(f"{s:<7}")
        print(f"{display:<32} | " + " | ".join(print_parts))
        lines.append("\t".join(cells))

    print(sep)

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.logs_dir / "benchmark_summary.tsv"
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote: {tsv}")


if __name__ == "__main__":
    main()
