#!/usr/bin/env python3
"""Plot train/test metric curves for the 7-model benchmark (fixed folder names)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dockbench.models import BENCHMARK_MODELS, LEGACY_ALIASES

METRIC_SPECS = [
    ("Affinity Loss", "affinity_loss"),
    ("Pose Loss", "pose_loss"),
    ("Pose Recall Pos", "pose_recall_pos"),
    ("Pose Recall Neg", "pose_recall_neg"),
    ("Balanced Accuracy", "balanced_accuracy"),
    ("PR AUC", "pr_auc"),
    ("MCC", "mcc"),
    ("MAE", "mae"),
    ("RMSE", "rmse"),
    ("Pearson R", "pearson_r"),
    ("Spearman Rho", "spearman_rho"),
    ("C-index", "c_index"),
    ("EF_1pct", "ef_1pct"),
    ("EF_5pct", "ef_5pct"),
    ("Success_at_1", "success_at_1"),
    ("Success_at_5", "success_at_5"),
    ("Success_at_10", "success_at_10"),
]


def resolve_models(only_model: str) -> list[str]:
    if only_model in {"all", "benchmark", "full", ""}:
        return list(BENCHMARK_MODELS)
    return [LEGACY_ALIASES.get(only_model, only_model)]


def load_metric_frame(model_dir: Path, split: str) -> pd.DataFrame | None:
    path = model_dir / f"training_metrics_{split}.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return frame if "Epoch" in frame.columns else None


def plot_metric(
    metric_name: str,
    metric_slug: str,
    models_dir: Path,
    selected_models: list[str],
    out_dir: Path,
) -> bool:
    plotted = False
    plt.figure(figsize=(10, 6))
    for model_name in selected_models:
        model_dir = models_dir / model_name
        if not model_dir.is_dir():
            continue
        for split, style, width in (("train", "-", 1.5), ("test", "--", 2.0)):
            frame = load_metric_frame(model_dir, split)
            if frame is None or metric_name not in frame.columns:
                continue
            plt.plot(
                frame["Epoch"],
                frame[metric_name],
                linewidth=width,
                linestyle=style,
                label=f"{model_name}_{split}",
            )
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric_slug}.png", dpi=220)
    plt.close()
    return True


def generate_plots(models_dir: Path, out_dir: Path, selected_models: list[str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for metric_name, metric_slug in METRIC_SPECS:
        if plot_metric(metric_name, metric_slug, models_dir, selected_models, out_dir):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=Path, default=Path("results/models"))
    parser.add_argument("--plots_dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--only_model", default="")
    parser.add_argument("--also_geo_only", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    selected = resolve_models(args.only_model.strip())

    n = generate_plots(models_dir, plots_dir, selected)
    print(f"Generated {n} plots at: {plots_dir}")

    if args.also_geo_only and "geoformerdock" in BENCHMARK_MODELS:
        geo_dir = plots_dir / "geoformerdock_only"
        n_geo = generate_plots(models_dir, geo_dir, ["geoformerdock"])
        if n_geo:
            print(f"Generated {n_geo} GeoFormerDock-only plots at: {geo_dir}")


if __name__ == "__main__":
    main()
