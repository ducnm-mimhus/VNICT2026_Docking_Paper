#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRAIN_FILE = "examples/Francoeur2020/data/types/ref_uff_train0.types"
DEFAULT_TEST_FILE = "examples/Francoeur2020/data/types/ref_uff_test0.types"
DEFAULT_DATA_ROOT = "examples/Francoeur2020/data"
STRUCTURE_SUFFIXES = (
    ".gninatypes",
    ".sdf",
    ".mol2",
    ".pdb",
    ".pdbqt",
    ".types",
)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def clean_token(token: str) -> str:
    return token.strip().strip("\"'")


def looks_like_structure_token(token: str) -> bool:
    token_lower = token.lower()
    return any(token_lower.endswith(suffix) for suffix in STRUCTURE_SUFFIXES)


def candidate_paths(project_root: Path, data_root: Path, token: str) -> list[Path]:
    token = token.replace("\\", "/")
    token_path = Path(token)
    if token_path.is_absolute():
        return [token_path]

    return [
        project_root / token,
        project_root / data_root / token,
        project_root / data_root / "PDBbind2016" / token,
    ]


def resolve_existing(project_root: Path, data_root: Path, token: str) -> Path | None:
    for candidate in candidate_paths(project_root, data_root, token):
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def infer_complex_id(tokens: list[str]) -> str:
    for token in tokens:
        parts = Path(token.replace("\\", "/")).parts
        for part in parts:
            normalized = part.lower()
            if len(normalized) == 4 and normalized.isalnum():
                return normalized
    if tokens:
        return Path(tokens[-1].replace("\\", "/")).parent.name.lower() or "unknown"
    return "unknown"


def parse_types_file(
    types_file: Path,
    split: str,
    project_root: Path,
    data_root: Path,
    label_pos: int,
    affinity_pos: int | None,
    rmsd_pos: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_numeric_labels = 0
    max_structure_refs = 0

    with types_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            tokens = [clean_token(tok) for tok in line.split()]
            numeric_tokens: list[float] = []
            rest_start = 0
            for idx, token in enumerate(tokens):
                if is_float_token(token):
                    numeric_tokens.append(float(token))
                    rest_start = idx + 1
                else:
                    break

            rest_tokens = tokens[rest_start:]
            structure_tokens = [tok for tok in rest_tokens if looks_like_structure_token(tok)]
            resolved = [
                resolve_existing(project_root=project_root, data_root=data_root, token=tok)
                for tok in structure_tokens
            ]

            row: dict[str, Any] = {
                "split": split,
                "source_file": str(types_file),
                "line_no": line_no,
                "raw_line": raw_line.rstrip("\n"),
                "n_numeric_labels": len(numeric_tokens),
                "n_structure_refs": len(structure_tokens),
                "complex_id": infer_complex_id(structure_tokens),
                "all_paths_exist": all(path is not None for path in resolved) if structure_tokens else False,
                "missing_path_count": sum(path is None for path in resolved),
            }

            if label_pos < len(numeric_tokens):
                row["pose_label"] = int(round(numeric_tokens[label_pos]))
            else:
                row["pose_label"] = np.nan

            if affinity_pos is not None and affinity_pos < len(numeric_tokens):
                row["affinity"] = numeric_tokens[affinity_pos]
            else:
                row["affinity"] = np.nan

            if rmsd_pos is not None and rmsd_pos < len(numeric_tokens):
                row["rmsd"] = numeric_tokens[rmsd_pos]
            else:
                row["rmsd"] = np.nan

            for idx, value in enumerate(numeric_tokens):
                row[f"numeric_label_{idx}"] = value
            for idx, token in enumerate(structure_tokens, 1):
                row[f"structure_ref_{idx}"] = token
                row[f"structure_ref_{idx}_exists"] = resolved[idx - 1] is not None
                row[f"structure_ref_{idx}_resolved"] = str(resolved[idx - 1]) if resolved[idx - 1] else ""

            max_numeric_labels = max(max_numeric_labels, len(numeric_tokens))
            max_structure_refs = max(max_structure_refs, len(structure_tokens))
            rows.append(row)

    frame = pd.DataFrame(rows)
    for idx in range(max_numeric_labels):
        col = f"numeric_label_{idx}"
        if col not in frame.columns:
            frame[col] = np.nan
    for idx in range(1, max_structure_refs + 1):
        for suffix, default in [("", ""), ("_exists", False), ("_resolved", "")]:
            col = f"structure_ref_{idx}{suffix}"
            if col not in frame.columns:
                frame[col] = default
    return frame


def describe_numeric(frame: pd.DataFrame, value_col: str, group_col: str = "split") -> pd.DataFrame:
    if value_col not in frame.columns or frame[value_col].dropna().empty:
        return pd.DataFrame()
    return (
        frame.groupby(group_col)[value_col]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
        .rename(columns={"50%": "median"})
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_split_count_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    counts = frame["split"].value_counts().reindex(["train", "test"]).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Number of samples")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].axis("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "split_counts.png", dpi=240)
    plt.close(fig)


def save_pose_distribution_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    if "pose_label" not in frame.columns or frame["pose_label"].dropna().empty:
        return
    table = pd.crosstab(frame["pose_label"], frame["split"]).sort_index()
    ax = table.plot(kind="bar", figsize=(8, 5), color=["#4C78A8", "#F58518"])
    ax.set_xlabel("Pose label")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pose_class_distribution.png", dpi=240)
    plt.close()

    ratio = table.div(table.sum(axis=0), axis=1) * 100.0
    ax = ratio.plot(kind="bar", figsize=(8, 5), color=["#4C78A8", "#F58518"])
    ax.set_xlabel("Pose label")
    ax.set_ylabel("Percent of split")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pose_class_ratio.png", dpi=240)
    plt.close()


def save_histogram_by_split(frame: pd.DataFrame, column: str, out_dir: Path, filename: str, xlabel: str) -> None:
    values = frame[column].dropna() if column in frame.columns else pd.Series(dtype=float)
    if values.empty:
        return

    plt.figure(figsize=(9, 5))
    for split, split_frame in frame.groupby("split"):
        split_values = split_frame[column].dropna()
        if split_values.empty:
            continue
        plt.hist(split_values, bins=50, alpha=0.45, label=split)
    plt.xlabel(xlabel)
    plt.ylabel("Number of samples")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=240)
    plt.close()


def save_boxplot_by_split(frame: pd.DataFrame, column: str, out_dir: Path, filename: str, ylabel: str) -> None:
    if column not in frame.columns or frame[column].dropna().empty:
        return
    splits = [split for split in ["train", "test"] if not frame.loc[frame["split"] == split, column].dropna().empty]
    if not splits:
        return
    data = [frame.loc[frame["split"] == split, column].dropna().to_numpy() for split in splits]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=splits, showfliers=False)
    plt.xlabel("Split")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=240)
    plt.close()


def save_ecdf_by_split(frame: pd.DataFrame, column: str, out_dir: Path, filename: str, xlabel: str) -> None:
    if column not in frame.columns or frame[column].dropna().empty:
        return
    plt.figure(figsize=(8, 5))
    for split, split_frame in frame.groupby("split"):
        values = np.sort(split_frame[column].dropna().to_numpy())
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, y, label=split, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative fraction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=240)
    plt.close()


def save_path_status_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    table = pd.crosstab(frame["split"], frame["all_paths_exist"]).rename(
        columns={False: "missing_or_unresolved", True: "all_resolved"}
    )
    if table.empty:
        return
    ax = table.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#E45756", "#54A24B"])
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "path_resolution_status.png", dpi=240)
    plt.close()


def save_complex_plots(frame: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    if "complex_id" not in frame.columns:
        return
    top_complexes = frame["complex_id"].value_counts().head(top_n)
    if not top_complexes.empty:
        plt.figure(figsize=(10, max(5, top_n * 0.28)))
        top_complexes.sort_values().plot(kind="barh", color="#72B7B2")
        plt.xlabel("Number of samples")
        plt.ylabel("Complex ID")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "top_complexes_by_sample_count.png", dpi=240)
        plt.close()

    by_split = frame.groupby("split")["complex_id"].apply(set).to_dict()
    train_ids = by_split.get("train", set())
    test_ids = by_split.get("test", set())
    overlap = {
        "train_only": len(train_ids - test_ids),
        "test_only": len(test_ids - train_ids),
        "in_both": len(train_ids & test_ids),
    }
    plt.figure(figsize=(8, 5))
    plt.bar(overlap.keys(), overlap.values(), color=["#4C78A8", "#F58518", "#54A24B"])
    plt.xlabel("Complex split membership")
    plt.ylabel("Number of unique complexes")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "complex_overlap_train_test.png", dpi=240)
    plt.close()


def save_numeric_correlation_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = [col for col in frame.columns if col.startswith("numeric_label_")]
    numeric_cols = [col for col in numeric_cols if frame[col].dropna().nunique() > 1]
    if len(numeric_cols) < 2:
        return
    corr = frame[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(1.4 * len(numeric_cols) + 3, 1.2 * len(numeric_cols) + 3))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for row_idx in range(len(numeric_cols)):
        for col_idx in range(len(numeric_cols)):
            ax.text(col_idx, row_idx, f"{corr.iloc[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_label_correlation.png", dpi=240)
    plt.close(fig)


def generate_tables(frame: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(tables_dir / "parsed_types_rows.csv", index=False)

    split_summary = (
        frame.groupby("split")
        .agg(
            samples=("split", "size"),
            unique_complexes=("complex_id", "nunique"),
            missing_path_refs=("missing_path_count", "sum"),
            rows_with_all_paths_resolved=("all_paths_exist", "sum"),
        )
        .reset_index()
    )
    total_samples = split_summary["samples"].sum()
    split_summary["percent"] = split_summary["samples"] / total_samples * 100.0
    split_summary.to_csv(tables_dir / "split_summary.csv", index=False)

    pose_counts = (
        frame.groupby(["split", "pose_label"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values(["split", "pose_label"])
    )
    pose_counts["percent_within_split"] = pose_counts["samples"] / pose_counts.groupby("split")["samples"].transform("sum") * 100.0
    pose_counts.to_csv(tables_dir / "pose_class_distribution.csv", index=False)

    affinity_summary = describe_numeric(frame, "affinity")
    if not affinity_summary.empty:
        affinity_summary.to_csv(tables_dir / "affinity_summary.csv", index=False)

    rmsd_summary = describe_numeric(frame, "rmsd")
    if not rmsd_summary.empty:
        rmsd_summary.to_csv(tables_dir / "rmsd_summary.csv", index=False)

    numeric_summaries = []
    for col in [name for name in frame.columns if name.startswith("numeric_label_")]:
        summary = describe_numeric(frame, col)
        if not summary.empty:
            summary.insert(0, "numeric_label", col)
            numeric_summaries.append(summary)
    if numeric_summaries:
        pd.concat(numeric_summaries, ignore_index=True).to_csv(tables_dir / "numeric_label_summary.csv", index=False)

    path_cols = [col for col in frame.columns if col.startswith("structure_ref_") and col.endswith("_exists")]
    path_rows = []
    for col in path_cols:
        path_rows.append(
            {
                "path_slot": col.replace("_exists", ""),
                "resolved": int(frame[col].fillna(False).sum()),
                "missing": int((~frame[col].fillna(False)).sum()),
            }
        )
    if path_rows:
        pd.DataFrame(path_rows).to_csv(tables_dir / "path_resolution_summary.csv", index=False)

    by_split = frame.groupby("split")["complex_id"].apply(set).to_dict()
    train_ids = by_split.get("train", set())
    test_ids = by_split.get("test", set())
    overlap_frame = pd.DataFrame(
        [
            {"category": "train_only", "unique_complexes": len(train_ids - test_ids)},
            {"category": "test_only", "unique_complexes": len(test_ids - train_ids)},
            {"category": "in_both", "unique_complexes": len(train_ids & test_ids)},
        ]
    )
    overlap_frame.to_csv(tables_dir / "complex_overlap.csv", index=False)

    return {
        "total_samples": int(len(frame)),
        "splits": split_summary.to_dict(orient="records"),
        "pose_labels": pose_counts.to_dict(orient="records"),
        "affinity": affinity_summary.to_dict(orient="records") if not affinity_summary.empty else [],
        "rmsd": rmsd_summary.to_dict(orient="records") if not rmsd_summary.empty else [],
        "unique_complexes_total": int(frame["complex_id"].nunique()),
        "complex_overlap": overlap_frame.to_dict(orient="records"),
        "duplicate_raw_rows": int(frame.duplicated(subset=["raw_line"]).sum()),
        "missing_path_refs": int(frame["missing_path_count"].sum()),
    }


def generate_plots(frame: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_split_count_plot(frame, plots_dir)
    save_pose_distribution_plot(frame, plots_dir)
    save_histogram_by_split(frame, "affinity", plots_dir, "affinity_distribution.png", "Affinity")
    save_boxplot_by_split(frame, "affinity", plots_dir, "affinity_boxplot.png", "Affinity")
    save_ecdf_by_split(frame, "affinity", plots_dir, "affinity_ecdf.png", "Affinity")
    save_histogram_by_split(frame, "rmsd", plots_dir, "rmsd_distribution.png", "RMSD")
    save_boxplot_by_split(frame, "rmsd", plots_dir, "rmsd_boxplot.png", "RMSD")
    save_path_status_plot(frame, plots_dir)
    save_complex_plots(frame, plots_dir, top_n=top_n)
    save_numeric_correlation_plot(frame, plots_dir)


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Create thesis-ready dataset statistics and plots from GNINA/CrossDocked .types files."
    )
    parser.add_argument("--project-root", default=str(project_root), help="Project root used to resolve paths.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Training .types file.")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Test .types file.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Dataset root for relative structure paths.")
    parser.add_argument("--out-dir", default="data_plots/outputs", help="Output directory for tables and plots.")
    parser.add_argument("--label-pos", type=int, default=0, help="Numeric label index used as pose label.")
    parser.add_argument("--affinity-pos", type=int, default=1, help="Numeric label index used as affinity.")
    parser.add_argument("--rmsd-pos", type=int, default=-1, help="Numeric label index used as RMSD; set -1 to disable.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top complexes shown in bar plots.")
    parser.add_argument("--no-plots", action="store_true", help="Only export tables and summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    train_file = (project_root / args.train_file).resolve() if not Path(args.train_file).is_absolute() else Path(args.train_file)
    test_file = (project_root / args.test_file).resolve() if not Path(args.test_file).is_absolute() else Path(args.test_file)

    missing_inputs = [path for path in [train_file, test_file] if not path.exists()]
    if missing_inputs:
        print("[ERROR] Missing input .types file(s):")
        for path in missing_inputs:
            print(f"  - {path}")
        print("Download the dataset first: cd examples/Francoeur2020 && bash 00_download.sh")
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rmsd_pos = args.rmsd_pos if args.rmsd_pos >= 0 else None
    affinity_pos = args.affinity_pos if args.affinity_pos >= 0 else None
    frames = [
        parse_types_file(train_file, "train", project_root, data_root, args.label_pos, affinity_pos, rmsd_pos),
        parse_types_file(test_file, "test", project_root, data_root, args.label_pos, affinity_pos, rmsd_pos),
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        print("[ERROR] No rows were parsed from the input .types files.")
        return 3

    summary = generate_tables(frame, out_dir)
    summary["inputs"] = {
        "train_file": str(train_file),
        "test_file": str(test_file),
        "project_root": str(project_root),
        "data_root": str(project_root / data_root),
        "label_pos": args.label_pos,
        "affinity_pos": affinity_pos,
        "rmsd_pos": rmsd_pos,
    }
    save_json(out_dir / "summary.json", summary)

    if not args.no_plots:
        generate_plots(frame, out_dir, top_n=args.top_n)

    print(f"[OK] Wrote dataset statistics to: {out_dir}")
    print(f"     Tables: {out_dir / 'tables'}")
    if not args.no_plots:
        print(f"     Plots : {out_dir / 'plots'}")
    print(f"     Summary JSON: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
