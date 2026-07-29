#!/usr/bin/env python3
"""Kiem tra gia thiet:

    pose_label == 0  <=>  affinity < 0   (decoy / bad pose -> affinity bi lat dau)
    pose_label == 1  <=>  affinity > 0   (crystal / good pose -> affinity duong)

Co the chay theo 2 che do:

1) Doc tu `parsed_types_rows.csv` (mac dinh, nhanh, da co san):
       python data_plots/check_label_affinity_sign.py

2) Doc truc tiep tu cac file .types:
       python data_plots/check_label_affinity_sign.py --from-types \
           --train-file examples/Francoeur2020/data/types/ref_uff_train0.types \
           --test-file examples/Francoeur2020/data/types/ref_uff_test0.types \
           --label-pos 0 --affinity-pos 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PARSED_CSV = (
    Path(__file__).resolve().parent
    / "outputs"
    / "20260504_165251"
    / "tables"
    / "parsed_types_rows.csv"
)


def load_from_parsed_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file parsed CSV: {csv_path}\n"
            "Hay chay `python data_plots/plot_data_statistics.py` truoc, "
            "hoac dung --from-types de doc truc tiep tu file .types."
        )
    df = pd.read_csv(csv_path, usecols=["split", "pose_label", "affinity", "complex_id"])
    return df


def load_from_types_files(
    train_file: Path | None,
    test_file: Path | None,
    label_pos: int,
    affinity_pos: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    sources = [("train", train_file), ("test", test_file)]
    for split, path in sources:
        if path is None:
            continue
        if not path.exists():
            print(f"[WARN] Bo qua, khong ton tai: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                tokens = line.split()
                numeric: list[float] = []
                for tok in tokens:
                    try:
                        numeric.append(float(tok))
                    except ValueError:
                        break
                if label_pos >= len(numeric) or affinity_pos >= len(numeric):
                    continue
                rows.append(
                    {
                        "split": split,
                        "pose_label": int(round(numeric[label_pos])),
                        "affinity": numeric[affinity_pos],
                        "complex_id": "",
                        "line_no": line_no,
                    }
                )
    return pd.DataFrame(rows)


def sign_bucket(values: pd.Series) -> pd.Series:
    out = pd.Series("zero", index=values.index, dtype=object)
    out[values > 0] = "positive"
    out[values < 0] = "negative"
    return out


def cross_check(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["pose_label", "affinity"]).copy()
    df["pose_label"] = df["pose_label"].astype(int)
    df["aff_sign"] = sign_bucket(df["affinity"])

    print("=" * 72)
    print(f"Tong so dong hop le: {len(df):,}")
    print("Phan bo pose_label:")
    print(df["pose_label"].value_counts().sort_index().to_string())
    print()

    print("Bang cheo pose_label x dau cua affinity:")
    crosstab = pd.crosstab(
        df["pose_label"],
        df["aff_sign"],
        margins=True,
        margins_name="TONG",
    )
    for col in ("negative", "zero", "positive"):
        if col not in crosstab.columns:
            crosstab[col] = 0
    crosstab = crosstab[["negative", "zero", "positive", "TONG"]]
    print(crosstab.to_string())
    print()

    label0 = df[df["pose_label"] == 0]
    label1 = df[df["pose_label"] == 1]

    n0 = len(label0)
    n1 = len(label1)
    n0_neg = int((label0["affinity"] < 0).sum())
    n0_zero = int((label0["affinity"] == 0).sum())
    n0_pos = int((label0["affinity"] > 0).sum())
    n1_pos = int((label1["affinity"] > 0).sum())
    n1_zero = int((label1["affinity"] == 0).sum())
    n1_neg = int((label1["affinity"] < 0).sum())

    def pct(x: int, total: int) -> str:
        return f"{(x / total * 100):.4f}%" if total else "n/a"

    print("--- Gia thiet 1: pose_label == 0 -> affinity < 0 ---")
    print(f"  am     : {n0_neg:>8,} / {n0:,}  ({pct(n0_neg, n0)})")
    print(f"  bang 0 : {n0_zero:>8,} / {n0:,}  ({pct(n0_zero, n0)})")
    print(f"  duong  : {n0_pos:>8,} / {n0:,}  ({pct(n0_pos, n0)})  <-- vi pham gia thiet")
    print()
    print("--- Gia thiet 2: pose_label == 1 -> affinity > 0 ---")
    print(f"  duong  : {n1_pos:>8,} / {n1:,}  ({pct(n1_pos, n1)})")
    print(f"  bang 0 : {n1_zero:>8,} / {n1:,}  ({pct(n1_zero, n1)})")
    print(f"  am     : {n1_neg:>8,} / {n1:,}  ({pct(n1_neg, n1)})  <-- vi pham gia thiet")
    print()

    holds_label0 = (n0_pos == 0) and (n0_zero == 0)
    holds_label1 = (n1_neg == 0) and (n1_zero == 0)

    print("=" * 72)
    if holds_label0 and holds_label1:
        print("KET LUAN: dung tuyet doi - moi label 0 -> aff am, moi label 1 -> aff duong.")
    elif (n0_pos == 0) and (n1_neg == 0):
        print(
            "KET LUAN: dung neu coi affinity == 0 la 'mat nhan'."
            " Khong co dong nao bi dao dau."
        )
    else:
        print("KET LUAN: KHONG dung tuyet doi. Co dong vi pham gia thiet, xem o duoi.")
    print("=" * 72)
    print()

    print("--- Thong ke |affinity| theo pose_label (ky vong: cung phan bo) ---")
    abs_summary = (
        df.assign(abs_aff=df["affinity"].abs())
        .groupby("pose_label")["abs_aff"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .rename(columns={"50%": "median"})
    )
    print(abs_summary.to_string())
    print()

    if "complex_id" in df.columns and df["complex_id"].astype(str).str.len().gt(0).any():
        per_complex = df.groupby(["complex_id", "pose_label"])["affinity"].agg(
            lambda s: s.abs().mean()
        )
        per_complex = per_complex.unstack("pose_label").dropna()
        if not per_complex.empty and 0 in per_complex.columns and 1 in per_complex.columns:
            diff = (per_complex[0] - per_complex[1]).abs()
            n_match = int((diff < 1e-3).sum())
            print(
                f"Trong {len(per_complex):,} complex co ca label 0 va label 1: "
                f"{n_match:,} complex co |aff(label0)| ~= aff(label1) "
                f"(sai khac < 1e-3) -> chung minh la cung 1 gia tri pK bi lat dau."
            )
            print()

    violations = pd.concat(
        [
            label0[label0["affinity"] >= 0].assign(rule="label0_phai_am"),
            label1[label1["affinity"] <= 0].assign(rule="label1_phai_duong"),
        ]
    )
    if violations.empty:
        print("Khong co dong vi pham.")
    else:
        print(f"Co {len(violations):,} dong vi pham. Hien 20 dong dau:")
        cols = [c for c in ("split", "complex_id", "line_no", "pose_label", "affinity", "rule") if c in violations.columns]
        print(violations[cols].head(20).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--parsed-csv",
        type=Path,
        default=DEFAULT_PARSED_CSV,
        help=f"Duong dan parsed_types_rows.csv (mac dinh: {DEFAULT_PARSED_CSV})",
    )
    parser.add_argument(
        "--from-types",
        action="store_true",
        help="Doc truc tiep tu file .types thay vi tu CSV.",
    )
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--label-pos", type=int, default=0)
    parser.add_argument("--affinity-pos", type=int, default=1)
    args = parser.parse_args()

    if args.from_types:
        df = load_from_types_files(
            train_file=args.train_file,
            test_file=args.test_file,
            label_pos=args.label_pos,
            affinity_pos=args.affinity_pos,
        )
        if df.empty:
            print("[ERROR] Khong doc duoc dong nao tu file .types.", file=sys.stderr)
            return 2
    else:
        df = load_from_parsed_csv(args.parsed_csv)
        print(f"Doc tu: {args.parsed_csv}")

    cross_check(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
