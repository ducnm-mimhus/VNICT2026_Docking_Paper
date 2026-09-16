#!/usr/bin/env python3
"""
Tach tap validation tu tap train theo RECEPTOR (khong tach ngau nhien tung dong)
- dung de sua VD9 (chon checkpoint/early-stopping dang dua tren chinh test set).

Dinh dang 1 dong trong .types (CrossDocked2020 / dockbench):
    label affinity rmsd receptor_path ligand_path [# comment]
Vi du:
    0 -6.05 5.35864 4kqp/4kqp_rec_0.gninatypes 4kqp/4kqp_docked_0.gninatypes # 4.17234

receptor_path luon co dang "<ma_pdb>/<ten_file>.gninatypes" -> lay <ma_pdb> lam
khoa nhom (khong dung tach random tung pose, vi nhieu pose/ligand cung 1 receptor
se ri ri sang ca train va val neu tach ngau nhien tung dong -> van con leak).

Cach dung:
    python3 tools/make_val_split.py \
        --train data/types/ref_uff_train0.types \
        --out_train data/types/ref_uff_train0_split.types \
        --out_val data/types/ref_uff_val0.types \
        --val_frac 0.15 --seed 2026
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def receptor_key(line: str) -> str:
    parts = line.split()
    # cot 3 (0-indexed) la receptor_path theo dinh dang chuan cua dockbench
    receptor_path = parts[3]
    return receptor_path.split("/")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out_train", type=Path, required=True)
    ap.add_argument("--out_val", type=Path, required=True)
    ap.add_argument("--val_frac", type=float, default=0.15, help="Ty le RECEPTOR (khong phai dong) danh cho validation")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    with args.train.open() as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    by_receptor = defaultdict(list)
    bad = 0
    for ln in lines:
        try:
            key = receptor_key(ln)
        except IndexError:
            bad += 1
            continue
        by_receptor[key].append(ln)

    if bad:
        print(f"[CANH BAO] {bad} dong khong parse duoc receptor, da bo qua")

    receptors = sorted(by_receptor.keys())
    rng = random.Random(args.seed)
    rng.shuffle(receptors)

    n_val_receptors = max(1, round(len(receptors) * args.val_frac))
    val_receptors = set(receptors[:n_val_receptors])
    train_receptors = set(receptors[n_val_receptors:])

    train_lines, val_lines = [], []
    for r in train_receptors:
        train_lines.extend(by_receptor[r])
    for r in val_receptors:
        val_lines.extend(by_receptor[r])

    # xao dong trong tung file (khong xao receptor giua 2 file) de tranh mau theo thu tu file goc
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_train.write_text("\n".join(train_lines) + "\n")
    args.out_val.write_text("\n".join(val_lines) + "\n")

    # Kiem tra khong leak: giao 2 tap receptor phai rong
    overlap = train_receptors & val_receptors
    assert not overlap, f"LOI: {len(overlap)} receptor bi lap giua train/val — khong duoc xay ra"

    print("=" * 70)
    print(f"Tong so receptor (protein) trong train goc : {len(receptors)}")
    print(f"  -> train_split : {len(train_receptors)} receptor, {len(train_lines)} dong")
    print(f"  -> val         : {len(val_receptors)} receptor, {len(val_lines)} dong")
    print(f"Giao receptor train/val (phai = 0)          : {len(overlap)}")
    print(f"Da ghi: {args.out_train}")
    print(f"Da ghi: {args.out_val}")
    print("=" * 70)


if __name__ == "__main__":
    main()
