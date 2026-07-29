#!/usr/bin/env python3
"""
Q4 — chay inference tu mot checkpoint da luu (best_model.pt / final_model.pt / cac
ban ablation B1, B2) tren mot file .types, xuat du doan TUNG MAU ra CSV.

Dung lam dau vao cho:
  - tools/exact_metrics.py       (A1 — C-index dung du cap, khong lay mau ngau nhien)
  - tools/paired_bootstrap.py    (VD4 — CI cho hieu so giua 2 mo hinh)

Tai su dung dung logic dung grid/model nhu dockbench/training.py (setup.py,
dataloaders.py, models/registry.py) de dam bao inference khop chinh xac voi
luc train — khong tu viet lai pipeline rieng.

Yeu cau: data/ (PDBbind2016 + .types) da tai xong, molgrid + torch + ignite
da cai (vd conda env "dockbench" da dung o buoc smoketest).

Vi du chay (mot model):
    python3 tools/run_inference.py \
        --model geoformerdock \
        --checkpoint results/models/geoformerdock/best_model.pt \
        --testfile data/types/ref_uff_test0.types \
        --data_root data \
        --out results/predictions/geoformerdock.csv

Voi checkpoint tu ablation B1 (uncertainty head) — BAT BUOC them --uncertainty,
neu khong kien truc se khong khop voi state_dict da luu:
    python3 tools/run_inference.py \
        --model geoformerdock --uncertainty \
        --checkpoint results/models/geoformerdock_uncertainty/best_model.pt \
        --testfile data/types/ref_uff_test0.types --data_root data \
        --out results/predictions/geoformerdock_uncertainty.csv

Smoke test (khong can data/ day du, dung demo_inference co san trong repo):
    python3 tools/run_inference.py \
        --model geoformerdock \
        --checkpoint results/models/geoformerdock/best_model.pt \
        --testfile demo_inference/types/demo.types \
        --data_root demo_inference \
        --rmsd_pos 2 \
        --batch_size 2 \
        --out /tmp/smoketest_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dockbench import setup, utils
from dockbench.dataloaders import GriddedExamplesLoader
from dockbench.models import build_model, canonical_name
from dockbench.target_normalizer import TargetNormalizer


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Ten model (vd geoformerdock, gnina_dense, ...)")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Duong dan .pt (best_model.pt/final_model.pt)")
    parser.add_argument("--testfile", required=True, type=Path, help="File .types can chay inference")
    parser.add_argument("-d", "--data_root", type=str, default="", help="Thu muc goc chua .gninatypes")
    parser.add_argument("--label_pos", type=int, default=0)
    parser.add_argument("--affinity_pos", type=int, default=1)
    parser.add_argument("--rmsd_pos", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dimension", type=float, default=23.5)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--ligmolcache", type=str, default="")
    parser.add_argument("--recmolcache", type=str, default="")
    parser.add_argument("--no_cache", action="store_false", dest="cache_structures")
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    # GeoFormerDock-only kien truc — PHAI khop voi luc train checkpoint nay
    parser.add_argument("--max_pseudo_atoms", type=int, default=12)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument(
        "--uncertainty", action="store_true",
        help="BAT BUOC neu checkpoint la ban ablation B1 (--geoformer_uncertainty luc train)",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = options(argv)

    device = utils.set_device(args.gpu)
    print(f"  Device: {device}")

    example_provider = setup.setup_example_provider(str(args.testfile), args, training=False)
    grid_maker = setup.setup_grid_maker(args)

    loader = GriddedExamplesLoader(
        example_provider=example_provider,
        grid_maker=grid_maker,
        label_pos=args.label_pos,
        affinity_pos=args.affinity_pos,
        rmsd_pos=args.rmsd_pos,
        random_translation=0.0,
        random_rotation=False,
        device=device,
    )
    print(f"  So mau: {loader.num_examples_tot}, dims: {loader.dims}")

    canonical = canonical_name(args.model)
    geoformer_kwargs: Optional[dict] = None
    if canonical == "geoformerdock":
        geoformer_kwargs = {
            "max_pseudo_atoms": args.max_pseudo_atoms,
            "num_transformer_layers": args.num_transformer_layers,
            "uncertainty": args.uncertainty,
        }

    model = build_model(
        args.model, loader.dims, affinity=True, flex=False, geoformer_kwargs=geoformer_kwargs,
    ).to(device)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    print(f"  Da nap checkpoint: epoch={payload.get('epoch')}, model={payload.get('model')}")

    target_normalizer = None
    norm_state = payload.get("target_normalizer")
    if payload.get("normalize_targets") and norm_state is not None:
        target_normalizer = TargetNormalizer()
        target_normalizer.mean = norm_state["mean"]
        target_normalizer.std = norm_state["std"]
        target_normalizer.fitted = norm_state["fitted"]
        print(f"  Target normalizer: mean={target_normalizer.mean:.4f}, std={target_normalizer.std:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "label", "affinity_true", "p_good", "affinity_pred"])

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    grids, labels, affinities, _rmsd = batch
                else:
                    grids, labels, affinities = batch

                pose_log, affinities_pred = model(grids)
                if target_normalizer is not None and target_normalizer.fitted:
                    affinities_pred = target_normalizer.denormalize(affinities_pred)

                p_good = torch.exp(pose_log)[:, 1]

                labels_np = labels.detach().cpu().tolist()
                aff_true_np = affinities.detach().cpu().tolist()
                p_good_np = p_good.detach().cpu().tolist()
                aff_pred_np = affinities_pred.detach().cpu().tolist()

                for lbl, aff_t, pg, aff_p in zip(labels_np, aff_true_np, p_good_np, aff_pred_np):
                    writer.writerow([n_written, lbl, aff_t, pg, aff_p])
                    n_written += 1

    print(f"  Da ghi {n_written} du doan vao: {args.out}")


if __name__ == "__main__":
    main()
