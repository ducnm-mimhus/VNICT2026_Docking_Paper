#!/usr/bin/env python3
"""
Hoan tat 1 run bi Kaggle giet giua chung (co best_model.pt nhung KHONG co
summary.json, vi phien "Save & Run All" cham tran thoi gian truoc khi
dockbench.training chay het vong lap/early-stop) — chay lai CHINH XAC 3
buoc CUOI cua dockbench.training.training() tu checkpoint da luu san,
KHONG train lai tu dau (chi vai phut, khong can hang gio):

  1. Chon pose_threshold tren VALIDATION tai checkpoint tot nhat.
  2. Danh gia lai model (weights da load tu checkpoint) tren test/val.
  3. Ghi summary.json + luu lai pose_threshold vao checkpoint — dung KHOP
     schema ma dockbench.training.training() se ghi neu chay het binh
     thuong (xem cuoi ham training() trong dockbench/training.py).

QUAN TRONG: build_train_argv() duoi day PHAI khop NGUYEN VAN voi lenh training
that da chay (vd scripts/run_overnight_valsplit.sh train_one(), hoac cac cell
Track A trong notebooks/) — dac biet cac tham so anh huong kien truc
model/data loader (batch_size, max_pseudo_atoms chi cho geoformerdock,
normalize_targets, seed...). Sai 1 trong so nay se cho metrics/threshold
KHONG khop voi checkpoint da train. Dung --model/--batch_size de chay cho
gnina_dense/gnina_default2018/pafnucy/geoformerdock (--batch_size phai khop
dung muc da THANH CONG trong log training — xem "THANH CONG voi
batch_size=..." trong log, vi co the da fallback 256->128->64 do OOM).

Cach dung (tren may co GPU + torch + molgrid, vd Kaggle):
    cd VNICT2026_Docking_Paper
    python3 tools/finalize_from_checkpoint.py \
        --model geoformerdock --batch_size 256 \
        --checkpoint results/models/geoformerdock_valsplit_s2026/best_model.pt \
        --out_dir results/models/geoformerdock_valsplit_s2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from dockbench import metrics, setup  # noqa: E402
from dockbench.dataloaders import GriddedExamplesLoader  # noqa: E402
from dockbench.losses import CombinedAffinityLoss, ScaledNLLLoss  # noqa: E402
from dockbench.models import build_model, canonical_name  # noqa: E402
from dockbench.target_normalizer import TargetNormalizer  # noqa: E402
from dockbench.training import (  # noqa: E402
    _estimate_pose_label_stats,
    _select_pose_threshold,
    _setup_evaluator,
    fit_target_normalizer,
    options,
)

# Sao chep NGUYEN VAN tu train_one() (scripts/run_overnight_valsplit.sh), seed=2026.
# --max_pseudo_atoms chi co tac dung voi geoformerdock (build_model() bo qua
# geoformer_kwargs cho cac model khac) nhung de nguyen trong argv cho ca 4
# model, giong het cach train_one() luon truyen GEO_ARGS chi khi MODEL=geoformerdock
# — o day don gian hoa bang cach luon truyen, vo hai voi 3 model con lai.
def build_train_argv(model: str, batch_size: int) -> list:
    return [
        "data/types/ref_uff_train0_split.types",
        "--testfile", "data/types/ref_uff_test0.types",
        "--valfile", "data/types/ref_uff_val0.types",
        "-d", "data",
        "-m", model,
        "--label_pos", "0", "--affinity_pos", "1",
        "--base_lr", "0.001", "--weight_decay", "0.01",
        "--batch_size", str(batch_size),
        "--random_translation", "1.0", "--clip_gradients", "5.0",
        "-i", "100",
        "--iteration_scheme", "small",
        "--lr_dynamic", "--warmup_epochs", "2",
        "--test_every", "2", "--checkpoint_every", "100",
        "--no_roc_auc",
        "--scale_affinity_loss", "1.0", "--delta_affinity_loss", "1.0",
        "--scale_ranking", "0.05", "--ranking_temperature", "1.0", "--ranking_num_pairs", "128",
        "--hard_neg_fraction", "0.3",
        "--rank_warmup_epochs", "10", "--rank_rampup_epochs", "15",
        "--scale_pose_coupling", "0.00", "--lambda_pose", "1.2",
        "--pose_warmup_epochs", "0", "--pose_only_epochs", "4",
        "--pose_loss_type", "focal", "--pose_focal_gamma", "2.0", "--pose_focal_alpha", "0.75",
        "--pose_class_normalize", "--pose_balance_batch", "--pose_balance_target_per_class", "32",
        "--pose_prior_logit_scale", "0.25", "--disable_pose_prior_init",
        "--pose_loss_scale", "0.5", "--pose_total_weight", "0.85", "--aff_total_weight", "0.15",
        "--metric_ema_alpha", "0.3",
        "--early_stop_metric", "composite_cidx_balacc", "--early_stop_composite_w_cidx", "0.5",
        "--early_stop_patience", "25", "--early_stop_min_delta", "0.0001",
        "--scale_dist_constraint", "0.02", "--scale_anchor_loss", "0.01",
        "--normalize_targets", "--seed", "2026",
        "--max_pseudo_atoms", "12",
        "--use_amp",
    ]

# Cung gia tri hardcode nhu trong dockbench.training.training() (khong phai
# tham so CLI, xem dong ~1318-1319 cua dockbench/training.py).
ENABLE_PR_AUC = True
ENABLE_POSE_CONFIDENCE_METRICS = True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="best_model.pt bi dang do (co epoch/score, chua co pose_threshold)")
    p.add_argument("--out_dir", required=True, help="Thu muc se ghi summary.json (thuong = thu muc chua checkpoint)")
    p.add_argument("--model", default="geoformerdock",
                    choices=["geoformerdock", "gnina_dense", "gnina_default2018", "pafnucy"])
    p.add_argument("--batch_size", type=int, default=256,
                    help="PHAI khop dung muc da THANH CONG trong log training that (256/128/64)")
    p.add_argument("--device", default="cuda:0")
    cli = p.parse_args()

    args = options(build_train_argv(cli.model, cli.batch_size))
    args.out_dir = cli.out_dir
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    affinity = args.affinity_pos is not None
    flex = args.flexlabel_pos is not None
    assert affinity and not flex, "Ca 4 model Track A deu affinity=True, flex=False - kiem tra lai --affinity_pos/--flexlabel_pos"

    # ---- Data loaders (cung code path voi dockbench.training.training()) ----
    print("Dang doc data (chi de fit target_normalizer + lay dims, KHONG train)...")
    train_example_provider = setup.setup_example_provider(args.trainfile, args, training=True)
    test_example_provider = setup.setup_example_provider(args.testfile, args, training=False)
    val_example_provider = setup.setup_example_provider(args.valfile, args, training=False)
    grid_maker = setup.setup_grid_maker(args)

    def _loader(example_provider):
        return GriddedExamplesLoader(
            example_provider=example_provider,
            grid_maker=grid_maker,
            label_pos=args.label_pos,
            affinity_pos=args.affinity_pos,
            rmsd_pos=args.rmsd_pos,
            flexlabel_pos=args.flexlabel_pos,
            random_translation=0.0,
            random_rotation=False,
            device=device,
        )

    train_loader = _loader(train_example_provider)
    test_loader = _loader(test_example_provider)
    val_loader = _loader(val_example_provider)
    assert test_loader.dims == train_loader.dims == val_loader.dims

    # ---- Target normalizer: fit tren train, giong het luc training that ----
    target_normalizer = None
    if args.normalize_targets:
        target_normalizer = TargetNormalizer()
        fit_target_normalizer(target_normalizer, train_loader)
        print(f"[NORM] pK mean={target_normalizer.mean:.3f}, std={target_normalizer.std:.3f}")

    # ---- Model + load checkpoint ----
    geoformer_kwargs = None
    if canonical_name(args.model) == "geoformerdock":
        geoformer_kwargs = {
            "max_pseudo_atoms": args.max_pseudo_atoms,
            "num_transformer_layers": args.num_transformer_layers,
            "uncertainty": args.geoformer_uncertainty,
            "geo_ablation": args.geo_ablation,
        }
    model = build_model(
        args.model, train_loader.dims, affinity=affinity, flex=flex,
        geoformer_kwargs=geoformer_kwargs,
    ).to(device)

    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    best_epoch = ckpt["epoch"]
    best_score = ckpt.get("score")
    print(f"Da load checkpoint: epoch={best_epoch}, score={best_score}, model={ckpt.get('model')}")

    # ---- Loss objects (chi de metrics.setup_metrics tinh 'Pose Loss'/'Affinity
    # Loss' — khong dung de train, chi de danh gia) ----
    pose_loss_eval = ScaledNLLLoss(scale=1.0).to(device)
    affinity_loss = CombinedAffinityLoss(
        scale_affinity=args.scale_affinity_loss,
        huber_delta=args.delta_affinity_loss,
        scale_ranking=args.scale_ranking,
        ranking_temperature=args.ranking_temperature,
        ranking_num_pairs=args.ranking_num_pairs,
        hard_neg_fraction=args.hard_neg_fraction,
        scale_pose_coupling=args.scale_pose_coupling,
        use_gradient_alignment=args.use_gradient_alignment,
        scale_align=args.scale_align,
        scale_dist=args.scale_dist_constraint,
        scale_anchor=args.scale_anchor_loss,
        rank_warmup_epochs=args.rank_warmup_epochs,
        rank_rampup_epochs=args.rank_rampup_epochs,
        full_loss_epoch=args.full_loss_epoch,
    )

    allmetrics = metrics.setup_metrics(
        affinity, flex, pose_loss_eval, affinity_loss, None,
        args.roc_auc, device,
        pr_auc=ENABLE_PR_AUC,
        pose_confidence_metrics=ENABLE_POSE_CONFIDENCE_METRICS,
    )

    # ---- Danh gia lai tren test + val voi checkpoint da load (khong train) ----
    print("Dang chay test_evaluator...")
    test_evaluator = _setup_evaluator(model, allmetrics, affinity=affinity, flex=flex, target_normalizer=target_normalizer)
    test_evaluator.run(test_loader)
    final_test = {k: float(v) for k, v in test_evaluator.state.metrics.items()}
    print("Test metrics:", json.dumps(final_test, indent=2))

    print("Dang chay val_evaluator...")
    val_evaluator = _setup_evaluator(model, allmetrics, affinity=affinity, flex=flex, target_normalizer=target_normalizer)
    val_evaluator.run(val_loader)
    val_metrics_at_best = {k: float(v) for k, v in val_evaluator.state.metrics.items()}
    print("Val metrics:", json.dumps(val_metrics_at_best, indent=2))

    # ---- A1c: chon pose_threshold tren validation (giong het dockbench.training) ----
    print("Dang chon pose_threshold tren validation...")
    pose_threshold, pose_threshold_val_balacc, pose_threshold_n_pos, pose_threshold_n_neg = (
        _select_pose_threshold(model, val_loader, device, affinity=affinity)
    )
    print(
        f"[NGUONG] pose_threshold={pose_threshold}, BalAcc(val)={pose_threshold_val_balacc}, "
        f"n_pos(val)={pose_threshold_n_pos}, n_neg(val)={pose_threshold_n_neg}"
    )
    if pose_threshold is not None:
        ckpt["pose_threshold"] = pose_threshold
        torch.save(ckpt, cli.checkpoint)
        print(f"Da luu pose_threshold vao {cli.checkpoint}")

    # ---- Pose sample stats (metadata, giong dockbench.training) ----
    pose_stats = _estimate_pose_label_stats(
        train_loader, rmsd_low=args.rmsd_good_max, rmsd_high=args.rmsd_bad_min,
    )

    best_m = final_test

    summary = {
        "model": args.model,
        "strategy": "5_loss_pipeline",
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "selection_split": "val",
        "best_val_score": float(best_score) if best_score is not None else None,
        "val_metrics_at_best": val_metrics_at_best,
        "pose_threshold": pose_threshold,
        "pose_threshold_val_balacc": pose_threshold_val_balacc,
        "best_weights": cli.checkpoint,
        "final_weights": None,
        "final_acc": float(best_m.get("Accuracy", 0.0)),
        "final_bal_acc": float(best_m.get("Balanced Accuracy", 0.0)),
        "final_pose_recall_neg": float(best_m.get("Pose Recall Neg", 0.0)),
        "final_pose_recall_pos": float(best_m.get("Pose Recall Pos", 0.0)),
        "final_pr_auc": float(best_m.get("PR AUC", 0.0)),
        "final_mcc": float(best_m.get("MCC", 0.0)),
        "final_mae": float(best_m.get("MAE", 0.0)),
        "final_rmse": float(best_m.get("RMSE", 0.0)),
        "final_pearson_r": float(best_m.get("Pearson R", 0.0)),
        "final_spearman_rho": float(best_m.get("Spearman Rho", 0.0)),
        "final_c_index": float(best_m.get("C-index", 0.5)),
        "final_ef_1pct": float(best_m.get("EF_1pct", 0.0)),
        "final_ef_5pct": float(best_m.get("EF_5pct", 0.0)),
        "final_success_at_1": float(best_m.get("Success_at_1", 0.0)),
        "final_success_at_5": float(best_m.get("Success_at_5", 0.0)),
        "final_success_at_10": float(best_m.get("Success_at_10", 0.0)),
        "scale_affinity": args.scale_affinity_loss,
        "scale_ranking": args.scale_ranking,
        "scale_pose_coupling": args.scale_pose_coupling,
        "use_gradient_alignment": args.use_gradient_alignment,
        "scale_align": args.scale_align,
        "scale_dist": args.scale_dist_constraint,
        "scale_anchor": args.scale_anchor_loss,
        "rank_warmup_epochs": args.rank_warmup_epochs,
        "rank_rampup_epochs": args.rank_rampup_epochs,
        "full_loss_epoch": args.full_loss_epoch,
        "use_sam": args.use_sam,
        "use_amp": args.use_amp,
        "lambda_pose": args.lambda_pose,
        "pose_total_weight": args.pose_total_weight,
        "aff_total_weight": args.aff_total_weight,
        "pose_warmup_epochs": args.pose_warmup_epochs,
        "pose_only_epochs": args.pose_only_epochs,
        "pose_loss_scale": args.pose_loss_scale,
        "pose_loss_type": args.pose_loss_type,
        "pose_focal_gamma": args.pose_focal_gamma,
        "pose_focal_alpha": args.pose_focal_alpha,
        "pr_auc": ENABLE_PR_AUC,
        "pose_confidence_metrics": ENABLE_POSE_CONFIDENCE_METRICS,
        "balanced": False,
        "pose_class_normalize": args.pose_class_normalize,
        "pose_balance_batch": args.pose_balance_batch,
        "pose_balance_target_per_class": args.pose_balance_target_per_class,
        "pose_pos_prior_estimate": None,
        "pose_bias_prior_init_applied": None,
        "pose_prior_logit_scale": args.pose_prior_logit_scale,
        "disable_pose_prior_init": args.disable_pose_prior_init,
        "pose_samples_total_seen": pose_stats["total_seen"],
        "pose_samples_valid": pose_stats["valid"],
        "pose_samples_ignored": pose_stats["ignored"],
        "pose_samples_bad": pose_stats["bad"],
        "pose_samples_good": pose_stats["good"],
        "use_pose_to_affinity_gate": args.use_pose_to_affinity_gate,
        "stop_grad_pose_to_affinity": args.stop_grad_pose_to_affinity,
        "normalize_targets": args.normalize_targets,
        "affinity_calib_frozen": None,
        "clip_gradients": args.clip_gradients,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_composite_w_cidx": (
            args.early_stop_composite_w_cidx
            if args.early_stop_metric == "composite_cidx_balacc" else None
        ),
        "max_pseudo_atoms": args.max_pseudo_atoms,
        "num_transformer_layers": args.num_transformer_layers,
        "geo_ablation": args.geo_ablation,
        "finalized_from_interrupted_checkpoint": True,
    }

    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== DA GHI {summary_path} ===")


if __name__ == "__main__":
    main()
