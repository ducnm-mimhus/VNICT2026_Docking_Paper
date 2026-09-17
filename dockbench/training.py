"""
DockBench training script (code_docking project).

Multi-task training pipeline:
  1. ConfidenceAwareRegressionLoss (uncertainty-aware robust regression)
  2. SoftProbabilisticRankingLoss (temperature-stabilized logistic ranking)
  3. DistributionAlignmentLoss (mean/std matching)
  4. AnchorLoss (median-anchor stabilization)
  5. Pose focal + label smoothing (auxiliary, weighted by lambda_pose)

Total step: L = L_aff + 0.3 * (lambda_pose * pose_loss_scale * L_pose) after pose warmup.

Training features:
  - SAM (Sharpness-Aware Minimization) optimizer wrapper
  - Mixed-precision training (AMP)
  - Linear warmup + cosine decay LR schedule
  - Target (pK) standardisation
  - Stable ranking + distribution + anchor affinity losses
  - Gradient clipping (default max_norm=5.0)
  - Per-component loss logging for debugging
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from typing import List, Optional, Tuple

import ignite
import molgrid
import numpy as np
import pandas as pd
import torch
from ignite.contrib.handlers.mlflow_logger import MLflowLogger, global_step_from_engine
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, timing
from torch import nn, optim
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = None

from dockbench import metrics, setup, utils
from dockbench.dataloaders import GriddedExamplesLoader
from dockbench.target_normalizer import TargetNormalizer
from dockbench.losses import (
    CombinedAffinityLoss,
    PoseFocalLoss,
    ScaledNLLLoss,
    rmsd_pose_targets_and_ignore,
)
from dockbench.models import GEO_ABLATION_CHOICES, build_model, canonical_name, model_choices


# ============================================================================
# SAM (Sharpness-Aware Minimization) wrapper
# ============================================================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (Foret et al., ICLR 2021).
    Wraps any base optimizer.
    """

    def __init__(self, params, base_optimizer=torch.optim.AdamW,
                 rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        return self.base_optimizer.step(closure)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ============================================================================
# Target normalisation helper
# ============================================================================

def _positive_affinity_chunks(loader, max_batches: Optional[int] = None):
    """CPU tensors of positive affinity values (full loader if max_batches is None)."""
    chunks = []
    for i, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and i >= max_batches:
            break
        if len(batch) >= 3:
            aff = batch[2]
            good = aff > 0
            if good.any():
                chunks.append(aff[good].detach().cpu())
    return chunks


def fit_target_normalizer(
    normalizer: TargetNormalizer,
    loader,
    max_batches: Optional[int] = None,
) -> TargetNormalizer:
    """Estimate μ, σ from positive affinity labels in ``loader``."""
    vals = _positive_affinity_chunks(loader, max_batches)
    if vals:
        all_vals = torch.cat(vals)
        normalizer.mean = all_vals.mean().item()
        normalizer.std = all_vals.std(unbiased=False).item()
        if normalizer.std < 1e-6:
            normalizer.std = 1.0
    normalizer.fitted = True
    return normalizer


# ============================================================================
# CLI options
# ============================================================================

def options(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="DockBench multi-task scoring (code_docking)")

    # Data
    parser.add_argument("trainfile", type=str, help="Training file")
    parser.add_argument("--testfile", type=str, default=None, help="Test file")
    parser.add_argument("--valfile", type=str, default=None,
                        help="Validation file (held out, cluster-split from train). "
                             "If given, checkpoint selection/early stopping use THIS "
                             "instead of --testfile, keeping the test set untouched "
                             "until final reporting. If omitted, behavior is unchanged "
                             "(selection on --testfile, as before).")
    parser.add_argument("-d", "--data_root", type=str, default="",
                        help="Root folder for relative paths in train files")
    # NOTE: --balanced is intentionally removed from this pipeline.
    # We now balance pose classes at minibatch/loss level for leak-safe control.
    parser.add_argument("--no_shuffle", action="store_false",
                        help="Deactivate random shuffling", dest="shuffle")
    parser.add_argument("--label_pos", type=int, default=0)
    parser.add_argument("--affinity_pos", type=int, default=None)
    parser.add_argument(
        "--rmsd_pos",
        type=int,
        default=None,
        help="Optional types column index for RMSD (Å): relabel pose + ignore 1.5–3 Å band",
    )
    parser.add_argument("--flexlabel_pos", type=int, default=None)
    parser.add_argument("--stratify_receptor", action="store_true")
    parser.add_argument("--stratify_pos", type=int, default=1)
    parser.add_argument("--stratify_max", type=float, default=0)
    parser.add_argument("--stratify_min", type=float, default=0)
    parser.add_argument("--stratify_step", type=float, default=0)
    parser.add_argument("--ligmolcache", type=str, default="")
    parser.add_argument("--recmolcache", type=str, default="")
    parser.add_argument("-o", "--out_dir", type=str, default=os.getcwd())
    parser.add_argument("--log_file", type=str, default="training.log")

    # Model
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gnina_dense",
        choices=model_choices(),
        help="Benchmark model id or legacy alias (default2017, default2018)",
    )
    parser.add_argument("--dimension", type=float, default=23.5)
    parser.add_argument("--resolution", type=float, default=0.5)

    # Learning — core
    parser.add_argument("--base_lr", type=float, default=1.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_random_rotation", action="store_false",
                        dest="random_rotation")
    parser.add_argument("--random_translation", type=float, default=2.0)
    parser.add_argument("-i", "--iterations", type=int, default=250000,
                        help="Number of epochs")
    parser.add_argument("--iteration_scheme", type=str, default="small",
                        choices=setup._iteration_schemes.keys())

    # LR schedule
    parser.add_argument("--lr_dynamic", action="store_true",
                        help="Use warmup + cosine decay LR schedule")
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_reduce", type=float, default=0.1)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Gradient & optimisation
    parser.add_argument("--clip_gradients", type=float, default=5.0)
    parser.add_argument("--use_sam", action="store_true",
                        help="Use SAM (Sharpness-Aware Minimization)")
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed-precision training (AMP)")
    parser.add_argument("--use_pose_to_affinity_gate", action="store_true",
                        help="Inject pose feature as one-way gate into affinity branch")
    parser.add_argument("--stop_grad_pose_to_affinity", action="store_true",
                        help="Stop affinity gradient from flowing back into pose branch")
    parser.add_argument("--metric_ema_alpha", type=float, default=0.3,
                        help="EMA smoothing alpha for logged metrics (0 disables)")
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="c_index",
        choices=[
            "c_index",
            "rmse",
            "mae",
            "balanced_accuracy",
            "accuracy",
            "composite_cidx_balacc",
        ],
        help=(
            "Early stopping on test metrics. "
            "composite_cidx_balacc = w*C-index + (1-w)*Balanced Accuracy (see --early_stop_composite_w_cidx)"
        ),
    )
    parser.add_argument(
        "--early_stop_composite_w_cidx",
        type=float,
        default=0.5,
        help="For composite_cidx_balacc only: weight on C-index in [0,1]; rest on Balanced Accuracy",
    )
    parser.add_argument("--early_stop_patience", type=int, default=8,
                        help="Stop if no improvement for N evals")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum change to qualify as improvement")

    # === 5 Losses ===
    # Loss 1: Confidence-aware regression
    parser.add_argument("--scale_affinity_loss", type=float, default=1.0,
                        help="Affinity regression weight (default 1.0)")
    parser.add_argument("--delta_affinity_loss", type=float, default=1.0,
                        help="Huber delta for fallback regression (default 1.0)")
    # Loss 2: Soft probabilistic ranking
    parser.add_argument("--scale_ranking", type=float, default=0.05,
                        help="Ranking loss weight (default 0.05)")
    parser.add_argument("--ranking_num_pairs", type=int, default=256,
                        help="Pairs per batch for ranking (default 256)")
    parser.add_argument("--ranking_temperature", type=float, default=1.0,
                        help="Ranking logistic temperature (default 1.0)")
    parser.add_argument("--hard_neg_fraction", type=float, default=0.3,
                        help="Fraction of hard-negative pairs (default 0.3)")
    parser.add_argument("--rank_warmup_epochs", type=int, default=5,
                        help="Epochs of pure regression before ranking (default 5)")
    parser.add_argument("--rank_rampup_epochs", type=int, default=10,
                        help="Epochs to ramp ranking from 0→full (default 10)")
    # Loss 3: Pose-aware affinity coupling
    parser.add_argument("--scale_pose_coupling", type=float, default=0.5,
                        help="α for pose-aware coupling (default 0.5)")
    parser.add_argument("--lambda_pose", type=float, default=0.3,
                        help="Weight on pose term: L_aff + lambda_pose * (scale * L_pose_focal)")
    parser.add_argument("--pose_total_weight", type=float, default=0.3,
                        help="Total-loss weight for pose term after warmup (default 0.3)")
    parser.add_argument("--aff_total_weight", type=float, default=1.0,
                        help="Total-loss weight for affinity term (default 1.0)")
    parser.add_argument("--pose_warmup_epochs", type=int, default=5,
                        help="Epochs 1..N: train affinity losses only (no pose in total)")
    parser.add_argument("--pose_only_epochs", type=int, default=0,
                        help="Epochs 1..N: optimize pose objective only (stabilize pose branch)")
    parser.add_argument("--pose_loss_scale", type=float, default=0.1,
                        help="Scale focal pose loss before lambda_pose (stability)")
    parser.add_argument("--pose_focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--pose_focal_alpha",
        type=float,
        default=0.87,
        help="Focal alpha for the positive/good-pose class y=1 (default 0.87)",
    )
    parser.add_argument(
        "--pose_class_normalize",
        action="store_true",
        help="Use per-class normalized pose focal loss (0.5*(L_bad+L_good))",
    )
    parser.add_argument(
        "--pose_balance_batch",
        action="store_true",
        help="Balance pose classes inside each train minibatch (good/bad)",
    )
    parser.add_argument(
        "--pose_balance_target_per_class",
        type=int,
        default=0,
        help="If >0, samples per class for pose-balanced minibatch; 0=auto",
    )
    parser.add_argument(
        "--pose_prior_logit_scale",
        type=float,
        default=1.0,
        help="Scale factor for pose prior bias logit init (1.0=full prior logit, 0=disabled effect).",
    )
    parser.add_argument(
        "--disable_pose_prior_init",
        action="store_true",
        help="Disable pose-head bias initialization from empirical class prior.",
    )
    parser.add_argument(
        "--pose_loss_type",
        type=str,
        default="focal",
        choices=["focal"],
        help="Pose training loss is fixed to focal for thesis runs",
    )
    parser.add_argument("--rmsd_good_max", type=float, default=1.5,
                        help="With --rmsd_pos: label good pose if RMSD below this (Å)")
    parser.add_argument("--rmsd_bad_min", type=float, default=3.0,
                        help="With --rmsd_pos: label bad pose if RMSD above this (Å)")
    # Loss 4: Gradient alignment
    parser.add_argument("--use_gradient_alignment", action="store_true",
                        help="Enable gradient alignment loss")
    parser.add_argument("--scale_align", type=float, default=0.1,
                        help="λ4: gradient alignment weight (default 0.1)")
    parser.add_argument("--full_loss_epoch", type=int, default=15,
                        help="Epoch to enable gradient alignment (default 15)")
    # Loss 5: Distribution alignment
    parser.add_argument("--scale_dist_constraint", type=float, default=0.02,
                        help="Distribution alignment weight (default 0.02)")
    parser.add_argument("--scale_anchor_loss", type=float, default=0.01,
                        help="Anchor loss weight (default 0.01)")

    # Target normalisation
    parser.add_argument("--normalize_targets", action="store_true",
                        help="Standardise pK targets (z-score)")

    # GeoFormerDock only
    parser.add_argument(
        "--max_pseudo_atoms",
        type=int,
        default=12,
        help="GeoFormerDock: pseudo-atoms cap for pose geometry (0 = disabled / no geometry tokens; lower = stabler, less VRAM)",
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=2,
        help="GeoFormerDock: pocket transformer depth (default 2)",
    )
    parser.add_argument(
        "--geoformer_uncertainty",
        action="store_true",
        help="GeoFormerDock: use aleatoric uncertainty head (default: off)",
    )
    parser.add_argument(
        "--geo_ablation",
        type=str,
        default="none",
        choices=list(GEO_ABLATION_CHOICES),
        help="GeoFormerDock (B0, docs/revision_plan_reviews.md Bang IV): "
             "none = kien truc day du (mac dinh, giu nguyen 1.594.573 tham so); "
             "no_geometry = bo luong hinh hoc pseudo-atom/RBF o nhanh tu the; "
             "concat_fusion = thay cong hop nhat bang noi + chieu tuyen tinh; "
             "no_key_bias = bo hang cong B_pocket trong attention (khong con thien vi "
             "chu y theo khoa); simple_affinity = bo tokenizer + Transformer o nhanh "
             "ai luc, chi con GAP + MLP.",
    )

    # Flex pose
    parser.add_argument("--scale_flexpose_loss", type=float, default=1.0)

    # Misc
    parser.add_argument("-t", "--test_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--num_checkpoints", type=int, default=1)
    parser.add_argument("--checkpoint_prefix", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument("--no_roc_auc", action="store_false", dest="roc_auc")
    parser.add_argument("--no_cache", action="store_false",
                        dest="cache_structures")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--silent", action="store_true")

    return parser.parse_args(args)


# ============================================================================
# Training steps
# ============================================================================

def _make_pose_balanced_indices(
    labels: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
    target_per_class: int = 0,
) -> Optional[torch.Tensor]:
    """
    Create balanced indices for pose classes from one minibatch.

    Returns None if balancing cannot be applied (e.g., one class absent).
    """
    y = labels.long()
    if ignore_mask is not None:
        valid = ~ignore_mask.bool()
    else:
        valid = torch.ones_like(y, dtype=torch.bool, device=y.device)
    if valid.sum() == 0:
        return None

    idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(1)
    yv = y[idx_valid]
    idx0 = idx_valid[yv == 0]
    idx1 = idx_valid[yv == 1]
    if idx0.numel() == 0 or idx1.numel() == 0:
        return None

    if target_per_class > 0:
        k = target_per_class
    else:
        k = max(idx0.numel(), idx1.numel())

    dev = labels.device
    p0 = torch.randint(0, idx0.numel(), (k,), device=dev)
    p1 = torch.randint(0, idx1.numel(), (k,), device=dev)
    out = torch.cat([idx0[p0], idx1[p1]], dim=0)
    perm = torch.randperm(out.numel(), device=dev)
    return out[perm]

def _train_step_pose(
    trainer: Engine, batch, model: nn.Module, optimizer,
    pose_loss: nn.Module, clip_gradients: float,
) -> float:
    model.train()
    optimizer.zero_grad()
    grids, labels = batch
    pose_log = model(grids)
    loss = pose_loss(pose_log, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
    optimizer.step()
    return loss.item()


def _train_step_pose_and_affinity(
    trainer: Engine, batch, model: nn.Module, optimizer,
    pose_loss: nn.Module, affinity_loss: CombinedAffinityLoss,
    clip_gradients: float,
    scaler=None, use_amp: bool = False,
    use_sam: bool = False,
    target_normalizer=None,
    lambda_pose: float = 0.3,
    pose_total_weight: float = 0.5,
    aff_total_weight: float = 0.5,
    pose_warmup_epochs: int = 5,
    pose_only_epochs: int = 0,
    pose_loss_scale: float = 0.1,
    rmsd_low: float = 1.5,
    rmsd_high: float = 3.0,
    pose_balance_batch: bool = False,
    pose_balance_target_per_class: int = 0,
) -> float:
    """
    Training step: L = L_aff + lambda_pose * (pose_scale * L_pose) after warmup;
    pose uses focal loss; optional RMSD relabel + ignore band.
    """
    model.train()
    optimizer.zero_grad()

    if len(batch) == 4:
        grids, labels, affinities, rmsd_vals = batch
    else:
        grids, labels, affinities = batch
        rmsd_vals = None

    ignore_pose = None
    if rmsd_vals is not None:
        labels, ignore_pose = rmsd_pose_targets_and_ignore(
            rmsd_vals, low=rmsd_low, high=rmsd_high,
        )

    mask = (affinities > 0).float()

    # Normalise targets if requested
    if target_normalizer is not None and target_normalizer.fitted:
        affinities_norm = target_normalizer.normalize(affinities)
    else:
        affinities_norm = affinities

    epoch = trainer.state.epoch
    include_pose = epoch > pose_warmup_epochs
    pose_only_phase = epoch <= max(int(pose_only_epochs), 0)

    def _forward_loss():
        amp_on = use_amp and scaler is not None
        ctx = (
            (autocast(device_type="cuda") if _AMP_DEVICE else autocast())
            if amp_on
            else nullcontext()
        )
        with ctx:
            pose_log, affinities_pred = model(grids)
            log_sigma = (
                model.get_log_sigma() if hasattr(model, "get_log_sigma") else None
            )
            pose_log_for_loss = pose_log
            labels_for_loss = labels
            ignore_for_loss = ignore_pose

            if pose_balance_batch:
                balanced_idx = _make_pose_balanced_indices(
                    labels_for_loss,
                    ignore_mask=ignore_for_loss,
                    target_per_class=pose_balance_target_per_class,
                )
                if balanced_idx is not None:
                    pose_log_for_loss = pose_log_for_loss[balanced_idx]
                    labels_for_loss = labels_for_loss[balanced_idx]
                    ignore_for_loss = None

            L_pose_raw = pose_loss(
                pose_log_for_loss,
                labels_for_loss,
                ignore_mask=ignore_for_loss,
            )
            L_pose = pose_loss_scale * L_pose_raw
            aff_result = affinity_loss(
                affinities_pred, affinities_norm, mask=mask,
                log_sigma=log_sigma,
                pose_log=pose_log, labels=labels,
                pose_loss_val=L_pose if include_pose else None,
                model=model,
            )
            L_aff = aff_result[0] if isinstance(aff_result, tuple) else aff_result
            pose_obj = lambda_pose * L_pose
            if pose_only_phase:
                total = pose_obj
            elif include_pose:
                wp = max(float(pose_total_weight), 0.0)
                wa = max(float(aff_total_weight), 0.0)
                total = wa * L_aff + wp * pose_obj
            else:
                total = L_aff
            return L_pose, L_aff, total

    if use_sam:
        L_pose, L_aff, loss = _forward_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        optimizer.first_step(zero_grad=True)

        L_pose2, L_aff2, loss2 = _forward_loss()
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        optimizer.second_step(zero_grad=True)
        return loss.item()
    if use_amp and scaler is not None:
        L_pose, L_aff, loss = _forward_loss()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    L_pose, L_aff, loss = _forward_loss()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
    optimizer.step()
    return loss.item()


def _train_step_flex(
    trainer: Engine, batch, model: nn.Module, optimizer,
    pose_loss: nn.Module, flexpose_loss: nn.Module,
    clip_gradients: float,
) -> float:
    model.train()
    optimizer.zero_grad()
    grids, labels, flexlabels = batch
    pose_log, flexpose_log = model(grids)
    loss = pose_loss(pose_log, labels) + flexpose_loss(flexpose_log, flexlabels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
    optimizer.step()
    return loss.item()


def _setup_trainer(
    model, optimizer, pose_loss, affinity_loss, flexpose_loss,
    clip_gradients: float,
    scaler=None, use_amp: bool = False,
    use_sam: bool = False,
    target_normalizer=None,
    lambda_pose: float = 0.3,
    pose_total_weight: float = 0.5,
    aff_total_weight: float = 0.5,
    pose_warmup_epochs: int = 5,
    pose_only_epochs: int = 0,
    pose_loss_scale: float = 0.1,
    rmsd_low: float = 1.5,
    rmsd_high: float = 3.0,
    pose_balance_batch: bool = False,
    pose_balance_target_per_class: int = 0,
) -> Engine:
    assert affinity_loss is None or flexpose_loss is None

    if affinity_loss is not None:
        trainer = Engine(
            lambda trainer, batch: _train_step_pose_and_affinity(
                trainer, batch, model, optimizer,
                pose_loss=pose_loss,
                affinity_loss=affinity_loss,
                clip_gradients=clip_gradients,
                scaler=scaler,
                use_amp=use_amp,
                use_sam=use_sam,
                target_normalizer=target_normalizer,
                lambda_pose=lambda_pose,
                pose_total_weight=pose_total_weight,
                aff_total_weight=aff_total_weight,
                pose_warmup_epochs=pose_warmup_epochs,
                pose_only_epochs=pose_only_epochs,
                pose_loss_scale=pose_loss_scale,
                rmsd_low=rmsd_low,
                rmsd_high=rmsd_high,
                pose_balance_batch=pose_balance_batch,
                pose_balance_target_per_class=pose_balance_target_per_class,
            )
        )
    elif flexpose_loss is not None:
        trainer = Engine(
            lambda trainer, batch: _train_step_flex(
                trainer, batch, model, optimizer,
                pose_loss=pose_loss,
                flexpose_loss=flexpose_loss,
                clip_gradients=clip_gradients,
            )
        )
    else:
        trainer = Engine(
            lambda trainer, batch: _train_step_pose(
                trainer, batch, model, optimizer,
                pose_loss=pose_loss,
                clip_gradients=clip_gradients,
            )
        )

    return trainer


# ============================================================================
# Evaluation steps
# ============================================================================

def _evaluation_step_pose_and_affinity(evaluator: Engine, batch, model, target_normalizer=None):
    model.eval()
    with torch.no_grad():
        grids, labels, affinities = batch[0], batch[1], batch[2]
        pose_log, affinities_pred = model(grids)
        # If training used normalized targets, convert predictions back to raw pK
        # so MAE/RMSE/Pearson/Spearman/C-index are reported on the true scale.
        if target_normalizer is not None and target_normalizer.fitted:
            affinities_pred = target_normalizer.denormalize(affinities_pred)
    return {
        "pose_log": pose_log,
        "affinities_pred": affinities_pred,
        "labels": labels,
        "affinities": affinities,
    }


def _evaluation_step_pose(evaluator: Engine, batch, model):
    model.eval()
    with torch.no_grad():
        grids, labels = batch
        pose_log = model(grids)
    return {"pose_log": pose_log, "labels": labels}


def _evaluation_step_flex(evaluator: Engine, batch, model):
    model.eval()
    with torch.no_grad():
        grids, labels, flexlabels = batch
        pose_log, flexpose_log = model(grids)
    return {
        "pose_log": pose_log,
        "flexpose_log": flexpose_log,
        "labels": labels,
        "flexlabels": flexlabels,
    }


def _select_pose_threshold(
    model: nn.Module,
    loader,
    device: torch.device,
    affinity: bool = True,
    n_thresholds: int = 197,
) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Sua VD9 phan nguong (docs/revision_plan_reviews.md QD-2): quet nguong quyet dinh
    tren P(good) de toi da hoa Balanced Accuracy TREN LOADER DUOC TRUYEN VAO. Chi
    duoc goi voi val_loader — KHONG BAO GIO voi test_loader, de tap kiem tra khong
    tham gia vao bat ky quyet dinh nao (dung nguyen tac voi viec chon checkpoint).

    Tra ve (best_threshold, best_balacc, n_pos, n_neg). Neu mot lop hoan toan vang
    mat tren loader (n_pos=0 hoac n_neg=0) thi khong the chon nguong co y nghia —
    tra ve (None, None, n_pos, n_neg), noi goi se giu nguyen mac dinh 0.5.
    """
    model.eval()
    p_good_batches, label_batches = [], []
    with torch.no_grad():
        for batch in loader:
            grids, labels = batch[0], batch[1]
            outputs = model(grids)
            pose_log = outputs[0] if affinity else outputs
            p_good_batches.append(pose_log[:, 1].exp().detach().cpu())
            label_batches.append(labels.detach().cpu())
    p_good = torch.cat(p_good_batches).numpy()
    labels = torch.cat(label_batches).numpy().astype(int)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None, None, n_pos, n_neg

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_t, best_balacc = 0.5, -1.0
    for t in thresholds:
        pred_good = p_good >= t
        recall_pos = float((pred_good & (labels == 1)).sum()) / n_pos
        recall_neg = float((~pred_good & (labels == 0)).sum()) / n_neg
        balacc = 0.5 * (recall_pos + recall_neg)
        if balacc > best_balacc:
            best_balacc = balacc
            best_t = float(t)
    return best_t, best_balacc, n_pos, n_neg


def _setup_evaluator(model, metrics_dict, affinity=False, flex=False, target_normalizer=None):
    assert not (affinity and flex)
    if affinity:
        evaluator = Engine(
            lambda evaluator, batch: _evaluation_step_pose_and_affinity(
                evaluator, batch, model, target_normalizer=target_normalizer
            )
        )
    elif flex:
        evaluator = Engine(
            lambda evaluator, batch: _evaluation_step_flex(
                evaluator, batch, model
            )
        )
    else:
        evaluator = Engine(
            lambda evaluator, batch: _evaluation_step_pose(
                evaluator, batch, model
            )
        )
    for name, metric in metrics_dict.items():
        metric.attach(evaluator, name)
    return evaluator


# ============================================================================
# Debug helpers
# ============================================================================

def _estimate_pose_class_weights(
    loader,
    device: torch.device,
    rmsd_low: float,
    rmsd_high: float,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Inverse-frequency weights [w0, w1] so rare class is upweighted (balanced CE).
    Uses RMSD relabel + ignore band when batch includes RMSD (same as training).
    """
    n0 = n1 = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and i >= max_batches:
            break
        if len(batch) == 4:
            _, labels, _, rmsd_vals = batch
            labels, ignore = rmsd_pose_targets_and_ignore(
                rmsd_vals, low=rmsd_low, high=rmsd_high,
            )
            valid = ~ignore
        elif len(batch) >= 2:
            labels = batch[1]
            valid = torch.ones(
                labels.shape[0], dtype=torch.bool, device=labels.device,
            )
        else:
            continue
        if not valid.any():
            continue
        lb = labels[valid]
        n0 += int((lb == 0).sum().item())
        n1 += int((lb == 1).sum().item())
    total = max(n0 + n1, 1)
    w0 = total / max(2 * n0, 1)
    w1 = total / max(2 * n1, 1)
    return torch.tensor([w0, w1], device=device, dtype=torch.float32)


def _estimate_pose_positive_prior(
    loader,
    rmsd_low: float,
    rmsd_high: float,
    max_batches: Optional[int] = None,
) -> float:
    """Estimate P(y=1) from train loader (with RMSD relabel/ignore if available)."""
    n0 = n1 = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and i >= max_batches:
            break
        if len(batch) == 4:
            _, labels, _, rmsd_vals = batch
            labels, ignore = rmsd_pose_targets_and_ignore(
                rmsd_vals, low=rmsd_low, high=rmsd_high,
            )
            valid = ~ignore
        elif len(batch) >= 2:
            labels = batch[1]
            valid = torch.ones_like(labels, dtype=torch.bool)
        else:
            continue
        if not valid.any():
            continue
        lb = labels[valid]
        n0 += int((lb == 0).sum().item())
        n1 += int((lb == 1).sum().item())
    total = max(n0 + n1, 1)
    return float(n1) / float(total)


def _estimate_pose_label_stats(
    loader,
    rmsd_low: float,
    rmsd_high: float,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Estimate pose-label sample counts from the training loader.

    Returns counts for:
      - total_seen: total rows observed
      - valid: rows used for pose supervision
      - ignored: rows ignored by RMSD gray-zone (if rmsd_pos is used)
      - bad: valid class-0 rows
      - good: valid class-1 rows
    """
    total_seen = valid = ignored = n0 = n1 = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and max_batches > 0 and i >= max_batches:
            break
        if len(batch) == 4:
            _, labels, _, rmsd_vals = batch
            labels, ignore = rmsd_pose_targets_and_ignore(
                rmsd_vals, low=rmsd_low, high=rmsd_high,
            )
            vmask = ~ignore
            ignored += int(ignore.sum().item())
        elif len(batch) >= 2:
            labels = batch[1]
            vmask = torch.ones_like(labels, dtype=torch.bool)
        else:
            continue

        total_seen += int(labels.numel())
        valid += int(vmask.sum().item())
        if vmask.any():
            lb = labels[vmask]
            n0 += int((lb == 0).sum().item())
            n1 += int((lb == 1).sum().item())

    return {
        "total_seen": total_seen,
        "valid": valid,
        "ignored": ignored,
        "bad": n0,
        "good": n1,
    }


def _set_pose_bias_from_prior(
    model: nn.Module,
    pos_prior: float,
    logit_scale: float = 1.0,
) -> bool:
    """
    Initialize final pose logits bias from class prior:
      b = log(pi / (1 - pi))
    """
    pi = max(min(pos_prior, 1.0 - 1e-4), 1e-4)
    logit = math.log(pi / (1.0 - pi)) * float(logit_scale)

    candidates = []
    if hasattr(model, "pose_head") and isinstance(model.pose_head, nn.Sequential):
        for m in reversed(model.pose_head):
            if isinstance(m, nn.Linear) and m.out_features == 2 and m.bias is not None:
                candidates.append(m)
                break
    if hasattr(model, "pose") and isinstance(model.pose, nn.Sequential):
        for m in model.pose:
            if isinstance(m, nn.Linear) and m.out_features == 2 and m.bias is not None:
                candidates.append(m)
                break

    if not candidates:
        return False

    with torch.no_grad():
        for layer in candidates:
            layer.bias.fill_(0.0)
            layer.bias[1] = logit
    return True


def _pose_bias_logit_from_prior(pos_prior: float) -> float:
    pi = max(min(pos_prior, 1.0 - 1e-4), 1e-4)
    return math.log(pi / (1.0 - pi))


def _print_target_stats(loader, outstreams, max_batches: Optional[int] = None):
    vals = _positive_affinity_chunks(loader, max_batches)
    if not vals:
        for s in outstreams:
            print("  [DEBUG] No positive affinity targets found!", file=s, flush=True)
        return
    all_vals = torch.cat(vals)
    pcts = [1, 5, 25, 50, 75, 95, 99]
    pct_vals = np.percentile(all_vals.numpy(), pcts)
    for s in outstreams:
        print(f"  [DEBUG] Affinity target stats (N={len(all_vals)}):", file=s, flush=True)
        print(f"    mean={all_vals.mean():.3f}  std={all_vals.std():.3f}", file=s, flush=True)
        for p, v in zip(pcts, pct_vals):
            print(f"    p{p:02d}={v:.3f}", end="  ", file=s, flush=True)
        print("", file=s, flush=True)


def _print_grad_norms(model, outstreams):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            module_name = name.split('.')[0]
            if module_name not in norms:
                norms[module_name] = []
            norms[module_name].append(param.grad.data.norm(2).item())
    for s in outstreams:
        print("  [DEBUG] Grad norms:", file=s, flush=True)
        for mod, ns in norms.items():
            total = (sum(n ** 2 for n in ns)) ** 0.5
            print(f"    {mod}: {total:.4f}", file=s, flush=True)


# ============================================================================
# Main training function
# ============================================================================

_VOXEL_CNN_MEMORY_HEAVY = frozenset({"gnina_dense", "gnina_default2018", "pafnucy"})


def _maybe_adjust_voxel_cnn_training_args(args, outstreams) -> None:
    """
    3D voxel CNNs (esp. DenseNet concat) use large activation memory vs batch size.
    Keep user-specified batch size unchanged; optionally auto-enable AMP unless
    disabled via GNINA_CNN_AUTO_AMP=0.
    """
    if args.model not in _VOXEL_CNN_MEMORY_HEAVY:
        return
    auto_amp = os.environ.get("GNINA_CNN_AUTO_AMP", "1").strip() != "0"
    if auto_amp and not args.use_amp and torch.cuda.is_available():
        args.use_amp = True
        msg = (
            f"  [VRAM] {args.model}: enabled mixed precision (same as --use_amp) "
            f"for 3D CNN memory (set GNINA_CNN_AUTO_AMP=0 to disable)"
        )
        for s in outstreams:
            print(msg, file=s, flush=True)


def training(args):
    """Main training function with 5-loss pipeline."""
    assert args.affinity_pos is None or args.flexlabel_pos is None

    os.makedirs(args.out_dir, exist_ok=True)

    logfilename = os.path.join(args.out_dir, args.log_file)
    logfile = open(logfilename, "w")
    outstreams = [sys.stdout, logfile] if not args.silent else [logfile]

    _maybe_adjust_voxel_cnn_training_args(args, outstreams)

    mlflogger = MLflowLogger()

    params = vars(args)
    params.update({
        "pytorch": torch.__version__,
        "ignite": ignite.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "None",
    })
    mlflogger.log_params(params)

    utils.print_args(args, "--- DOCKBENCH TRAINING (5-Loss Pipeline) ---", stream=logfile)
    if not args.silent:
        print(f"--- DOCKBENCH TRAINING (5-Loss Pipeline): {args.model} ---", flush=True)

    # Seed
    if args.seed is not None:
        molgrid.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = utils.set_device(args.gpu)

    # ---- Data loaders ----
    train_example_provider = setup.setup_example_provider(
        args.trainfile, args, training=True
    )
    best_epoch = None
    best_metrics = None
    best_val_metrics = None

    if args.testfile is not None:
        test_example_provider = setup.setup_example_provider(
            args.testfile, args, training=False
        )
    if args.valfile is not None:
        val_example_provider = setup.setup_example_provider(
            args.valfile, args, training=False
        )

    grid_maker = setup.setup_grid_maker(args)

    train_loader = GriddedExamplesLoader(
        example_provider=train_example_provider,
        grid_maker=grid_maker,
        label_pos=args.label_pos,
        affinity_pos=args.affinity_pos,
        rmsd_pos=args.rmsd_pos,
        flexlabel_pos=args.flexlabel_pos,
        random_translation=args.random_translation,
        random_rotation=args.random_rotation,
        device=device,
    )

    train_eval_example_provider = setup.setup_example_provider(
        args.trainfile, args, training=True
    )
    train_eval_loader = GriddedExamplesLoader(
        example_provider=train_eval_example_provider,
        grid_maker=grid_maker,
        label_pos=args.label_pos,
        affinity_pos=args.affinity_pos,
        rmsd_pos=args.rmsd_pos,
        flexlabel_pos=args.flexlabel_pos,
        random_translation=0.0,
        random_rotation=False,
        device=device,
    )

    if args.testfile is not None:
        test_loader = GriddedExamplesLoader(
            example_provider=test_example_provider,
            grid_maker=grid_maker,
            label_pos=args.label_pos,
            affinity_pos=args.affinity_pos,
            rmsd_pos=args.rmsd_pos,
            flexlabel_pos=args.flexlabel_pos,
            random_translation=0.0,
            random_rotation=False,
            device=device,
        )
        assert test_loader.dims == train_loader.dims

    if args.valfile is not None:
        val_loader = GriddedExamplesLoader(
            example_provider=val_example_provider,
            grid_maker=grid_maker,
            label_pos=args.label_pos,
            affinity_pos=args.affinity_pos,
            rmsd_pos=args.rmsd_pos,
            flexlabel_pos=args.flexlabel_pos,
            random_translation=0.0,
            random_rotation=False,
            device=device,
        )
        assert val_loader.dims == train_loader.dims

    affinity: bool = args.affinity_pos is not None
    flex: bool = args.flexlabel_pos is not None

    # ---- Debug: print target stats ----
    if affinity:
        for s in outstreams:
            print("\n=== TARGET STATISTICS ===", file=s, flush=True)
        _print_target_stats(train_loader, outstreams)
        pose_stats = _estimate_pose_label_stats(
            train_loader,
            rmsd_low=args.rmsd_good_max,
            rmsd_high=args.rmsd_bad_min,
        )
        for s in outstreams:
            total = pose_stats["total_seen"]
            valid = pose_stats["valid"]
            ignored = pose_stats["ignored"]
            bad = pose_stats["bad"]
            good = pose_stats["good"]
            print(
                f"  [DEBUG] Pose sample stats: total={total}, valid={valid}, "
                f"ignored={ignored}, bad={bad}, good={good}",
                file=s,
                flush=True,
            )
            if valid > 0:
                print(
                    f"          valid ratio: bad={bad/valid:.4f}, good={good/valid:.4f}",
                    file=s,
                    flush=True,
                )

    # ---- Target normalisation ----
    target_normalizer = None
    if affinity and args.normalize_targets:
        target_normalizer = TargetNormalizer()
        fit_target_normalizer(target_normalizer, train_loader)
        for s in outstreams:
            print(f"  [NORM] pK mean={target_normalizer.mean:.3f}, "
                  f"std={target_normalizer.std:.3f}", file=s, flush=True)

    # ---- Model ----
    geoformer_kwargs = None
    if canonical_name(args.model) == "geoformerdock":
        geoformer_kwargs = {
            "max_pseudo_atoms": args.max_pseudo_atoms,
            "num_transformer_layers": args.num_transformer_layers,
            "uncertainty": args.geoformer_uncertainty,
            "geo_ablation": args.geo_ablation,
        }
    model = build_model(
        args.model,
        train_loader.dims,
        affinity=affinity,
        flex=flex,
        geoformer_kwargs=geoformer_kwargs,
    ).to(device)
    if geoformer_kwargs is not None:
        for s in outstreams:
            print(
                f"  GeoFormerDock arch: num_transformer_layers={geoformer_kwargs['num_transformer_layers']}, "
                f"max_pseudo_atoms={geoformer_kwargs['max_pseudo_atoms']}, "
                f"geo_ablation={geoformer_kwargs['geo_ablation']}",
                file=s,
                flush=True,
            )
    if hasattr(model, "set_gradient_routing"):
        model.set_gradient_routing(
            use_pose_to_affinity_gate=args.use_pose_to_affinity_gate,
            stop_grad_pose_to_affinity=args.stop_grad_pose_to_affinity,
        )

    def _kaiming_init(m: nn.Module):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(_kaiming_init)

    # Pose classifier bias initialization from class prior:
    # b_good = log(pi_good / (1 - pi_good)).
    pose_pos_prior = _estimate_pose_positive_prior(
        train_loader,
        rmsd_low=args.rmsd_good_max,
        rmsd_high=args.rmsd_bad_min,
    )
    if args.disable_pose_prior_init:
        pose_bias_set = False
        for s in outstreams:
            print("  Pose prior init: disabled by --disable_pose_prior_init", file=s, flush=True)
    else:
        pose_bias_set = _set_pose_bias_from_prior(
            model,
            pose_pos_prior,
            logit_scale=args.pose_prior_logit_scale,
        )
    for s in outstreams:
        if pose_bias_set:
            print(
                f"  Pose prior init: pi_good={pose_pos_prior:.4f}, "
                f"b_good={_pose_bias_logit_from_prior(pose_pos_prior) * args.pose_prior_logit_scale:.4f} "
                f"(scale={args.pose_prior_logit_scale:.3f})",
                file=s,
                flush=True,
            )
        else:
            print("  Pose prior init: skipped (no final 2-logit pose layer found)", file=s, flush=True)

    calib_frozen = False
    # Re-initialize calibration head
    if hasattr(model, 'aff_scale') and hasattr(model, 'aff_bias'):
        if target_normalizer is not None and target_normalizer.fitted:
            with torch.no_grad():
                model.aff_scale.fill_(1.0)
                model.aff_bias.fill_(0.0)
            # Keep predictions in z-score space; learnable scale/bias would drift to raw pK
            # and break both the loss (vs z-targets) and eval denormalize.
            model.aff_scale.requires_grad_(False)
            model.aff_bias.requires_grad_(False)
            calib_frozen = True
            for s in outstreams:
                print(
                    "  Calibration head: scale=1.0, bias=0.0 (z-space; frozen — no drift)",
                    file=s,
                    flush=True,
                )
        else:
            vals = _positive_affinity_chunks(train_loader, 50)
            if vals:
                all_vals = torch.cat(vals)
                train_mean = all_vals.mean().item()
                train_std = all_vals.std(unbiased=False).item()
                with torch.no_grad():
                    model.aff_scale.fill_(max(train_std, 0.5))
                    model.aff_bias.fill_(train_mean)
                for s in outstreams:
                    print(f"  Calibration head: scale={train_std:.3f}, bias={train_mean:.3f} "
                          f"(from data statistics)", file=s, flush=True)
            else:
                with torch.no_grad():
                    model.aff_scale.fill_(2.0)
                    model.aff_bias.fill_(6.3)
                for s in outstreams:
                    print(f"  Calibration head: scale=2.0, bias=6.3 (fallback)",
                          file=s, flush=True)

    for s in outstreams:
        print(f"  Using eager mode for {args.model}", file=s, flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    for s in outstreams:
        print(f"  Model parameters: {n_params:,}", file=s, flush=True)

    # ---- Optimizer ----
    if args.use_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=args.sam_rho,
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        for s in outstreams:
            print(f"  Optimizer: SAM(AdamW, rho={args.sam_rho})", file=s, flush=True)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        for s in outstreams:
            print(f"  Optimizer: AdamW(lr={args.base_lr}, wd={args.weight_decay})",
                  file=s, flush=True)

    # ---- AMP scaler ----
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    if args.use_amp:
        for s in outstreams:
            print("  Mixed-precision (AMP) enabled", file=s, flush=True)
    if args.use_pose_to_affinity_gate:
        for s in outstreams:
            print("  Pose->Affinity gate enabled", file=s, flush=True)
            print(f"  Stop-grad Pose->Affinity: {args.stop_grad_pose_to_affinity}",
                  file=s, flush=True)

    # ---- Pose classification settings ----
    ENABLE_PR_AUC = True
    ENABLE_POSE_CONFIDENCE_METRICS = True

    # ---- Loss functions (5-loss pipeline) ----
    # Thesis configuration: pose branch is optimized with Focal Loss.
    pose_loss = PoseFocalLoss(
        gamma=args.pose_focal_gamma,
        alpha=args.pose_focal_alpha,
        scale=1.0,
        class_normalize=args.pose_class_normalize,
    ).to(device)
    # Keep evaluation "Pose Loss" on plain NLL for stable, comparable logging.
    pose_loss_eval = ScaledNLLLoss(scale=1.0).to(device)
    for s in outstreams:
        print(
            f"  Loss (Pose): Focal(γ={args.pose_focal_gamma}, "
            f"α_good={args.pose_focal_alpha}, α_bad={1.0 - args.pose_focal_alpha}); "
            f"λ_pose={args.lambda_pose}, pose_scale={args.pose_loss_scale}, "
            f"w_pose={args.pose_total_weight}, w_aff={args.aff_total_weight}, "
            f"warmup_epochs={args.pose_warmup_epochs}, pose_only_epochs={args.pose_only_epochs}, "
            f"class_norm={args.pose_class_normalize}, "
            f"batch_balance={args.pose_balance_batch}",
            file=s,
            flush=True,
        )

    if affinity:
        affinity_loss = CombinedAffinityLoss(
            # Loss 1: Confidence-aware regression
            scale_affinity=args.scale_affinity_loss,
            huber_delta=args.delta_affinity_loss,
            # Loss 2: Soft probabilistic ranking
            scale_ranking=args.scale_ranking,
            ranking_temperature=args.ranking_temperature,
            ranking_num_pairs=args.ranking_num_pairs,
            hard_neg_fraction=args.hard_neg_fraction,
            # Loss 3: Pose-aware coupling
            scale_pose_coupling=args.scale_pose_coupling,
            # Loss 4: Gradient alignment
            use_gradient_alignment=args.use_gradient_alignment,
            scale_align=args.scale_align,
            # Loss 5: Distribution alignment
            scale_dist=args.scale_dist_constraint,
            scale_anchor=args.scale_anchor_loss,
            # Schedule
            rank_warmup_epochs=args.rank_warmup_epochs,
            rank_rampup_epochs=args.rank_rampup_epochs,
            full_loss_epoch=args.full_loss_epoch,
        )
        for s in outstreams:
            print(f"  Loss 1 (Regression): ConfidenceAware(scale={args.scale_affinity_loss})",
                  file=s, flush=True)
            print(f"  Loss 2 (Ranking): SoftProbabilistic(scale={args.scale_ranking}, "
                  f"τ={args.ranking_temperature}, pairs={args.ranking_num_pairs})",
                  file=s, flush=True)
            print(f"  Loss 3 (DistAlign): scale={args.scale_dist_constraint}",
                  file=s, flush=True)
            print(f"  Loss 4 (Anchor): scale={args.scale_anchor_loss}", file=s, flush=True)
    else:
        affinity_loss = None

    flexpose_loss = (
        ScaledNLLLoss(scale=args.scale_flexpose_loss)
        if flex else None
    )

    # ---- Trainer engine ----
    trainer = _setup_trainer(
        model, optimizer,
        pose_loss=pose_loss,
        affinity_loss=affinity_loss,
        flexpose_loss=flexpose_loss,
        clip_gradients=args.clip_gradients,
        scaler=scaler,
        use_amp=args.use_amp,
        use_sam=args.use_sam,
        target_normalizer=target_normalizer,
        lambda_pose=args.lambda_pose,
        pose_total_weight=args.pose_total_weight,
        aff_total_weight=args.aff_total_weight,
        pose_warmup_epochs=args.pose_warmup_epochs,
        pose_only_epochs=args.pose_only_epochs,
        pose_loss_scale=args.pose_loss_scale,
        rmsd_low=args.rmsd_good_max,
        rmsd_high=args.rmsd_bad_min,
        pose_balance_batch=args.pose_balance_batch,
        pose_balance_target_per_class=args.pose_balance_target_per_class,
    )

    mlflogger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_STARTED,
        optimizer=optimizer, param_name="lr",
    )

    allmetrics = metrics.setup_metrics(
        affinity,
        flex,
        pose_loss_eval,
        affinity_loss,
        flexpose_loss,
        args.roc_auc,
        device,
        pr_auc=ENABLE_PR_AUC,
        pose_confidence_metrics=ENABLE_POSE_CONFIDENCE_METRICS,
    )

    metrics_train = defaultdict(list)
    metrics_test = defaultdict(list)
    metrics_val = defaultdict(list)
    metrics_train_ema = {}
    metrics_test_ema = {}

    train_evaluator = _setup_evaluator(
        model, allmetrics, affinity=affinity, flex=flex, target_normalizer=target_normalizer
    )
    test_evaluator = _setup_evaluator(
        model, allmetrics, affinity=affinity, flex=flex, target_normalizer=target_normalizer
    )
    if args.valfile is not None:
        val_evaluator = _setup_evaluator(
            model, allmetrics, affinity=affinity, flex=flex, target_normalizer=target_normalizer
        )

    mlflogger.attach_output_handler(
        train_evaluator, event_name=Events.EPOCH_COMPLETED,
        tag="Train", metric_names=list(allmetrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )
    mlflogger.attach_output_handler(
        test_evaluator, event_name=Events.EPOCH_COMPLETED,
        tag="Test", metric_names=list(allmetrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )

    # ---- LR scheduler: linear warmup + cosine decay ----
    if args.lr_dynamic:
        warmup_epochs = args.warmup_epochs
        total_epochs = args.iterations

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * progress)),
                       args.lr_min / args.base_lr)

        torch_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_weights_path = os.path.join(args.out_dir, "best_model.pt")
    final_weights_path = os.path.join(args.out_dir, "final_model.pt")

    def _save_weights(path: str, epoch: int, metrics_dict=None, score=None):
        normalizer_state = None
        if target_normalizer is not None and target_normalizer.fitted:
            normalizer_state = {
                "mean": float(target_normalizer.mean),
                "std": float(target_normalizer.std),
                "fitted": bool(target_normalizer.fitted),
            }
        payload = {
            "model": args.model,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": (
                {k: float(v) for k, v in metrics_dict.items()}
                if metrics_dict is not None
                else {}
            ),
            "early_stop_metric": args.early_stop_metric,
            "score": float(score) if score is not None else None,
            "target_normalizer": normalizer_state,
            "normalize_targets": bool(args.normalize_targets),
        }
        if args.lr_dynamic:
            payload["scheduler_state_dict"] = torch_scheduler.state_dict()
        torch.save(payload, path)

    # Timer
    elapsed_time = timing.Timer()
    elapsed_time.attach(
        trainer,
        start=Events.STARTED,
        resume=Events.EPOCH_STARTED,
        pause=Events.EPOCH_COMPLETED,
        step=Events.EPOCH_COMPLETED,
    )
    # Shared tqdm state so logging handlers can close bars
    iter_bar = {"bar": None}
    epoch_bar = {"bar": None}

    def _close_iter_bar_if_open():
        bar = iter_bar.get("bar")
        if bar is not None:
            bar.close()
            iter_bar["bar"] = None

    # ---- Progressive loss scheduling: update epoch counter ----
    @trainer.on(Events.EPOCH_STARTED)
    def update_loss_schedule(trainer):
        epoch = trainer.state.epoch
        if affinity_loss is not None and hasattr(affinity_loss, 'set_epoch'):
            affinity_loss.set_epoch(epoch)

    # ---- Debug: print grad norms every N epochs ----
    @trainer.on(Events.EPOCH_COMPLETED(every=max(1, args.test_every)))
    def debug_grad_norms(trainer):
        if trainer.state.epoch <= 3 or trainer.state.epoch % (args.test_every * 5) == 0:
            _print_grad_norms(model, [logfile])

    # ---- Evaluation & logging ----
    @trainer.on(Events.EPOCH_COMPLETED(every=args.test_every))
    def log_training_results(trainer):
        _close_iter_bar_if_open()
        train_evaluator.run(train_eval_loader)

        for outstream in outstreams:
            utils.log_print(
                train_evaluator.state.metrics,
                title="Train Results",
                epoch=trainer.state.epoch,
                epoch_time=trainer.state.times["EPOCH_COMPLETED"],
                elapsed_time=elapsed_time.total,
                stream=outstream,
            )

        mts = train_evaluator.state.metrics
        metrics_train["Epoch"].append(trainer.state.epoch)
        for key, value in mts.items():
            metrics_train[key].append(value)
            if args.metric_ema_alpha > 0:
                prev = metrics_train_ema.get(key, float(value))
                ema_v = args.metric_ema_alpha * float(value) + (1.0 - args.metric_ema_alpha) * prev
                metrics_train_ema[key] = ema_v

        # LR step
        if args.lr_dynamic:
            torch_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    lr: {current_lr:.6f}", file=logfile)
            if not args.silent:
                print(f"    lr: {current_lr:.6f}", flush=True)

        if args.metric_ema_alpha > 0 and metrics_train_ema:
            # BalAcc + PR-AUC + precision/recall; raw Acc is secondary under imbalance.
            pr_v = metrics_train_ema.get("PR AUC", float("nan"))
            rneg_v = metrics_train_ema.get("Pose Recall Neg", float("nan"))
            rpos_v = metrics_train_ema.get("Pose Recall Pos", float("nan"))
            ema_msg = (
                "    train_ema: "
                f"Pose={metrics_train_ema.get('Pose Loss', float('nan')):.4f}, "
                f"BalAcc={metrics_train_ema.get('Balanced Accuracy', float('nan')):.4f}, "
                f"PR={pr_v:.4f}, Rneg={rneg_v:.4f}, Rpos={rpos_v:.4f}, "
                f"MAE={metrics_train_ema.get('MAE', float('nan')):.4f}, "
                f"RMSE={metrics_train_ema.get('RMSE', float('nan')):.4f}, "
                f"r={metrics_train_ema.get('Pearson R', float('nan')):.4f}, "
                f"Cidx={metrics_train_ema.get('C-index', float('nan')):.4f}"
            )
            print(ema_msg, file=logfile, flush=True)
            if not args.silent:
                print(ema_msg, flush=True)

    if args.testfile is not None:
        for s in outstreams:
            if args.early_stop_metric == "composite_cidx_balacc":
                wc = float(args.early_stop_composite_w_cidx)
                wc = max(0.0, min(1.0, wc))
                print(
                    f"  Early stopping: composite = {wc:.3f}*C-index + {1.0 - wc:.3f}*Balanced Accuracy "
                    f"(min_delta={args.early_stop_min_delta}, patience={args.early_stop_patience})",
                    file=s,
                    flush=True,
                )
            else:
                print(
                    f"  Early stopping: metric={args.early_stop_metric} "
                    f"(min_delta={args.early_stop_min_delta}, patience={args.early_stop_patience})",
                    file=s,
                    flush=True,
                )
        best_score = None
        bad_eval_count = 0

        def _extract_score(metrics_dict):
            mname = args.early_stop_metric
            if mname == "composite_cidx_balacc":
                ci = float(metrics_dict.get("C-index", 0.5))
                ba = float(metrics_dict.get("Balanced Accuracy", 0.5))
                w = float(args.early_stop_composite_w_cidx)
                w = max(0.0, min(1.0, w))
                return w * ci + (1.0 - w) * ba
            metric_key = {
                "c_index": "C-index",
                "rmse": "RMSE",
                "mae": "MAE",
                "balanced_accuracy": "Balanced Accuracy",
                "accuracy": "Accuracy",
            }[mname]
            val = float(metrics_dict.get(metric_key, 0.0))
            # For rmse/mae: lower is better -> negate for unified "higher is better"
            return -val if mname in ("rmse", "mae") else val

        @trainer.on(Events.EPOCH_COMPLETED(every=args.test_every))
        def log_test_results(trainer):
            nonlocal best_score, bad_eval_count, best_epoch, best_metrics, best_val_metrics
            _close_iter_bar_if_open()
            test_evaluator.run(test_loader)
            for outstream in outstreams:
                utils.log_print(
                    test_evaluator.state.metrics,
                    title="Test Results",
                    epoch=trainer.state.epoch,
                    stream=outstream,
                )
            metrics_test["Epoch"].append(trainer.state.epoch)
            for key, value in test_evaluator.state.metrics.items():
                metrics_test[key].append(value)
                if args.metric_ema_alpha > 0:
                    prev = metrics_test_ema.get(key, float(value))
                    ema_v = args.metric_ema_alpha * float(value) + (1.0 - args.metric_ema_alpha) * prev
                    metrics_test_ema[key] = ema_v

            if args.metric_ema_alpha > 0 and metrics_test_ema:
                pr_v = metrics_test_ema.get("PR AUC", float("nan"))
                rneg_v = metrics_test_ema.get("Pose Recall Neg", float("nan"))
                rpos_v = metrics_test_ema.get("Pose Recall Pos", float("nan"))
                ema_msg = (
                    "    test_ema: "
                    f"Pose={metrics_test_ema.get('Pose Loss', float('nan')):.4f}, "
                    f"BalAcc={metrics_test_ema.get('Balanced Accuracy', float('nan')):.4f}, "
                    f"PR={pr_v:.4f}, Rneg={rneg_v:.4f}, Rpos={rpos_v:.4f}, "
                    f"MAE={metrics_test_ema.get('MAE', float('nan')):.4f}, "
                    f"RMSE={metrics_test_ema.get('RMSE', float('nan')):.4f}, "
                    f"r={metrics_test_ema.get('Pearson R', float('nan')):.4f}, "
                    f"Cidx={metrics_test_ema.get('C-index', float('nan')):.4f}"
                )
                print(ema_msg, file=logfile, flush=True)
                if not args.silent:
                    print(ema_msg, flush=True)

            # Model selection / early stopping: use --valfile if given (held out from
            # train, cluster-split by receptor) so the test set stays untouched until
            # final reporting. If --valfile is not given, fall back to selecting on
            # --testfile exactly as before (unchanged legacy behavior).
            if args.valfile is not None:
                val_evaluator.run(val_loader)
                for outstream in outstreams:
                    utils.log_print(
                        val_evaluator.state.metrics,
                        title="Val Results",
                        epoch=trainer.state.epoch,
                        stream=outstream,
                    )
                metrics_val["Epoch"].append(trainer.state.epoch)
                for key, value in val_evaluator.state.metrics.items():
                    metrics_val[key].append(value)
                selection_metrics = val_evaluator.state.metrics
            else:
                selection_metrics = test_evaluator.state.metrics

            current_score = _extract_score(selection_metrics)
            if best_score is None or current_score > best_score + args.early_stop_min_delta:
                best_score = current_score
                best_epoch = trainer.state.epoch
                best_metrics = {
                    k: float(v) for k, v in test_evaluator.state.metrics.items()
                }
                if args.valfile is not None:
                    best_val_metrics = {
                        k: float(v) for k, v in val_evaluator.state.metrics.items()
                    }
                _save_weights(
                    best_weights_path,
                    epoch=trainer.state.epoch,
                    metrics_dict=test_evaluator.state.metrics,
                    score=current_score,
                )
                bad_eval_count = 0
            else:
                bad_eval_count += 1

            if bad_eval_count >= args.early_stop_patience:
                msg = (
                    f"Early stopping triggered at epoch {trainer.state.epoch}: "
                    f"no {args.early_stop_metric} improvement for {bad_eval_count} evals"
                )
                print(msg, file=logfile, flush=True)
                if not args.silent:
                    print(msg, flush=True)
                trainer.terminate()

    # ---- Checkpointing ----
    to_save = {"model": model, "optimizer": optimizer}
    checkpoint = Checkpoint(
        to_save,
        os.path.join(args.out_dir, args.checkpoint_dir),
        filename_prefix=args.checkpoint_prefix,
        n_saved=args.num_checkpoints,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=args.checkpoint_every), checkpoint
    )

    # Progress bars intentionally disabled; keep plain epoch logs only.

    # ---- Run training ----
    trainer.run(train_loader, max_epochs=args.iterations)
    _save_weights(
        final_weights_path,
        epoch=trainer.state.epoch,
        metrics_dict=(
            test_evaluator.state.metrics
            if args.testfile and hasattr(test_evaluator.state, 'metrics')
            and test_evaluator.state.metrics
            else (
                train_evaluator.state.metrics
                if hasattr(train_evaluator.state, 'metrics')
                and train_evaluator.state.metrics
                else None
            )
        ),
        score=None,
    )

    # ---- Save results ----
    log_root = os.path.splitext(args.log_file)[0]

    metrics_train_outfile = os.path.join(args.out_dir, f"{log_root}_metrics_train.csv")
    pd.DataFrame(metrics_train).to_csv(
        metrics_train_outfile, float_format="%.5f", index=False,
    )
    mlflogger.log_artifact(metrics_train_outfile)

    if args.testfile is not None:
        metrics_test_outfile = os.path.join(
            args.out_dir, f"{log_root}_metrics_test.csv"
        )
        pd.DataFrame(metrics_test).to_csv(
            metrics_test_outfile, float_format="%.5f", index=False,
        )
        mlflogger.log_artifact(metrics_test_outfile)

    if args.valfile is not None:
        metrics_val_outfile = os.path.join(
            args.out_dir, f"{log_root}_metrics_val.csv"
        )
        pd.DataFrame(metrics_val).to_csv(
            metrics_val_outfile, float_format="%.5f", index=False,
        )
        mlflogger.log_artifact(metrics_val_outfile)

    # ---- A1c (docs/revision_plan_reviews.md QD-2, sua VD9 phan nguong) ----
    # Chon nguong quyet dinh cua bo phan loai tu the TREN VALIDATION, quet BalAcc
    # tai chinh checkpoint TOT NHAT (khong phai epoch cuoi cung). Neu khong co
    # --valfile, khong lam gi ca — nguong mac dinh 0.5 o downstream giu nguyen y
    # het hanh vi cu.
    pose_threshold = None
    pose_threshold_val_balacc = None
    pose_threshold_n_pos = pose_threshold_n_neg = None
    if args.valfile is not None and os.path.exists(best_weights_path):
        _ckpt_for_threshold = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(_ckpt_for_threshold["model_state_dict"])
        (
            pose_threshold, pose_threshold_val_balacc,
            pose_threshold_n_pos, pose_threshold_n_neg,
        ) = _select_pose_threshold(model, val_loader, device, affinity=affinity)
        threshold_msg = (
            f"  [NGUONG] Chon tren validation tai checkpoint tot nhat (epoch {best_epoch}): "
            f"pose_threshold={pose_threshold}, BalAcc(val)={pose_threshold_val_balacc}, "
            f"n_pos(val)={pose_threshold_n_pos}, n_neg(val)={pose_threshold_n_neg}"
        )
        for s in outstreams:
            print(threshold_msg, file=s, flush=True)
        if not args.silent:
            print(threshold_msg, flush=True)
        if pose_threshold is not None:
            _ckpt_for_threshold["pose_threshold"] = pose_threshold
            torch.save(_ckpt_for_threshold, best_weights_path)
        else:
            for s in outstreams:
                print(
                    "  [CANH BAO] Validation chi co 1 lop (n_pos=0 hoac n_neg=0) — "
                    "khong chon duoc nguong co y nghia, giu mac dinh 0.5.",
                    file=s, flush=True,
                )

    # ---- Summary JSON ----
    final_train = (
        {k: float(v) for k, v in train_evaluator.state.metrics.items()}
        if hasattr(train_evaluator.state, 'metrics') and train_evaluator.state.metrics
        else {}
    )
    final_test = (
        {k: float(v) for k, v in test_evaluator.state.metrics.items()}
        if args.testfile and hasattr(test_evaluator.state, 'metrics')
        and test_evaluator.state.metrics
        else {}
    )
    # Use best test metrics for reporting when available; fallback to final metrics.
    best_m = best_metrics if best_metrics is not None else (final_test if final_test else final_train)

    summary = {
        "model": args.model,
        "strategy": "5_loss_pipeline",
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        # Sua VD9 (docs/revision_plan_reviews.md): "val" nghia la checkpoint/early
        # stopping duoc chon tren tap validation doc lap, TEST khong tham gia chon
        # gi ca — chi dung de bao cao cuoi cung. "test" = hanh vi cu (khong co
        # --valfile), giu de phan biet ro cac run cu voi cac run da sua.
        "selection_split": "val" if args.valfile is not None else "test",
        "best_val_score": (
            float(best_score) if (args.valfile is not None and best_score is not None) else None
        ),
        "val_metrics_at_best": best_val_metrics if args.valfile is not None else None,
        "pose_threshold": pose_threshold,
        "pose_threshold_val_balacc": pose_threshold_val_balacc,
        "best_weights": best_weights_path if os.path.exists(best_weights_path) else None,
        "final_weights": final_weights_path if os.path.exists(final_weights_path) else None,
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
        # 5-loss config
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
        # Training config
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
        "pose_pos_prior_estimate": pose_pos_prior,
        "pose_bias_prior_init_applied": pose_bias_set,
        "pose_prior_logit_scale": args.pose_prior_logit_scale,
        "disable_pose_prior_init": args.disable_pose_prior_init,
        "pose_samples_total_seen": pose_stats["total_seen"] if affinity else None,
        "pose_samples_valid": pose_stats["valid"] if affinity else None,
        "pose_samples_ignored": pose_stats["ignored"] if affinity else None,
        "pose_samples_bad": pose_stats["bad"] if affinity else None,
        "pose_samples_good": pose_stats["good"] if affinity else None,
        "use_pose_to_affinity_gate": args.use_pose_to_affinity_gate,
        "stop_grad_pose_to_affinity": args.stop_grad_pose_to_affinity,
        "normalize_targets": args.normalize_targets,
        "affinity_calib_frozen": calib_frozen,
        "clip_gradients": args.clip_gradients,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_composite_w_cidx": (
            args.early_stop_composite_w_cidx
            if args.early_stop_metric == "composite_cidx_balacc"
            else None
        ),
    }
    if canonical_name(args.model) == "geoformerdock":
        summary["max_pseudo_atoms"] = args.max_pseudo_atoms
        summary["num_transformer_layers"] = args.num_transformer_layers
        summary["geo_ablation"] = args.geo_ablation

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logfile.close()
    mlflogger.log_artifact(logfilename)


if __name__ == "__main__":
    args = options()
    training(args)
