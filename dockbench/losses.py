"""
GeoFormerDock Loss v2 (SOTA + Stable).

Affinity core:
  L_aff = L_reg + 0.05*L_rank + 0.02*L_dist + 0.01*L_anchor

Pose auxiliary:
  L_pose = focal

Total:
  L_total = L_aff + 0.3 * L_pose
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _resolve_mask(target: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return a boolean mask. Uses ``mask`` if given, else ``target > 0``."""
    if mask is not None:
        return mask.bool()
    return target > 0


# ============================================================================
# Pose classification losses (train + eval)
# ============================================================================

class LabelSmoothingPoseLoss(nn.Module):
    """
    Pose classification loss with label smoothing and confidence penalty.

    L = (1 - ε) · NLL + ε · mean(-input) + β · KL(p || uniform)
    """

    def __init__(self, scale: float = 1.0, smoothing: float = 0.1,
                 confidence_penalty: float = 0.1):
        super().__init__()
        self.scale = scale
        self.smoothing = smoothing
        self.confidence_penalty = confidence_penalty
        self.nll = nn.NLLLoss(reduction="mean")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n_classes = input.size(1)
        nll_loss = self.nll(input, target)
        smooth_loss = -input.mean()
        p = torch.exp(input)
        kl_uniform = (p * input).sum(dim=1).mean() + math.log(n_classes)
        total = ((1.0 - self.smoothing) * nll_loss
                 + self.smoothing * smooth_loss
                 + self.confidence_penalty * kl_uniform)
        return self.scale * total


class ScaledNLLLoss(nn.Module):
    """Standard NLL loss with scale factor — used for evaluation metrics."""

    def __init__(self, scale: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.scale = scale
        self.loss = nn.NLLLoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.scale * self.loss(input, target)


class PoseFocalLoss(nn.Module):
    """
    Focal loss for pose classification from log-probabilities.

    The model is expected to pass log-softmax outputs:
        log_p = log_softmax(o_pose)

    For each sample:
        p_t = exp(log_p[y])
        L_NLL = -log_p[y]
        alpha_t = alpha if y == 1 else 1 - alpha
        L = alpha_t * (1 - p_t)^gamma * L_NLL

    Optional ``ignore_mask`` [B] bool: True = skip sample (no gradient).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        scale: float = 1.0,
        class_normalize: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.scale = scale
        self.class_normalize = class_normalize

    def forward(
        self,
        log_p: Tensor,
        target: Tensor,
        ignore_mask: Optional[Tensor] = None,
    ) -> Tensor:
        target = target.long()
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8, max=1.0 - 1e-8)
        nll = -log_pt
        focal = (1.0 - pt).pow(self.gamma) * nll
        alpha_t = torch.where(
            target == 1,
            torch.full_like(focal, self.alpha),
            torch.full_like(focal, 1.0 - self.alpha),
        )
        loss = alpha_t * focal

        if ignore_mask is not None:
            valid = ~ignore_mask.bool()
        else:
            valid = torch.ones_like(target, dtype=torch.bool, device=target.device)
        if valid.sum() == 0:
            return (log_p * 0).sum()

        lv = loss[valid]
        yv = target[valid]

        if not self.class_normalize:
            return self.scale * lv.mean()

        # Per-class normalization: 0.5*(mean_bad + mean_good) on classes present.
        present_losses = []
        for cls in (0, 1):
            m = yv == cls
            if m.any():
                present_losses.append(lv[m].mean())
        if not present_losses:
            return (log_p * 0).sum()
        return self.scale * torch.stack(present_losses).mean()


class PoseNLLLoss(nn.Module):
    """
    Plain NLL on pose log-probabilities — **same objective** as logged ``Pose Loss`` / Ignite metric.

    Use this for training when focal loss tracks poorly vs evaluation (multi-task).
    Optional ``ignore_mask`` matches :class:`PoseFocalLoss`.

    ``log_p_min`` clamps log-probabilities before gather (default -10) to limit loss spikes;
    set to ``None`` to disable.
    """

    def __init__(
        self,
        scale: float = 1.0,
        log_p_min: Optional[float] = -10.0,
        label_smoothing: float = 0.0,
        class_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.scale = scale
        self.log_p_min = log_p_min
        self.label_smoothing = label_smoothing
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float(), persistent=False)
        else:
            self.class_weight = None  # type: ignore[assignment]

    def forward(
        self,
        log_p: Tensor,
        target: Tensor,
        ignore_mask: Optional[Tensor] = None,
    ) -> Tensor:
        target = target.long()
        lp = log_p
        if self.log_p_min is not None:
            lp = lp.clamp(min=self.log_p_min)
        n_classes = lp.size(1)
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            smooth = torch.zeros_like(lp).scatter(1, target.unsqueeze(1), 1.0)
            smooth = smooth * (1.0 - eps) + eps / float(n_classes)
            nll = -(smooth * lp).sum(dim=1)
        else:
            nll = -lp.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.class_weight is not None:
            nll = nll * self.class_weight[target]
        if ignore_mask is not None:
            valid = ~ignore_mask.bool()
            if valid.sum() == 0:
                return (log_p * 0).sum()
            return self.scale * nll[valid].mean()
        return self.scale * nll.mean()


def rmsd_pose_targets_and_ignore(
    rmsd: Tensor,
    low: float = 1.5,
    high: float = 3.0,
) -> Tuple[Tensor, Tensor]:
    """
    Relabel pose from RMSD (Å): good if RMSD < low, bad if RMSD > high, else ignore.

    Returns
    -------
    labels : LongTensor [B]   1 = near-native, 0 = decoy
    ignore : BoolTensor [B]   True = do not supervise pose on this example
    """
    labels = torch.zeros(rmsd.shape[0], dtype=torch.long, device=rmsd.device)
    ignore = torch.zeros(rmsd.shape[0], dtype=torch.bool, device=rmsd.device)
    labels = torch.where(rmsd < low, torch.ones_like(labels), labels)
    labels = torch.where(rmsd > high, torch.zeros_like(labels), labels)
    ignore = (rmsd >= low) & (rmsd <= high)
    return labels, ignore


# ============================================================================
# Loss 1: ConfidenceAwareRegressionLoss — uncertainty-aware robust regression
# ============================================================================

class ConfidenceAwareRegressionLoss(nn.Module):
    r"""L_reg = 0.5*exp(-2*logσ)*(y-ŷ)^2 + logσ, fallback to Huber."""

    def __init__(self, scale: float = 1.0, huber_delta: float = 1.0,
                 log_sigma_min: float = -6.0, log_sigma_max: float = 6.0,
                 reg_weight: float = 0.0):
        super().__init__()
        self.scale = scale
        self.huber_delta = huber_delta
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.reg_weight = reg_weight

    def per_sample_loss(
        self,
        pred: Tensor,
        target: Tensor,
        log_sigma: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Per-example regression loss [B] (0 on invalid rows) and boolean mask ``good`` [B].
        """
        good = _resolve_mask(target, mask)
        B = pred.shape[0]
        device, dtype = pred.device, pred.dtype
        out = torch.zeros(B, device=device, dtype=dtype)

        if good.sum() == 0:
            return out, good

        err = pred - target
        abs_err = err.abs()

        if log_sigma is not None:
            ls = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
            elem = 0.5 * torch.exp(-2.0 * ls) * (err ** 2) + ls + self.reg_weight * (ls ** 2)
        else:
            d = self.huber_delta
            quad = 0.5 * abs_err ** 2
            lin = d * (abs_err - 0.5 * d)
            elem = torch.where(abs_err <= d, quad, lin)

        out = self.scale * elem * good.float()
        return out, good

    def forward(self, pred: Tensor, target: Tensor,
                log_sigma: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        elem, good = self.per_sample_loss(pred, target, log_sigma=log_sigma, mask=mask)
        if good.sum() == 0:
            return (pred * 0).sum()
        return elem[good].mean()


# ============================================================================
# Loss 2: SoftProbabilisticRankingLoss — stable pairwise logistic ranking
# ============================================================================

class SoftProbabilisticRankingLoss(nn.Module):
    r"""L_rank = mean softplus(-sign(y_i-y_j) * clamp((ŷ_i-ŷ_j)/τ, -5, 5))."""

    def __init__(self, temperature: float = 1.0, num_pairs: int = 512, min_diff: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.num_pairs = num_pairs
        self.min_diff = min_diff
        self.scale = scale

    def _sample_pairs(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        n = pred.size(0)
        dev = pred.device
        k = min(max(self.num_pairs * 4, self.num_pairs), n * max(n - 1, 1))
        i_idx = torch.randint(0, n, (k,), device=dev)
        j_idx = torch.randint(0, n, (k,), device=dev)
        valid = (i_idx != j_idx) & ((target[i_idx] - target[j_idx]).abs() > self.min_diff)
        i_idx, j_idx = i_idx[valid][: self.num_pairs], j_idx[valid][: self.num_pairs]
        return i_idx, j_idx

    def forward(self, pred: Tensor, target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        good = _resolve_mask(target, mask)
        if good.sum() < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p, t = pred[good], target[good]
        i, j = self._sample_pairs(p, t)
        if i.size(0) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff = torch.clamp((p[i] - p[j]) / self.temperature, -5.0, 5.0)
        s_ij = torch.sign(t[i] - t[j])
        loss = F.softplus(-s_ij * diff)
        return self.scale * loss.mean()


# ============================================================================
# Loss 3: DistributionAlignmentLoss + Loss 4: Anchor loss
# ============================================================================

class DistributionAlignmentLoss(nn.Module):
    r"""L_dist = |mu_pred-mu_true| + |std_pred-std_true|."""

    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale

    def forward(self, pred: Tensor, target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        good = _resolve_mask(target, mask)
        if good.sum() < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p, t = pred[good], target[good]
        return self.scale * ((p.mean() - t.mean()).abs()
                             + (p.std(unbiased=False) - t.std(unbiased=False)).abs())


class AnchorLoss(nn.Module):
    """L_anchor = mean |pred - median(target)| on valid affinity samples."""

    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        good = _resolve_mask(target, mask)
        if good.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p, t = pred[good], target[good]
        anchor = t.median()
        return self.scale * (p - anchor).abs().mean()


# ============================================================================
# CombinedAffinityLoss — orchestrator for all 5 losses
# ============================================================================

class CombinedAffinityLoss(nn.Module):
    """GeoFormerDock Loss v2 affinity bundle."""

    def __init__(self,
                 # Loss 1: Regression
                 scale_affinity: float = 1.0,
                 huber_delta: float = 1.0,
                 scale_ranking: float = 0.05,
                 ranking_temperature: float = 1.0,
                 ranking_num_pairs: int = 128,
                 hard_neg_fraction: float = 0.0,
                 scale_pose_coupling: float = 0.0,
                 scale_align: float = 0.0,
                 use_gradient_alignment: bool = False,
                 scale_dist: float = 0.02,
                 scale_anchor: float = 0.01,
                 rank_warmup_epochs: int = 0,
                 rank_rampup_epochs: int = 0,
                 full_loss_epoch: int = 0,
                 ):
        super().__init__()

        self.regression = ConfidenceAwareRegressionLoss(
            scale=scale_affinity, huber_delta=huber_delta)
        self.ranking = SoftProbabilisticRankingLoss(
            temperature=ranking_temperature,
            num_pairs=ranking_num_pairs,
            min_diff=0.0,
            scale=scale_ranking)
        self.dist_alignment = DistributionAlignmentLoss(scale=scale_dist)
        self.anchor_loss = AnchorLoss(scale=scale_anchor)
        self._current_epoch = 0  # keep for compatibility

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def forward(self, pred: Tensor, target: Tensor,
                mask: Optional[Tensor] = None,
                log_sigma: Optional[Tensor] = None,
                pose_log: Optional[Tensor] = None,
                labels: Optional[Tensor] = None,
                pose_loss_val: Optional[Tensor] = None,
                model: Optional[nn.Module] = None,
                ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Parameters
        ----------
        pred : Tensor           Predicted affinities [B].
        target : Tensor         Target affinities [B].
        mask : Tensor, optional Boolean mask for valid samples.
        log_sigma : Tensor, optional  Log-sigma from uncertainty head [B].
        pose_log : Tensor, optional   Pose log-softmax [B, 2].
        labels : Tensor, optional     Pose labels [B].
        pose_loss_val : Tensor, optional  Scalar pose loss (for gradient alignment).
        model : nn.Module, optional   Model (for gradient alignment).

        Returns
        -------
        total : Tensor          Scalar total loss.
        info : Dict[str, float] Per-component loss values for logging.
        """

        L_reg_per, good = self.regression.per_sample_loss(
            pred, target, log_sigma=log_sigma, mask=mask)
        if good.sum() == 0:
            L_reg = (pred * 0).sum()
        else:
            L_reg = L_reg_per[good].mean()
        L_rank = self.ranking(pred, target, mask=mask)
        L_dist = self.dist_alignment(pred, target, mask=mask)
        L_anchor = self.anchor_loss(pred, target, mask=mask)
        total = L_reg + L_rank + L_dist + L_anchor

        info = {
            'L_reg': L_reg.item(),
            'L_rank': L_rank.item(),
            'L_dist': L_dist.item(),
            'L_anchor': L_anchor.item(),
            'total': total.item(),
        }
        return total, info
