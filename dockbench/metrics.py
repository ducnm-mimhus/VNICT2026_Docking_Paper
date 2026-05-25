"""
Evaluation metrics for protein-ligand docking.

This module consolidates all metrics (previously split across
``metrics.py`` and ``advanced_metrics.py``) into a single file.

Provides three layers of API:

1. **Standalone functions** (pure torch / numpy, no Ignite dependency):
       rmse, mae, pearson_r, spearman_rho, concordance_index,
       top_k_success_rate, pose_rmsd

2. **DockingMetrics** accumulator class (batch-wise update → compute):
       Convenient for notebooks / offline evaluation.

3. **Ignite-compatible metrics** (used by ``training.py``):
       PearsonR, SpearmanRho, CIndex  (accumulated across batches)
       + ``setup_metrics()`` factory called by the training loop.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ignite import metrics as igm
from ignite.contrib.metrics import ROC_AUC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from dockbench import transforms


# ====================================================================
# 1. Standalone metric functions (pure torch, no Ignite dependency)
# ====================================================================

def rmse(pred: Tensor, target: Tensor,
         mask: Optional[Tensor] = None) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(mean((pred − target)²))

    Parameters
    ----------
    pred : Tensor [N]
    target : Tensor [N]
    mask : Tensor [N], optional
        Boolean or float mask.  If boolean, used for indexing;
        if float, used as multiplicative weight.

    Returns
    -------
    float
    """
    sq = (pred - target) ** 2
    if mask is not None:
        if mask.dtype == torch.bool:
            sq = sq[mask]
        else:
            sq = sq * mask
            return torch.sqrt(sq.sum() / (mask.sum() + 1e-8)).item()
    if sq.numel() == 0:
        return 0.0
    return torch.sqrt(sq.mean()).item()


def mae(pred: Tensor, target: Tensor,
        mask: Optional[Tensor] = None) -> float:
    """
    Mean Absolute Error.

    MAE = mean(|pred − target|)

    Parameters
    ----------
    pred : Tensor [N]
    target : Tensor [N]
    mask : Tensor [N], optional

    Returns
    -------
    float
    """
    ae = torch.abs(pred - target)
    if mask is not None:
        if mask.dtype == torch.bool:
            ae = ae[mask]
        else:
            ae = ae * mask
            return (ae.sum() / (mask.sum() + 1e-8)).item()
    if ae.numel() == 0:
        return 0.0
    return ae.mean().item()


def pearson_r(pred: Tensor, target: Tensor,
              mask: Optional[Tensor] = None, eps: float = 1e-8) -> float:
    """
    Pearson correlation coefficient.

    Numerically stable: centres first, guards against zero-variance.

    Parameters
    ----------
    pred : Tensor [N]
    target : Tensor [N]
    mask : Tensor [N], optional   Boolean mask.
    eps : float

    Returns
    -------
    float   in [-1, 1]
    """
    if mask is not None:
        if mask.dtype == torch.bool:
            pred, target = pred[mask], target[mask]
        else:
            pred, target = pred[mask.bool()], target[mask.bool()]
    if pred.numel() < 2:
        return 0.0
    p = pred - pred.mean()
    t = target - target.mean()
    num = (p * t).sum()
    den = torch.sqrt((p ** 2).sum() * (t ** 2).sum())
    if den < eps:
        return 0.0
    return torch.clamp(num / den, -1.0, 1.0).item()


# Alias used by advanced_metrics consumers
pearson_correlation = pearson_r


def spearman_rho(pred: Tensor, target: Tensor,
                 mask: Optional[Tensor] = None) -> float:
    """
    Spearman rank correlation (Pearson on ranks).

    Falls back to scipy for correct tie handling.

    Parameters
    ----------
    pred : Tensor [N]
    target : Tensor [N]
    mask : Tensor [N], optional

    Returns
    -------
    float   in [-1, 1]
    """
    if mask is not None:
        if mask.dtype == torch.bool:
            pred, target = pred[mask], target[mask]
        else:
            pred, target = pred[mask.bool()], target[mask.bool()]
    if pred.numel() < 2:
        return 0.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(pred.detach().cpu().numpy(),
                       target.detach().cpu().numpy())
    return 0.0 if np.isnan(rho) else float(rho)


# Alias
spearman_correlation = spearman_rho


def concordance_index(pred: Tensor, target: Tensor,
                      mask: Optional[Tensor] = None,
                      max_pairs: int = 50000) -> float:
    """
    Concordance index (C-index).

    Probability that for a random pair (i,j) with y_i > y_j the model
    also predicts ŷ_i > ŷ_j.  0.5 = random, 1.0 = perfect.

    Samples up to *max_pairs* for efficiency.

    Parameters
    ----------
    pred : Tensor [N]
    target : Tensor [N]
    mask : Tensor [N], optional
    max_pairs : int

    Returns
    -------
    float   in [0, 1]
    """
    if mask is not None:
        if mask.dtype == torch.bool:
            pred, target = pred[mask], target[mask]
        else:
            pred, target = pred[mask.bool()], target[mask.bool()]
    N = pred.size(0)
    if N < 2:
        return 0.5
    device = pred.device

    P = min(max_pairs, N * (N - 1) // 2)
    i = torch.randint(0, N, (P * 2,), device=device)
    j = torch.randint(0, N, (P * 2,), device=device)
    keep = (i != j) & (torch.abs(target[i] - target[j]) > 1e-6)
    i, j = i[keep][:P], j[keep][:P]
    if i.size(0) == 0:
        return 0.5

    yd = target[i] - target[j]
    pd_ = pred[i] - pred[j]
    concordant = (yd * pd_) > 0
    tied = torch.abs(pd_) < 1e-6
    return (concordant.float() + 0.5 * tied.float()).mean().item()


def top_k_success_rate(pred_scores: Tensor,
                       true_labels: Tensor,
                       k: int = 1,
                       threshold: float = 0.5) -> float:
    """
    Top-k success rate for pose prediction.

    For each complex, check if any of the top-k predicted poses
    is a good pose (label = 1 or score > threshold).

    Parameters
    ----------
    pred_scores : Tensor [N]
        Predicted pose scores (higher = better).
    true_labels : Tensor [N]
        Binary labels (1 = good pose, 0 = bad pose)
        or continuous scores where > threshold = good.
    k : int
        Number of top predictions to consider (default 1).
    threshold : float
        Threshold for considering a pose "good".

    Returns
    -------
    float   in [0, 1]
    """
    N = pred_scores.size(0)
    if N == 0:
        return 0.0
    _, top_k_idx = torch.topk(pred_scores, min(k, N), largest=True)
    top_k_labels = true_labels[top_k_idx]
    success = (top_k_labels > threshold).any()
    return float(success)


def pose_rmsd(pred_coords: Tensor,
              true_coords: Tensor,
              align: bool = True) -> float:
    """
    Root Mean Squared Deviation between predicted and true ligand coordinates.

    RMSD = sqrt(mean(||pred − true||²))

    Optionally performs optimal alignment (Kabsch algorithm) before RMSD.

    Parameters
    ----------
    pred_coords : Tensor [M, 3]
    true_coords : Tensor [M, 3]
    align : bool
        Whether to perform optimal alignment (default True).

    Returns
    -------
    float   RMSD in Ångströms.
    """
    if align:
        pred_c = pred_coords - pred_coords.mean(dim=0, keepdim=True)
        true_c = true_coords - true_coords.mean(dim=0, keepdim=True)

        H = pred_c.T @ true_c
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        pred_aligned = pred_c @ R
        sq_dist = ((pred_aligned - true_c) ** 2).sum(dim=1)
    else:
        sq_dist = ((pred_coords - true_coords) ** 2).sum(dim=1)

    return torch.sqrt(sq_dist.mean()).item()


# ====================================================================
# 2. DockingMetrics accumulator (batch-wise update → compute)
# ====================================================================

class DockingMetrics:
    """
    Container for computing all docking metrics at once.

    Usage::

        dm = DockingMetrics()
        for batch in loader:
            dm.update(pred, target)
        results = dm.compute()   # dict of metric values
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds: list = []
        self.targets: list = []
        self.masks: list = []

    def update(self, pred: Tensor, target: Tensor,
               mask: Optional[Tensor] = None):
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
        if mask is not None:
            self.masks.append(mask.detach().cpu())
        else:
            self.masks.append(torch.ones_like(pred, dtype=torch.bool))

    def compute(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        pred = torch.cat(self.preds, dim=0)
        target = torch.cat(self.targets, dim=0)
        mask = torch.cat(self.masks, dim=0)
        return {
            "RMSE": rmse(pred, target, mask),
            "MAE": mae(pred, target, mask),
            "Pearson": pearson_r(pred, target, mask),
            "Spearman": spearman_rho(pred, target, mask),
            "C-index": concordance_index(pred, target, mask),
        }

    def __repr__(self):
        results = self.compute()
        return "\n".join(f"{k}: {v:.4f}" for k, v in results.items())


# ====================================================================
# 3. Custom Ignite metrics (accumulate across batches, compute at epoch end)
# ====================================================================

class _AffinityAccumulator(Metric):
    """
    Base class that accumulates (pred, target) tensors for good-pose
    samples (target > 0) across all batches, then computes a scalar
    metric at epoch end.

    Subclasses override ``_compute_from_accumulated``.
    """

    def __init__(self, output_transform, **kwargs):
        super().__init__(output_transform=output_transform, **kwargs)

    @reinit__is_reduced
    def reset(self):
        self._preds: list = []
        self._targets: list = []

    @reinit__is_reduced
    def update(self, output):
        pred, target = output
        good = target > 0
        if good.any():
            self._preds.append(pred[good].detach())
            self._targets.append(target[good].detach())

    def compute(self):
        if len(self._preds) == 0:
            raise NotComputableError(
                "No good-pose samples accumulated. "
                "Check that target affinities contain positive values."
            )
        pred = torch.cat(self._preds)
        target = torch.cat(self._targets)
        return self._compute_from_accumulated(pred, target)

    def _compute_from_accumulated(self, pred: Tensor, target: Tensor) -> float:
        raise NotImplementedError


class PearsonR(_AffinityAccumulator):
    """Pearson correlation on good-pose affinities (accumulated)."""
    def _compute_from_accumulated(self, pred, target):
        return pearson_r(pred, target)


class SpearmanRho(_AffinityAccumulator):
    """Spearman rank correlation on good-pose affinities (accumulated)."""
    def _compute_from_accumulated(self, pred, target):
        return spearman_rho(pred, target)


class CIndex(_AffinityAccumulator):
    """Concordance index on good-pose affinities (accumulated)."""
    def _compute_from_accumulated(self, pred, target):
        return concordance_index(pred, target)


# --------------------------------------------------------------------
# Pose classification: PR-AUC & confidence (softmax statistics)
# --------------------------------------------------------------------

class _PoseBinaryScoreAccumulator(Metric):
    """
    Accumulate positive-class scores and integer labels for sklearn metrics (e.g. PR-AUC).
    """

    def __init__(self, output_transform=lambda x: x, device=torch.device("cpu")):
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._scores: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    @reinit__is_reduced
    def update(self, output):
        scores, labels = output
        self._scores.append(scores.detach())
        self._labels.append(labels.detach().long())

    def compute(self):
        if not self._scores:
            raise NotComputableError("No pose classification samples accumulated.")
        s = torch.cat(self._scores).cpu().numpy()
        y = torch.cat(self._labels).cpu().numpy()
        return s, y


class PoseAveragePrecision(_PoseBinaryScoreAccumulator):
    """Average precision (PR-AUC) for the positive pose class."""

    def compute(self):
        from sklearn.metrics import average_precision_score

        s, y = super().compute()
        if len(np.unique(y)) < 2:
            return 0.5
        return float(average_precision_score(y, s))


class PoseRecallForLabel(_PoseBinaryScoreAccumulator):
    """
    Recall for class ``pos_label`` (0 = negative/bad pose, 1 = positive/good pose).
    Hard predictions: y_hat = 1 iff P(good) >= 0.5 (same as PR-AUC positive score).
    """

    def __init__(
        self,
        pos_label: int,
        output_transform=lambda x: x,
        device=torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self._pos_label = int(pos_label)

    def compute(self):
        from sklearn.metrics import recall_score

        s, y = super().compute()
        y = y.astype(np.int64, copy=False)
        y_pred = (s >= 0.5).astype(np.int64)
        if len(np.unique(y)) < 2:
            return 0.5
        return float(
            recall_score(
                y, y_pred, pos_label=self._pos_label, zero_division=0
            )
        )


class PoseMCC(_PoseBinaryScoreAccumulator):
    """Matthews correlation coefficient for pose classification at P(good) >= 0.5."""

    def compute(self):
        from sklearn.metrics import matthews_corrcoef

        s, y = super().compute()
        y = y.astype(np.int64, copy=False)
        y_pred = (s >= 0.5).astype(np.int64)
        if len(np.unique(y)) < 2:
            return 0.0
        return float(matthews_corrcoef(y, y_pred))


class _ScreeningAccumulator(Metric):
    """
    Accumulate affinity predictions for ranked screening metrics.

    Positives are samples with target affinity > 0, matching the GNINA convention
    used elsewhere in this project for good/valid poses.
    """

    def __init__(self, output_transform=lambda x: x, device=torch.device("cpu")):
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._scores: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

    @reinit__is_reduced
    def update(self, output):
        pred, target = output
        self._scores.append(pred.detach())
        self._labels.append((target.detach() > 0).long())

    def _score_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._scores:
            raise NotComputableError("No screening samples accumulated.")
        scores = torch.cat(self._scores).flatten()
        labels = torch.cat(self._labels).flatten().long()
        return scores, labels


class EnrichmentFactor(_ScreeningAccumulator):
    """Enrichment factor in the top fraction of predictions (e.g. 0.01 for EF@1%)."""

    def __init__(
        self,
        fraction: float,
        output_transform=lambda x: x,
        device=torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self._fraction = float(fraction)

    def compute(self):
        scores, labels = self._score_and_labels()
        n = int(labels.numel())
        total_pos = int(labels.sum().item())
        if n == 0 or total_pos == 0:
            return 0.0
        k = max(1, int(math.ceil(n * self._fraction)))
        order = torch.argsort(scores, descending=True)
        hits_top = labels[order[:k]].float().sum().item()
        expected_hits = k * (total_pos / n)
        if expected_hits <= 0:
            return 0.0
        return float(hits_top / expected_hits)


class SuccessAtK(_ScreeningAccumulator):
    """Whether at least one positive appears in the top-k affinity-ranked predictions."""

    def __init__(
        self,
        k: int,
        output_transform=lambda x: x,
        device=torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self._k = int(k)

    def compute(self):
        scores, labels = self._score_and_labels()
        if labels.numel() == 0 or labels.sum().item() == 0:
            return 0.0
        k = max(1, min(self._k, int(labels.numel())))
        order = torch.argsort(scores, descending=True)
        return float(labels[order[:k]].bool().any().item())


# ====================================================================
# 4. setup_metrics — called by training.py
# ====================================================================

def setup_metrics(
    affinity: bool,
    flex: bool,
    pose_loss: nn.Module,
    affinity_loss: nn.Module,
    flexpose_loss: nn.Module,
    roc_auc: bool,
    device: torch.device,
    pr_auc: bool = True,
    pose_confidence_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Build the dictionary of Ignite metrics attached to evaluator engines.

    Parameters
    ----------
    affinity : bool
        Whether affinity prediction is enabled.
    flex : bool
        Whether flexible-residue pose prediction is enabled.
    pose_loss, affinity_loss, flexpose_loss : nn.Module or None
        Loss modules (used as metrics to track loss curves).
    roc_auc : bool
        Whether to compute ROC-AUC for pose classification.
    device : torch.device
    pr_auc : bool
        Whether to compute average precision (PR-AUC) on the positive pose class.
    pose_confidence_metrics : bool
        Whether to log per-class pose recall (neg / pos) at 0.5 threshold on P(good).

    Returns
    -------
    Dict[str, Metric]
    """
    assert not (affinity and flex)

    pose_softmax_tf = transforms.output_transform_select_pose

    # --- Pose classification metrics (Balanced Accuracy first: primary under imbalance) ---
    m: Dict[str, Any] = {
        "Balanced Accuracy": igm.Recall(
            average=True,
            output_transform=pose_softmax_tf,
        ),
        "Accuracy": igm.Accuracy(
            output_transform=pose_softmax_tf,
        ),
    }

    if pose_loss is not None:
        m["Pose Loss"] = igm.Loss(
            pose_loss,
            output_transform=transforms.output_transform_select_log_pose,
        )

    if pose_confidence_metrics:
        roc_tf = lambda o: transforms.output_transform_ROC(o)
        m["Pose Recall Neg"] = PoseRecallForLabel(
            0,
            output_transform=roc_tf,
            device=device,
        )
        m["Pose Recall Pos"] = PoseRecallForLabel(
            1,
            output_transform=roc_tf,
            device=device,
        )
        m["MCC"] = PoseMCC(
            output_transform=roc_tf,
            device=device,
        )

    if roc_auc:
        m["ROC AUC"] = ROC_AUC(
            output_transform=lambda o: transforms.output_transform_ROC(o),
            device=device,
        )

    if pr_auc:
        m["PR AUC"] = PoseAveragePrecision(
            output_transform=lambda o: transforms.output_transform_ROC(o),
            device=device,
        )

    # --- Affinity regression metrics ---
    if affinity:
        aff_transform = transforms.output_transform_select_affinity

        m["MAE"] = igm.MeanAbsoluteError(
            output_transform=transforms.output_transform_affinity_good,
        )
        m["RMSE"] = igm.RootMeanSquaredError(
            output_transform=transforms.output_transform_affinity_good,
        )

        # Correlation & ranking metrics (accumulated across batches)
        m["Pearson R"] = PearsonR(output_transform=aff_transform)
        m["Spearman Rho"] = SpearmanRho(output_transform=aff_transform)
        m["C-index"] = CIndex(output_transform=aff_transform)
        m["EF_1pct"] = EnrichmentFactor(
            0.01,
            output_transform=aff_transform,
            device=device,
        )
        m["EF_5pct"] = EnrichmentFactor(
            0.05,
            output_transform=aff_transform,
            device=device,
        )
        m["Success_at_1"] = SuccessAtK(
            1,
            output_transform=aff_transform,
            device=device,
        )
        m["Success_at_5"] = SuccessAtK(
            5,
            output_transform=aff_transform,
            device=device,
        )
        m["Success_at_10"] = SuccessAtK(
            10,
            output_transform=aff_transform,
            device=device,
        )

        # Affinity loss as a metric (for tracking the loss curve)
        # CombinedAffinityLoss returns (loss, info_dict) — extract scalar
        if affinity_loss is not None:
            def _scalar_affinity_loss(pred, target):
                result = affinity_loss(pred, target)
                if isinstance(result, tuple):
                    return result[0]
                return result

            m["Affinity Loss"] = igm.Loss(
                _scalar_affinity_loss,
                output_transform=aff_transform,
            )

    # --- Flexible residue metrics ---
    if flex:
        m["Flex Accuracy"] = igm.Accuracy(
            output_transform=transforms.output_transform_select_flex,
        )
        m["Flex Balanced Accuracy"] = igm.Recall(
            average=True,
            output_transform=transforms.output_transform_select_flex,
        )
        if roc_auc:
            m["Flex ROC AUC"] = ROC_AUC(
                output_transform=lambda o: transforms.output_transform_ROC_flex(o),
                device=device,
            )
        if flexpose_loss is not None:
            m["Flex Pose Loss"] = igm.Loss(
                flexpose_loss,
                output_transform=transforms.output_transform_select_log_flex,
            )

    return m
