"""
Shared building blocks and utilities reused across model files.

Keep this module small and dependency-free so every architecture file imports
the exact same helpers (normalisation, initialisation, pseudo-atom extraction,
pose→affinity gating). Any change here is visible to every baseline, which is
important for a fair, like-for-like comparison in the thesis benchmark.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def StableBatchNorm3d(num_features: int) -> nn.BatchNorm3d:
    """
    BatchNorm3d with slower running-stat momentum for small-batch stability.

    Default PyTorch momentum=0.1 causes running stats to swing with small
    batches; momentum=0.01 gives much steadier training, which matters for
    graph-like baselines (PotentialNet, EquiBind, TankBind)
    and for stratified / non-IID sampling.
    """
    return nn.BatchNorm3d(num_features, momentum=0.01, track_running_stats=True)


def weights_and_biases_init(m: nn.Module) -> None:
    """Xavier-uniform init for Conv3d / Linear, zero bias. (Shared by all models.)"""
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def apply_pose_to_affinity_gating(
    pose_feat: torch.Tensor,
    affinity_feat: torch.Tensor,
    enabled: bool,
    stop_grad: bool,
    scale: float = 0.2,
) -> torch.Tensor:
    """One-way pose→affinity gating with optional stop-gradient."""
    if not enabled:
        return affinity_feat
    pose_ctx = pose_feat.detach() if stop_grad else pose_feat
    sim = F.cosine_similarity(affinity_feat, pose_ctx, dim=1).unsqueeze(1)
    gate = torch.sigmoid(sim)
    return affinity_feat * (1.0 + scale * gate)


def extract_pseudo_atoms_from_grid(
    voxel_grid: torch.Tensor, max_atoms: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dense voxel grid into a sparse set of pseudo-atoms.

    Selects the top-K voxels by channel-wise L2 norm as pseudo-atom sites;
    every atom carries its original channel vector as an embedding and a
    normalised [-1, 1]³ coordinate. Used by every *graph-style* baseline
    (PotentialNet, EquiBind, TankBind, GeoFormerDock)
    so they all consume the exact same input as GNINA / Pafnucy.

    Parameters
    ----------
    voxel_grid : torch.Tensor  [B, C, D, H, W]
    max_atoms : int
        If <= 0, returns empty feature/position tensors (K=0).

    Returns
    -------
    features  : torch.Tensor  [B, K, C]
    positions : torch.Tensor  [B, K, 3]   in [-1, 1]
    """
    B, C, D, H, W = voxel_grid.shape
    if max_atoms <= 0:
        device, dtype = voxel_grid.device, voxel_grid.dtype
        return (
            torch.empty(B, 0, C, device=device, dtype=dtype),
            torch.empty(B, 0, 3, device=device, dtype=dtype),
        )

    K = min(max_atoms, D * H * W)

    importance = voxel_grid.pow(2).sum(dim=1)                  # [B, D, H, W]
    importance_flat = importance.view(B, -1)                   # [B, N]
    _, topk_idx = importance_flat.topk(K, dim=1)               # [B, K]

    feat_flat = voxel_grid.view(B, C, -1)                      # [B, C, N]
    idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)          # [B, C, K]
    features = feat_flat.gather(2, idx_exp).transpose(1, 2)    # [B, K, C]

    z = topk_idx // (H * W)
    y = (topk_idx % (H * W)) // W
    x_coord = topk_idx % W
    positions = torch.stack([
        2.0 * x_coord.float() / max(W - 1, 1) - 1.0,
        2.0 * y.float() / max(H - 1, 1) - 1.0,
        2.0 * z.float() / max(D - 1, 1) - 1.0,
    ], dim=-1)                                                 # [B, K, 3]

    return features, positions


class AffinityCalibration(nn.Module):
    """
    Learnable scale+bias applied to raw affinity logits.

    Every model in the benchmark uses this same calibration head so the final
    pK range stays comparable regardless of backbone (prevents one model being
    advantaged by better-initialised output scaling).
    """

    def __init__(self, init_scale: float = 2.0, init_bias: float = 6.3, clamp: float = 20.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        self.clamp = clamp

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        out = self.scale * raw + self.bias
        return torch.clamp(out, -self.clamp, self.clamp)
