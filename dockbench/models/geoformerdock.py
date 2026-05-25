"""
GeoFormerDock — row 7 of the thesis benchmark ("Ours", full model).

Hybrid voxel–graph architecture for protein–ligand scoring with:

* Shared 3D CNN backbone + lightweight residual block
* Early task decoupling (separate pose / affinity branches)
* Pose branch = local CNN + top-K pseudo-atom geometry (RBF) + gated fusion
* Affinity branch = global CNN context + voxel-tokenised Pocket-Aware Transformer
* Optional uncertainty-aware (Bayesian) affinity head
* Learnable affinity calibration (scale + bias)

This file concentrates every GeoFormerDock component in one module so the
benchmark can be read end-to-end and compared easily against the other six
baselines in ``dockbench/models/``.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import StableBatchNorm3d, extract_pseudo_atoms_from_grid


# ---------- 1. Voxel Tokenization ----------

class VoxelTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


# ---------- 2. Pocket-Aware Transformer ----------

class PocketAwareAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.pocket_gate = nn.Sequential(
            nn.Linear(embed_dim, num_heads),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        pocket_scores = self.pocket_gate(x).permute(0, 2, 1).unsqueeze(-1)
        pocket_bias = pocket_scores.transpose(-2, -1)
        attn = attn + pocket_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PocketAwareAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------- 3. Uncertainty Head ----------

class UncertaintyHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.mean_head(h).squeeze(-1), self.log_var_head(h).squeeze(-1)


# ---------- 4. Lightweight residual block ----------

class LightweightResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = StableBatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = StableBatchNorm3d(out_channels)
        self.act = nn.ReLU()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                StableBatchNorm3d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


# ---------- 5. Simple geometry encoder (top-K pseudo-atoms + RBF) ----------

class SimpleGeometryEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, num_rbf: int = 16, max_atoms: int = 12):
        super().__init__()
        self.max_atoms = max_atoms
        self._geom_feat_dim = out_dim
        self.rbf_centers = nn.Parameter(torch.linspace(0.0, 3.0, num_rbf))
        self.rbf_widths = nn.Parameter(torch.ones(num_rbf) * 0.5)
        self.out_proj = nn.Sequential(
            nn.Linear(in_channels + num_rbf, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def _gaussian_rbf(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.rbf_centers) ** 2) / (2 * self.rbf_widths ** 2))

    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        atom_features, positions = extract_pseudo_atoms_from_grid(voxel_features, self.max_atoms)
        if atom_features.size(1) == 0:
            B = voxel_features.size(0)
            return torch.zeros(
                B,
                self._geom_feat_dim,
                device=voxel_features.device,
                dtype=voxel_features.dtype,
            )
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)
        rbf = self._gaussian_rbf(distances)
        geom_feat = rbf.mean(dim=(1, 2))
        atom_feat = atom_features.mean(dim=1)
        return self.out_proj(torch.cat([atom_feat, geom_feat], dim=-1))


# ---------- 6. Gated feature fusion ----------

class GatedFeatureFusion(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, out_dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, out_dim)
        self.proj_b = nn.Linear(dim_b, out_dim)
        self.gate = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        a = self.proj_a(feat_a)
        b = self.proj_b(feat_b)
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return self.norm(g * a + (1.0 - g) * b)


# ---------- 7. GeoFormerDock (full model) ----------

class GeoFormerDock(nn.Module):
    """Hybrid voxel–graph architecture for protein–ligand scoring."""

    def __init__(
        self,
        input_dims: Tuple,
        embed_dim: int = 128,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_rbf: int = 16,
        max_pseudo_atoms: int = 12,
        patch_size: int = 4,
        dropout: float = 0.1,
        uncertainty: bool = False,
    ):
        super().__init__()
        assert len(input_dims) == 4, "Input dimensions must be (channels, depth, height, width)"
        self.input_dims = input_dims
        self.uncertainty = uncertainty

        C, _, _, _ = input_dims
        backbone_mid = 48
        backbone_out = 64
        pose_channels = 64
        aff_channels = 64

        self.backbone = nn.Sequential(
            nn.Conv3d(C, 32, kernel_size=3, stride=1, padding=1, bias=False),
            StableBatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, backbone_mid, kernel_size=3, stride=2, padding=1, bias=False),
            StableBatchNorm3d(backbone_mid),
            nn.ReLU(),
            LightweightResidualBlock(backbone_mid, backbone_out),
        )

        # Pose branch
        self.pose_local = nn.Sequential(
            nn.Conv3d(backbone_out, pose_channels, kernel_size=3, padding=1, bias=False),
            StableBatchNorm3d(pose_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(pose_channels, pose_channels, kernel_size=3, padding=1, bias=False),
            StableBatchNorm3d(pose_channels),
            nn.ReLU(),
        )
        self.pose_geometry = SimpleGeometryEncoder(
            in_channels=pose_channels,
            out_dim=hidden_dim,
            num_rbf=num_rbf,
            max_atoms=max_pseudo_atoms,
        )
        self.pose_gap = nn.AdaptiveAvgPool3d(1)
        self.pose_local_proj = nn.Sequential(
            nn.Linear(pose_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pose_norm = nn.BatchNorm1d(hidden_dim)
        self.pose_fusion = GatedFeatureFusion(hidden_dim, hidden_dim, hidden_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # Affinity branch
        self.aff_local = nn.Sequential(
            nn.Conv3d(backbone_out, aff_channels, kernel_size=3, padding=1, bias=False),
            StableBatchNorm3d(aff_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.tokenizer = VoxelTokenizer(aff_channels, embed_dim, patch_size=patch_size)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(embed_dim)
        self.aff_gap = nn.AdaptiveAvgPool3d(1)
        self.aff_global_proj = nn.Sequential(
            nn.Linear(aff_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.aff_token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.aff_fusion = GatedFeatureFusion(hidden_dim, hidden_dim, hidden_dim)

        if uncertainty:
            self.affinity_head = UncertaintyHead(hidden_dim, hidden_dim)
        else:
            self.affinity_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.aff_scale = nn.Parameter(torch.tensor(2.0))
        self.aff_bias = nn.Parameter(torch.tensor(6.3))

        self.features_out_size = hidden_dim

    def set_gradient_routing(self, use_pose_to_affinity_gate: bool, stop_grad_pose_to_affinity: bool):
        return None

    def _extract_features(self, x: torch.Tensor):
        shared = self.backbone(x)
        pose_in = shared.clone()
        aff_in = shared.clone()

        pose_vol = self.pose_local(pose_in)
        pose_local_feat = self.pose_gap(pose_vol).view(x.size(0), -1)
        pose_local_feat = self.pose_norm(self.pose_local_proj(pose_local_feat))
        pose_geom_feat = self.pose_geometry(pose_vol)
        pose_feat = self.pose_fusion(pose_local_feat, pose_geom_feat)

        aff_vol = self.aff_local(aff_in)
        tokens = self.tokenizer(aff_vol)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.transformer_norm(tokens)
        aff_token_feat = self.aff_token_proj(tokens.mean(dim=1))
        aff_global_feat = self.aff_global_proj(self.aff_gap(aff_vol).view(x.size(0), -1))
        aff_feat = self.aff_fusion(aff_token_feat, aff_global_feat)

        return pose_feat, aff_feat, tokens

    def forward(self, x: torch.Tensor):
        pose_feat, affinity_feat, _ = self._extract_features(x)
        pose_raw = self.pose_head(pose_feat)
        pose_log = F.log_softmax(pose_raw, dim=1)
        self._last_pose_log = pose_log

        _aff_clamp = 20.0

        if self.uncertainty:
            affinity_mean, affinity_log_var = self.affinity_head(affinity_feat)
            affinity_mean = self.aff_scale * affinity_mean + self.aff_bias
            affinity_mean = torch.clamp(affinity_mean, -_aff_clamp, _aff_clamp)
            self._last_log_var = affinity_log_var
            return pose_log, affinity_mean
        raw_aff = self.affinity_head(affinity_feat).squeeze(-1)
        affinity = self.aff_scale * raw_aff + self.aff_bias
        affinity = torch.clamp(affinity, -_aff_clamp, _aff_clamp)
        return pose_log, affinity

    def get_uncertainty(self) -> Optional[torch.Tensor]:
        if self.uncertainty and hasattr(self, '_last_log_var'):
            return self._last_log_var
        return None

    def get_log_sigma(self) -> Optional[torch.Tensor]:
        if self.uncertainty and hasattr(self, '_last_log_var'):
            return 0.5 * self._last_log_var
        return None

    def get_pose_log(self) -> Optional[torch.Tensor]:
        return getattr(self, '_last_pose_log', None)
