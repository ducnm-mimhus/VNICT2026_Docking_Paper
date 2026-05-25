"""
GNINA Dense — 3D DenseNet baseline.

Row 1 of the thesis benchmark: "GNINA Dense / 3D CNN / strong baseline".

Architecture follows Francoeur et al., *J. Chem. Inf. Model.* 2020, 60,
4200–4215 (the "dense" model of GNINA): a stem 3D convolution, 3 dense
blocks with bottleneck layers and transition-down modules between them,
global average pooling and two prediction heads (pose, affinity).

This is an **honest port** of the DenseNet-3D design into PyTorch — not an
alias of the older Ragoza 2017 MaxPool→Conv stack. Channel growth, bottleneck
factor and block depth follow the published configuration; numbers can be
tuned via the class kwargs.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import AffinityCalibration, StableBatchNorm3d


class _DenseLayer(nn.Module):
    """BN-ReLU-1x1x1 (bottleneck) → BN-ReLU-3x3x3 → concat."""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        self.bn1 = StableBatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = StableBatchNorm3d(bn_size * growth_rate)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int,
                 bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth_rate, bn_size=bn_size, dropout=dropout))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Transition(nn.Module):
    """BN-ReLU-1x1x1 (compression) → AvgPool 2x."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn = StableBatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(F.relu(self.bn(x), inplace=True)))


class GNINADense(nn.Module):
    """
    GNINA Dense (3D DenseNet).

    Input
    -----
    x : [B, C, D, H, W] voxel grid from molgrid.

    Output
    ------
    (pose_log, affinity) if ``affinity=True``, else ``pose_log`` only.

    Defaults (``init_features=32``, ``growth_rate=16``, ``block_config=(4, 4, 4)``,
    ``compression=0.5``) keep the parameter count near the GNINA 3D CNN family
    (~0.5–1M parameters) for fair comparison.
    """

    def __init__(
        self,
        input_dims: Tuple,
        init_features: int = 32,
        growth_rate: int = 16,
        block_config: Tuple[int, ...] = (4, 4, 4),
        bn_size: int = 4,
        compression: float = 0.5,
        dropout: float = 0.0,
        affinity: bool = True,
        hidden_dim: int = 128,
    ):
        super().__init__()
        assert len(input_dims) == 4, "Input dimensions must be (channels, depth, height, width)"
        self.input_dims = input_dims
        self.affinity = affinity

        C = input_dims[0]
        # Stem: 3x3x3 conv + pool (keeps receptive field comparable to default2018)
        self.stem = nn.Sequential(
            nn.Conv3d(C, init_features, kernel_size=3, stride=1, padding=1, bias=False),
            StableBatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        ch = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, ch, growth_rate, bn_size=bn_size, dropout=dropout)
            self.blocks.append(block)
            ch = block.out_channels
            if i < len(block_config) - 1:
                out_ch = max(1, int(ch * compression))
                self.transitions.append(_Transition(ch, out_ch))
                ch = out_ch

        self.final_norm = StableBatchNorm3d(ch)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.pose_head = nn.Sequential(
            nn.Linear(ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 2),
        )
        if affinity:
            self.affinity_head = nn.Sequential(
                nn.Linear(ch, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, 1),
            )
            self.log_sigma_head = nn.Sequential(
                nn.Linear(ch, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            self.calib = AffinityCalibration()

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = F.relu(self.final_norm(x), inplace=True)
        return self.gap(x).flatten(1)

    def forward(self, x: torch.Tensor):
        feat = self._extract(x)
        pose_log = F.log_softmax(self.pose_head(feat), dim=1)
        self._last_pose_log = pose_log
        if not self.affinity:
            return pose_log
        raw_aff = self.affinity_head(feat).squeeze(-1)
        affinity = self.calib(raw_aff)
        log_sigma = self.log_sigma_head(feat).squeeze(-1)
        self._last_log_sigma = log_sigma
        return pose_log, affinity

    def get_log_sigma(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_log_sigma", None)

    def get_pose_log(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_pose_log", None)


class GNINADensePose(GNINADense):
    """Pose-only variant (for registry entry with affinity=False)."""

    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims, affinity=False)


class GNINADenseAffinity(GNINADense):
    """Pose + affinity variant (registry entry with affinity=True)."""

    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims, affinity=True)
