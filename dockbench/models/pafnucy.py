"""
Pafnucy — row 3 of the thesis benchmark (3D CNN affinity baseline).

Reference: M. M. Stępniewska-Dziubińska, P. Zielenkiewicz, P. Siedlecki,
"Development and evaluation of a deep learning model for protein–ligand
binding affinity prediction", *Bioinformatics* 34(21), 2018.

Original Pafnucy architecture:
    3 × [Conv3D (k=5, C: 64→128→256) → ReLU → MaxPool(k=2)]
    Flatten
    3 × [Linear(1000 → 500 → 200) → ReLU + Dropout]
    Linear(1) → affinity (pK)

This port keeps the block structure (5×5×5 convs, channel growth 64→128→256,
three FC layers of width 1000/500/200) but adapts to the shared GNINA pipeline:

* Works on the same ``(C, D, H, W)`` molgrid voxel input as every other
  benchmark model — no separate data loader, no grid-size change.
* Channel counts at FC are auto-sized from ``input_dims`` so it remains within
  a reasonable parameter budget for a 48³ grid.
* Adds a parallel 2-way pose head so it can train with the same multi-task
  loss as GNINA / GeoFormerDock. The pose head is small; the original Pafnucy
  output (scalar affinity) is unchanged in spirit.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import AffinityCalibration


class Pafnucy(nn.Module):
    """
    3D CNN affinity predictor in the style of Pafnucy (Bioinformatics 2018).
    """

    def __init__(
        self,
        input_dims: Tuple,
        conv_channels: Tuple[int, int, int] = (64, 128, 256),
        fc_dims: Tuple[int, int, int] = (1000, 500, 200),
        conv_kernel: int = 5,
        dropout: float = 0.5,
        adaptive_pool_size: int = 2,
    ):
        super().__init__()
        assert len(input_dims) == 4, "Input dimensions must be (channels, depth, height, width)"
        self.input_dims = input_dims
        C, D, H, W = input_dims
        pad = conv_kernel // 2

        layers = []
        prev = C
        for out_c in conv_channels:
            layers += [
                nn.Conv3d(prev, out_c, kernel_size=conv_kernel, padding=pad),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            ]
            prev = out_c
        # Adaptive pool keeps the FC input size constant across grid sizes, so
        # Pafnucy stays at a comparable parameter count (~few M) to every
        # other row in the benchmark — avoids the FC input blowing up to 55k+
        # on a 48³ molgrid and dominating the comparison with 60M+ params.
        layers.append(nn.AdaptiveAvgPool3d(adaptive_pool_size))
        self.features = nn.Sequential(*layers)

        flat = prev * (adaptive_pool_size ** 3)
        self.flat_dim = flat

        fc_layers = []
        prev = flat
        for out_c in fc_dims:
            fc_layers += [nn.Linear(prev, out_c), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = out_c
        self.fc = nn.Sequential(*fc_layers)
        self.fc_out = prev

        self.affinity_head = nn.Linear(self.fc_out, 1)
        self.pose_head = nn.Linear(self.fc_out, 2)

        self.calib = AffinityCalibration()

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(1)
        h = self.fc(h)

        pose_log = F.log_softmax(self.pose_head(h), dim=1)
        self._last_pose_log = pose_log

        raw_aff = self.affinity_head(h).squeeze(-1)
        affinity = self.calib(raw_aff)
        return pose_log, affinity

    def get_pose_log(self):
        return getattr(self, "_last_pose_log", None)


class PafnucyAffinity(Pafnucy):
    """Alias kept for symmetry with Default2018Affinity etc."""

    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
