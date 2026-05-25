"""
GNINA Default2018 — row 2 of the thesis benchmark.

Classic GNINA 2020-family 3D CNN (AvgPool → Conv3D → ReLU → Conv 1×1 stack)
ported verbatim from the original monolithic ``models.py`` so legacy GNINA
examples (PDBbind Crystal / Docked scripts) and loaded Caffe checkpoints
keep working unchanged.
"""

from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Default2018(nn.Module):
    """GNINA default2018 feature backbone (shared by pose / affinity variants)."""

    def __init__(self, input_dims: Tuple):
        super().__init__()
        assert len(input_dims) == 4, "Input dimensions must be (channels, depth, height, width)"
        self.input_dims = input_dims

        self.features = nn.Sequential(OrderedDict([
            ("unit1_pool", nn.AvgPool3d(kernel_size=2, stride=2)),
            ("unit1_conv", nn.Conv3d(input_dims[0], 32, kernel_size=3, stride=1, padding=1)),
            ("unit1_func", nn.ReLU()),
            ("unit2_conv", nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0)),
            ("unit2_func", nn.ReLU()),
            ("unit3_pool", nn.AvgPool3d(kernel_size=2, stride=2)),
            ("unit3_conv", nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)),
            ("unit3_func", nn.ReLU()),
            ("unit4_conv", nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0)),
            ("unit4_func", nn.ReLU()),
            ("unit5_pool", nn.AvgPool3d(kernel_size=2, stride=2)),
            ("unit5_conv", nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)),
            ("unit5_func", nn.ReLU()),
        ]))
        self.features_out_size = input_dims[1] // 8 * input_dims[2] // 8 * input_dims[3] // 8 * 128

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class Default2018Pose(Default2018):
    """Default2018 — pose classification only."""

    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
        self.pose = nn.Sequential(OrderedDict([
            ("pose_output", nn.Linear(self.features_out_size, 2)),
        ]))

    def forward(self, x: torch.Tensor):
        x = self.features(x).view(-1, self.features_out_size)
        return F.log_softmax(self.pose(x), dim=1)


class Default2018Affinity(Default2018Pose):
    """Default2018 — pose + affinity with learnable calibration and uncertainty head."""

    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
        self.affinity = nn.Sequential(OrderedDict([
            ("affinity_output", nn.Linear(self.features_out_size, 1)),
        ]))
        self.log_sigma_head = nn.Sequential(
            nn.Linear(self.features_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.aff_scale = nn.Parameter(torch.tensor(2.0))
        self.aff_bias = nn.Parameter(torch.tensor(6.3))

    def forward(self, x: torch.Tensor):
        x = self.features(x).view(-1, self.features_out_size)
        pose_log = F.log_softmax(self.pose(x), dim=1)
        raw_aff = self.affinity(x).squeeze(-1)
        affinity = self.aff_scale * raw_aff + self.aff_bias
        log_sigma = self.log_sigma_head(x).squeeze(-1)
        self._last_log_sigma = log_sigma
        self._last_pose_log = pose_log
        return pose_log, affinity

    def get_log_sigma(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_log_sigma", None)

    def get_pose_log(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_pose_log", None)


# --- Legacy alias for Ragoza 2017 Default2017 (MaxPool→Conv→ReLU stack) ---
# Kept for backward-compatibility with PDBbind / CSAR example scripts and with
# GNINA Caffe checkpoints. Not part of the 7-row thesis table — that row is
# GNINADense in ``gnina_dense.py``.


class Default2017(nn.Module):
    """GNINA default2017 backbone (Ragoza 2017)."""

    def __init__(self, input_dims: Tuple):
        super().__init__()
        assert len(input_dims) == 4
        self.input_dims = input_dims
        self.features = nn.Sequential(OrderedDict([
            ("unit1_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
            ("unit1_conv1", nn.Conv3d(input_dims[0], 32, kernel_size=3, stride=1, padding=1)),
            ("unit1_relu1", nn.ReLU()),
            ("unit2_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
            ("unit2_conv1", nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)),
            ("unit2_relu1", nn.ReLU()),
            ("unit3_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
            ("unit3_conv1", nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)),
            ("unit3_relu1", nn.ReLU()),
        ]))
        self.features_out_size = input_dims[1] // 8 * input_dims[2] // 8 * input_dims[3] // 8 * 128

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class Default2017Pose(Default2017):
    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
        self.pose = nn.Sequential(OrderedDict([
            ("pose_output", nn.Linear(self.features_out_size, 2)),
        ]))

    def forward(self, x: torch.Tensor):
        x = self.features(x).view(-1, self.features_out_size)
        return F.log_softmax(self.pose(x), dim=1)


class Default2017Affinity(Default2017Pose):
    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
        self.affinity = nn.Sequential(OrderedDict([
            ("affinity_output", nn.Linear(self.features_out_size, 1)),
        ]))
        self.log_sigma_head = nn.Sequential(
            nn.Linear(self.features_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.aff_scale = nn.Parameter(torch.tensor(2.0))
        self.aff_bias = nn.Parameter(torch.tensor(6.3))

    def forward(self, x: torch.Tensor):
        x = self.features(x).view(-1, self.features_out_size)
        pose_log = F.log_softmax(self.pose(x), dim=1)
        raw_aff = self.affinity(x).squeeze(-1)
        affinity = self.aff_scale * raw_aff + self.aff_bias
        log_sigma = self.log_sigma_head(x).squeeze(-1)
        self._last_log_sigma = log_sigma
        self._last_pose_log = pose_log
        return pose_log, affinity

    def get_log_sigma(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_log_sigma", None)

    def get_pose_log(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_pose_log", None)
