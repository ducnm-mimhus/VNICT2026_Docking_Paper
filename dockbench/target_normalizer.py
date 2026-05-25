"""pK target normalization (no molgrid dependency)."""

from __future__ import annotations

import torch
from torch import Tensor


class TargetNormalizer:
    """Standardise pK targets: z = (y − μ) / σ."""

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.fitted = False

    def normalize(self, y: Tensor) -> Tensor:
        if not self.fitted:
            return y
        mask = y > 0
        out = y.clone()
        out[mask] = (y[mask] - self.mean) / self.std
        return out

    def denormalize(self, z: Tensor) -> Tensor:
        return z * self.std + self.mean
