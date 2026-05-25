"""
EquiBind — row 5 of the thesis benchmark (geometric / equivariant GNN).

Reference: H. Stärk et al.,
"EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction",
ICML 2022.

Original EquiBind uses an SE(3)-equivariant graph neural network over two
graphs (protein pocket graph + ligand graph) with coordinate updates that
are equivariant under 3D rotations/translations. The network outputs an
updated ligand conformation (pose) and, via its embeddings, a binding score.

This port keeps the **equivariant message-passing + coordinate update** core
and adapts it to the shared molgrid voxel input for fair comparison:

1. CNN stem on the voxel grid produces per-voxel embeddings.
2. ``extract_pseudo_atoms_from_grid`` yields ``K`` pseudo-atom embeddings and
   normalised coordinates.
3. A stack of *E(3)-invariant* message-passing + equivariant coordinate
   update layers refines ``(h_i, x_i)`` jointly:

        m_ij  = φ_m([h_i, h_j, d_ij_rbf])
        h_i   ← GRU(h_i, Σ_j m_ij)
        x_i   ← x_i + Σ_{j≠i} φ_x(m_ij) · (x_i − x_j)

   (Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021;
    used internally by EquiBind.)
4. Pose logits are read from coordinate displacement magnitude (near-native
   poses drift less) and node-feature summary, matching EquiBind's design
   intuition. Affinity is an MLP over pooled embeddings.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import AffinityCalibration, StableBatchNorm3d, extract_pseudo_atoms_from_grid


class _EGNNLayer(nn.Module):
    """One E(3)-equivariant update step (scalar features + coords)."""

    def __init__(self, node_dim: int, num_rbf: int, coord_scale: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + num_rbf, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
        )
        self.node_gru = nn.GRUCell(node_dim, node_dim)
        self.coord_scale = coord_scale

    def forward(self, h: torch.Tensor, x: torch.Tensor, rbf: torch.Tensor):
        B, K, D = h.shape
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)
        pair = torch.cat([h_i, h_j, rbf], dim=-1)
        m_ij = self.edge_mlp(pair)                                     # [B,K,K,D]

        # Node update: GRU on aggregated message
        agg = m_ij.sum(dim=2)                                          # [B,K,D]
        h_new = self.node_gru(agg.reshape(B * K, D), h.reshape(B * K, D)).view(B, K, D)

        # Coordinate update (E(3)-equivariant)
        coord_weight = self.coord_mlp(m_ij).squeeze(-1)                # [B,K,K]
        diff = x.unsqueeze(2) - x.unsqueeze(1)                         # [B,K,K,3]
        # mask diagonal to avoid NaN; weight zero at i==j
        eye = torch.eye(K, device=x.device).unsqueeze(0).unsqueeze(-1)
        diff = diff * (1.0 - eye)
        coord_update = (coord_weight.unsqueeze(-1) * diff).sum(dim=2)  # [B,K,3]
        x_new = x + self.coord_scale * coord_update

        return h_new, x_new


class EquiBind(nn.Module):
    """
    Equivariant GNN with coordinate refinement (EquiBind-style).
    """

    def __init__(
        self,
        input_dims: Tuple,
        node_dim: int = 96,
        num_rbf: int = 16,
        num_layers: int = 4,
        max_atoms: int = 48,
        rbf_max_dist: float = 2.0,    # in normalised coords
        coord_scale: float = 0.1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(input_dims) == 4
        self.input_dims = input_dims
        self.max_atoms = max_atoms
        C = input_dims[0]

        self.stem = nn.Sequential(
            nn.Conv3d(C, 32, kernel_size=3, padding=1, bias=False),
            StableBatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, node_dim, kernel_size=3, stride=2, padding=1, bias=False),
            StableBatchNorm3d(node_dim),
            nn.SiLU(inplace=True),
        )

        self.rbf_centers = nn.Parameter(torch.linspace(0.0, rbf_max_dist, num_rbf))
        self.rbf_widths = nn.Parameter(torch.ones(num_rbf) * (rbf_max_dist / num_rbf))

        self.layers = nn.ModuleList([
            _EGNNLayer(node_dim, num_rbf, coord_scale=coord_scale) for _ in range(num_layers)
        ])

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Pose head takes (pooled node feature, coord displacement magnitude)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.calib = AffinityCalibration()

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.rbf_centers) ** 2) / (2.0 * self.rbf_widths ** 2 + 1e-8))

    def forward(self, x_vox: torch.Tensor):
        feat_vol = self.stem(x_vox)
        h, x = extract_pseudo_atoms_from_grid(feat_vol, self.max_atoms)
        x_init = x

        for layer in self.layers:
            diff = x.unsqueeze(2) - x.unsqueeze(1)
            dist = torch.norm(diff + 1e-8, dim=-1)
            rbf = self._rbf(dist)
            h, x = layer(h, x, rbf)

        # Pool node features
        pooled = self.node_proj(h).mean(dim=1)                                 # [B, hidden]

        # Coordinate displacement magnitude (||Δx||₂ summed per atom, batch mean).
        # EquiBind uses pose quality ≈ how much refinement was needed.
        disp = (x - x_init).norm(dim=-1).mean(dim=1, keepdim=True)             # [B,1]

        pose_log = F.log_softmax(self.pose_head(torch.cat([pooled, disp], dim=-1)), dim=1)
        self._last_pose_log = pose_log

        raw_aff = self.affinity_head(pooled).squeeze(-1)
        affinity = self.calib(raw_aff)
        return pose_log, affinity

    def get_pose_log(self):
        return getattr(self, "_last_pose_log", None)


class EquiBindAffinity(EquiBind):
    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
