"""
PotentialNet — row 4 of the thesis benchmark (GNN affinity baseline).

Reference: E. N. Feinberg et al.,
"PotentialNet for molecular property prediction",
*ACS Central Science* 4(11), 1520–1530, 2018.

Original PotentialNet is a two-stage gated graph neural network on a
protein–ligand graph: stage 1 operates on *covalent* bonds only, stage 2
adds *spatial* (non-covalent) edges weighted by distance cutoff, followed
by a graph-gather readout and an affinity MLP.

This port reproduces the two-stage gated message-passing design while
consuming the shared molgrid voxel input for fair comparison:

1. A lightweight 3D CNN stem produces per-voxel features.
2. ``extract_pseudo_atoms_from_grid`` yields ``K`` pseudo-atoms with
   ``[K, feat]`` embeddings and ``[K, 3]`` normalised coords.
3. Stage 1 (``bonded``): short-range message passing over the ``k_bonded``
   nearest neighbours (proxy for covalent/near-covalent edges).
4. Stage 2 (``spatial``): long-range message passing over pairs within a
   distance cutoff (non-covalent contacts).
5. Graph-gather + MLP → pose & affinity heads.

The gated update follows Li et al., "Gated Graph Sequence Neural Networks"
(ICLR 2016) as used by the original paper.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import AffinityCalibration, StableBatchNorm3d, extract_pseudo_atoms_from_grid


class _GatedGraphLayer(nn.Module):
    """One gated message-passing update step with edge-weighted aggregation."""

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )
        # GRU-style gated update: h' = (1-z)*h + z*tanh(candidate)
        self.gru = nn.GRUCell(node_dim, node_dim)

    def forward(
        self,
        h: torch.Tensor,          # [B, K, D]
        edge_feat: torch.Tensor,  # [B, K, K, De]
        mask: torch.Tensor,       # [B, K, K]
    ) -> torch.Tensor:
        B, K, D = h.shape
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)                      # [B,K,K,D]
        msg_in = torch.cat([edge_feat, h_j], dim=-1)                    # [B,K,K,De+D]
        msg = self.edge_mlp(msg_in)                                     # [B,K,K,D]
        msg = msg * mask.unsqueeze(-1)
        agg = msg.sum(dim=2)                                            # [B,K,D]

        updated = self.gru(agg.reshape(B * K, D), h.reshape(B * K, D))
        return updated.view(B, K, D)


class PotentialNet(nn.Module):
    """
    Two-stage gated graph network PotentialNet baseline.
    """

    def __init__(
        self,
        input_dims: Tuple,
        node_dim: int = 96,
        num_rbf: int = 16,
        k_bonded: int = 4,
        spatial_cutoff: float = 1.4,          # normalised coords ∈ [-1,1]
        num_bonded_layers: int = 2,
        num_spatial_layers: int = 2,
        max_atoms: int = 48,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(input_dims) == 4
        self.input_dims = input_dims
        self.max_atoms = max_atoms
        self.k_bonded = k_bonded
        self.spatial_cutoff = spatial_cutoff
        C = input_dims[0]

        self.stem = nn.Sequential(
            nn.Conv3d(C, 32, kernel_size=3, padding=1, bias=False),
            StableBatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, node_dim, kernel_size=3, stride=2, padding=1, bias=False),
            StableBatchNorm3d(node_dim),
            nn.ReLU(inplace=True),
        )

        # Learnable RBF kernels for distance encoding
        self.rbf_centers = nn.Parameter(torch.linspace(0.0, spatial_cutoff, num_rbf))
        self.rbf_widths = nn.Parameter(torch.ones(num_rbf) * (spatial_cutoff / num_rbf))

        self.bonded_layers = nn.ModuleList([
            _GatedGraphLayer(node_dim, num_rbf) for _ in range(num_bonded_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            _GatedGraphLayer(node_dim, num_rbf) for _ in range(num_spatial_layers)
        ])

        # Graph-gather readout: sigmoid-gated sum (Li et al. 2016 / Duvenaud 2015)
        self.gate = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.calib = AffinityCalibration()

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.rbf_centers) ** 2) / (2.0 * self.rbf_widths ** 2 + 1e-8))

    def _bonded_mask(self, distances: torch.Tensor) -> torch.Tensor:
        """k-nearest-neighbour mask, excluding self."""
        B, K, _ = distances.shape
        eye = torch.eye(K, device=distances.device).unsqueeze(0)
        d = distances + eye * 1e6
        k = min(self.k_bonded, K - 1)
        _, idx = torch.topk(-d, k, dim=-1)
        mask = torch.zeros_like(distances)
        mask.scatter_(-1, idx, 1.0)
        return mask

    def _spatial_mask(self, distances: torch.Tensor) -> torch.Tensor:
        mask = (distances > 0) & (distances <= self.spatial_cutoff)
        return mask.float()

    def forward(self, x: torch.Tensor):
        feat_vol = self.stem(x)
        atom_feat, positions = extract_pseudo_atoms_from_grid(feat_vol, self.max_atoms)

        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.norm(diff + 1e-8, dim=-1)
        edge_feat = self._rbf(dist)

        bonded_mask = self._bonded_mask(dist)
        spatial_mask = self._spatial_mask(dist)

        h = atom_feat
        for layer in self.bonded_layers:
            h = layer(h, edge_feat, bonded_mask)
        for layer in self.spatial_layers:
            h = layer(h, edge_feat, spatial_mask)

        gate = self.gate(h)
        node_proj = self.node_proj(h)
        graph_feat = (gate * node_proj).sum(dim=1)

        pose_log = F.log_softmax(self.pose_head(graph_feat), dim=1)
        self._last_pose_log = pose_log

        raw_aff = self.affinity_head(graph_feat).squeeze(-1)
        affinity = self.calib(raw_aff)
        return pose_log, affinity

    def get_pose_log(self):
        return getattr(self, "_last_pose_log", None)


class PotentialNetAffinity(PotentialNet):
    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
