"""
TankBind — row 6 of the thesis benchmark (geometry + interaction docking).

Reference: W. Lu et al.,
"TankBind: Trigonometry-aware Neural NetworKs for Drug-Protein Binding
Structure Prediction", NeurIPS 2022.

Original TankBind predicts a *pair distance map* between ligand atoms and
protein residues using a "trigonometry-aware" attention module inspired by
AlphaFold2's triangle attention (outgoing / incoming triangle updates over
a pairwise embedding ``z_ij``), followed by an interaction scoring head.

This port captures the essentials while consuming the shared voxel input
for fair benchmarking:

1. A lightweight 3D CNN stem provides per-voxel features.
2. ``extract_pseudo_atoms_from_grid`` returns ``K`` pseudo-atoms. We split
   them in half — the upper half acts as "ligand" tokens, the lower half as
   "protein" tokens — purely as a same-input analogue of TankBind's two
   node sets (no extra data loading required).
3. A pairwise embedding ``z_ij`` is initialised from an RBF expansion of
   ligand–protein distances, then refined by ``num_layers`` trigonometry
   attention blocks (outgoing multiplicative update + simple pair self-
   attention, following the spirit of AF2's triangle updates used by
   TankBind).
4. The interaction matrix is reduced by a small attention pooling into a
   graph-level interaction vector feeding pose & affinity heads.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import AffinityCalibration, StableBatchNorm3d, extract_pseudo_atoms_from_grid


class _TriangleMultiplicativeUpdate(nn.Module):
    """
    Outgoing-edge triangle multiplicative update (AF2 / TankBind-style).

        z_ij ← LN(g_ij ⊙ σ(a_ik) · σ(b_jk))

    This captures trigonometric consistency — updates to an edge z_ij use all
    third-node interactions (i,k) and (j,k).
    """

    def __init__(self, pair_dim: int):
        super().__init__()
        hidden = pair_dim
        self.norm_in = nn.LayerNorm(pair_dim)
        self.norm_out = nn.LayerNorm(hidden)
        self.lin_a = nn.Linear(pair_dim, hidden)
        self.lin_b = nn.Linear(pair_dim, hidden)
        self.gate_a = nn.Linear(pair_dim, hidden)
        self.gate_b = nn.Linear(pair_dim, hidden)
        self.gate_out = nn.Linear(pair_dim, hidden)
        self.lin_out = nn.Linear(hidden, pair_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zn = self.norm_in(z)
        a = torch.sigmoid(self.gate_a(zn)) * self.lin_a(zn)   # [B, L, P, H]
        b = torch.sigmoid(self.gate_b(zn)) * self.lin_b(zn)   # [B, L, P, H]
        # Outgoing update: sum over a third axis k (protein side)
        #   out_ij = Σ_k a_ik * b_jk  → we reuse the protein axis as k
        out = torch.einsum('bikh,bjkh->bijh', a, b)           # [B, L, L, H]
        # Map back to original ligand×protein shape using protein attention
        # (keeps tensor rank the same as input for residual connection)
        pool = out.mean(dim=2, keepdim=True).expand(-1, -1, z.size(2), -1)
        gate_o = torch.sigmoid(self.gate_out(zn))
        upd = gate_o * self.lin_out(self.norm_out(pool))
        return z + upd


class _PairSelfAttention(nn.Module):
    """Lightweight self-attention over ligand tokens, conditioned on pair bias."""

    def __init__(self, node_dim: int, pair_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=False)
        self.pair_bias = nn.Linear(pair_dim, num_heads)
        self.out = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, D],  z: [B, L, P, H]  — we mean-pool z over P for bias
        h = self.norm(tokens)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Pair bias: use ligand self-pair by collapsing protein axis
        z_self = z.mean(dim=2, keepdim=False)                # [B, L, H]
        bias = self.pair_bias(z_self)                        # [B, L, heads]
        bias = bias.permute(0, 2, 1).unsqueeze(-1)           # [B, heads, L, 1]
        attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return tokens + self.out(out)


class _TrigBlock(nn.Module):
    def __init__(self, node_dim: int, pair_dim: int, num_heads: int = 4):
        super().__init__()
        self.pair_upd = _TriangleMultiplicativeUpdate(pair_dim)
        self.tok_attn = _PairSelfAttention(node_dim, pair_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )

    def forward(self, ligand: torch.Tensor, protein: torch.Tensor, z: torch.Tensor):
        z = self.pair_upd(z)
        ligand = self.tok_attn(ligand, z)
        ligand = ligand + self.ffn(ligand)
        protein = self.tok_attn(protein, z.transpose(1, 2))
        protein = protein + self.ffn(protein)
        return ligand, protein, z


class TankBind(nn.Module):
    """
    Trigonometry-aware docking baseline inspired by TankBind.
    """

    def __init__(
        self,
        input_dims: Tuple,
        node_dim: int = 64,
        pair_dim: int = 32,
        num_rbf: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        max_atoms: int = 32,
        rbf_max_dist: float = 2.0,
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
            nn.GELU(),
            nn.Conv3d(32, node_dim, kernel_size=3, stride=2, padding=1, bias=False),
            StableBatchNorm3d(node_dim),
            nn.GELU(),
        )

        self.rbf_centers = nn.Parameter(torch.linspace(0.0, rbf_max_dist, num_rbf))
        self.rbf_widths = nn.Parameter(torch.ones(num_rbf) * (rbf_max_dist / num_rbf))
        self.pair_in = nn.Linear(num_rbf, pair_dim)

        self.blocks = nn.ModuleList([
            _TrigBlock(node_dim, pair_dim, num_heads) for _ in range(num_layers)
        ])

        # Attention pooling over the ligand × protein pair map
        self.pool_att = nn.Linear(pair_dim, 1)
        self.pair_proj = nn.Linear(pair_dim, hidden_dim)

        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.calib = AffinityCalibration()

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.rbf_centers) ** 2) / (2.0 * self.rbf_widths ** 2 + 1e-8))

    def forward(self, x_vox: torch.Tensor):
        feat_vol = self.stem(x_vox)
        atoms, positions = extract_pseudo_atoms_from_grid(feat_vol, self.max_atoms)
        B, K, D = atoms.shape
        half = K // 2
        ligand = atoms[:, :half, :]
        protein = atoms[:, half:, :]
        pos_l = positions[:, :half, :]
        pos_p = positions[:, half:, :]

        diff = pos_l.unsqueeze(2) - pos_p.unsqueeze(1)
        dist = torch.norm(diff + 1e-8, dim=-1)
        z = self.pair_in(self._rbf(dist))                  # [B, L, P, pair_dim]

        for block in self.blocks:
            ligand, protein, z = block(ligand, protein, z)

        # Attention pool z → global interaction vector
        att = F.softmax(self.pool_att(z).flatten(1), dim=-1).view(B, ligand.size(1), protein.size(1), 1)
        z_global = (att * self.pair_proj(z)).sum(dim=(1, 2))                 # [B, hidden]
        node_global = torch.cat([ligand.mean(dim=1), protein.mean(dim=1)], dim=-1)
        node_global = node_global.view(B, -1).mean(dim=-1, keepdim=True).expand(-1, ligand.size(-1))
        # use simpler pooling: mean of all atoms as node feature
        node_feat = atoms.mean(dim=1)                                        # [B, D]
        fused = torch.cat([z_global, node_feat], dim=-1)

        pose_log = F.log_softmax(self.pose_head(fused), dim=1)
        self._last_pose_log = pose_log
        raw_aff = self.affinity_head(fused).squeeze(-1)
        affinity = self.calib(raw_aff)
        return pose_log, affinity

    def get_pose_log(self):
        return getattr(self, "_last_pose_log", None)


class TankBindAffinity(TankBind):
    def __init__(self, input_dims: Tuple):
        super().__init__(input_dims)
