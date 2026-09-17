"""
``dockbench.models`` — thesis benchmark architectures and registry.

Benchmark rows (7)
------------------
1. ``gnina_dense``       — 3D DenseNet (GNINA dense)
2. ``gnina_default2018`` — 3D CNN (GNINA default 2018)
3. ``pafnucy``           — 3D CNN (Pafnucy-style)
4. ``potentialnet``      — Gated GNN (PotentialNet-style)
5. ``equibind``          — Equivariant GNN (EquiBind-style)
6. ``tankbind``          — Triangle-aware GNN (TankBind-style)
7. ``geoformerdock``     — GeoFormerDock (proposed)

Use :func:`build_model` or ``models_dict`` for training; see :mod:`registry`.
"""

from .common import (
    AffinityCalibration,
    StableBatchNorm3d,
    apply_pose_to_affinity_gating,
    extract_pseudo_atoms_from_grid,
    weights_and_biases_init,
)
from .equibind import EquiBind, EquiBindAffinity
from .geoformerdock import GEO_ABLATION_CHOICES, GeoFormerDock
from .gnina_default2018 import (
    Default2017,
    Default2017Affinity,
    Default2017Pose,
    Default2018,
    Default2018Affinity,
    Default2018Pose,
)
from .gnina_dense import GNINADense, GNINADenseAffinity, GNINADensePose
from .pafnucy import Pafnucy, PafnucyAffinity
from .potentialnet import PotentialNet, PotentialNetAffinity
from .registry import (
    BENCHMARK_MODELS,
    LEGACY_ALIASES,
    MODEL_DISPLAY_NAMES,
    ModelKey,
    build_model,
    canonical_name,
    model_choices,
    models_dict,
)
from .tankbind import TankBind, TankBindAffinity

__all__ = [
    # registry
    "ModelKey",
    "models_dict",
    "BENCHMARK_MODELS",
    "LEGACY_ALIASES",
    "MODEL_DISPLAY_NAMES",
    "canonical_name",
    "model_choices",
    "build_model",
    "GEO_ABLATION_CHOICES",
    # shared
    "AffinityCalibration",
    "StableBatchNorm3d",
    "apply_pose_to_affinity_gating",
    "extract_pseudo_atoms_from_grid",
    "weights_and_biases_init",
    # gnina dense
    "GNINADense",
    "GNINADensePose",
    "GNINADenseAffinity",
    # gnina default / legacy 2017
    "Default2018",
    "Default2018Pose",
    "Default2018Affinity",
    "Default2017",
    "Default2017Pose",
    "Default2017Affinity",
    # baselines
    "Pafnucy",
    "PafnucyAffinity",
    "PotentialNet",
    "PotentialNetAffinity",
    "EquiBind",
    "EquiBindAffinity",
    "TankBind",
    "TankBindAffinity",
    "GeoFormerDock",
]
