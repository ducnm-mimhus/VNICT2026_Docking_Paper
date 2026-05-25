"""
Model registry for the 7-row thesis benchmark and legacy GNINA aliases.

Training resolves models via :func:`build_model` using
``(canonical_name, affinity, flex)`` keys. Benchmark runs use
``affinity=True``, ``flex=False`` (multi-task pose + affinity).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, Optional, Tuple, Type

import torch.nn as nn

from .equibind import EquiBind, EquiBindAffinity
from .geoformerdock import GeoFormerDock
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
from .tankbind import TankBind, TankBindAffinity

ModelKey = namedtuple("ModelKey", ["name", "affinity", "flex"])

# Canonical benchmark model ids (CLI: ``-m`` / ``ONLY_MODEL``)
BENCHMARK_MODELS: Tuple[str, ...] = (
    "gnina_dense",
    "gnina_default2018",
    "pafnucy",
    "potentialnet",
    "equibind",
    "tankbind",
    "geoformerdock",
)

# Legacy GNINA names accepted by training CLI
LEGACY_ALIASES: Dict[str, str] = {
    "default2017": "gnina_dense",
    "default2018": "gnina_default2018",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "gnina_dense": "GNINA Dense",
    "gnina_default2018": "GNINA Default2018",
    "pafnucy": "Pafnucy",
    "potentialnet": "PotentialNet",
    "equibind": "EquiBind",
    "tankbind": "TankBind",
    "geoformerdock": "GeoFormerDock",
}


def canonical_name(name: str) -> str:
    """Map legacy CLI names to canonical benchmark ids."""
    return LEGACY_ALIASES.get(name, name)


def model_choices() -> Tuple[str, ...]:
    """All valid ``-m`` values (benchmark + legacy aliases)."""
    return tuple(sorted(set(BENCHMARK_MODELS) | set(LEGACY_ALIASES.keys())))


def _build_models_dict() -> Dict[ModelKey, Type[nn.Module]]:
    """Registry: (name, affinity, flex) → constructor(input_dims)."""
    return {
        # --- benchmark (multi-task: pose + affinity) ---
        ModelKey("gnina_dense", True, False): GNINADenseAffinity,
        ModelKey("gnina_default2018", True, False): Default2018Affinity,
        ModelKey("pafnucy", True, False): PafnucyAffinity,
        ModelKey("potentialnet", True, False): PotentialNetAffinity,
        ModelKey("equibind", True, False): EquiBindAffinity,
        ModelKey("tankbind", True, False): TankBindAffinity,
        ModelKey("geoformerdock", True, False): GeoFormerDock,
        # --- pose-only variants (affinity=False) ---
        ModelKey("gnina_dense", False, False): GNINADensePose,
        ModelKey("gnina_default2018", False, False): Default2018Pose,
        # --- legacy GNINA 2017 / 2018 aliases ---
        ModelKey("default2017", False, False): Default2017Pose,
        ModelKey("default2017", True, False): Default2017Affinity,
        ModelKey("default2018", False, False): Default2018Pose,
        ModelKey("default2018", True, False): Default2018Affinity,
    }


models_dict: Dict[ModelKey, Type[nn.Module]] = _build_models_dict()


def build_model(
    name: str,
    input_dims: Tuple,
    *,
    affinity: bool = True,
    flex: bool = False,
    geoformer_kwargs: Optional[dict] = None,
) -> nn.Module:
    """
    Instantiate a model by canonical or legacy name.

    Parameters
    ----------
    name
        Benchmark id or legacy alias (e.g. ``default2017``).
    input_dims
        ``(C, D, H, W)`` from the data loader.
    affinity, flex
        Must match a key in ``models_dict`` (benchmark: ``True, False``).
    geoformer_kwargs
        Optional overrides for :class:`GeoFormerDock` only
        (``max_pseudo_atoms``, ``num_transformer_layers``, ``uncertainty``).
    """
    canonical = canonical_name(name)
    key = ModelKey(canonical, affinity, flex)
    if key not in models_dict:
        raise KeyError(
            f"No model registered for {key!r}. "
            f"Valid names: {', '.join(BENCHMARK_MODELS)}"
        )

    if canonical == "geoformerdock" and geoformer_kwargs:
        return GeoFormerDock(input_dims, **geoformer_kwargs)

    return models_dict[key](input_dims)
