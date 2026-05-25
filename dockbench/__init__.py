"""DockBench — multi-task protein–ligand scoring (7-model thesis benchmark)."""

from . import _version

__version__ = _version.get_versions()["version"]
__git_revision__ = _version.get_versions().get("full-revisionid", "")

# Public API: version + model registry
from .models import (
    BENCHMARK_MODELS,
    GeoFormerDock,
    build_model,
    canonical_name,
    model_choices,
    models_dict,
)

__all__ = [
    "__version__",
    "__git_revision__",
    "BENCHMARK_MODELS",
    "GeoFormerDock",
    "build_model",
    "canonical_name",
    "model_choices",
    "models_dict",
]
