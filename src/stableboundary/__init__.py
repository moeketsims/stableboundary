"""Public interface for :mod:`stableboundary`.

The package uses Nolan's continuous ``S0`` parameterization throughout.
Private numerical backends are intentionally absent from this facade.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._exceptions import (
    ConvergenceError,
    InfiniteMomentError,
    NumericalProbabilityError,
    StableBoundaryError,
    UnidentifiedParameterError,
    ValidationError,
)
from .cells import CellCounts, CellProbabilities
from .design import KnownNuisance, LocalDesign, LocalPrior, NuisanceMode
from .parameters import LocalCoordinates, SignedTailGap, StableParams
from .simulation import simulate

try:
    __version__ = version("stableboundary")
except PackageNotFoundError:  # pragma: no cover - supports an uninstalled source tree
    __version__ = "0.1.0"

__all__ = [
    "CellCounts",
    "CellProbabilities",
    "ConvergenceError",
    "InfiniteMomentError",
    "KnownNuisance",
    "LocalCoordinates",
    "LocalDesign",
    "LocalPrior",
    "NumericalProbabilityError",
    "NuisanceMode",
    "SignedTailGap",
    "StableBoundaryError",
    "StableParams",
    "UnidentifiedParameterError",
    "ValidationError",
    "__version__",
    "simulate",
]
