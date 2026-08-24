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

try:
    __version__ = version("stableboundary")
except PackageNotFoundError:  # pragma: no cover - supports an uninstalled source tree
    __version__ = "0.1.0"

__all__ = [
    "ConvergenceError",
    "InfiniteMomentError",
    "NumericalProbabilityError",
    "StableBoundaryError",
    "UnidentifiedParameterError",
    "ValidationError",
    "__version__",
]
