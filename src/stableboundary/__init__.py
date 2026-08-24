"""Public interface for :mod:`stableboundary`.

The package uses Nolan's continuous ``S0`` parameterization throughout.
Private numerical backends are intentionally absent from this facade.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._exceptions import (
    ConvergenceError,
    InfiniteMomentError,
    InfiniteVarianceError,
    NumericalProbabilityError,
    StableBoundaryError,
    UnidentifiedParameterError,
    ValidationError,
)
from .api import fit_known_nuisance
from .cells import CellCounts, CellProbabilities
from .design import KnownNuisance, LocalDesign, LocalPrior, NuisanceMode
from .parameters import LocalCoordinates, SignedTailGap, StableParams
from .posterior import PosteriorGrid, QuadratureConfig
from .result import (
    CredibleInterval,
    ExpectedExceedanceCounts,
    IdentificationDiagnostics,
    KnownNuisanceFit,
    ParameterSummary,
    PosteriorPredictiveSample,
    PredictiveQuantileEstimate,
    SignedTailPrediction,
)
from .simulation import simulate

try:
    __version__ = version("stableboundary")
except PackageNotFoundError:  # pragma: no cover - supports an uninstalled source tree
    __version__ = "0.1.0"

__all__ = [
    "CellCounts",
    "CellProbabilities",
    "ConvergenceError",
    "CredibleInterval",
    "ExpectedExceedanceCounts",
    "IdentificationDiagnostics",
    "InfiniteMomentError",
    "InfiniteVarianceError",
    "KnownNuisance",
    "KnownNuisanceFit",
    "LocalCoordinates",
    "LocalDesign",
    "LocalPrior",
    "NumericalProbabilityError",
    "NuisanceMode",
    "ParameterSummary",
    "PosteriorGrid",
    "PosteriorPredictiveSample",
    "PredictiveQuantileEstimate",
    "QuadratureConfig",
    "SignedTailGap",
    "SignedTailPrediction",
    "StableBoundaryError",
    "StableParams",
    "UnidentifiedParameterError",
    "ValidationError",
    "__version__",
    "fit_known_nuisance",
    "simulate",
]
