"""Package-owned protocol for Nolan ``S0`` stable-law calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

BackendResult = float | NDArray[np.float64]
type BackendSetting = str | int | float | None


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    """Immutable description of a stable-law numerical implementation."""

    method: str
    tolerance: float
    parameterization: str = "S0"
    library: str | None = None
    library_version: str | None = None
    effective_settings: tuple[tuple[str, BackendSetting], ...] = ()


@runtime_checkable
class StableBackend(Protocol):
    """Numerical operations required by the finite stable experiment.

    Probability methods follow NumPy/SciPy broadcasting for ``x``, ``alpha``,
    and ``beta``.  Implementations return a Python ``float`` for scalar input
    and a ``float64`` array otherwise.
    """

    @property
    def metadata(self) -> BackendMetadata:
        """Return immutable method and tolerance metadata."""

    def logpdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        """Evaluate the stable log density directly."""

    def cdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        """Evaluate the lower-tail probability directly."""

    def sf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        """Evaluate the upper-tail probability directly."""

    def logcdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        """Evaluate the lower-tail log probability directly."""

    def logsf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        """Evaluate the upper-tail log probability directly."""

    def rvs(
        self,
        alpha: float,
        beta: float,
        *,
        loc: float,
        scale: float,
        size: int,
        random_state: Generator,
    ) -> NDArray[np.float64]:
        """Draw stable variates with the supplied NumPy generator."""


__all__ = [
    "BackendMetadata",
    "BackendResult",
    "BackendSetting",
    "StableBackend",
]
