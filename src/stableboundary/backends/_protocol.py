"""Package-owned protocol for Nolan ``S0`` stable-law calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from stableboundary._exceptions import ValidationError

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

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValidationError("backend metadata must name its method")
        if isinstance(self.tolerance, bool) or not isinstance(self.tolerance, Real):
            raise ValidationError("backend tolerance must be a real number")
        tolerance = float(self.tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValidationError(
                "backend tolerance must be finite and strictly positive"
            )
        if not isinstance(self.parameterization, str) or not self.parameterization:
            raise ValidationError("backend parameterization must be named")
        for name, value in (
            ("library", self.library),
            ("library_version", self.library_version),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValidationError(f"backend {name} must be a nonempty string")
        setting_names: set[str] = set()
        for setting in self.effective_settings:
            if not isinstance(setting, tuple) or len(setting) != 2:
                raise ValidationError(
                    "backend effective settings must be name-value pairs"
                )
            name, value = setting
            if not isinstance(name, str) or not name.strip() or name in setting_names:
                raise ValidationError(
                    "backend effective setting names must be unique nonempty strings"
                )
            if isinstance(value, bool) or not isinstance(
                value, (str, int, float, type(None))
            ):
                raise ValidationError("backend effective setting values are invalid")
            if isinstance(value, (int, float)) and not isfinite(float(value)):
                raise ValidationError("backend effective setting values must be finite")
            setting_names.add(name)
        object.__setattr__(self, "tolerance", tolerance)


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


def validate_s0_metadata(candidate: object) -> BackendMetadata:
    """Validate one immutable metadata record for a Nolan S0 backend."""
    metadata = candidate
    if not isinstance(metadata, BackendMetadata):
        raise ValidationError("backend metadata must be a BackendMetadata object")
    if metadata.parameterization != "S0":
        raise ValidationError(
            "backend parameterization must be Nolan S0; "
            f"received {metadata.parameterization!r}"
        )
    return metadata


def validate_s0_backend(candidate: object) -> tuple[StableBackend, BackendMetadata]:
    """Validate a backend and return one immutable S0 metadata snapshot."""
    if not isinstance(candidate, StableBackend):
        raise ValidationError("backend must satisfy StableBackend")
    backend = candidate
    metadata = validate_s0_metadata(backend.metadata)
    return backend, metadata


__all__ = [
    "BackendMetadata",
    "BackendResult",
    "BackendSetting",
    "StableBackend",
    "validate_s0_backend",
    "validate_s0_metadata",
]
