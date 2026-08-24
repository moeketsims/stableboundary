"""Prespecified local designs, compact priors, and nuisance provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, log, sqrt
from numbers import Integral
from typing import ClassVar, Self, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import lambertw  # type: ignore[import-untyped]

from ._exceptions import ValidationError
from .parameters import LocalCoordinates, _finite_float, _positive_float


@dataclass(frozen=True, slots=True, init=False)
class LocalDesign:
    """A theorem-prescribed design derived only from sample size and ``c``.

    Instances can be created only with :meth:`from_sample_size`, preventing a
    caller from supplying internally inconsistent values for ``r`` or the
    moving threshold.
    """

    FORMULA_ID: ClassVar[str] = "critical-rate-lambertw-loglog-threshold"
    FORMULA_VERSION: ClassVar[int] = 1
    MAX_RELATIVE_RESIDUAL: ClassVar[float] = 1e-12

    n: int
    c: float
    r: float
    threshold: float
    formula_id: str
    formula_version: int
    critical_rate_relative_residual: float

    def __init__(self) -> None:
        """Prevent construction without validated derived quantities."""
        raise TypeError("use LocalDesign.from_sample_size() to construct a design")

    @classmethod
    def from_sample_size(cls, n: int, c: float = 1.0) -> Self:
        """Construct the critical-rate design before observations are inspected.

        The principal Lambert-W branch gives
        ``r = (8*c/n) * W(n/(8*c))``.  The threshold is
        ``2*sqrt(log(1/r) + 2*log(log(1/r)))``.
        """
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise ValidationError("n must be an integer sample size")
        n_value = int(n)
        if n_value <= 0:
            raise ValidationError("n must be strictly positive")
        c_value = _positive_float("c", c)

        try:
            argument = n_value / (8.0 * c_value)
        except OverflowError as error:
            raise ValidationError("n/(8*c) must be finite") from error
        if not isfinite(argument) or argument <= 0.0:
            raise ValidationError("n/(8*c) must be finite and positive")

        solution = complex(lambertw(argument, k=0))
        if (
            not isfinite(solution.real)
            or not isfinite(solution.imag)
            or solution.imag != 0.0
            or solution.real <= 0.0
        ):
            raise ValidationError(
                "principal Lambert-W evaluation must be real and positive"
            )

        r = (8.0 * c_value / n_value) * solution.real
        if not isfinite(r) or not 0.0 < r < 1.0:
            raise ValidationError("critical-rate r must lie strictly inside (0, 1)")
        log_inverse_r = log(1.0 / r)
        threshold_argument = log_inverse_r + 2.0 * log(log_inverse_r)
        if not isfinite(threshold_argument) or threshold_argument <= 0.0:
            raise ValidationError(
                "sample size and c do not yield a real positive log-log threshold"
            )
        threshold = 2.0 * sqrt(threshold_argument)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValidationError("derived threshold must be finite and positive")

        lhs = n_value * r / log_inverse_r
        target = 8.0 * c_value
        relative_residual = abs(lhs - target) / target
        if (
            not isfinite(relative_residual)
            or relative_residual > cls.MAX_RELATIVE_RESIDUAL
        ):
            raise ValidationError("critical-rate solution failed its residual check")

        design = object.__new__(cls)
        object.__setattr__(design, "n", n_value)
        object.__setattr__(design, "c", c_value)
        object.__setattr__(design, "r", r)
        object.__setattr__(design, "threshold", threshold)
        object.__setattr__(design, "formula_id", cls.FORMULA_ID)
        object.__setattr__(design, "formula_version", cls.FORMULA_VERSION)
        object.__setattr__(
            design,
            "critical_rate_relative_residual",
            relative_residual,
        )
        return design


@dataclass(frozen=True, slots=True)
class LocalPrior:
    """A proper uniform prior on a compact theorem-interior rectangle.

    The default support is ``h in [0.25, 4.0]`` and
    ``p in [0.05, 0.95]``.  The design is retained so support validity is
    checked on the same scale ``r`` used by the experiment.
    """

    design: LocalDesign
    h_min: float = 0.25
    h_max: float = 4.0
    p_min: float = 0.05
    p_max: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.design, LocalDesign):
            raise ValidationError("design must be a LocalDesign")
        h_min = _positive_float("h_min", self.h_min)
        h_max = _positive_float("h_max", self.h_max)
        p_min = _finite_float("p_min", self.p_min)
        p_max = _finite_float("p_max", self.p_max)
        if h_min >= h_max:
            raise ValidationError("h bounds must have positive width")
        if not 0.0 < p_min < p_max < 1.0:
            raise ValidationError("p bounds must have positive width inside (0, 1)")

        # Endpoint validation proves the full monotone h interval stays in
        # 0 < alpha = 2-r*h < 2 without post-hoc clipping.
        LocalCoordinates(r=self.design.r, h=h_min, p=p_min)
        LocalCoordinates(r=self.design.r, h=h_max, p=p_max)

        area = (h_max - h_min) * (p_max - p_min)
        if not isfinite(area) or area <= 0.0:
            raise ValidationError("prior rectangle must have finite positive area")
        object.__setattr__(self, "h_min", h_min)
        object.__setattr__(self, "h_max", h_max)
        object.__setattr__(self, "p_min", p_min)
        object.__setattr__(self, "p_max", p_max)

    @classmethod
    def default(cls, design: LocalDesign) -> Self:
        """Return the documented default compact prior for ``design``."""
        return cls(design=design)

    @property
    def area(self) -> float:
        """Lebesgue area of the support rectangle."""
        return (self.h_max - self.h_min) * (self.p_max - self.p_min)

    @overload
    def log_density(self, h: float, p: float) -> float: ...

    @overload
    def log_density(self, h: ArrayLike, p: ArrayLike) -> NDArray[np.float64]: ...

    def log_density(
        self,
        h: float | ArrayLike,
        p: float | ArrayLike,
    ) -> float | NDArray[np.float64]:
        """Evaluate the normalized log density, broadcasting array inputs."""
        try:
            h_values, p_values = np.broadcast_arrays(
                np.asarray(h, dtype=np.float64),
                np.asarray(p, dtype=np.float64),
            )
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "h and p must be broadcastable numeric values"
            ) from error
        inside = (
            np.isfinite(h_values)
            & np.isfinite(p_values)
            & (h_values >= self.h_min)
            & (h_values <= self.h_max)
            & (p_values >= self.p_min)
            & (p_values <= self.p_max)
        )
        values = np.where(inside, -log(self.area), -np.inf).astype(
            np.float64,
            copy=False,
        )
        if values.ndim == 0:
            return float(values)
        return values


class NuisanceMode(StrEnum):
    """Closed provenance modes for location and scale."""

    EXTERNALLY_KNOWN = "externally_known"
    PILOT_CONDITIONED = "pilot_conditioned"
    PLUGIN_ESTIMATE = "plugin_estimate"


@dataclass(frozen=True, slots=True)
class KnownNuisance:
    """Recorded location/scale values with explicit provenance."""

    loc: float
    scale: float
    mode: NuisanceMode
    provenance: str

    def __post_init__(self) -> None:
        loc = _finite_float("loc", self.loc)
        scale = _positive_float("scale", self.scale)
        if not isinstance(self.mode, NuisanceMode):
            raise ValidationError("mode must be a NuisanceMode value")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValidationError("provenance must be a nonempty string")
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "provenance", self.provenance.strip())

    @classmethod
    def externally_known(
        cls,
        *,
        loc: float,
        scale: float,
        provenance: str,
    ) -> Self:
        """Construct nuisance values known independently of the main sample."""
        return cls(
            loc=loc,
            scale=scale,
            mode=NuisanceMode.EXTERNALLY_KNOWN,
            provenance=provenance,
        )

    def require_externally_known(self) -> None:
        """Refuse nuisance modes outside the Phase 1 theorem-faithful workflow."""
        if self.mode is not NuisanceMode.EXTERNALLY_KNOWN:
            raise ValidationError(
                "Phase 1 fitting requires independently externally known "
                "nuisance values"
            )


__all__ = ["KnownNuisance", "LocalDesign", "LocalPrior", "NuisanceMode"]
