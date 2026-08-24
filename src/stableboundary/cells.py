"""Immutable exact finite three-cell experiment for standardized stable laws."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike

from ._exceptions import NumericalProbabilityError, ValidationError
from .backends import (
    ScipyS0Backend,
    StableBackend,
    validate_s0_backend,
)
from .design import KnownNuisance, LocalDesign
from .parameters import LocalCoordinates, StableParams


@dataclass(frozen=True, slots=True, init=False)
class CellCounts:
    """Counts bound to their complete design and nuisance provenance."""

    n_minus: int
    n_zero: int
    n_plus: int
    design: LocalDesign
    nuisance: KnownNuisance

    def __init__(self) -> None:
        """Prevent construction without validated observations and provenance."""
        raise TypeError("use CellCounts.from_observations() to construct counts")

    @classmethod
    def from_observations(
        cls,
        observations: ArrayLike,
        *,
        nuisance: KnownNuisance,
        design: LocalDesign,
    ) -> Self:
        """Standardize once and partition observations using ``design``."""
        if not isinstance(nuisance, KnownNuisance):
            raise ValidationError("nuisance must be a KnownNuisance object")
        nuisance.require_externally_known()
        if not isinstance(design, LocalDesign):
            raise ValidationError("design must be a LocalDesign object")

        try:
            raw = np.asarray(observations)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError("observations must be a numeric array") from error
        if raw.ndim != 1:
            raise ValidationError("observations must be one-dimensional")
        if raw.dtype.kind not in "iuf":
            raise ValidationError("observations must have a real numeric dtype")
        values = np.asarray(raw, dtype=np.float64)
        if values.size == 0:
            raise ValidationError("observations must not be empty")
        if values.size != design.n:
            raise ValidationError(
                "observation count must equal the prespecified design sample size"
            )
        if not np.all(np.isfinite(values)):
            raise ValidationError("observations must contain only finite values")

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                standardized = (values - nuisance.loc) / nuisance.scale
            except FloatingPointError as error:
                raise ValidationError(
                    "known-nuisance standardization produced nonfinite values"
                ) from error
        if not np.all(np.isfinite(standardized)):
            raise ValidationError(
                "known-nuisance standardization produced nonfinite values"
            )

        negative = standardized < -design.threshold
        positive = standardized > design.threshold
        n_minus = int(np.count_nonzero(negative))
        n_plus = int(np.count_nonzero(positive))
        n = int(standardized.size)
        n_zero = n - n_minus - n_plus
        if n_zero < 0 or n_minus + n_zero + n_plus != n:
            raise NumericalProbabilityError("cell partition failed to preserve n")

        result = object.__new__(cls)
        object.__setattr__(result, "n_minus", n_minus)
        object.__setattr__(result, "n_zero", n_zero)
        object.__setattr__(result, "n_plus", n_plus)
        object.__setattr__(result, "design", design)
        object.__setattr__(result, "nuisance", nuisance)
        return result

    @property
    def threshold(self) -> float:
        """Return the threshold from the retained immutable design."""
        return self.design.threshold

    @property
    def n(self) -> int:
        """Return the sample size from the retained immutable design."""
        return self.design.n


@dataclass(frozen=True, slots=True)
class CellProbabilities:
    """Validated exact-model probabilities for the three disjoint cells."""

    SIMPLEX_TOLERANCE: ClassVar[float] = 1e-12

    q_minus: float
    q_zero: float
    q_plus: float
    log_q_minus: float
    log_q_plus: float
    method: str
    tolerance: float
    parameterization: str = "S0"
    backend_library: str | None = None
    backend_library_version: str | None = None
    backend_effective_settings: tuple[tuple[str, str | int | float | None], ...] = ()

    def __post_init__(self) -> None:
        probabilities = (self.q_minus, self.q_zero, self.q_plus)
        if any(not isfinite(value) for value in probabilities):
            raise NumericalProbabilityError("cell probabilities must be finite")
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise NumericalProbabilityError(
                "cell probabilities must each lie inside [0, 1]"
            )
        total = sum(probabilities)
        if not isfinite(total) or abs(total - 1.0) > self.SIMPLEX_TOLERANCE:
            raise NumericalProbabilityError(
                "cell probabilities do not normalize within the declared tolerance"
            )
        if self.q_minus == 0.0 or self.q_plus == 0.0:
            raise NumericalProbabilityError(
                "interior finite-threshold stable tails must be strictly positive"
            )
        if not isfinite(self.log_q_minus) or not isfinite(self.log_q_plus):
            raise NumericalProbabilityError(
                "cell log-tail probabilities must be finite"
            )
        if self.log_q_minus > 0.0 or self.log_q_plus > 0.0:
            raise NumericalProbabilityError(
                "cell log-tail probabilities cannot be positive"
            )
        if (
            abs(log(self.q_minus) - self.log_q_minus) > self.SIMPLEX_TOLERANCE
            or abs(log(self.q_plus) - self.log_q_plus) > self.SIMPLEX_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "cell probabilities disagree with their direct log-tail values"
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise NumericalProbabilityError("probability method must be named")
        if not isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise NumericalProbabilityError(
                "probability tolerance must be finite and strictly positive"
            )
        if self.parameterization != "S0":
            raise NumericalProbabilityError("cell probabilities require Nolan S0")


def _scalar_log_probability(operation: str, value: object) -> float:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise NumericalProbabilityError(
            f"{operation} returned a nonnumeric log probability"
        ) from error
    if array.ndim != 0:
        raise NumericalProbabilityError(
            f"{operation} returned a nonscalar log probability"
        )
    result = float(array)
    if not isfinite(result) or result > 0.0:
        raise NumericalProbabilityError(
            f"{operation} returned an invalid log probability: {result!r}"
        )
    return result


def exact_cell_probabilities(
    local: LocalCoordinates,
    design: LocalDesign,
    backend: StableBackend | None = None,
) -> CellProbabilities:
    """Evaluate exact finite S0 cell probabilities at one local point."""
    if not isinstance(local, LocalCoordinates):
        raise ValidationError("local must be a LocalCoordinates object")
    if not isinstance(design, LocalDesign):
        raise ValidationError("design must be a LocalDesign object")
    if local.r != design.r:
        raise ValidationError("local coordinates must use the design's exact r")
    candidate: object = ScipyS0Backend() if backend is None else backend
    evaluator, metadata = validate_s0_backend(candidate)

    params = StableParams(
        alpha=local.alpha,
        beta=local.beta,
        loc=0.0,
        scale=1.0,
    )
    log_q_minus = _scalar_log_probability(
        "logcdf",
        evaluator.logcdf(
            -design.threshold,
            params.alpha,
            params.beta,
            loc=0.0,
            scale=1.0,
        ),
    )
    log_q_plus = _scalar_log_probability(
        "logsf",
        evaluator.logsf(
            design.threshold,
            params.alpha,
            params.beta,
            loc=0.0,
            scale=1.0,
        ),
    )

    q_minus = exp(log_q_minus)
    q_plus = exp(log_q_plus)
    if q_minus == 0.0 or q_plus == 0.0:
        raise NumericalProbabilityError(
            "direct interior stable tail underflowed to zero"
        )
    if (
        not isfinite(q_minus)
        or not isfinite(q_plus)
        or q_minus < 0.0
        or q_plus < 0.0
        or q_minus > 1.0
        or q_plus > 1.0
    ):
        raise NumericalProbabilityError("direct stable tails are invalid")
    tail_total = q_minus + q_plus
    if not isfinite(tail_total) or tail_total >= 1.0:
        raise NumericalProbabilityError(
            "direct stable tails leave no positive central-cell probability"
        )
    q_zero = 1.0 - tail_total
    if not isfinite(q_zero) or q_zero <= 0.0:
        raise NumericalProbabilityError("central-cell probability is invalid")

    return CellProbabilities(
        q_minus=q_minus,
        q_zero=q_zero,
        q_plus=q_plus,
        log_q_minus=log_q_minus,
        log_q_plus=log_q_plus,
        method=metadata.method,
        tolerance=metadata.tolerance,
        parameterization=metadata.parameterization,
        backend_library=metadata.library,
        backend_library_version=metadata.library_version,
        backend_effective_settings=metadata.effective_settings,
    )


__all__ = ["CellCounts", "CellProbabilities"]
