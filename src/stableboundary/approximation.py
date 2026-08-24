"""Explicit compactly truncated signed-Poisson Gamma--Beta approximation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log
from numbers import Real
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad  # type: ignore[import-untyped]
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import (  # type: ignore[import-untyped]
    betainc,
    betaincc,
    betaln,
    gammainc,
    gammaincc,
    gammaln,
    logsumexp,
    roots_legendre,
)

from ._exceptions import NumericalProbabilityError, ValidationError
from .cells import CellCounts
from .design import LocalDesign, LocalPrior
from .result import CredibleInterval, ParameterSummary

_NODES: Final = 96
_INTERVAL_MASS: Final = 0.90
_QUADRATURE_RELATIVE_TOLERANCE: Final = 2e-12
_QUADRATURE_LIMIT: Final = 200
_CDF_ROUNDOFF_TOLERANCE: Final = 2e-11
_QUANTITIES: Final = (
    "h",
    "p",
    "alpha",
    "beta",
    "tau_plus",
    "tau_minus",
)


def _support_value(name: str, value: float, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or not lower <= result <= upper:
        raise ValidationError(f"{name} must lie in the compact prior support")
    return result


def _axis(
    lower: float, upper: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = roots_legendre(_NODES)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return midpoint + half_width * nodes, half_width * weights


def _normalized_mass(
    log_density: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    log_weight = log_density + np.log(weights)
    log_normalizer = float(logsumexp(log_weight))
    if not isfinite(log_normalizer):
        raise NumericalProbabilityError(
            "truncated limiting posterior normalization is nonfinite"
        )
    mass = np.exp(log_weight - log_normalizer)
    if (
        not np.all(np.isfinite(mass))
        or np.any(mass < 0.0)
        or abs(float(np.sum(mass)) - 1.0) > 1e-12
    ):
        raise NumericalProbabilityError(
            "truncated limiting posterior mass failed normalization"
        )
    return mass, log_normalizer


def _positive_interval_difference(
    lower_cdf: float,
    upper_cdf: float,
    lower_survival: float,
    upper_survival: float,
) -> float | None:
    """Choose the less cancellation-prone representation of interval mass."""
    candidates: list[tuple[float, float]] = []
    for high, low in (
        (upper_cdf, lower_cdf),
        (lower_survival, upper_survival),
    ):
        difference = high - low
        scale = max(abs(high), abs(low), np.finfo(np.float64).tiny)
        if isfinite(difference) and difference > 0.0:
            candidates.append((difference / scale, difference))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _integral(
    integrand: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    points: Sequence[float] = (),
) -> float:
    """Evaluate a nonnegative one-dimensional integral with explicit checks."""
    if lower >= upper:
        return 0.0
    internal_points = sorted({point for point in points if lower < point < upper})
    value, error = quad(
        integrand,
        lower,
        upper,
        epsabs=0.0,
        epsrel=_QUADRATURE_RELATIVE_TOLERANCE,
        limit=_QUADRATURE_LIMIT,
        points=internal_points or None,
    )
    result = float(value)
    estimated_error = float(error)
    tolerance = max(
        np.finfo(np.float64).tiny,
        _CDF_ROUNDOFF_TOLERANCE * abs(result),
    )
    if not isfinite(result) or not isfinite(estimated_error) or result < -tolerance:
        raise NumericalProbabilityError(
            "continuous limiting-posterior integration failed"
        )
    return max(0.0, result)


@dataclass(frozen=True, slots=True)
class _TruncatedContinuousDistribution:
    """A continuously normalized one-dimensional law on compact support."""

    lower: float
    upper: float
    peak: float
    log_kernel: Callable[[float], float]
    base_cdf: Callable[[float], float]
    base_survival: Callable[[float], float]
    _peak_log_kernel: float = field(init=False, repr=False)
    _scaled_normalizer: float = field(init=False, repr=False)
    _base_interval_mass: float | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        peak = min(self.upper, max(self.lower, self.peak))
        peak_log_kernel = float(self.log_kernel(peak))
        if (
            not isfinite(self.lower)
            or not isfinite(self.upper)
            or self.lower >= self.upper
            or not isfinite(peak_log_kernel)
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior support is invalid"
            )
        object.__setattr__(self, "peak", peak)
        object.__setattr__(self, "_peak_log_kernel", peak_log_kernel)
        normalizer = _integral(
            self._scaled_density,
            self.lower,
            self.upper,
            points=(peak,),
        )
        if not isfinite(normalizer) or normalizer <= 0.0:
            raise NumericalProbabilityError(
                "continuous limiting-posterior normalization is nonpositive"
            )
        object.__setattr__(self, "_scaled_normalizer", normalizer)
        object.__setattr__(
            self,
            "_base_interval_mass",
            self._base_mass(self.lower, self.upper),
        )

    def _scaled_density(self, value: float) -> float:
        if value < self.lower or value > self.upper:
            return 0.0
        log_value = float(self.log_kernel(value)) - self._peak_log_kernel
        if not isfinite(log_value):
            raise NumericalProbabilityError(
                "continuous limiting-posterior density is nonfinite"
            )
        return exp(min(0.0, log_value))

    def _base_mass(self, lower: float, upper: float) -> float | None:
        return _positive_interval_difference(
            float(self.base_cdf(lower)),
            float(self.base_cdf(upper)),
            float(self.base_survival(lower)),
            float(self.base_survival(upper)),
        )

    def _adaptive_probability(self, value: float, *, lower_tail: bool) -> float:
        lower_mass = _integral(
            self._scaled_density,
            self.lower,
            value,
            points=(self.peak,),
        )
        upper_mass = _integral(
            self._scaled_density,
            value,
            self.upper,
            points=(self.peak,),
        )
        total = lower_mass + upper_mass
        if not isfinite(total) or total <= 0.0:
            raise NumericalProbabilityError(
                "continuous limiting-posterior CDF failed normalization"
            )
        return (lower_mass if lower_tail else upper_mass) / total

    def cdf(self, value: float) -> float:
        if value <= self.lower:
            return 0.0
        if value >= self.upper:
            return 1.0
        numerator = self._base_mass(self.lower, value)
        denominator = self._base_interval_mass
        if numerator is None or denominator is None or denominator <= 0.0:
            probability = self._adaptive_probability(value, lower_tail=True)
        else:
            probability = numerator / denominator
        if (
            not isfinite(probability)
            or probability < -_CDF_ROUNDOFF_TOLERANCE
            or probability > 1.0 + _CDF_ROUNDOFF_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior CDF is outside [0, 1]"
            )
        return min(1.0, max(0.0, probability))

    def survival(self, value: float) -> float:
        if value <= self.lower:
            return 1.0
        if value >= self.upper:
            return 0.0
        numerator = self._base_mass(value, self.upper)
        denominator = self._base_interval_mass
        if numerator is None or denominator is None or denominator <= 0.0:
            probability = self._adaptive_probability(value, lower_tail=False)
        else:
            probability = numerator / denominator
        if (
            not isfinite(probability)
            or probability < -_CDF_ROUNDOFF_TOLERANCE
            or probability > 1.0 + _CDF_ROUNDOFF_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior survival function is outside [0, 1]"
            )
        return min(1.0, max(0.0, probability))

    def quantile(self, probability: float) -> float:
        if probability <= 0.0:
            return self.lower
        if probability >= 1.0:
            return self.upper
        width = self.upper - self.lower
        root = brentq(
            lambda value: self.cdf(value) - probability,
            self.lower,
            self.upper,
            xtol=max(np.finfo(np.float64).tiny, 1e-13 * width),
            rtol=8.0 * np.finfo(np.float64).eps,
        )
        return float(root)

    def expectation(
        self,
        function: Callable[[float], float],
        points: Sequence[float],
    ) -> float:
        numerator = _integral(
            lambda value: self._scaled_density(value) * function(value),
            self.lower,
            self.upper,
            points=(self.peak, *points),
        )
        probability = numerator / self._scaled_normalizer
        if (
            not isfinite(probability)
            or probability < -_CDF_ROUNDOFF_TOLERANCE
            or probability > 1.0 + _CDF_ROUNDOFF_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "continuous product-distribution CDF is outside [0, 1]"
            )
        return min(1.0, max(0.0, probability))


def _product_quantile(
    h_distribution: _TruncatedContinuousDistribution,
    p_distribution: _TruncatedContinuousDistribution,
    r: float,
    probability: float,
    *,
    positive: bool,
) -> float:
    """Invert the continuous CDF of ``r*H*P`` or ``r*H*(1-P)``."""
    allocation_lower = p_distribution.lower if positive else 1.0 - p_distribution.upper
    allocation_upper = p_distribution.upper if positive else 1.0 - p_distribution.lower
    lower = r * h_distribution.lower * allocation_lower
    upper = r * h_distribution.upper * allocation_upper
    if probability <= 0.0:
        return lower
    if probability >= 1.0:
        return upper

    def product_cdf(value: float) -> float:
        if value <= lower:
            return 0.0
        if value >= upper:
            return 1.0

        def conditional_probability(h: float) -> float:
            allocation = value / (r * h)
            if positive:
                return p_distribution.cdf(allocation)
            return p_distribution.survival(1.0 - allocation)

        breakpoints = (
            value / (r * allocation_lower),
            value / (r * allocation_upper),
        )
        return h_distribution.expectation(conditional_probability, breakpoints)

    root = brentq(
        lambda value: product_cdf(value) - probability,
        lower,
        upper,
        xtol=max(np.finfo(np.float64).tiny, 1e-12 * (upper - lower)),
        rtol=8.0 * np.finfo(np.float64).eps,
    )
    return float(root)


@dataclass(frozen=True, slots=True)
class SignedPoissonIntensities:
    """Limiting independent signed-Poisson means."""

    positive: float
    negative: float

    @property
    def lambda_plus(self) -> float:
        return self.positive

    @property
    def lambda_minus(self) -> float:
        return self.negative


@dataclass(frozen=True, slots=True)
class CompactApproximationSupport:
    """Exact compact rectangle retained by the limiting benchmark."""

    h_min: float
    h_max: float
    p_min: float
    p_max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "h_min": self.h_min,
            "h_max": self.h_max,
            "p_min": self.p_min,
            "p_max": self.p_max,
        }


@dataclass(frozen=True, slots=True)
class LimitingApproximationFit:
    """Distinct result for the compactly truncated analytic limit."""

    counts: CellCounts
    design: LocalDesign
    prior: LocalPrior
    h_nodes: NDArray[np.float64]
    p_nodes: NDArray[np.float64]
    mass: NDArray[np.float64]
    h_truncation_mass: float
    p_truncation_mass: float
    approximation: Literal[True] = field(default=True, init=False)
    method: Literal["signed_poisson_gamma_beta_limit"] = field(
        default="signed_poisson_gamma_beta_limit", init=False
    )
    assumptions: tuple[str, ...] = field(
        default=(
            "independent signed-Poisson count limit",
            "critical-rate local design",
            "uniform prior on the retained compact rectangle",
            "asymptotic benchmark only; not the exact finite-cell posterior",
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        if self.prior.design != self.design:
            raise ValidationError("approximation prior must use the supplied design")
        if self.counts.n != self.design.n:
            raise ValidationError("approximation counts must match the supplied design")
        shape = (_NODES, _NODES)
        retained: list[NDArray[np.float64]] = []
        for name in ("h_nodes", "p_nodes", "mass"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise NumericalProbabilityError(
                    f"limiting approximation {name} grid is invalid"
                )
            copied = np.array(values, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            retained.append(copied)
        if np.any(retained[2] < 0.0) or abs(float(np.sum(retained[2])) - 1.0) > 1e-12:
            raise NumericalProbabilityError(
                "limiting approximation joint mass must normalize"
            )
        if (
            not isfinite(self.h_truncation_mass)
            or not 0.0 < self.h_truncation_mass <= 1.0 + 1e-12
            or not isfinite(self.p_truncation_mass)
            or not 0.0 < self.p_truncation_mass <= 1.0 + 1e-12
        ):
            raise NumericalProbabilityError("truncation masses must be valid")
        for name, values in zip(("h_nodes", "p_nodes", "mass"), retained, strict=True):
            object.__setattr__(self, name, values)

    @property
    def support(self) -> CompactApproximationSupport:
        return CompactApproximationSupport(
            h_min=self.prior.h_min,
            h_max=self.prior.h_max,
            p_min=self.prior.p_min,
            p_max=self.prior.p_max,
        )

    @property
    def h_shape(self) -> float:
        return float(1 + self.counts.n_plus + self.counts.n_minus)

    @property
    def h_rate(self) -> float:
        return 2.0 * self.design.c

    @property
    def p_shape_positive(self) -> float:
        return float(1 + self.counts.n_plus)

    @property
    def p_shape_negative(self) -> float:
        return float(1 + self.counts.n_minus)

    @property
    def gamma_shape(self) -> float:
        return self.h_shape

    @property
    def gamma_rate(self) -> float:
        return self.h_rate

    @property
    def beta_shapes(self) -> tuple[float, float]:
        return self.p_shape_positive, self.p_shape_negative

    @property
    def evidence_status(self) -> str:
        total = self.counts.n_plus + self.counts.n_minus
        if total == 0:
            return "prior_dominated"
        if self.counts.n_plus == 0 or self.counts.n_minus == 0:
            return "one_sided_evidence"
        return "two_sided_evidence"

    def intensities(self, h: float, p: float) -> SignedPoissonIntensities:
        h_value = _support_value("h", h, self.prior.h_min, self.prior.h_max)
        p_value = _support_value("p", p, self.prior.p_min, self.prior.p_max)
        return SignedPoissonIntensities(
            positive=2.0 * self.design.c * h_value * p_value,
            negative=2.0 * self.design.c * h_value * (1.0 - p_value),
        )

    def _values(self, quantity: str) -> NDArray[np.float64]:
        if quantity == "h":
            return self.h_nodes
        if quantity == "p":
            return self.p_nodes
        if quantity == "alpha":
            return 2.0 - self.design.r * self.h_nodes
        if quantity == "beta":
            return 2.0 * self.p_nodes - 1.0
        if quantity == "tau_plus":
            return self.design.r * self.h_nodes * self.p_nodes
        if quantity == "tau_minus":
            return self.design.r * self.h_nodes * (1.0 - self.p_nodes)
        raise ValidationError(f"unknown approximation quantity: {quantity!r}")

    def _h_distribution(self) -> _TruncatedContinuousDistribution:
        shape = self.h_shape
        rate = self.h_rate
        mode = self.prior.h_min if shape <= 1.0 else (shape - 1.0) / rate
        return _TruncatedContinuousDistribution(
            lower=self.prior.h_min,
            upper=self.prior.h_max,
            peak=mode,
            log_kernel=lambda value: (shape - 1.0) * log(value) - rate * value,
            base_cdf=lambda value: float(gammainc(shape, rate * value)),
            base_survival=lambda value: float(gammaincc(shape, rate * value)),
        )

    def _p_distribution(self) -> _TruncatedContinuousDistribution:
        positive_shape = self.p_shape_positive
        negative_shape = self.p_shape_negative
        if positive_shape == 1.0 and negative_shape == 1.0:
            mode = 0.5 * (self.prior.p_min + self.prior.p_max)
        elif positive_shape == 1.0:
            mode = self.prior.p_min
        elif negative_shape == 1.0:
            mode = self.prior.p_max
        else:
            mode = (positive_shape - 1.0) / (positive_shape + negative_shape - 2.0)
        return _TruncatedContinuousDistribution(
            lower=self.prior.p_min,
            upper=self.prior.p_max,
            peak=mode,
            log_kernel=lambda value: (
                (positive_shape - 1.0) * log(value)
                + (negative_shape - 1.0) * np.log1p(-value)
            ),
            base_cdf=lambda value: float(
                betainc(positive_shape, negative_shape, value)
            ),
            base_survival=lambda value: float(
                betaincc(positive_shape, negative_shape, value)
            ),
        )

    def _continuous_quantile(self, quantity: str, probability: float) -> float:
        if quantity == "h":
            return self._h_distribution().quantile(probability)
        if quantity == "p":
            return self._p_distribution().quantile(probability)
        if quantity == "alpha":
            return 2.0 - self.design.r * self._h_distribution().quantile(
                1.0 - probability
            )
        if quantity == "beta":
            return 2.0 * self._p_distribution().quantile(probability) - 1.0
        if quantity in {"tau_plus", "tau_minus"}:
            return _product_quantile(
                self._h_distribution(),
                self._p_distribution(),
                self.design.r,
                probability,
                positive=quantity == "tau_plus",
            )
        raise ValidationError(f"unknown approximation quantity: {quantity!r}")

    def parameter_summary(self, quantity: str) -> ParameterSummary:
        values = self._values(quantity)
        tail = 0.5 * (1.0 - _INTERVAL_MASS)
        return ParameterSummary(
            mean=float(np.sum(self.mass * values)),
            median=self._continuous_quantile(quantity, 0.5),
            credible_interval=CredibleInterval(
                lower=self._continuous_quantile(quantity, tail),
                upper=self._continuous_quantile(quantity, 1.0 - tail),
                mass=_INTERVAL_MASS,
            ),
        )

    def summary(self) -> dict[str, object]:
        return {
            "approximation": self.approximation,
            "method": self.method,
            "assumptions": list(self.assumptions),
            "support": self.support.to_dict(),
            "evidence_status": self.evidence_status,
            "parameters": {
                name: self.parameter_summary(name).to_dict() for name in _QUANTITIES
            },
            "conjugate_parameters": {
                "h_gamma_shape": self.h_shape,
                "h_gamma_rate": self.h_rate,
                "p_beta_shape_positive": self.p_shape_positive,
                "p_beta_shape_negative": self.p_shape_negative,
            },
            "truncation_mass": {
                "h": self.h_truncation_mass,
                "p": self.p_truncation_mass,
            },
        }


def fit_limiting_approximation(
    counts: CellCounts,
    design: LocalDesign,
    prior: LocalPrior | None = None,
) -> LimitingApproximationFit:
    """Fit the explicit compact signed-Poisson Gamma--Beta limit."""
    if not isinstance(counts, CellCounts):
        raise ValidationError("counts must be a CellCounts object")
    if not isinstance(design, LocalDesign):
        raise ValidationError("design must be a LocalDesign object")
    selected_prior = LocalPrior.default(design) if prior is None else prior
    if not isinstance(selected_prior, LocalPrior) or selected_prior.design != design:
        raise ValidationError("prior must be a LocalPrior on the supplied design")
    if (
        counts.n != design.n
        or counts.n_minus + counts.n_zero + counts.n_plus != counts.n
    ):
        raise ValidationError("counts must form the supplied finite design")

    h_shape = float(1 + counts.n_plus + counts.n_minus)
    h_rate = 2.0 * design.c
    p_positive = float(1 + counts.n_plus)
    p_negative = float(1 + counts.n_minus)
    h_axis, h_weights = _axis(selected_prior.h_min, selected_prior.h_max)
    p_axis, p_weights = _axis(selected_prior.p_min, selected_prior.p_max)
    h_log_density = (
        h_shape * log(h_rate)
        - float(gammaln(h_shape))
        + (h_shape - 1.0) * np.log(h_axis)
        - h_rate * h_axis
    )
    p_log_density = (
        (p_positive - 1.0) * np.log(p_axis)
        + (p_negative - 1.0) * np.log1p(-p_axis)
        - float(betaln(p_positive, p_negative))
    )
    h_mass, h_log_normalizer = _normalized_mass(h_log_density, h_weights)
    p_mass, p_log_normalizer = _normalized_mass(p_log_density, p_weights)
    h_grid, p_grid = np.meshgrid(h_axis, p_axis, indexing="ij")
    joint_mass = np.multiply.outer(h_mass, p_mass)
    return LimitingApproximationFit(
        counts=counts,
        design=design,
        prior=selected_prior,
        h_nodes=h_grid,
        p_nodes=p_grid,
        mass=joint_mass,
        h_truncation_mass=float(np.exp(h_log_normalizer)),
        p_truncation_mass=float(np.exp(p_log_normalizer)),
    )


__all__ = ["fit_limiting_approximation"]
