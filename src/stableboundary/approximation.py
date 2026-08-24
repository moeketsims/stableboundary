"""Explicit compactly truncated signed-Poisson Gamma--Beta approximation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log, ulp
from numbers import Real
from typing import Final, Literal
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.integrate import IntegrationWarning, quad  # type: ignore[import-untyped]
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import (  # type: ignore[import-untyped]
    betaln,
    gammaln,
)

from ._exceptions import NumericalProbabilityError, ValidationError
from .cells import CellCounts
from .design import LocalDesign, LocalPrior
from .result import CredibleInterval, ParameterSummary

_INTERVAL_MASS: Final = 0.90
_QUADRATURE_RELATIVE_TOLERANCE: Final = 2e-12
_QUADRATURE_LIMIT: Final = 200
_CDF_ROUNDOFF_TOLERANCE: Final = 2e-11
_TRUNCATION_LOG_MASS_PROJECTION_TOLERANCE: Final = 2e-11
_PRODUCT_CDF_RESIDUAL_TOLERANCE: Final = 1e-9
_DENSITY_DROP_LEVELS: Final = (1.0, 4.0, 16.0, 64.0, 256.0, 700.0)
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
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as cause:
        raise ValidationError(f"{name} must be a finite real number") from cause
    if not isfinite(result) or not lower <= result <= upper:
        raise ValidationError(f"{name} must lie in the compact prior support")
    return result


def _integral(
    integrand: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    points: Sequence[float] = (),
    absolute_tolerance: float = 0.0,
) -> float:
    """Evaluate a nonnegative one-dimensional integral with explicit checks."""
    if not isfinite(lower) or not isfinite(upper):
        raise NumericalProbabilityError(
            "continuous limiting-posterior integration bounds are nonfinite"
        )
    if not isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise NumericalProbabilityError(
            "continuous limiting-posterior absolute tolerance is invalid"
        )
    if lower >= upper:
        return 0.0
    internal_points = sorted({point for point in points if lower < point < upper})
    try:
        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            value, reported_error = quad(
                integrand,
                lower,
                upper,
                epsabs=absolute_tolerance,
                epsrel=_QUADRATURE_RELATIVE_TOLERANCE,
                limit=_QUADRATURE_LIMIT,
                points=internal_points or None,
            )
    except (IntegrationWarning, FloatingPointError, OverflowError, ValueError) as cause:
        raise NumericalProbabilityError(
            "continuous limiting-posterior integration did not converge"
        ) from cause
    result = float(value)
    estimated_error = float(reported_error)
    convergence_tolerance = max(
        np.finfo(np.float64).tiny,
        absolute_tolerance,
        _QUADRATURE_RELATIVE_TOLERANCE * abs(result),
    )
    probability_tolerance = max(
        np.finfo(np.float64).tiny,
        _CDF_ROUNDOFF_TOLERANCE * abs(result),
    )
    if (
        not isfinite(result)
        or not isfinite(estimated_error)
        or estimated_error < 0.0
        or estimated_error > convergence_tolerance
        or result < -probability_tolerance
    ):
        raise NumericalProbabilityError(
            "continuous limiting-posterior integration did not converge"
        )
    return max(0.0, result)


@dataclass(frozen=True, slots=True)
class _TruncatedContinuousDistribution:
    """A continuously normalized one-dimensional law on compact support."""

    lower: float
    upper: float
    peak: float
    log_kernel: Callable[[float], float]
    _peak_log_kernel: float = field(init=False, repr=False)
    _integration_points: tuple[float, ...] = field(init=False, repr=False)
    _scaled_normalizer: float = field(init=False, repr=False)
    _log_truncation_mass: float = field(init=False, repr=False)
    _truncation_mass_projected: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isfinite(self.lower)
            or not isfinite(self.upper)
            or self.lower >= self.upper
            or not isfinite(self.peak)
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior support is invalid"
            )
        peak = min(self.upper, max(self.lower, self.peak))
        peak_log_kernel = float(self.log_kernel(peak))
        if not isfinite(peak_log_kernel):
            raise NumericalProbabilityError(
                "continuous limiting-posterior support is invalid"
            )
        object.__setattr__(self, "peak", peak)
        object.__setattr__(self, "_peak_log_kernel", peak_log_kernel)
        integration_points = self._density_level_points()
        object.__setattr__(self, "_integration_points", integration_points)
        normalizer = _integral(
            self._scaled_density,
            self.lower,
            self.upper,
            points=(*integration_points, peak),
        )
        if not isfinite(normalizer) or normalizer <= 0.0:
            raise NumericalProbabilityError(
                "continuous limiting-posterior normalization is nonpositive"
            )
        object.__setattr__(self, "_scaled_normalizer", normalizer)
        log_truncation_mass = peak_log_kernel + log(normalizer)
        if (
            not isfinite(log_truncation_mass)
            or log_truncation_mass > _TRUNCATION_LOG_MASS_PROJECTION_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior truncation mass is invalid"
            )
        projected = log_truncation_mass > 0.0
        if projected:
            log_truncation_mass = 0.0
        object.__setattr__(self, "_log_truncation_mass", log_truncation_mass)
        object.__setattr__(self, "_truncation_mass_projected", projected)

    @property
    def log_truncation_mass(self) -> float:
        """Return the base-law interval mass in the log domain."""
        return self._log_truncation_mass

    @property
    def truncation_mass(self) -> float:
        """Return the base-law interval mass, allowing honest underflow to zero."""
        return exp(self._log_truncation_mass)

    @property
    def truncation_mass_projected(self) -> bool:
        """Report whether roundoff above unit mass was projected to one."""
        return self._truncation_mass_projected

    def _scaled_density(self, value: float) -> float:
        if value < self.lower or value > self.upper:
            return 0.0
        log_value = float(self.log_kernel(value)) - self._peak_log_kernel
        if not isfinite(log_value):
            raise NumericalProbabilityError(
                "continuous limiting-posterior density is nonfinite"
            )
        if log_value > _CDF_ROUNDOFF_TOLERANCE:
            raise NumericalProbabilityError(
                "continuous limiting-posterior peak does not bound its density"
            )
        return exp(min(0.0, log_value))

    def _density_level_points(self) -> tuple[float, ...]:
        """Locate log-density contours so quadrature resolves narrow modes."""
        points: list[float] = []
        for boundary in (self.lower, self.upper):
            if boundary == self.peak:
                continue
            boundary_drop = float(self.log_kernel(boundary)) - self._peak_log_kernel
            if not isfinite(boundary_drop):
                raise NumericalProbabilityError(
                    "continuous limiting-posterior endpoint density is nonfinite"
                )
            for level in _DENSITY_DROP_LEVELS:
                if boundary_drop >= -level:
                    continue
                left, right = sorted((boundary, self.peak))
                try:
                    point = brentq(
                        lambda value, level=level: (
                            float(self.log_kernel(value))
                            - self._peak_log_kernel
                            + level
                        ),
                        left,
                        right,
                        xtol=max(
                            np.finfo(np.float64).tiny,
                            2e-14 * (self.upper - self.lower),
                        ),
                        rtol=8.0 * np.finfo(np.float64).eps,
                    )
                except (FloatingPointError, OverflowError, ValueError) as cause:
                    raise NumericalProbabilityError(
                        "continuous limiting-posterior density contour failed"
                    ) from cause
                points.append(float(point))
        return tuple(sorted(set(points)))

    def _probability(self, value: float, *, lower_tail: bool) -> float:
        numerator = _integral(
            self._scaled_density,
            self.lower if lower_tail else value,
            value if lower_tail else self.upper,
            points=(*self._integration_points, self.peak),
            absolute_tolerance=(
                _QUADRATURE_RELATIVE_TOLERANCE * self._scaled_normalizer
            ),
        )
        return numerator / self._scaled_normalizer

    @staticmethod
    def _validated_probability(probability: float, operation: str) -> float:
        if (
            not isfinite(probability)
            or probability < -_CDF_ROUNDOFF_TOLERANCE
            or probability > 1.0 + _CDF_ROUNDOFF_TOLERANCE
        ):
            raise NumericalProbabilityError(
                f"continuous limiting-posterior {operation} is outside [0, 1]"
            )
        return min(1.0, max(0.0, probability))

    def cdf(self, value: float) -> float:
        if not isfinite(value):
            raise NumericalProbabilityError(
                "continuous limiting-posterior CDF argument is nonfinite"
            )
        if value <= self.lower:
            return 0.0
        if value >= self.upper:
            return 1.0
        return self._validated_probability(
            self._probability(value, lower_tail=True), "CDF"
        )

    def survival(self, value: float) -> float:
        if not isfinite(value):
            raise NumericalProbabilityError(
                "continuous limiting-posterior survival argument is nonfinite"
            )
        if value <= self.lower:
            return 1.0
        if value >= self.upper:
            return 0.0
        return self._validated_probability(
            self._probability(value, lower_tail=False), "survival function"
        )

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
            xtol=max(np.finfo(np.float64).tiny, 2e-14 * width),
            rtol=8.0 * np.finfo(np.float64).eps,
        )
        result = float(root)
        cdf_residual = abs(self.cdf(result) - probability)
        survival_residual = abs(self.survival(result) - (1.0 - probability))
        if (
            not isfinite(cdf_residual)
            or not isfinite(survival_residual)
            or cdf_residual > _CDF_ROUNDOFF_TOLERANCE
            or survival_residual > _CDF_ROUNDOFF_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior quantile failed its CDF check"
            )
        return result

    def mean(self) -> float:
        """Integrate the conditional mean against the shared normalizer."""
        numerator = _integral(
            lambda value: self._scaled_density(value) * value,
            self.lower,
            self.upper,
            points=(*self._integration_points, self.peak),
            absolute_tolerance=(
                _QUADRATURE_RELATIVE_TOLERANCE * self._scaled_normalizer
            ),
        )
        result = numerator / self._scaled_normalizer
        if (
            not isfinite(result)
            or result < self.lower - _CDF_ROUNDOFF_TOLERANCE
            or result > self.upper + _CDF_ROUNDOFF_TOLERANCE
        ):
            raise NumericalProbabilityError(
                "continuous limiting-posterior mean is outside its support"
            )
        return min(self.upper, max(self.lower, result))

    def expectation(
        self,
        function: Callable[[float], float],
        points: Sequence[float],
    ) -> float:
        numerator = _integral(
            lambda value: self._scaled_density(value) * function(value),
            self.lower,
            self.upper,
            points=(*self._integration_points, self.peak, *points),
            absolute_tolerance=(_CDF_ROUNDOFF_TOLERANCE * self._scaled_normalizer),
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


@dataclass(frozen=True, slots=True)
class _ReflectedContinuousDistribution:
    """Expose a compact law in ``p = 1 - u`` without right-tail cancellation."""

    base: _TruncatedContinuousDistribution

    @property
    def lower(self) -> float:
        return 1.0 - self.base.upper

    @property
    def upper(self) -> float:
        return 1.0 - self.base.lower

    @property
    def log_truncation_mass(self) -> float:
        return self.base.log_truncation_mass

    @property
    def truncation_mass(self) -> float:
        return self.base.truncation_mass

    @property
    def truncation_mass_projected(self) -> bool:
        return self.base.truncation_mass_projected

    def cdf(self, value: float) -> float:
        return self.base.survival(1.0 - value)

    def survival(self, value: float) -> float:
        return self.base.cdf(1.0 - value)

    def quantile(self, probability: float) -> float:
        # Validate both probability tails before reflecting.  Near p=1, a single
        # binary64 p-ULP can span more than the CDF residual tolerance even though
        # the u-coordinate quantile itself is fully resolved.
        return 1.0 - self.base.quantile(1.0 - probability)

    def mean(self) -> float:
        return 1.0 - self.base.mean()

    def expectation(
        self,
        function: Callable[[float], float],
        points: Sequence[float],
    ) -> float:
        return self.base.expectation(
            lambda value: function(1.0 - value),
            tuple(1.0 - point for point in points),
        )


_ContinuousDistribution = (
    _TruncatedContinuousDistribution | _ReflectedContinuousDistribution
)


def _product_cdf_condition_on_h(
    value: float,
    h_distribution: _ContinuousDistribution,
    p_distribution: _ContinuousDistribution,
    r: float,
    *,
    positive: bool,
) -> float:
    """Evaluate a product CDF by conditioning on ``H``."""
    allocation_lower = p_distribution.lower if positive else 1.0 - p_distribution.upper
    allocation_upper = p_distribution.upper if positive else 1.0 - p_distribution.lower
    lower = r * h_distribution.lower * allocation_lower
    upper = r * h_distribution.upper * allocation_upper
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


def _product_cdf_condition_on_p(
    value: float,
    h_distribution: _ContinuousDistribution,
    p_distribution: _ContinuousDistribution,
    r: float,
    *,
    positive: bool,
) -> float:
    """Independently evaluate a product CDF by conditioning on ``P``."""
    allocation_lower = p_distribution.lower if positive else 1.0 - p_distribution.upper
    allocation_upper = p_distribution.upper if positive else 1.0 - p_distribution.lower
    lower = r * h_distribution.lower * allocation_lower
    upper = r * h_distribution.upper * allocation_upper
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0

    def conditional_probability(p: float) -> float:
        allocation = p if positive else 1.0 - p
        return h_distribution.cdf(value / (r * allocation))

    breakpoints = tuple(
        (value / (r * h_bound) if positive else 1.0 - value / (r * h_bound))
        for h_bound in (h_distribution.lower, h_distribution.upper)
    )
    return p_distribution.expectation(conditional_probability, breakpoints)


def _product_quantile(
    h_distribution: _ContinuousDistribution,
    p_distribution: _ContinuousDistribution,
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

    primary_cdf = (
        _product_cdf_condition_on_h if positive else _product_cdf_condition_on_p
    )
    independent_cdf = (
        _product_cdf_condition_on_p if positive else _product_cdf_condition_on_h
    )
    root = brentq(
        lambda value: (
            primary_cdf(
                value,
                h_distribution,
                p_distribution,
                r,
                positive=positive,
            )
            - probability
        ),
        lower,
        upper,
        xtol=max(np.finfo(np.float64).tiny, 2e-14 * (upper - lower)),
        rtol=8.0 * np.finfo(np.float64).eps,
    )
    result = float(root)
    direct_probability = primary_cdf(
        result,
        h_distribution,
        p_distribution,
        r,
        positive=positive,
    )
    independent_probability = independent_cdf(
        result,
        h_distribution,
        p_distribution,
        r,
        positive=positive,
    )
    if (
        not isfinite(direct_probability)
        or not isfinite(independent_probability)
        or abs(direct_probability - probability) > _PRODUCT_CDF_RESIDUAL_TOLERANCE
        or abs(independent_probability - probability) > _PRODUCT_CDF_RESIDUAL_TOLERANCE
    ):
        raise NumericalProbabilityError(
            "continuous product quantile failed its independent CDF check"
        )
    return result


def _gamma_distribution(
    shape: float,
    rate: float,
    lower: float,
    upper: float,
) -> _TruncatedContinuousDistribution:
    """Construct the normalized compact Gamma law used by the approximation."""
    mode = lower if shape <= 1.0 else (shape - 1.0) / rate
    log_normalizing_constant = shape * log(rate) - float(gammaln(shape))
    return _TruncatedContinuousDistribution(
        lower=lower,
        upper=upper,
        peak=mode,
        log_kernel=lambda value: (
            log_normalizing_constant + (shape - 1.0) * log(value) - rate * value
        ),
    )


def _beta_distribution(
    positive_shape: float,
    negative_shape: float,
    lower: float,
    upper: float,
) -> _ContinuousDistribution:
    """Construct the normalized compact Beta law used by the approximation."""
    if lower + upper > 1.0:
        reflected = _beta_distribution(
            negative_shape,
            positive_shape,
            1.0 - upper,
            1.0 - lower,
        )
        if not isinstance(reflected, _TruncatedContinuousDistribution):
            raise NumericalProbabilityError(
                "reflected Beta construction did not reach a stable coordinate"
            )
        return _ReflectedContinuousDistribution(reflected)
    if positive_shape == 1.0 and negative_shape == 1.0:
        mode = 0.5 * (lower + upper)
    elif positive_shape == 1.0:
        mode = lower
    elif negative_shape == 1.0:
        mode = upper
    else:
        mode = (positive_shape - 1.0) / (positive_shape + negative_shape - 2.0)
    log_normalizing_constant = -float(betaln(positive_shape, negative_shape))
    return _TruncatedContinuousDistribution(
        lower=lower,
        upper=upper,
        peak=mode,
        log_kernel=lambda value: (
            log_normalizing_constant
            + (positive_shape - 1.0) * log(value)
            + (negative_shape - 1.0) * np.log1p(-value)
        ),
    )


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


@dataclass(frozen=True, slots=True, init=False)
class LimitingApproximationFit:
    """Distinct result for the compactly truncated continuous analytic limit."""

    counts: CellCounts
    design: LocalDesign
    prior: LocalPrior
    h_truncation_mass: float
    p_truncation_mass: float
    h_log_truncation_mass: float
    p_log_truncation_mass: float
    h_truncation_mass_projected: bool
    p_truncation_mass_projected: bool
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

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Prevent public composition of independently produced components."""
        del args, kwargs
        raise TypeError(
            "use stableboundary.fit_limiting_approximation() to construct a fit"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.counts, CellCounts):
            raise ValidationError("approximation counts must be a CellCounts object")
        if not isinstance(self.design, LocalDesign):
            raise ValidationError("approximation design must be a LocalDesign object")
        if not isinstance(self.prior, LocalPrior):
            raise ValidationError("approximation prior must be a LocalPrior object")
        if self.prior.design != self.design:
            raise ValidationError("approximation prior must use the supplied design")
        if self.counts.design != self.design:
            raise ValidationError(
                "approximation counts must retain the full supplied design"
            )
        if (
            self.counts.n_minus + self.counts.n_zero + self.counts.n_plus
            != self.design.n
        ):
            raise ValidationError("approximation counts must form the supplied design")
        self._validate_truncation_mass(
            "h", self.h_truncation_mass, self.h_log_truncation_mass
        )
        self._validate_truncation_mass(
            "p", self.p_truncation_mass, self.p_log_truncation_mass
        )
        if not isinstance(self.h_truncation_mass_projected, bool) or not isinstance(
            self.p_truncation_mass_projected, bool
        ):
            raise NumericalProbabilityError(
                "truncation-mass projection evidence must be boolean"
            )
        for name, mass, log_mass, projected in (
            (
                "h",
                self.h_truncation_mass,
                self.h_log_truncation_mass,
                self.h_truncation_mass_projected,
            ),
            (
                "p",
                self.p_truncation_mass,
                self.p_log_truncation_mass,
                self.p_truncation_mass_projected,
            ),
        ):
            if projected and (mass != 1.0 or log_mass != 0.0):
                raise NumericalProbabilityError(
                    f"{name} projected truncation mass must be exactly one"
                )

    @staticmethod
    def _validate_truncation_mass(name: str, mass: float, log_mass: float) -> None:
        if (
            not isfinite(mass)
            or mass < 0.0
            or mass > 1.0
            or not isfinite(log_mass)
            or log_mass > 0.0
        ):
            raise NumericalProbabilityError(f"{name} truncation mass must be valid")
        expected_mass = exp(log_mass)
        if expected_mass == 0.0:
            consistent = mass == 0.0
        elif mass == 0.0:
            consistent = False
        else:
            consistent = abs(mass - expected_mass) <= max(ulp(mass), ulp(expected_mass))
        if not consistent:
            raise NumericalProbabilityError(
                f"{name} truncation mass disagrees with its log mass"
            )

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

    def _h_distribution(self) -> _TruncatedContinuousDistribution:
        return _gamma_distribution(
            self.h_shape,
            self.h_rate,
            self.prior.h_min,
            self.prior.h_max,
        )

    def _p_distribution(self) -> _ContinuousDistribution:
        return _beta_distribution(
            self.p_shape_positive,
            self.p_shape_negative,
            self.prior.p_min,
            self.prior.p_max,
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

    def _continuous_mean(self, quantity: str) -> float:
        h_mean = self._h_distribution().mean()
        if quantity == "h":
            return h_mean
        if quantity == "alpha":
            return 2.0 - self.design.r * h_mean
        p_mean = self._p_distribution().mean()
        if quantity == "p":
            return p_mean
        if quantity == "beta":
            return 2.0 * p_mean - 1.0
        if quantity == "tau_plus":
            return self.design.r * h_mean * p_mean
        if quantity == "tau_minus":
            return self.design.r * h_mean * (1.0 - p_mean)
        raise ValidationError(f"unknown approximation quantity: {quantity!r}")

    def parameter_summary(self, quantity: str) -> ParameterSummary:
        tail = 0.5 * (1.0 - _INTERVAL_MASS)
        return ParameterSummary(
            mean=self._continuous_mean(quantity),
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
            "log_truncation_mass": {
                "h": self.h_log_truncation_mass,
                "p": self.p_log_truncation_mass,
            },
            "truncation_mass_projected": {
                "h": self.h_truncation_mass_projected,
                "p": self.p_truncation_mass_projected,
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
    if counts.design != design:
        raise ValidationError("counts must retain the full supplied design")
    selected_prior = LocalPrior.default(design) if prior is None else prior
    if not isinstance(selected_prior, LocalPrior) or selected_prior.design != design:
        raise ValidationError("prior must be a LocalPrior on the supplied design")
    if counts.n_minus + counts.n_zero + counts.n_plus != design.n:
        raise ValidationError("counts must form the supplied finite design")

    h_shape = float(1 + counts.n_plus + counts.n_minus)
    h_rate = 2.0 * design.c
    p_positive = float(1 + counts.n_plus)
    p_negative = float(1 + counts.n_minus)
    h_distribution = _gamma_distribution(
        h_shape,
        h_rate,
        selected_prior.h_min,
        selected_prior.h_max,
    )
    p_distribution = _beta_distribution(
        p_positive,
        p_negative,
        selected_prior.p_min,
        selected_prior.p_max,
    )
    result = object.__new__(LimitingApproximationFit)
    object.__setattr__(result, "counts", counts)
    object.__setattr__(result, "design", design)
    object.__setattr__(result, "prior", selected_prior)
    object.__setattr__(result, "h_truncation_mass", h_distribution.truncation_mass)
    object.__setattr__(result, "p_truncation_mass", p_distribution.truncation_mass)
    object.__setattr__(
        result, "h_log_truncation_mass", h_distribution.log_truncation_mass
    )
    object.__setattr__(
        result, "p_log_truncation_mass", p_distribution.log_truncation_mass
    )
    object.__setattr__(
        result,
        "h_truncation_mass_projected",
        h_distribution.truncation_mass_projected,
    )
    object.__setattr__(
        result,
        "p_truncation_mass_projected",
        p_distribution.truncation_mass_projected,
    )
    object.__setattr__(result, "approximation", True)
    object.__setattr__(result, "method", "signed_poisson_gamma_beta_limit")
    object.__setattr__(
        result,
        "assumptions",
        (
            "independent signed-Poisson count limit",
            "critical-rate local design",
            "uniform prior on the retained compact rectangle",
            "asymptotic benchmark only; not the exact finite-cell posterior",
        ),
    )
    result.__post_init__()
    return result


__all__ = ["fit_limiting_approximation"]
