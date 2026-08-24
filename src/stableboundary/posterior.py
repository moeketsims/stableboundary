"""Deterministic quadrature for the exact finite three-cell posterior."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import (  # type: ignore[import-untyped]
    gammaln,
    logsumexp,
    roots_legendre,
    xlogy,
)

from ._exceptions import ConvergenceError, ValidationError
from .backends import (
    BackendMetadata,
    ScipyS0Backend,
    StableBackend,
    validate_s0_backend,
    validate_s0_metadata,
)
from .cells import CellCounts
from .design import KnownNuisance, LocalDesign, LocalPrior

_QUANTITIES: Final = (
    "h",
    "p",
    "alpha",
    "beta",
    "tau_plus",
    "tau_minus",
)
_PUSHFORWARD_NODES, _PUSHFORWARD_WEIGHTS = roots_legendre(8)
_CANONICAL_SCIPY_S0_TYPE: Final = ScipyS0Backend


def _bounded_integer(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValidationError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _bounded_real(
    name: str,
    value: float,
    *,
    lower: float,
    upper: float,
    upper_inclusive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    upper_valid = result <= upper if upper_inclusive else result < upper
    if not isfinite(result) or not lower < result or not upper_valid:
        closing = "]" if upper_inclusive else ")"
        raise ValidationError(f"{name} must lie in ({lower}, {upper}{closing}")
    return result


@dataclass(frozen=True, slots=True)
class QuadratureConfig:
    """Bounded deterministic controls for posterior quadrature and refinement."""

    base_nodes: int = 20
    refined_nodes: int = 32
    refinement_tolerance: float = 0.002
    interval_mass: float = 0.90
    common_grid_points: int = 65

    def __post_init__(self) -> None:
        base = _bounded_integer("base_nodes", self.base_nodes, minimum=2, maximum=256)
        refined = _bounded_integer(
            "refined_nodes", self.refined_nodes, minimum=2, maximum=384
        )
        common = _bounded_integer(
            "common_grid_points", self.common_grid_points, minimum=3, maximum=513
        )
        tolerance = _bounded_real(
            "refinement_tolerance",
            self.refinement_tolerance,
            lower=0.0,
            upper=1.0,
            upper_inclusive=True,
        )
        interval_mass = _bounded_real(
            "interval_mass",
            self.interval_mass,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        object.__setattr__(self, "base_nodes", base)
        object.__setattr__(self, "refined_nodes", refined)
        object.__setattr__(self, "common_grid_points", common)
        object.__setattr__(self, "refinement_tolerance", tolerance)
        object.__setattr__(self, "interval_mass", interval_mass)


@dataclass(frozen=True, slots=True)
class SummaryRefinement:
    """Support-normalized refinement changes for one posterior quantity."""

    quantity: str
    mean: float
    median: float
    interval_lower: float
    interval_upper: float

    @property
    def maximum(self) -> float:
        return max(self.mean, self.median, self.interval_lower, self.interval_upper)


@dataclass(frozen=True, slots=True)
class _PosteriorSummary:
    """One retained continuous-posterior summary used by the public result."""

    quantity: str
    mean: float
    median: float
    interval_lower: float
    interval_upper: float

    def __post_init__(self) -> None:
        if self.quantity not in _QUANTITIES:
            raise ConvergenceError(f"unknown posterior summary: {self.quantity!r}")
        values = (self.mean, self.median, self.interval_lower, self.interval_upper)
        if not all(isfinite(value) for value in values):
            raise ConvergenceError("posterior summaries must be finite")
        if not self.interval_lower <= self.median <= self.interval_upper:
            raise ConvergenceError("posterior interval must contain its median")


@dataclass(frozen=True, slots=True)
class PredictiveTailRefinement:
    """Symmetric relative changes in signed design-tail predictions."""

    positive: float
    negative: float

    @property
    def maximum(self) -> float:
        return max(self.positive, self.negative)


@dataclass(frozen=True, slots=True)
class RefinementDiagnostics:
    """Complete common-grid evidence for the base-to-refined comparison."""

    tolerance: float
    common_grid_points: int
    joint_total_variation: float
    log_normalizer_change: float
    summaries: tuple[SummaryRefinement, ...]
    predictive_tail: PredictiveTailRefinement
    converged: bool

    @property
    def maximum_component(self) -> float:
        values = [
            self.joint_total_variation,
            self.log_normalizer_change,
            self.predictive_tail.maximum,
        ]
        values.extend(item.maximum for item in self.summaries)
        return max(values)


def _validate_experiment_provenance(
    counts: CellCounts,
    design: LocalDesign,
    prior: LocalPrior,
) -> None:
    """Reject composition across finite experiments before numerical work."""
    retained_design = getattr(counts, "design", None)
    retained_nuisance = getattr(counts, "nuisance", None)
    if not isinstance(retained_design, LocalDesign) or retained_design != design:
        raise ValidationError("counts must retain the supplied full design")
    if counts.threshold != design.threshold or counts.n != design.n:
        raise ValidationError("counts threshold and sample size must match the design")
    if not isinstance(retained_nuisance, KnownNuisance):
        raise ValidationError("counts must retain known-nuisance provenance")
    retained_nuisance.require_externally_known()
    if prior.design != design:
        raise ValidationError("prior must be defined on the supplied design")
    values = (counts.n_minus, counts.n_zero, counts.n_plus)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValidationError("cell counts must be integers")
    if any(int(value) < 0 for value in values) or sum(map(int, values)) != design.n:
        raise ValidationError("cell counts must form the supplied finite design")


@dataclass(frozen=True, slots=True)
class PosteriorGrid:
    """Read-only refined tensor grid and normalized posterior probability mass."""

    h_nodes: NDArray[np.float64]
    p_nodes: NDArray[np.float64]
    mass: NDArray[np.float64]
    q_minus: NDArray[np.float64]
    q_plus: NDArray[np.float64]
    log_normalizer: float
    base_nodes: int
    refined_nodes: int
    interval_mass: float
    summaries: tuple[_PosteriorSummary, ...]
    design: LocalDesign
    prior: LocalPrior
    counts: CellCounts
    backend_metadata: BackendMetadata
    backend_origin: Literal["canonical_scipy_s0", "custom"]
    refinement: RefinementDiagnostics

    def __post_init__(self) -> None:
        shape = (self.refined_nodes, self.refined_nodes)
        arrays: list[NDArray[np.float64]] = []
        for name in ("h_nodes", "p_nodes", "mass", "q_minus", "q_plus"):
            raw = np.asarray(getattr(self, name), dtype=np.float64)
            if raw.shape != shape or not np.all(np.isfinite(raw)):
                raise ConvergenceError(f"{name} must be a finite {shape!r} array")
            value = np.array(raw, dtype=np.float64, copy=True)
            value.setflags(write=False)
            arrays.append(value)
        if np.any(arrays[2] < 0.0) or abs(float(np.sum(arrays[2])) - 1.0) > 1e-12:
            raise ConvergenceError("posterior mass must be nonnegative and normalize")
        if not isfinite(self.log_normalizer):
            raise ConvergenceError("posterior log normalizer must be finite")
        if tuple(summary.quantity for summary in self.summaries) != _QUANTITIES:
            raise ConvergenceError(
                "posterior summaries must cover each supported quantity once"
            )
        _validate_experiment_provenance(self.counts, self.design, self.prior)
        validate_s0_metadata(self.backend_metadata)
        if self.backend_origin not in {"canonical_scipy_s0", "custom"}:
            raise ValidationError("posterior backend origin is invalid")
        if (
            self.backend_origin == "canonical_scipy_s0"
            and self.backend_metadata != _CANONICAL_SCIPY_S0_TYPE().metadata
        ):
            raise ValidationError(
                "canonical SciPy posterior metadata does not match the package backend"
            )
        for name, value in zip(
            ("h_nodes", "p_nodes", "mass", "q_minus", "q_plus"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def backend_method(self) -> str:
        return self.backend_metadata.method

    @property
    def backend_tolerance(self) -> float:
        return self.backend_metadata.tolerance

    @property
    def backend_parameterization(self) -> str:
        return self.backend_metadata.parameterization

    @property
    def r(self) -> float:
        """Return the local rate from the retained full design."""
        return self.design.r

    def prediction_backend(self) -> StableBackend:
        """Reconstruct the canonical backend or refuse custom-backend prediction."""
        if self.backend_origin != "canonical_scipy_s0":
            raise ValidationError(
                "prediction is unavailable for a posterior fitted with a custom "
                "backend; refit with the package canonical SciPy S0 backend"
            )
        backend = _CANONICAL_SCIPY_S0_TYPE()
        _, live_metadata = validate_s0_backend(backend)
        if live_metadata != self.backend_metadata:
            raise ValidationError(
                "canonical SciPy backend metadata changed after inference"
            )
        return backend

    def summary_record(self, quantity: str) -> _PosteriorSummary:
        """Return the retained summary assessed by the refinement gate."""
        for summary in self.summaries:
            if summary.quantity == quantity:
                return summary
        raise ValidationError(f"unknown posterior quantity: {quantity!r}")

    def values(self, quantity: str) -> NDArray[np.float64]:
        """Return a read-only derived quantity on the retained grid."""
        if quantity == "h":
            values = self.h_nodes
        elif quantity == "p":
            values = self.p_nodes
        elif quantity == "alpha":
            values = 2.0 - self.r * self.h_nodes
        elif quantity == "beta":
            values = 2.0 * self.p_nodes - 1.0
        elif quantity == "tau_plus":
            values = self.r * self.h_nodes * self.p_nodes
        elif quantity == "tau_minus":
            values = self.r * self.h_nodes * (1.0 - self.p_nodes)
        else:
            raise ValidationError(f"unknown posterior quantity: {quantity!r}")
        result = np.array(values, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class _GridEvaluation:
    h_axis: NDArray[np.float64]
    p_axis: NDArray[np.float64]
    h_nodes: NDArray[np.float64]
    p_nodes: NDArray[np.float64]
    mass: NDArray[np.float64]
    log_density: NDArray[np.float64]
    q_minus: NDArray[np.float64]
    q_plus: NDArray[np.float64]
    log_normalizer: float


def _legendre_axis(
    lower: float, upper: float, nodes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    raw_nodes, raw_weights = roots_legendre(nodes)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return midpoint + half_width * raw_nodes, half_width * raw_weights


def _evaluate_grid(
    counts: CellCounts,
    design: LocalDesign,
    prior: LocalPrior,
    nodes: int,
    backend: StableBackend,
) -> _GridEvaluation:
    quadrature_h, h_weights = _legendre_axis(prior.h_min, prior.h_max, nodes)
    quadrature_p, p_weights = _legendre_axis(prior.p_min, prior.p_max, nodes)
    # Shared support endpoints eliminate extrapolation from the bilinear
    # refinement comparison while quadrature itself remains Gauss--Legendre.
    h_axis = np.concatenate(([prior.h_min], quadrature_h, [prior.h_max]))
    p_axis = np.concatenate(([prior.p_min], quadrature_p, [prior.p_max]))
    interpolation_h, interpolation_p = np.meshgrid(h_axis, p_axis, indexing="ij")
    h_grid = interpolation_h[1:-1, 1:-1]
    p_grid = interpolation_p[1:-1, 1:-1]
    alpha = 2.0 - design.r * interpolation_h
    beta = 2.0 * interpolation_p - 1.0
    try:
        log_q_minus = np.asarray(
            backend.logcdf(
                -design.threshold,
                alpha,
                beta,
                loc=0.0,
                scale=1.0,
            ),
            dtype=np.float64,
        )
        log_q_plus = np.asarray(
            backend.logsf(
                design.threshold,
                alpha,
                beta,
                loc=0.0,
                scale=1.0,
            ),
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ConvergenceError(
            "exact finite cell backend returned nonnumeric log tails"
        ) from error
    if (
        log_q_minus.shape != interpolation_h.shape
        or log_q_plus.shape != interpolation_h.shape
    ):
        raise ConvergenceError("exact finite cell backend returned a wrong-shaped grid")
    if (
        not np.all(np.isfinite(log_q_minus))
        or not np.all(np.isfinite(log_q_plus))
        or np.any(log_q_minus > 0.0)
        or np.any(log_q_plus > 0.0)
    ):
        raise ConvergenceError("exact finite cell backend returned invalid log tails")
    q_minus = np.exp(log_q_minus)
    q_plus = np.exp(log_q_plus)
    tail_total = q_minus + q_plus
    if (
        np.any(q_minus <= 0.0)
        or np.any(q_plus <= 0.0)
        or not np.all(np.isfinite(tail_total))
        or np.any(tail_total >= 1.0)
    ):
        raise ConvergenceError(
            "exact finite tail probabilities are outside the simplex"
        )
    log_q_zero = np.log1p(-tail_total)
    q_zero = np.exp(log_q_zero)
    if not np.all(np.isfinite(q_zero)) or np.any(q_zero <= 0.0):
        raise ConvergenceError("exact finite central-cell probabilities are invalid")

    coefficient = float(
        gammaln(counts.n + 1)
        - gammaln(counts.n_minus + 1)
        - gammaln(counts.n_zero + 1)
        - gammaln(counts.n_plus + 1)
    )
    log_likelihood = coefficient + (
        xlogy(counts.n_minus, q_minus)
        + xlogy(counts.n_zero, q_zero)
        + xlogy(counts.n_plus, q_plus)
    )
    log_prior = np.asarray(
        prior.log_density(interpolation_h, interpolation_p), dtype=np.float64
    )
    log_density = log_likelihood + log_prior
    quadrature_log_density = log_density[1:-1, 1:-1]
    log_measure = np.log(np.multiply.outer(h_weights, p_weights))
    log_weight = quadrature_log_density + log_measure
    log_normalizer = float(logsumexp(log_weight))
    if not isfinite(log_normalizer):
        raise ConvergenceError("posterior normalization is nonfinite")
    mass = np.exp(log_weight - log_normalizer)
    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise ConvergenceError("posterior normalization produced invalid mass")
    total = float(np.sum(mass))
    if not isfinite(total) or abs(total - 1.0) > 1e-12:
        raise ConvergenceError("posterior mass does not normalize within 1e-12")
    return _GridEvaluation(
        h_axis=h_axis,
        p_axis=p_axis,
        h_nodes=h_grid,
        p_nodes=p_grid,
        mass=mass,
        log_density=log_density,
        q_minus=q_minus[1:-1, 1:-1],
        q_plus=q_plus[1:-1, 1:-1],
        log_normalizer=log_normalizer,
    )


def _support_ranges(prior: LocalPrior, design: LocalDesign) -> dict[str, float]:
    h_range = prior.h_max - prior.h_min
    p_range = prior.p_max - prior.p_min
    corners_h = np.array([prior.h_min, prior.h_max, prior.h_min, prior.h_max])
    corners_p = np.array([prior.p_min, prior.p_min, prior.p_max, prior.p_max])
    return {
        "h": h_range,
        "p": p_range,
        "alpha": design.r * h_range,
        "beta": 2.0 * p_range,
        "tau_plus": float(np.ptp(design.r * corners_h * corners_p)),
        "tau_minus": float(np.ptp(design.r * corners_h * (1.0 - corners_p))),
    }


def _trapezoid_weights(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    weights = np.empty_like(axis)
    weights[0] = 0.5 * (axis[1] - axis[0])
    weights[-1] = 0.5 * (axis[-1] - axis[-2])
    weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return weights


@dataclass(frozen=True, slots=True)
class _CommonGrid:
    h_axis: NDArray[np.float64]
    p_axis: NDArray[np.float64]
    density: NDArray[np.float64]
    measure: NDArray[np.float64]


def _common_grid_density(
    evaluation: _GridEvaluation, prior: LocalPrior, points: int
) -> _CommonGrid:
    h_axis = np.linspace(prior.h_min, prior.h_max, points)
    p_axis = np.linspace(prior.p_min, prior.p_max, points)
    h_grid, p_grid = np.meshgrid(h_axis, p_axis, indexing="ij")
    locations = np.column_stack((h_grid.ravel(), p_grid.ravel()))
    interpolator = RegularGridInterpolator(
        (evaluation.h_axis, evaluation.p_axis),
        evaluation.log_density,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    common_log = np.asarray(interpolator(locations), dtype=np.float64).reshape(
        h_grid.shape
    )
    if not np.all(np.isfinite(common_log)):
        raise ConvergenceError("common-grid log posterior interpolation is nonfinite")
    centered = common_log - float(np.max(common_log))
    density = np.exp(centered)
    measure = np.multiply.outer(_trapezoid_weights(h_axis), _trapezoid_weights(p_axis))
    normalizer = float(np.sum(density * measure))
    if not isfinite(normalizer) or normalizer <= 0.0:
        raise ConvergenceError("common-grid posterior normalization is nonfinite")
    return _CommonGrid(
        h_axis=h_axis,
        p_axis=p_axis,
        density=density / normalizer,
        measure=measure,
    )


def _axis_quantile(
    axis: NDArray[np.float64],
    density: NDArray[np.float64],
    probability: float,
) -> float:
    """Invert the CDF of a linearly interpolated one-dimensional density."""
    if probability <= 0.0:
        return float(axis[0])
    if probability >= 1.0:
        return float(axis[-1])
    widths = np.diff(axis)
    segment_mass = 0.5 * (density[:-1] + density[1:]) * widths
    cumulative = np.concatenate(([0.0], np.cumsum(segment_mass)))
    total = float(cumulative[-1])
    if (
        not isfinite(total)
        or total <= 0.0
        or np.any(segment_mass < 0.0)
        or np.any(np.diff(cumulative) < 0.0)
    ):
        raise ConvergenceError("continuous marginal CDF is invalid")
    target = probability * total
    index = min(
        int(np.searchsorted(cumulative, target, side="right") - 1),
        axis.size - 2,
    )
    while index < segment_mass.size and segment_mass[index] <= 0.0:
        index += 1
    if index >= segment_mass.size:
        return float(axis[-1])
    local_target = target - float(cumulative[index])
    width = float(widths[index])
    f0 = float(density[index])
    f1 = float(density[index + 1])
    scale = max(abs(f0), abs(f1), np.finfo(np.float64).tiny)
    if abs(f1 - f0) <= 64.0 * np.finfo(np.float64).eps * scale:
        offset = local_target / f0
    else:
        slope = (f1 - f0) / width
        discriminant = max(0.0, f0 * f0 + 2.0 * slope * local_target)
        denominator = f0 + discriminant**0.5
        if denominator <= 0.0:
            raise ConvergenceError("continuous marginal CDF cannot be inverted")
        offset = 2.0 * local_target / denominator
    return float(axis[index] + min(width, max(0.0, offset)))


def _row_lower_integrals(
    common: _CommonGrid,
    prefix: NDArray[np.float64],
    row_total: NDArray[np.float64],
    rows: NDArray[np.intp],
    cuts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized exact partial integrals for piecewise-linear p rows."""
    clipped = np.clip(cuts, common.p_axis[0], common.p_axis[-1])
    columns = np.searchsorted(common.p_axis, clipped, side="right") - 1
    columns = np.clip(columns, 0, common.p_axis.size - 2)
    offsets = clipped - common.p_axis[columns]
    widths = common.p_axis[columns + 1] - common.p_axis[columns]
    f0 = common.density[rows, columns]
    slopes = (
        common.density[rows, columns + 1] - common.density[rows, columns]
    ) / widths
    values = prefix[rows, columns] + f0 * offsets + 0.5 * slopes * offsets**2
    values = np.where(cuts <= common.p_axis[0], 0.0, values)
    values = np.where(cuts >= common.p_axis[-1], row_total[rows], values)
    return np.asarray(values, dtype=np.float64)


def _tau_cdf(
    common: _CommonGrid,
    design: LocalDesign,
    prefix: NDArray[np.float64],
    row_total: NDArray[np.float64],
    value: float,
    *,
    positive: bool,
) -> float:
    """Integrate a continuous bilinear posterior below one signed intensity."""
    p_lower = float(common.p_axis[0])
    p_upper = float(common.p_axis[-1])
    allocation_lower = p_lower if positive else 1.0 - p_upper
    allocation_upper = p_upper if positive else 1.0 - p_lower
    support_lower = design.r * float(common.h_axis[0]) * allocation_lower
    support_upper = design.r * float(common.h_axis[-1]) * allocation_upper
    if value <= support_lower:
        return 0.0
    if value >= support_upper:
        return 1.0

    denominators = common.p_axis if positive else 1.0 - common.p_axis
    crossings = value / (design.r * denominators)
    internal = crossings[
        (crossings > common.h_axis[0]) & (crossings < common.h_axis[-1])
    ]
    breaks = np.unique(np.concatenate((common.h_axis, internal)))
    left = breaks[:-1]
    right = breaks[1:]
    half_widths = 0.5 * (right - left)
    midpoints = 0.5 * (right + left)
    h_values = (midpoints[:, None] + half_widths[:, None] * _PUSHFORWARD_NODES).ravel()
    integration_weights = (half_widths[:, None] * _PUSHFORWARD_WEIGHTS).ravel()
    h_rows = np.searchsorted(common.h_axis, h_values, side="right") - 1
    h_rows = np.asarray(np.clip(h_rows, 0, common.h_axis.size - 2), dtype=np.intp)
    h_widths = common.h_axis[h_rows + 1] - common.h_axis[h_rows]
    fractions = (h_values - common.h_axis[h_rows]) / h_widths
    cuts = value / (design.r * h_values)
    if not positive:
        cuts = 1.0 - cuts
    lower_left = _row_lower_integrals(common, prefix, row_total, h_rows, cuts)
    lower_right = _row_lower_integrals(common, prefix, row_total, h_rows + 1, cuts)
    conditional = (1.0 - fractions) * lower_left + fractions * lower_right
    if not positive:
        totals = (1.0 - fractions) * row_total[h_rows] + fractions * row_total[
            h_rows + 1
        ]
        conditional = totals - conditional
    integral = float(np.sum(integration_weights * conditional))

    normalizer = float(np.sum(_trapezoid_weights(common.h_axis) * row_total))
    if not isfinite(normalizer) or normalizer <= 0.0:
        raise ConvergenceError("continuous push-forward normalizer is invalid")
    probability = integral / normalizer
    roundoff = 512.0 * np.finfo(np.float64).eps
    if (
        not isfinite(probability)
        or probability < -roundoff
        or probability > 1.0 + roundoff
    ):
        raise ConvergenceError("continuous push-forward CDF is outside [0, 1]")
    return min(1.0, max(0.0, probability))


def _tau_quantile(
    common: _CommonGrid,
    design: LocalDesign,
    prefix: NDArray[np.float64],
    row_total: NDArray[np.float64],
    probability: float,
    *,
    positive: bool,
) -> float:
    p_lower = float(common.p_axis[0])
    p_upper = float(common.p_axis[-1])
    allocation_lower = p_lower if positive else 1.0 - p_upper
    allocation_upper = p_upper if positive else 1.0 - p_lower
    lower = design.r * float(common.h_axis[0]) * allocation_lower
    upper = design.r * float(common.h_axis[-1]) * allocation_upper
    if probability <= 0.0:
        return lower
    if probability >= 1.0:
        return upper
    root = brentq(
        lambda value: (
            _tau_cdf(
                common,
                design,
                prefix,
                row_total,
                value,
                positive=positive,
            )
            - probability
        ),
        lower,
        upper,
        xtol=max(np.finfo(np.float64).tiny, 1e-13 * (upper - lower)),
        rtol=8.0 * np.finfo(np.float64).eps,
    )
    return float(root)


def _posterior_summaries(
    evaluation: _GridEvaluation,
    common: _CommonGrid,
    design: LocalDesign,
    interval_mass: float,
) -> tuple[_PosteriorSummary, ...]:
    h_weights = _trapezoid_weights(common.h_axis)
    p_weights = _trapezoid_weights(common.p_axis)
    h_density = common.density @ p_weights
    p_density = h_weights @ common.density
    p_widths = np.diff(common.p_axis)
    prefix = np.zeros_like(common.density)
    prefix[:, 1:] = np.cumsum(
        0.5 * (common.density[:, :-1] + common.density[:, 1:]) * p_widths,
        axis=1,
    )
    row_total = prefix[:, -1]
    h = evaluation.h_nodes
    p = evaluation.p_nodes
    values = {
        "h": h,
        "p": p,
        "alpha": 2.0 - design.r * h,
        "beta": 2.0 * p - 1.0,
        "tau_plus": design.r * h * p,
        "tau_minus": design.r * h * (1.0 - p),
    }
    tail = 0.5 * (1.0 - interval_mass)

    def quantile(quantity: str, probability: float) -> float:
        if quantity == "h":
            return _axis_quantile(common.h_axis, h_density, probability)
        if quantity == "p":
            return _axis_quantile(common.p_axis, p_density, probability)
        if quantity == "alpha":
            return 2.0 - design.r * _axis_quantile(
                common.h_axis, h_density, 1.0 - probability
            )
        if quantity == "beta":
            return 2.0 * _axis_quantile(common.p_axis, p_density, probability) - 1.0
        if quantity == "tau_plus":
            return _tau_quantile(
                common,
                design,
                prefix,
                row_total,
                probability,
                positive=True,
            )
        if quantity == "tau_minus":
            return _tau_quantile(
                common,
                design,
                prefix,
                row_total,
                probability,
                positive=False,
            )
        raise ConvergenceError(f"unknown posterior summary: {quantity!r}")

    return tuple(
        _PosteriorSummary(
            quantity=name,
            mean=float(np.sum(evaluation.mass * value)),
            median=quantile(name, 0.5),
            interval_lower=quantile(name, tail),
            interval_upper=quantile(name, 1.0 - tail),
        )
        for name, value in values.items()
    )


def _symmetric_relative(first: float, second: float) -> float:
    denominator = abs(first) + abs(second)
    return 0.0 if denominator == 0.0 else 2.0 * abs(first - second) / denominator


def _joint_total_variation(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    measure: NDArray[np.float64],
) -> float:
    """Integrate joint density disagreement without marginalizing dependence."""
    return 0.5 * float(np.sum(np.abs(first - second) * measure))


def _refinement_diagnostics(
    base: _GridEvaluation,
    refined: _GridEvaluation,
    design: LocalDesign,
    prior: LocalPrior,
    config: QuadratureConfig,
) -> tuple[RefinementDiagnostics, tuple[_PosteriorSummary, ...]]:
    base_common = _common_grid_density(base, prior, config.common_grid_points)
    refined_common = _common_grid_density(refined, prior, config.common_grid_points)
    coarse_points = max(3, (config.common_grid_points + 1) // 2)
    refined_coarse = _common_grid_density(refined, prior, coarse_points)
    if not np.array_equal(base_common.measure, refined_common.measure):
        raise ConvergenceError("common-grid comparison did not use one measure")
    joint_tv = _joint_total_variation(
        base_common.density,
        refined_common.density,
        base_common.measure,
    )
    base_summaries = _posterior_summaries(
        base, base_common, design, config.interval_mass
    )
    refined_summaries = _posterior_summaries(
        refined, refined_common, design, config.interval_mass
    )
    coarse_summaries = _posterior_summaries(
        refined, refined_coarse, design, config.interval_mass
    )
    base_by_name = {summary.quantity: summary for summary in base_summaries}
    refined_by_name = {summary.quantity: summary for summary in refined_summaries}
    coarse_by_name = {summary.quantity: summary for summary in coarse_summaries}
    ranges = _support_ranges(prior, design)
    summary_changes = tuple(
        SummaryRefinement(
            quantity=name,
            mean=abs(base_by_name[name].mean - refined_by_name[name].mean)
            / ranges[name],
            median=max(
                abs(base_by_name[name].median - refined_by_name[name].median),
                abs(coarse_by_name[name].median - refined_by_name[name].median),
            )
            / ranges[name],
            interval_lower=max(
                abs(
                    base_by_name[name].interval_lower
                    - refined_by_name[name].interval_lower
                ),
                abs(
                    coarse_by_name[name].interval_lower
                    - refined_by_name[name].interval_lower
                ),
            )
            / ranges[name],
            interval_upper=max(
                abs(
                    base_by_name[name].interval_upper
                    - refined_by_name[name].interval_upper
                ),
                abs(
                    coarse_by_name[name].interval_upper
                    - refined_by_name[name].interval_upper
                ),
            )
            / ranges[name],
        )
        for name in _QUANTITIES
    )
    base_positive = float(np.sum(base.mass * base.q_plus))
    base_negative = float(np.sum(base.mass * base.q_minus))
    refined_positive = float(np.sum(refined.mass * refined.q_plus))
    refined_negative = float(np.sum(refined.mass * refined.q_minus))
    predictive = PredictiveTailRefinement(
        positive=_symmetric_relative(base_positive, refined_positive),
        negative=_symmetric_relative(base_negative, refined_negative),
    )
    log_change = abs(base.log_normalizer - refined.log_normalizer)
    provisional = RefinementDiagnostics(
        tolerance=config.refinement_tolerance,
        common_grid_points=config.common_grid_points,
        joint_total_variation=joint_tv,
        log_normalizer_change=log_change,
        summaries=summary_changes,
        predictive_tail=predictive,
        converged=False,
    )
    diagnostics = RefinementDiagnostics(
        tolerance=provisional.tolerance,
        common_grid_points=provisional.common_grid_points,
        joint_total_variation=provisional.joint_total_variation,
        log_normalizer_change=provisional.log_normalizer_change,
        summaries=provisional.summaries,
        predictive_tail=provisional.predictive_tail,
        converged=provisional.maximum_component <= config.refinement_tolerance,
    )
    return diagnostics, refined_summaries


def compute_exact_posterior(
    counts: CellCounts,
    design: LocalDesign,
    prior: LocalPrior,
    config: QuadratureConfig | None = None,
    *,
    backend: StableBackend | None = None,
) -> PosteriorGrid:
    """Compute and retain only a demonstrably converged exact finite posterior."""
    if not isinstance(counts, CellCounts):
        raise ValidationError("counts must be a CellCounts object")
    if not isinstance(design, LocalDesign):
        raise ValidationError("design must be a LocalDesign object")
    if not isinstance(prior, LocalPrior):
        raise ValidationError("prior must be defined on the supplied design")
    _validate_experiment_provenance(counts, design, prior)
    controls = QuadratureConfig() if config is None else config
    if not isinstance(controls, QuadratureConfig):
        raise ValidationError("config must be a QuadratureConfig object")
    if controls.refined_nodes <= controls.base_nodes:
        raise ConvergenceError(
            "refined_nodes must exceed base_nodes to demonstrate convergence"
        )
    candidate: object = ScipyS0Backend() if backend is None else backend
    evaluator, metadata = validate_s0_backend(candidate)
    backend_origin: Literal["canonical_scipy_s0", "custom"] = (
        "canonical_scipy_s0"
        if type(evaluator) is _CANONICAL_SCIPY_S0_TYPE
        else "custom"
    )
    base = _evaluate_grid(counts, design, prior, controls.base_nodes, evaluator)
    refined = _evaluate_grid(counts, design, prior, controls.refined_nodes, evaluator)
    diagnostics, summaries = _refinement_diagnostics(
        base, refined, design, prior, controls
    )
    if not diagnostics.converged:
        raise ConvergenceError(
            "posterior refinement failed: "
            f"maximum component {diagnostics.maximum_component:.6g} exceeds "
            f"tolerance {diagnostics.tolerance:.6g}"
        )
    return PosteriorGrid(
        h_nodes=refined.h_nodes,
        p_nodes=refined.p_nodes,
        mass=refined.mass,
        q_minus=refined.q_minus,
        q_plus=refined.q_plus,
        log_normalizer=refined.log_normalizer,
        base_nodes=controls.base_nodes,
        refined_nodes=controls.refined_nodes,
        interval_mass=controls.interval_mass,
        summaries=summaries,
        design=design,
        prior=prior,
        counts=counts,
        backend_metadata=metadata,
        backend_origin=backend_origin,
        refinement=diagnostics,
    )


__all__ = ["PosteriorGrid", "QuadratureConfig", "compute_exact_posterior"]
