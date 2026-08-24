"""Deterministic quadrature for the exact finite three-cell posterior."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]
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
)
from .cells import CellCounts
from .design import LocalDesign, LocalPrior

_QUANTITIES: Final = (
    "h",
    "p",
    "alpha",
    "beta",
    "tau_plus",
    "tau_minus",
)


def _bounded_integer(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValidationError(f"{name} must lie in [{minimum}, {maximum}]")
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
        tolerance = float(self.refinement_tolerance)
        interval_mass = float(self.interval_mass)
        if not isfinite(tolerance) or not 0.0 < tolerance <= 1.0:
            raise ValidationError("refinement_tolerance must lie in (0, 1]")
        if not isfinite(interval_mass) or not 0.0 < interval_mass < 1.0:
            raise ValidationError("interval_mass must lie strictly inside (0, 1)")
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
    interval_lower: float
    interval_upper: float

    @property
    def maximum(self) -> float:
        return max(self.mean, self.interval_lower, self.interval_upper)


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
    r: float
    interval_mass: float
    backend_metadata: BackendMetadata
    _backend: StableBackend = field(repr=False, compare=False)
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
        _, live_metadata = validate_s0_backend(self._backend)
        if live_metadata != self.backend_metadata:
            raise ValidationError("retained backend metadata changed during inference")
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

    def prediction_backend(self) -> StableBackend:
        """Return the fitted backend only while its metadata remains unchanged."""
        backend, live_metadata = validate_s0_backend(self._backend)
        if live_metadata != self.backend_metadata:
            raise ValidationError("fitted backend metadata changed after inference")
        return backend

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


def _weighted_quantile(
    values: NDArray[np.float64], mass: NDArray[np.float64], probability: float
) -> float:
    flat_values = values.ravel()
    flat_mass = mass.ravel()
    order = np.argsort(flat_values, kind="stable")
    cumulative = np.cumsum(flat_mass[order])
    return float(np.interp(probability, cumulative, flat_values[order]))


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
    h_nodes: NDArray[np.float64]
    p_nodes: NDArray[np.float64]
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
        h_nodes=h_grid,
        p_nodes=p_grid,
        density=density / normalizer,
        measure=measure,
    )


def _common_summary(
    common: _CommonGrid,
    design: LocalDesign,
    interval_mass: float,
) -> dict[str, tuple[float, float, float]]:
    mass = common.density * common.measure
    mass = mass / float(np.sum(mass))
    h = common.h_nodes
    p = common.p_nodes
    values = {
        "h": h,
        "p": p,
        "alpha": 2.0 - design.r * h,
        "beta": 2.0 * p - 1.0,
        "tau_plus": design.r * h * p,
        "tau_minus": design.r * h * (1.0 - p),
    }
    tail = 0.5 * (1.0 - interval_mass)
    return {
        name: (
            float(np.sum(mass * value)),
            _weighted_quantile(value, mass, tail),
            _weighted_quantile(value, mass, 1.0 - tail),
        )
        for name, value in values.items()
    }


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
) -> RefinementDiagnostics:
    base_common = _common_grid_density(base, prior, config.common_grid_points)
    refined_common = _common_grid_density(refined, prior, config.common_grid_points)
    if not np.array_equal(base_common.measure, refined_common.measure):
        raise ConvergenceError("common-grid comparison did not use one measure")
    joint_tv = _joint_total_variation(
        base_common.density,
        refined_common.density,
        base_common.measure,
    )
    base_summary = _common_summary(base_common, design, config.interval_mass)
    refined_summary = _common_summary(refined_common, design, config.interval_mass)
    ranges = _support_ranges(prior, design)
    summary_changes = tuple(
        SummaryRefinement(
            quantity=name,
            mean=abs(base_summary[name][0] - refined_summary[name][0]) / ranges[name],
            interval_lower=abs(base_summary[name][1] - refined_summary[name][1])
            / ranges[name],
            interval_upper=abs(base_summary[name][2] - refined_summary[name][2])
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
    return RefinementDiagnostics(
        tolerance=provisional.tolerance,
        common_grid_points=provisional.common_grid_points,
        joint_total_variation=provisional.joint_total_variation,
        log_normalizer_change=provisional.log_normalizer_change,
        summaries=provisional.summaries,
        predictive_tail=provisional.predictive_tail,
        converged=provisional.maximum_component <= config.refinement_tolerance,
    )


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
    if not isinstance(prior, LocalPrior) or prior.design != design:
        raise ValidationError("prior must be defined on the supplied design")
    controls = QuadratureConfig() if config is None else config
    if not isinstance(controls, QuadratureConfig):
        raise ValidationError("config must be a QuadratureConfig object")
    if controls.refined_nodes <= controls.base_nodes:
        raise ConvergenceError(
            "refined_nodes must exceed base_nodes to demonstrate convergence"
        )
    candidate: object = ScipyS0Backend() if backend is None else backend
    evaluator, metadata = validate_s0_backend(candidate)
    base = _evaluate_grid(counts, design, prior, controls.base_nodes, evaluator)
    refined = _evaluate_grid(counts, design, prior, controls.refined_nodes, evaluator)
    diagnostics = _refinement_diagnostics(base, refined, design, prior, controls)
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
        r=design.r,
        interval_mass=controls.interval_mass,
        backend_metadata=metadata,
        _backend=evaluator,
        refinement=diagnostics,
    )


__all__ = ["PosteriorGrid", "QuadratureConfig", "compute_exact_posterior"]
