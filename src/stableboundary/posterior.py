"""Deterministic quadrature for the exact finite three-cell posterior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from numbers import Integral, Real
from platform import python_version
from typing import Final, Literal

import numpy as np
import scipy  # type: ignore[import-untyped]
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


def _package_version() -> str:
    try:
        return version("stableboundary")
    except PackageNotFoundError:  # pragma: no cover - source-tree fallback
        return "0.1.0"


@dataclass(frozen=True, slots=True)
class _InferenceEnvironment:
    """Runtime versions captured once as part of the inferential result."""

    python: str
    numpy: str
    scipy: str
    stableboundary: str

    def __post_init__(self) -> None:
        for name in ("python", "numpy", "scipy", "stableboundary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"environment {name} version must be named")
            object.__setattr__(self, name, value.strip())

    def to_dict(self) -> dict[str, str]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "stableboundary": self.stableboundary,
        }


def _capture_environment() -> _InferenceEnvironment:
    return _InferenceEnvironment(
        python=python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        stableboundary=_package_version(),
    )


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
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be a finite real number") from error
    upper_valid = result <= upper if upper_inclusive else result < upper
    if not isfinite(result) or not lower < result or not upper_valid:
        closing = "]" if upper_inclusive else ")"
        raise ValidationError(f"{name} must lie in ({lower}, {upper}{closing}")
    return result


def _immutable_float64_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Copy an array into immutable bytes-backed storage."""
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConvergenceError(f"{name} must be a numeric array") from error
    if raw.shape != shape or not np.all(np.isfinite(raw)):
        raise ConvergenceError(f"{name} must be a finite {shape!r} array")
    payload = np.ascontiguousarray(raw, dtype=np.float64).tobytes(order="C")
    retained = np.frombuffer(payload, dtype=np.float64).reshape(shape)
    if retained.flags.writeable:  # pragma: no cover - bytes contract guard
        raise ConvergenceError(f"{name} immutable storage could not be established")
    return retained


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

    def __post_init__(self) -> None:
        if self.quantity not in _QUANTITIES:
            raise ConvergenceError(f"unknown refinement quantity: {self.quantity!r}")
        for name in ("mean", "median", "interval_lower", "interval_upper"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ConvergenceError(
                    "summary refinement changes must be finite and nonnegative"
                )

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

    def __post_init__(self) -> None:
        if any(
            not isfinite(value) or value < 0.0
            for value in (self.positive, self.negative)
        ):
            raise ConvergenceError(
                "predictive-tail refinement changes must be finite and nonnegative"
            )

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

    def __post_init__(self) -> None:
        tolerance = _bounded_real(
            "refinement tolerance",
            self.tolerance,
            lower=0.0,
            upper=1.0,
            upper_inclusive=True,
        )
        common = _bounded_integer(
            "common_grid_points",
            self.common_grid_points,
            minimum=3,
            maximum=513,
        )
        if (
            not isfinite(self.joint_total_variation)
            or not 0.0 <= self.joint_total_variation <= 1.0
            or not isfinite(self.log_normalizer_change)
            or self.log_normalizer_change < 0.0
        ):
            raise ConvergenceError("joint refinement diagnostics are invalid")
        if (
            not isinstance(self.summaries, tuple)
            or not all(isinstance(item, SummaryRefinement) for item in self.summaries)
            or tuple(item.quantity for item in self.summaries) != _QUANTITIES
        ):
            raise ConvergenceError("refinement summaries are incomplete")
        if not isinstance(self.predictive_tail, PredictiveTailRefinement):
            raise ConvergenceError("predictive-tail refinement is invalid")
        if not isinstance(self.converged, bool):
            raise ConvergenceError("refinement convergence must be boolean")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "common_grid_points", common)
        if self.converged and self.maximum_component > tolerance:
            raise ConvergenceError(
                "converged refinement exceeds its retained tolerance"
            )

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


@dataclass(frozen=True, slots=True, init=False)
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
    environment: _InferenceEnvironment
    refinement: RefinementDiagnostics

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Prevent public construction or ``dataclasses.replace`` forgery."""
        del args, kwargs
        raise TypeError("use compute_exact_posterior() to construct a posterior")

    def __post_init__(self) -> None:
        base_nodes = _bounded_integer(
            "base_nodes", self.base_nodes, minimum=2, maximum=256
        )
        refined_nodes = _bounded_integer(
            "refined_nodes", self.refined_nodes, minimum=2, maximum=384
        )
        if refined_nodes <= base_nodes:
            raise ConvergenceError("posterior refined_nodes must exceed base_nodes")
        interval_mass = _bounded_real(
            "interval_mass",
            self.interval_mass,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        shape = (refined_nodes, refined_nodes)
        arrays: list[NDArray[np.float64]] = []
        for name in ("h_nodes", "p_nodes", "mass", "q_minus", "q_plus"):
            arrays.append(
                _immutable_float64_array(name, getattr(self, name), shape=shape)
            )
        if np.any(arrays[2] < 0.0) or abs(float(np.sum(arrays[2])) - 1.0) > 1e-12:
            raise ConvergenceError("posterior mass must be nonnegative and normalize")
        if np.any((arrays[3] <= 0.0) | (arrays[3] >= 1.0)) or np.any(
            (arrays[4] <= 0.0) | (arrays[4] >= 1.0)
        ):
            raise ConvergenceError("posterior signed-tail probabilities are invalid")
        if np.any(arrays[3] + arrays[4] >= 1.0):
            raise ConvergenceError(
                "posterior signed-tail probabilities leave no center"
            )
        if isinstance(self.log_normalizer, bool) or not isinstance(
            self.log_normalizer, Real
        ):
            raise ConvergenceError("posterior log normalizer must be a real number")
        try:
            log_normalizer = float(self.log_normalizer)
        except (TypeError, ValueError, OverflowError) as error:
            raise ConvergenceError("posterior log normalizer must be finite") from error
        if not isfinite(log_normalizer):
            raise ConvergenceError("posterior log normalizer must be finite")
        if not isinstance(self.summaries, tuple) or not all(
            isinstance(summary, _PosteriorSummary) for summary in self.summaries
        ):
            raise ConvergenceError("posterior summaries must be retained records")
        if tuple(summary.quantity for summary in self.summaries) != _QUANTITIES:
            raise ConvergenceError(
                "posterior summaries must cover each supported quantity once"
            )
        _validate_experiment_provenance(self.counts, self.design, self.prior)
        expected_h_axis, _ = _legendre_axis(
            self.prior.h_min, self.prior.h_max, refined_nodes
        )
        expected_p_axis, _ = _legendre_axis(
            self.prior.p_min, self.prior.p_max, refined_nodes
        )
        expected_h, expected_p = np.meshgrid(
            expected_h_axis,
            expected_p_axis,
            indexing="ij",
        )
        if not np.array_equal(arrays[0], expected_h) or not np.array_equal(
            arrays[1], expected_p
        ):
            raise ConvergenceError(
                "posterior nodes do not match the retained prior and quadrature"
            )
        grid_values = {
            "h": arrays[0],
            "p": arrays[1],
            "alpha": 2.0 - self.design.r * arrays[0],
            "beta": 2.0 * arrays[1] - 1.0,
            "tau_plus": self.design.r * arrays[0] * arrays[1],
            "tau_minus": self.design.r * arrays[0] * (1.0 - arrays[1]),
        }
        support = {
            "h": (self.prior.h_min, self.prior.h_max),
            "p": (self.prior.p_min, self.prior.p_max),
            "alpha": (
                2.0 - self.design.r * self.prior.h_max,
                2.0 - self.design.r * self.prior.h_min,
            ),
            "beta": (2.0 * self.prior.p_min - 1.0, 2.0 * self.prior.p_max - 1.0),
            "tau_plus": (
                self.design.r * self.prior.h_min * self.prior.p_min,
                self.design.r * self.prior.h_max * self.prior.p_max,
            ),
            "tau_minus": (
                self.design.r * self.prior.h_min * (1.0 - self.prior.p_max),
                self.design.r * self.prior.h_max * (1.0 - self.prior.p_min),
            ),
        }
        for summary in self.summaries:
            lower, upper = support[summary.quantity]
            scale = max(1.0, upper - lower)
            slack = 1e-12 * scale
            if any(
                value < lower - slack or value > upper + slack
                for value in (
                    summary.mean,
                    summary.median,
                    summary.interval_lower,
                    summary.interval_upper,
                )
            ):
                raise ConvergenceError("posterior summary lies outside prior support")
            expected_mean = float(np.sum(arrays[2] * grid_values[summary.quantity]))
            if abs(summary.mean - expected_mean) > slack:
                raise ConvergenceError(
                    "posterior summary mean does not match retained posterior mass"
                )
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
        if not isinstance(self.environment, _InferenceEnvironment):
            raise ValidationError("posterior must retain its inference environment")
        if not isinstance(self.refinement, RefinementDiagnostics):
            raise ConvergenceError("posterior must retain refinement diagnostics")
        if (
            self.refinement.common_grid_points < 3
            or not self.refinement.converged
            or self.refinement.maximum_component > self.refinement.tolerance
        ):
            raise ConvergenceError("posterior refinement provenance is invalid")
        object.__setattr__(self, "base_nodes", base_nodes)
        object.__setattr__(self, "refined_nodes", refined_nodes)
        object.__setattr__(self, "interval_mass", interval_mass)
        object.__setattr__(self, "log_normalizer", log_normalizer)
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
        return _immutable_float64_array(
            quantity,
            values,
            shape=(self.refined_nodes, self.refined_nodes),
        )


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


def _axis_components(
    axis: NDArray[np.float64],
    density: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    if (
        axis.ndim != 1
        or density.shape != axis.shape
        or axis.size < 2
        or not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ConvergenceError("continuous marginal density is invalid")
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
    return widths, cumulative, total


def _axis_tail_probability(
    axis: NDArray[np.float64],
    density: NDArray[np.float64],
    value: float,
    *,
    upper_tail: bool,
) -> float:
    oriented_axis = -axis[::-1] if upper_tail else axis
    oriented_density = density[::-1] if upper_tail else density
    oriented_value = -value if upper_tail else value
    widths, cumulative, total = _axis_components(oriented_axis, oriented_density)
    if oriented_value <= oriented_axis[0]:
        return 0.0
    if oriented_value >= oriented_axis[-1]:
        return 1.0
    segment = min(
        int(np.searchsorted(oriented_axis, oriented_value, side="right") - 1),
        oriented_axis.size - 2,
    )
    local_offset = oriented_value - float(oriented_axis[segment])
    local_width = float(widths[segment])
    local_f0 = float(oriented_density[segment])
    slope = (float(oriented_density[segment + 1]) - local_f0) / local_width
    integral = (
        float(cumulative[segment])
        + local_f0 * local_offset
        + 0.5 * slope * local_offset**2
    )
    result = integral / total
    roundoff = 32.0 * np.finfo(np.float64).eps
    if not isfinite(result) or result < -roundoff or result > 1.0 + roundoff:
        raise ConvergenceError("continuous marginal CDF is invalid")
    return min(1.0, max(0.0, result))


def _axis_quantile(
    axis: NDArray[np.float64],
    density: NDArray[np.float64],
    probability: float,
    *,
    upper_tail: bool = False,
) -> float:
    """Invert one tail of a linearly interpolated one-dimensional density."""
    if (
        not isinstance(probability, Real)
        or isinstance(probability, bool)
        or not isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ConvergenceError("continuous marginal probability must lie in [0, 1]")
    requested = float(probability)
    if upper_tail:
        if requested == 0.0:
            return float(axis[-1])
        if requested == 1.0:
            return float(axis[0])
        target = requested
        reverse = True
    else:
        if requested == 0.0:
            return float(axis[0])
        if requested == 1.0:
            return float(axis[-1])
        reverse = requested > 0.5
        target = 1.0 - requested if reverse else requested

    oriented_axis = -axis[::-1] if reverse else axis
    oriented_density = density[::-1] if reverse else density
    widths, cumulative, total = _axis_components(oriented_axis, oriented_density)
    segment_mass = np.diff(cumulative)
    target_mass = target * total
    index = min(
        int(np.searchsorted(cumulative, target_mass, side="right") - 1),
        oriented_axis.size - 2,
    )
    while index < segment_mass.size and segment_mass[index] <= 0.0:
        index += 1
    if index >= segment_mass.size:
        raise ConvergenceError("continuous marginal quantile cannot be inverted")
    local_target = target_mass - float(cumulative[index])
    width = float(widths[index])
    f0 = float(oriented_density[index])
    f1 = float(oriented_density[index + 1])
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
    candidate = float(oriented_axis[index] + min(width, max(0.0, offset)))

    def tail_probability(value: float) -> float:
        return _axis_tail_probability(
            oriented_axis,
            oriented_density,
            value,
            upper_tail=False,
        )

    resolved = _validate_probability_inversion(
        "continuous marginal",
        candidate,
        float(oriented_axis[0]),
        float(oriented_axis[-1]),
        target,
        tail_probability,
    )
    return -resolved if reverse else resolved


def _validate_probability_inversion(
    name: str,
    candidate: float,
    lower: float,
    upper: float,
    target: float,
    probability: Callable[[float], float],
    *,
    allow_bracketed_rounding: bool = True,
    increasing: bool = True,
) -> float:
    """Require an interior float whose probability resolves the target tail."""
    candidates = {
        candidate,
        float(np.nextafter(candidate, lower)),
        float(np.nextafter(candidate, upper)),
    }
    interior = [value for value in candidates if lower < value < upper]
    if not interior:
        raise ConvergenceError(
            f"{name} quantile probability {target:.17g} is not numerically "
            "resolvable inside its support"
        )
    evaluated = sorted((value, probability(value)) for value in interior)
    residual, result = min((abs(actual - target), value) for value, actual in evaluated)
    probability_ulp = abs(float(np.spacing(np.float64(target))))
    tolerance = max(512.0 * probability_ulp, 1e-9 * target)
    bracketed = any(
        (
            evaluated[index][1] <= target <= evaluated[index + 1][1]
            if increasing
            else evaluated[index][1] >= target >= evaluated[index + 1][1]
        )
        for index in range(len(evaluated) - 1)
    )
    if not isfinite(residual) or (
        residual > tolerance and (not allow_bracketed_rounding or not bracketed)
    ):
        raise ConvergenceError(
            f"{name} quantile probability {target:.17g} is not numerically "
            f"resolvable (best probability residual {residual:.3g})"
        )
    return result


def _affine_axis_quantile(
    axis: NDArray[np.float64],
    density: NDArray[np.float64],
    probability: float,
    *,
    upper_tail: bool,
    offset: float,
    slope: float,
    name: str,
) -> float:
    """Validate a marginal quantile after its reported affine transformation."""
    source_upper_tail = upper_tail if slope > 0.0 else not upper_tail
    source_candidate = _axis_quantile(
        axis,
        density,
        probability,
        upper_tail=source_upper_tail,
    )
    candidate = offset + slope * source_candidate
    mapped_endpoints = (
        offset + slope * float(axis[0]),
        offset + slope * float(axis[-1]),
    )
    lower = min(mapped_endpoints)
    upper = max(mapped_endpoints)

    def mapped_tail_probability(value: float) -> float:
        source_value = (value - offset) / slope
        return _axis_tail_probability(
            axis,
            density,
            source_value,
            upper_tail=source_upper_tail,
        )

    return _validate_probability_inversion(
        f"continuous {name}",
        candidate,
        lower,
        upper,
        probability,
        mapped_tail_probability,
        allow_bracketed_rounding=False,
        increasing=not upper_tail,
    )


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


def _row_upper_integrals(
    common: _CommonGrid,
    suffix: NDArray[np.float64],
    rows: NDArray[np.intp],
    cuts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized exact upper integrals for piecewise-linear p rows."""
    clipped = np.clip(cuts, common.p_axis[0], common.p_axis[-1])
    columns = np.searchsorted(common.p_axis, clipped, side="right") - 1
    columns = np.clip(columns, 0, common.p_axis.size - 2)
    offsets = clipped - common.p_axis[columns]
    widths = common.p_axis[columns + 1] - common.p_axis[columns]
    f0 = common.density[rows, columns]
    f1 = common.density[rows, columns + 1]
    slopes = (f1 - f0) / widths
    f_cut = f0 + slopes * offsets
    remaining = widths - offsets
    values = suffix[rows, columns + 1] + 0.5 * (f_cut + f1) * remaining
    values = np.where(cuts <= common.p_axis[0], suffix[rows, 0], values)
    values = np.where(cuts >= common.p_axis[-1], 0.0, values)
    return np.asarray(values, dtype=np.float64)


def _row_upper_suffix(common: _CommonGrid) -> NDArray[np.float64]:
    p_widths = np.diff(common.p_axis)
    segment_mass = 0.5 * (common.density[:, :-1] + common.density[:, 1:]) * p_widths
    suffix = np.zeros_like(common.density)
    suffix[:, :-1] = np.cumsum(segment_mass[:, ::-1], axis=1)[:, ::-1]
    return suffix


def _tau_tail_probability(
    common: _CommonGrid,
    design: LocalDesign,
    prefix: NDArray[np.float64],
    row_total: NDArray[np.float64],
    suffix: NDArray[np.float64],
    value: float,
    *,
    positive: bool,
    upper_tail: bool,
) -> float:
    """Integrate one direct tail of a continuous bilinear push-forward."""
    p_lower = float(common.p_axis[0])
    p_upper = float(common.p_axis[-1])
    allocation_lower = p_lower if positive else 1.0 - p_upper
    allocation_upper = p_upper if positive else 1.0 - p_lower
    support_lower = design.r * float(common.h_axis[0]) * allocation_lower
    support_upper = design.r * float(common.h_axis[-1]) * allocation_upper
    if value <= support_lower:
        return 1.0 if upper_tail else 0.0
    if value >= support_upper:
        return 0.0 if upper_tail else 1.0

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
    use_lower_integral = positive != upper_tail
    if use_lower_integral:
        conditional_left = _row_lower_integrals(common, prefix, row_total, h_rows, cuts)
        conditional_right = _row_lower_integrals(
            common, prefix, row_total, h_rows + 1, cuts
        )
        normalization_rows = row_total
    else:
        conditional_left = _row_upper_integrals(common, suffix, h_rows, cuts)
        conditional_right = _row_upper_integrals(common, suffix, h_rows + 1, cuts)
        normalization_rows = suffix[:, 0]
    conditional = (1.0 - fractions) * conditional_left + fractions * conditional_right
    integral = float(np.sum(integration_weights * conditional))

    normalizer = float(np.sum(_trapezoid_weights(common.h_axis) * normalization_rows))
    if not isfinite(normalizer) or normalizer <= 0.0:
        raise ConvergenceError("continuous push-forward normalizer is invalid")
    probability = integral / normalizer
    roundoff = 512.0 * np.finfo(np.float64).eps
    if (
        not isfinite(probability)
        or probability < -roundoff
        or probability > 1.0 + roundoff
    ):
        raise ConvergenceError(
            "continuous push-forward tail probability is outside [0, 1]"
        )
    return min(1.0, max(0.0, probability))


def _tau_cdf(
    common: _CommonGrid,
    design: LocalDesign,
    prefix: NDArray[np.float64],
    row_total: NDArray[np.float64],
    value: float,
    *,
    positive: bool,
) -> float:
    """Integrate the lower tail of one signed-intensity push-forward."""
    return _tau_tail_probability(
        common,
        design,
        prefix,
        row_total,
        _row_upper_suffix(common),
        value,
        positive=positive,
        upper_tail=False,
    )


def _tau_quantile(
    common: _CommonGrid,
    design: LocalDesign,
    prefix: NDArray[np.float64],
    row_total: NDArray[np.float64],
    probability: float,
    *,
    positive: bool,
    upper_tail: bool = False,
) -> float:
    p_lower = float(common.p_axis[0])
    p_upper = float(common.p_axis[-1])
    allocation_lower = p_lower if positive else 1.0 - p_upper
    allocation_upper = p_upper if positive else 1.0 - p_lower
    lower = design.r * float(common.h_axis[0]) * allocation_lower
    upper = design.r * float(common.h_axis[-1]) * allocation_upper
    if (
        not isinstance(probability, Real)
        or isinstance(probability, bool)
        or not isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ConvergenceError("continuous push-forward probability must lie in [0, 1]")
    requested = float(probability)
    if upper_tail:
        if requested == 0.0:
            return upper
        if requested == 1.0:
            return lower
        target = requested
        invert_upper_tail = True
    else:
        if requested == 0.0:
            return lower
        if requested == 1.0:
            return upper
        invert_upper_tail = requested > 0.5
        target = 1.0 - requested if invert_upper_tail else requested

    suffix = _row_upper_suffix(common)

    def tail_probability(value: float) -> float:
        return _tau_tail_probability(
            common,
            design,
            prefix,
            row_total,
            suffix,
            value,
            positive=positive,
            upper_tail=invert_upper_tail,
        )

    try:
        root = brentq(
            lambda value: tail_probability(value) - target,
            lower,
            upper,
            xtol=float(np.nextafter(0.0, 1.0)),
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=256,
        )
    except (RuntimeError, ValueError, OverflowError) as error:
        raise ConvergenceError(
            "continuous push-forward quantile cannot be inverted"
        ) from error
    signed_name = "tau_plus" if positive else "tau_minus"
    return _validate_probability_inversion(
        f"continuous {signed_name}",
        float(root),
        lower,
        upper,
        target,
        tail_probability,
        increasing=not invert_upper_tail,
    )


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

    def quantile(
        quantity: str,
        probability: float,
        *,
        upper_tail: bool = False,
    ) -> float:
        if quantity == "h":
            return _axis_quantile(
                common.h_axis,
                h_density,
                probability,
                upper_tail=upper_tail,
            )
        if quantity == "p":
            return _axis_quantile(
                common.p_axis,
                p_density,
                probability,
                upper_tail=upper_tail,
            )
        if quantity == "alpha":
            return _affine_axis_quantile(
                common.h_axis,
                h_density,
                probability,
                upper_tail=upper_tail,
                offset=2.0,
                slope=-design.r,
                name="alpha",
            )
        if quantity == "beta":
            return _affine_axis_quantile(
                common.p_axis,
                p_density,
                probability,
                upper_tail=upper_tail,
                offset=-1.0,
                slope=2.0,
                name="beta",
            )
        if quantity == "tau_plus":
            return _tau_quantile(
                common,
                design,
                prefix,
                row_total,
                probability,
                positive=True,
                upper_tail=upper_tail,
            )
        if quantity == "tau_minus":
            return _tau_quantile(
                common,
                design,
                prefix,
                row_total,
                probability,
                positive=False,
                upper_tail=upper_tail,
            )
        raise ConvergenceError(f"unknown posterior summary: {quantity!r}")

    return tuple(
        _PosteriorSummary(
            quantity=name,
            mean=float(np.sum(evaluation.mass * value)),
            median=quantile(name, 0.5),
            interval_lower=quantile(name, tail),
            interval_upper=quantile(name, tail, upper_tail=True),
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
    mean_changes = {
        name: abs(base_by_name[name].mean - refined_by_name[name].mean) / ranges[name]
        for name in ("h", "p", "tau_plus", "tau_minus")
    }
    mean_changes["alpha"] = mean_changes["h"]
    mean_changes["beta"] = mean_changes["p"]
    summary_changes = tuple(
        SummaryRefinement(
            quantity=name,
            mean=mean_changes[name],
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
    environment = _capture_environment()
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
    _, final_metadata = validate_s0_backend(evaluator)
    if final_metadata != metadata:
        raise ValidationError("backend metadata changed during posterior inference")
    if not diagnostics.converged:
        raise ConvergenceError(
            "posterior refinement failed: "
            f"maximum component {diagnostics.maximum_component:.6g} exceeds "
            f"tolerance {diagnostics.tolerance:.6g}"
        )
    # Construction is deliberately local to the computation that produced the
    # evidence.  The supported public API exposes no reusable rebinding factory.
    posterior = object.__new__(PosteriorGrid)
    for name, value in (
        ("h_nodes", refined.h_nodes),
        ("p_nodes", refined.p_nodes),
        ("mass", refined.mass),
        ("q_minus", refined.q_minus),
        ("q_plus", refined.q_plus),
        ("log_normalizer", refined.log_normalizer),
        ("base_nodes", controls.base_nodes),
        ("refined_nodes", controls.refined_nodes),
        ("interval_mass", controls.interval_mass),
        ("summaries", summaries),
        ("design", design),
        ("prior", prior),
        ("counts", counts),
        ("backend_metadata", metadata),
        ("backend_origin", backend_origin),
        ("environment", environment),
        ("refinement", diagnostics),
    ):
        object.__setattr__(posterior, name, value)
    posterior.__post_init__()
    return posterior


__all__ = ["PosteriorGrid", "QuadratureConfig", "compute_exact_posterior"]
