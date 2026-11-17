"""Full stable-likelihood reference inference and observed-data comparisons.

Both likelihoods use the same compact local prior and numerical machinery.
Agreement is an observed-data diagnostic, not independent backend validation
or a finite-sample guarantee for other samples.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._exceptions import ConvergenceError, NumericalProbabilityError, ValidationError
from .backends import ScipyS0Backend, StableBackend
from .cells import CellCounts
from .design import KnownNuisance, LocalDesign, LocalPrior
from .posterior import (
    _CANONICAL_SCIPY_S0_TYPE,
    PosteriorGrid,
    QuadratureConfig,
    _bounded_integer,
    _bounded_real,
    _common_grid_density,
    _compute_posterior,
    _GridEvaluation,
)
from .result import CredibleInterval, KnownNuisanceFit, ParameterSummary

_MAX_DENSITY_VALUES = 131_072


def _full_log_likelihood(
    values: NDArray[np.float64],
    multiplicities: NDArray[np.int64],
    nuisance: KnownNuisance,
    backend: StableBackend,
    chunk_size: int,
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum raw-observation log densities, retaining the location-scale Jacobian.

    Only exactly equal float64 observations are coalesced. Their multiplicities
    multiply log densities, so this is the full IID likelihood: this function
    does not bin or round. The canonical backend still applies SciPy's own
    rounding of observations very close to zeta (``piecewise_x_tol_near_zeta``).
    Chunking bounds temporary memory even for large quadrature grids.
    """
    step = min(chunk_size, max(1, _MAX_DENSITY_VALUES // alpha.size))
    total = np.zeros_like(alpha)
    for start in range(0, values.size, step):
        block = values[start : start + step]
        try:
            log_pdf = np.asarray(
                backend.logpdf(
                    block[:, None, None],
                    alpha[None, :, :],
                    beta[None, :, :],
                    loc=nuisance.loc,
                    scale=nuisance.scale,
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise NumericalProbabilityError(
                "full likelihood backend returned nonnumeric log densities"
            ) from error
        if log_pdf.shape != (block.size, *alpha.shape) or not np.all(
            np.isfinite(log_pdf)
        ):
            raise NumericalProbabilityError(
                "full likelihood backend returned nonfinite or wrong-shaped "
                "log densities"
            )
        with np.errstate(over="raise", invalid="raise"):
            try:
                total += np.sum(
                    log_pdf * multiplicities[start : start + step, None, None],
                    axis=0,
                )
            except FloatingPointError as error:
                raise NumericalProbabilityError(
                    "full log likelihood accumulation overflowed"
                ) from error
    return total


@dataclass(frozen=True, slots=True, init=False)
class FullKnownNuisanceFit:
    """Refinement-checked full-likelihood reference on a compact local prior.

    Location and scale remain fixed and independently known. This reference uses
    every observation, but still excludes the exact Gaussian and one-sided shape
    boundaries. It does not assess frequentist calibration or certify a reduction.
    """

    posterior: PosteriorGrid
    observation_chunk_size: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("use fit_full_known_nuisance() to construct a full fit")

    @property
    def nuisance(self) -> KnownNuisance:
        return self.posterior.counts.nuisance

    @property
    def design(self) -> LocalDesign:
        return self.posterior.design

    @property
    def prior(self) -> LocalPrior:
        return self.posterior.prior

    @property
    def counts(self) -> CellCounts:
        """Diagnostic counts, insufficient to determine the full likelihood."""
        return self.posterior.counts

    @property
    def method(self) -> Literal["full_stable_likelihood"]:
        return "full_stable_likelihood"

    @property
    def status(self) -> Literal["research_uncertified"]:
        return "research_uncertified"

    @property
    def warnings(self) -> tuple[str, ...]:
        return (
            "research_uncertified: ordinary floating-point refinement is not "
            "a proof certificate.",
            "Full-likelihood reference on the declared compact local prior; "
            "parameter precision has not been calibrated.",
        )

    def parameter_summary(self, quantity: str) -> ParameterSummary:
        retained = self.posterior.summary_record(quantity)
        return ParameterSummary(
            mean=retained.mean,
            median=retained.median,
            credible_interval=CredibleInterval(
                lower=retained.interval_lower,
                upper=retained.interval_upper,
                mass=self.posterior.interval_mass,
            ),
        )

    def mean(self, quantity: str) -> float:
        return self.parameter_summary(quantity).mean

    def median(self, quantity: str) -> float:
        return self.parameter_summary(quantity).median

    def credible_interval(self, quantity: str) -> CredibleInterval:
        return self.parameter_summary(quantity).credible_interval

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "method": self.method,
            "parameterization": "S0",
            "parameters": {
                item.quantity: self.parameter_summary(item.quantity).to_dict()
                for item in self.posterior.summaries
            },
            "warnings": list(self.warnings),
        }

    def audit_record(self) -> dict[str, object]:
        """Return JSON-safe fit-time provenance without retaining raw observations.

        The normalizer is for the density of the ordered raw sample (including its
        scale Jacobian). It is not comparable as a Bayes factor to the probability
        of the grouped counts, which is defined on a different observation space.
        """
        posterior = self.posterior
        return {
            "schema_version": 1,
            "status": self.status,
            "method": self.method,
            "parameterization": "S0",
            "environment": posterior.environment.to_dict(),
            "known_nuisance": {
                "loc": self.nuisance.loc,
                "scale": self.nuisance.scale,
                "mode": self.nuisance.mode.value,
                "provenance": self.nuisance.provenance,
            },
            "design": asdict(self.design),
            "prior": {
                "family": "compact_uniform_rectangle",
                "h_min": self.prior.h_min,
                "h_max": self.prior.h_max,
                "p_min": self.prior.p_min,
                "p_max": self.prior.p_max,
            },
            "data": {
                "n": self.design.n,
                "sha256": posterior.data_sha256,
                "encoding": "ordered-little-endian-float64",
                "observation_chunk_size": self.observation_chunk_size,
                "likelihood": "product_of_raw_observation_densities",
                "duplicate_handling": "exact_values_with_multiplicity",
            },
            "counts": {
                "n_minus": self.counts.n_minus,
                "n_zero": self.counts.n_zero,
                "n_plus": self.counts.n_plus,
            },
            "quadrature": {
                "base_nodes": posterior.base_nodes,
                "refined_nodes": posterior.refined_nodes,
                "interval_mass": posterior.interval_mass,
                "log_normalizer": posterior.log_normalizer,
            },
            "refinement": asdict(posterior.refinement),
            "backend": {
                **asdict(posterior.backend_metadata),
                "origin": posterior.backend_origin,
            },
            "rng": None,
            "warnings": list(self.warnings),
        }


def _require_validated_canonical_magnitude(
    observations: NDArray[np.float64], nuisance: KnownNuisance
) -> None:
    """Refuse observations too extreme for the canonical backend's density.

    The refinement checks may fail to detect a density that is wrong at one
    extreme observation, so this runs before any density evaluation. The limit
    is an engineering cutoff from spot checks, not a proven accuracy bound.
    """
    limit = _CANONICAL_SCIPY_S0_TYPE.MAX_VALIDATED_ABS_Z
    # CellCounts.from_observations has already refused nonfinite standardization.
    standardized = (observations - nuisance.loc) / nuisance.scale
    largest = float(np.max(np.abs(standardized)))
    if not isfinite(largest) or largest > limit:
        raise ValidationError(
            f"largest standardized observation |z| = {largest:.6g} exceeds the "
            f"{limit:g} limit of the canonical SciPy S0 density, which loses "
            "accuracy for extreme arguments; check loc and scale"
        )


def fit_full_known_nuisance(
    x: ArrayLike,
    loc: float,
    scale: float,
    design: LocalDesign,
    prior: LocalPrior | None = None,
    *,
    provenance: str = "caller-declared",
    quadrature: QuadratureConfig | None = None,
    observation_chunk_size: int = 64,
) -> FullKnownNuisanceFit:
    """Fit the full S0 likelihood with independently known location and scale.

    Uses the same uniform compact (h, p) prior as ``fit_known_nuisance``. All
    observations must be finite, one-dimensional, and match ``design.n``. Work
    grows with the number of distinct observations times the quadrature grid
    size; use this as a reference calculation. Numerical failures raise rather
    than returning an unrefined posterior or switching to the grouped likelihood.
    A prior whose support reaches ``alpha < 1.10`` is refused with
    ``ValidationError`` before any density evaluation, because the SciPy backend
    is unreliable near ``alpha = 1``. So is a sample whose largest standardized
    observation exceeds ``1e4`` scale units, where the SciPy density loses
    accuracy. Both limits are empirical; see "Known backend limits" in
    ``docs/validation.md``.
    """
    if not isinstance(design, LocalDesign):
        raise ValidationError("design must be a LocalDesign object")
    selected_prior = LocalPrior.default(design) if prior is None else prior
    if not isinstance(selected_prior, LocalPrior) or selected_prior.design != design:
        raise ValidationError("prior must be a LocalPrior on the supplied design")
    chunk_size = _bounded_integer(
        "observation_chunk_size", observation_chunk_size, minimum=1, maximum=4096
    )
    nuisance = KnownNuisance.externally_known(
        loc=loc, scale=scale, provenance=provenance
    )
    try:
        snapshot = np.array(x, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("observations must be a numeric array") from error
    counts = CellCounts.from_observations(snapshot, nuisance=nuisance, design=design)
    observations = np.ascontiguousarray(snapshot, dtype="<f8")
    data_sha256 = hashlib.sha256(observations.tobytes(order="C")).hexdigest()
    unique, multiplicities = np.unique(observations, return_counts=True)
    backend = ScipyS0Backend()
    if type(backend) is _CANONICAL_SCIPY_S0_TYPE:
        _require_validated_canonical_magnitude(observations, nuisance)

    def likelihood(
        alpha: NDArray[np.float64], beta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _full_log_likelihood(
            unique,
            np.asarray(multiplicities, dtype=np.int64),
            nuisance,
            backend,
            chunk_size,
            alpha,
            beta,
        )

    posterior = _compute_posterior(
        counts,
        design,
        selected_prior,
        quadrature,
        backend=backend,
        log_likelihood=likelihood,
        data_sha256=data_sha256,
    )
    result = object.__new__(FullKnownNuisanceFit)
    object.__setattr__(result, "posterior", posterior)
    object.__setattr__(result, "observation_chunk_size", chunk_size)
    return result


def _retained_evaluation(posterior: PosteriorGrid) -> _GridEvaluation:
    """Reconstruct the interpolation grid without recomputing either likelihood."""
    return _GridEvaluation(
        h_axis=np.concatenate(
            ([posterior.prior.h_min], posterior.h_nodes[:, 0], [posterior.prior.h_max])
        ),
        p_axis=np.concatenate(
            ([posterior.prior.p_min], posterior.p_nodes[0, :], [posterior.prior.p_max])
        ),
        h_nodes=posterior.h_nodes,
        p_nodes=posterior.p_nodes,
        mass=posterior.mass,
        log_density=posterior.log_density,
        q_minus=posterior.q_minus,
        q_plus=posterior.q_plus,
        log_normalizer=posterior.log_normalizer,
    )


def _posterior_distances(
    full: PosteriorGrid, reduced: PosteriorGrid, points: int
) -> tuple[float, float]:
    full_common = _common_grid_density(_retained_evaluation(full), full.prior, points)
    reduced_common = _common_grid_density(
        _retained_evaluation(reduced), reduced.prior, points
    )
    measure = full_common.measure
    tv = 0.5 * float(
        np.sum(np.abs(full_common.density - reduced_common.density) * measure)
    )
    h2 = 0.5 * float(
        np.sum(
            (np.sqrt(full_common.density) - np.sqrt(reduced_common.density)) ** 2
            * measure
        )
    )
    if any(
        not isfinite(value) or not 0.0 <= value <= 1.0 + 1e-12 for value in (tv, h2)
    ):
        raise ConvergenceError("posterior comparison produced invalid distances")
    return min(1.0, tv), sqrt(min(1.0, h2))


@dataclass(frozen=True, slots=True, init=False)
class PosteriorComparison:
    """Observed-data posterior distances, with explicit grid-refinement evidence.

    Hellinger distance uses ``sqrt(0.5 * integral((sqrt(p)-sqrt(q))**2))`` and
    lies in [0, 1]. Distances concern normalized posteriors on the same prior
    support. They are numerical estimates, not error bounds or safety decisions.
    Comparison-grid refinement checks numerical integration of the two fixed
    interpolated posterior surfaces; it does not independently measure their
    interpolation error or backend error.
    """

    full: FullKnownNuisanceFit
    reduced: KnownNuisanceFit
    total_variation: float
    hellinger: float
    coarse_total_variation: float
    coarse_hellinger: float
    grid_points: int
    refined_grid_points: int
    refinement_change: float
    tolerance: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("use compare_posteriors() to construct a comparison")

    def summary(self) -> dict[str, object]:
        return {
            "status": "research_uncertified",
            "scope": "observed_data_same_compact_prior",
            "full_method": self.full.method,
            "reduced_method": self.reduced.method,
            "data_sha256": self.full.posterior.data_sha256,
            "total_variation": self.total_variation,
            "hellinger": self.hellinger,
            "coarse_total_variation": self.coarse_total_variation,
            "coarse_hellinger": self.coarse_hellinger,
            "grid_points": self.grid_points,
            "refined_grid_points": self.refined_grid_points,
            "refinement_change": self.refinement_change,
            "tolerance": self.tolerance,
            "warnings": [
                "Observed-data comparison only; no automatic safety or "
                "fallback decision.",
                "Both fits share the SciPy S0 backend and quadrature machinery; "
                "agreement is not independent numerical validation.",
                "Full-density and count-probability normalizers are not "
                "a Bayes factor.",
            ],
        }


def compare_posteriors(
    full: FullKnownNuisanceFit,
    reduced: KnownNuisanceFit,
    *,
    grid_points: int = 65,
    tolerance: float = 0.002,
) -> PosteriorComparison:
    """Compare full and grouped posteriors on two common interpolation grids.

    Design, prior, nuisance provenance, and cell counts must match. Matching
    counts suffice to identify the reduced likelihood: it cannot distinguish
    raw datasets within those cells. Both fits must use the canonical backend.
    The comparison tolerance limits numerical grid disagreement, not the distance
    between the posteriors. A large distance is a valid and informative result.
    """
    if not isinstance(full, FullKnownNuisanceFit) or not isinstance(
        reduced, KnownNuisanceFit
    ):
        raise ValidationError("comparison requires a full fit and a reduced fit")
    if (
        full.design != reduced.design
        or full.prior != reduced.prior
        or full.counts != reduced.counts
    ):
        raise ValidationError(
            "comparison requires matching design, prior, nuisance and counts"
        )
    if (
        full.posterior.backend_origin != "canonical_scipy_s0"
        or reduced.posterior.backend_origin != "canonical_scipy_s0"
        or full.posterior.backend_metadata != reduced.posterior.backend_metadata
    ):
        raise ValidationError("comparison requires the same canonical SciPy S0 backend")
    points = _bounded_integer("grid_points", grid_points, minimum=3, maximum=257)
    tolerance_value = _bounded_real(
        "tolerance", tolerance, lower=0.0, upper=1.0, upper_inclusive=True
    )
    coarse_tv, coarse_h = _posterior_distances(
        full.posterior, reduced.posterior, points
    )
    refined_points = 2 * points - 1
    tv, h = _posterior_distances(full.posterior, reduced.posterior, refined_points)
    change = max(abs(tv - coarse_tv), abs(h - coarse_h))
    if change > tolerance_value:
        raise ConvergenceError(
            f"posterior comparison grid refinement failed: {change:.6g} "
            f"exceeds tolerance {tolerance_value:.6g}"
        )
    result = object.__new__(PosteriorComparison)
    for name, value in (
        ("full", full),
        ("reduced", reduced),
        ("total_variation", tv),
        ("hellinger", h),
        ("coarse_total_variation", coarse_tv),
        ("coarse_hellinger", coarse_h),
        ("grid_points", points),
        ("refined_grid_points", refined_points),
        ("refinement_change", change),
        ("tolerance", tolerance_value),
    ):
        object.__setattr__(result, name, value)
    return result
