"""Immutable summaries, diagnostics, audit data, and posterior prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from numbers import Integral, Real
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre, xlogy  # type: ignore[import-untyped]

from ._exceptions import (
    InfiniteVarianceError,
    NumericalProbabilityError,
    ValidationError,
)
from .cells import CellCounts
from .design import KnownNuisance, LocalDesign, LocalPrior
from .posterior import PosteriorGrid

_QUANTITIES: Final = (
    "h",
    "p",
    "alpha",
    "beta",
    "tau_plus",
    "tau_minus",
)
_MAX_PREDICTIVE_DRAWS: Final = 1_000_000
_PREDICTIVE_BATCHES: Final = 8

EvidenceStatus = Literal[
    "prior_dominated",
    "one_sided_evidence",
    "two_sided_evidence",
]
PrecisionStatus = Literal["unidentified", "not_assessed"]


def _package_version() -> str:
    try:
        return version("stableboundary")
    except PackageNotFoundError:  # pragma: no cover - source-tree fallback
        return "0.1.0"


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValidationError(f"{name} must lie strictly inside (0, 1)")
    return result


def _positive_integer(name: str, value: int, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValidationError(f"{name} must lie in [1, {maximum}]")
    return result


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError("seed must be an integer")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint64).max:
        raise ValidationError("seed must fit an unsigned 64-bit integer")
    return result


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError("threshold must be a real number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValidationError("threshold must be finite and strictly positive")
    return result


def _weighted_quantile(
    values: NDArray[np.float64],
    mass: NDArray[np.float64],
    probability: float,
) -> float:
    order = np.argsort(values.ravel(), kind="stable")
    ordered_values = values.ravel()[order]
    cumulative = np.cumsum(mass.ravel()[order])
    return float(np.interp(probability, cumulative, ordered_values))


@dataclass(frozen=True, slots=True)
class CredibleInterval:
    """Equal-tail posterior interval with its retained probability mass."""

    lower: float
    upper: float
    mass: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        mass = _probability("mass", self.mass)
        if not isfinite(lower) or not isfinite(upper) or lower > upper:
            raise NumericalProbabilityError("credible interval endpoints are invalid")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "mass", mass)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_dict(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper, "mass": self.mass}


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    """Weighted posterior mean, median, and equal-tail interval."""

    mean: float
    median: float
    credible_interval: CredibleInterval

    def __post_init__(self) -> None:
        if not isfinite(self.mean) or not isfinite(self.median):
            raise NumericalProbabilityError("posterior summaries must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "median": self.median,
            "credible_interval": self.credible_interval.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class IdentificationDiagnostics:
    """Evidence labels plus quantitative p-marginal information diagnostics."""

    evidence_status: EvidenceStatus
    precision_status: PrecisionStatus
    p_kl_divergence: float
    p_interval_width_contraction: float

    def __post_init__(self) -> None:
        if not isfinite(self.p_kl_divergence) or self.p_kl_divergence < 0.0:
            raise NumericalProbabilityError(
                "p KL divergence must be finite and nonnegative"
            )
        if not isfinite(self.p_interval_width_contraction):
            raise NumericalProbabilityError("p interval contraction must be finite")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "evidence_status": self.evidence_status,
            "precision_status": self.precision_status,
            "p_kl_divergence": self.p_kl_divergence,
            "p_interval_width_contraction": self.p_interval_width_contraction,
        }


@dataclass(frozen=True, slots=True)
class SignedTailPrediction:
    """Posterior predictive probabilities beyond a raw symmetric threshold."""

    threshold: float
    negative: float
    positive: float
    backend_method: str

    def __post_init__(self) -> None:
        threshold = _threshold(self.threshold)
        for name in ("negative", "positive"):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise NumericalProbabilityError(
                    f"predictive {name} probability must lie in [0, 1]"
                )
            object.__setattr__(self, name, value)
        if self.negative + self.positive > 1.0 + 1e-12:
            raise NumericalProbabilityError("signed predictive tails exceed unit mass")
        object.__setattr__(self, "threshold", threshold)

    @property
    def q_minus(self) -> float:
        return self.negative

    @property
    def q_plus(self) -> float:
        return self.positive


@dataclass(frozen=True, slots=True)
class ExpectedExceedanceCounts:
    """Expected future signed exceedance counts at a raw threshold."""

    future_size: int
    threshold: float
    negative: float
    positive: float

    @property
    def n_minus(self) -> float:
        return self.negative

    @property
    def n_plus(self) -> float:
        return self.positive


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveSample:
    """Read-only seeded draws from the full stable posterior mixture."""

    values: NDArray[np.float64]
    draw_count: int
    seed: int
    bit_generator: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (self.draw_count,) or not np.all(np.isfinite(values)):
            raise NumericalProbabilityError("posterior predictive draws are invalid")
        retained = np.array(values, dtype=np.float64, copy=True)
        retained.setflags(write=False)
        object.__setattr__(self, "values", retained)


@dataclass(frozen=True, slots=True)
class PredictiveQuantileEstimate:
    """Seeded posterior predictive quantile with batch Monte Carlo error."""

    probability: float
    value: float
    monte_carlo_standard_error: float
    draw_count: int
    seed: int
    bit_generator: str
    batches: int = _PREDICTIVE_BATCHES

    def __post_init__(self) -> None:
        probability = _probability("probability", self.probability)
        if not isfinite(self.value):
            raise NumericalProbabilityError("predictive quantile must be finite")
        if (
            not isfinite(self.monte_carlo_standard_error)
            or self.monte_carlo_standard_error < 0.0
        ):
            raise NumericalProbabilityError(
                "predictive quantile Monte Carlo error must be finite and nonnegative"
            )
        if self.batches != _PREDICTIVE_BATCHES:
            raise NumericalProbabilityError(
                "predictive quantiles require eight batches"
            )
        object.__setattr__(self, "probability", probability)

    @property
    def mcse(self) -> float:
        return self.monte_carlo_standard_error

    @property
    def requested_probability(self) -> float:
        return self.probability

    @property
    def monte_carlo_error(self) -> float:
        return self.monte_carlo_standard_error

    def audit_record(self) -> dict[str, str | int | float]:
        return {
            "probability": self.probability,
            "value": self.value,
            "monte_carlo_standard_error": self.monte_carlo_standard_error,
            "draw_count": self.draw_count,
            "seed": self.seed,
            "bit_generator": self.bit_generator,
            "batches": self.batches,
        }


@dataclass(frozen=True, slots=True)
class KnownNuisanceFit:
    """Exact finite-cell Bayesian fit with known location and scale."""

    nuisance: KnownNuisance
    design: LocalDesign
    prior: LocalPrior
    counts: CellCounts
    posterior: PosteriorGrid
    status: Literal["research_uncertified"] = field(
        default="research_uncertified", init=False
    )
    method: Literal["exact_finite_three_cell"] = field(
        default="exact_finite_three_cell", init=False
    )

    def __post_init__(self) -> None:
        self.nuisance.require_externally_known()
        if self.prior.design != self.design:
            raise ValidationError("fit prior must use the retained design")
        if self.counts.n != self.design.n:
            raise ValidationError("fit counts must match the retained design")
        if self.posterior.r != self.design.r:
            raise ValidationError("posterior and design must retain the same r")

    @property
    def r(self) -> float:
        return self.design.r

    @property
    def evidence_status(self) -> EvidenceStatus:
        return self.identification.evidence_status

    @property
    def precision_status(self) -> PrecisionStatus:
        return self.identification.precision_status

    @property
    def warnings(self) -> tuple[str, ...]:
        base = (
            "research_uncertified: ordinary floating-point refinement is not "
            "a proof certificate.",
        )
        if self.counts.n_minus + self.counts.n_plus == 0:
            return base + (
                "No signed-tail events: beta and p are prior-dominated and "
                "unidentified.",
            )
        if self.counts.n_minus == 0 or self.counts.n_plus == 0:
            return base + (
                "Only one signed tail has events: evidence is one-sided and "
                "precision is not assessed.",
            )
        return base + (
            "Signed-tail evidence is two-sided; precision is not assessed "
            "before calibration.",
        )

    def parameter_summary(self, quantity: str) -> ParameterSummary:
        values = self.posterior.values(quantity)
        mass = self.posterior.mass
        interval_mass = self.posterior.interval_mass
        tail = 0.5 * (1.0 - interval_mass)
        return ParameterSummary(
            mean=float(np.sum(mass * values)),
            median=_weighted_quantile(values, mass, 0.5),
            credible_interval=CredibleInterval(
                lower=_weighted_quantile(values, mass, tail),
                upper=_weighted_quantile(values, mass, 1.0 - tail),
                mass=interval_mass,
            ),
        )

    def mean(self, quantity: str) -> float:
        """Return a weighted posterior mean for one supported quantity."""
        return self.parameter_summary(quantity).mean

    def median(self, quantity: str) -> float:
        """Return a weighted posterior median for one supported quantity."""
        return self.parameter_summary(quantity).median

    def credible_interval(self, quantity: str) -> CredibleInterval:
        """Return the configured equal-tail interval for one quantity."""
        return self.parameter_summary(quantity).credible_interval

    @property
    def identification(self) -> IdentificationDiagnostics:
        total_tail = self.counts.n_minus + self.counts.n_plus
        if total_tail == 0:
            evidence: EvidenceStatus = "prior_dominated"
            precision: PrecisionStatus = "unidentified"
        elif self.counts.n_minus == 0 or self.counts.n_plus == 0:
            evidence = "one_sided_evidence"
            precision = "not_assessed"
        else:
            evidence = "two_sided_evidence"
            precision = "not_assessed"

        p_mass = np.sum(self.posterior.mass, axis=0)
        _, raw_weights = roots_legendre(self.posterior.refined_nodes)
        p_weights = 0.5 * (self.prior.p_max - self.prior.p_min) * raw_weights
        posterior_density = p_mass / p_weights
        prior_density = 1.0 / (self.prior.p_max - self.prior.p_min)
        kl = float(np.sum(xlogy(p_mass, posterior_density / prior_density)))
        if not isfinite(kl):
            raise NumericalProbabilityError("p-marginal KL divergence is nonfinite")
        kl = max(0.0, kl)
        p_interval = self.parameter_summary("p").credible_interval
        prior_interval_width = self.posterior.interval_mass * (
            self.prior.p_max - self.prior.p_min
        )
        contraction = 1.0 - p_interval.width / prior_interval_width
        return IdentificationDiagnostics(
            evidence_status=evidence,
            precision_status=precision,
            p_kl_divergence=kl,
            p_interval_width_contraction=contraction,
        )

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "method": self.method,
            "parameterization": self.posterior.backend_parameterization,
            "r": self.r,
            "counts": {
                "n_minus": self.counts.n_minus,
                "n_zero": self.counts.n_zero,
                "n_plus": self.counts.n_plus,
                "n": self.counts.n,
            },
            "identification": self.identification.to_dict(),
            "parameters": {
                name: self.parameter_summary(name).to_dict() for name in _QUANTITIES
            },
            "warnings": list(self.warnings),
        }

    def audit_record(self) -> dict[str, object]:
        refinement = self.posterior.refinement
        return {
            "schema_version": 1,
            "package_version": _package_version(),
            "status": self.status,
            "method": self.method,
            "parameterization": self.posterior.backend_parameterization,
            "known_nuisance": {
                "loc": self.nuisance.loc,
                "scale": self.nuisance.scale,
                "mode": self.nuisance.mode.value,
                "provenance": self.nuisance.provenance,
            },
            "design": {
                "n": self.design.n,
                "c": self.design.c,
                "r": self.design.r,
                "threshold": self.design.threshold,
                "formula_id": self.design.formula_id,
                "formula_version": self.design.formula_version,
                "critical_rate_relative_residual": (
                    self.design.critical_rate_relative_residual
                ),
            },
            "prior": {
                "family": "compact_uniform_rectangle",
                "h_min": self.prior.h_min,
                "h_max": self.prior.h_max,
                "p_min": self.prior.p_min,
                "p_max": self.prior.p_max,
            },
            "counts": {
                "n_minus": self.counts.n_minus,
                "n_zero": self.counts.n_zero,
                "n_plus": self.counts.n_plus,
                "n": self.counts.n,
                "threshold": self.counts.threshold,
            },
            "quadrature": {
                "base_nodes": self.posterior.base_nodes,
                "refined_nodes": self.posterior.refined_nodes,
                "interval_mass": self.posterior.interval_mass,
                "log_normalizer": self.posterior.log_normalizer,
            },
            "refinement": {
                "tolerance": refinement.tolerance,
                "common_grid_points": refinement.common_grid_points,
                "joint_total_variation": refinement.joint_total_variation,
                "log_normalizer_change": refinement.log_normalizer_change,
                "summary_changes": {
                    item.quantity: {
                        "mean": item.mean,
                        "interval_lower": item.interval_lower,
                        "interval_upper": item.interval_upper,
                    }
                    for item in refinement.summaries
                },
                "predictive_tail": {
                    "negative": refinement.predictive_tail.negative,
                    "positive": refinement.predictive_tail.positive,
                },
                "converged": refinement.converged,
            },
            "backend": {
                "method": self.posterior.backend_method,
                "tolerance": self.posterior.backend_tolerance,
                "parameterization": self.posterior.backend_parameterization,
                "library": self.posterior.backend_metadata.library,
                "library_version": self.posterior.backend_metadata.library_version,
                "effective_settings": dict(
                    self.posterior.backend_metadata.effective_settings
                ),
            },
            "rng": None,
            "identification": self.identification.to_dict(),
            "warnings": list(self.warnings),
        }

    def tail_probabilities(self, threshold: float) -> SignedTailPrediction:
        raw_threshold = _threshold(threshold)
        alpha = self.posterior.values("alpha")
        beta = self.posterior.values("beta")
        backend = self.posterior.prediction_backend()
        log_negative = np.asarray(
            backend.logcdf(
                -raw_threshold,
                alpha,
                beta,
                loc=self.nuisance.loc,
                scale=self.nuisance.scale,
            ),
            dtype=np.float64,
        )
        log_positive = np.asarray(
            backend.logsf(
                raw_threshold,
                alpha,
                beta,
                loc=self.nuisance.loc,
                scale=self.nuisance.scale,
            ),
            dtype=np.float64,
        )
        if (
            log_negative.shape != self.posterior.mass.shape
            or log_positive.shape != self.posterior.mass.shape
            or not np.all(np.isfinite(log_negative))
            or not np.all(np.isfinite(log_positive))
            or np.any(log_negative > 0.0)
            or np.any(log_positive > 0.0)
        ):
            raise NumericalProbabilityError(
                "posterior predictive backend returned invalid log tails"
            )
        negative = float(np.sum(self.posterior.mass * np.exp(log_negative)))
        positive = float(np.sum(self.posterior.mass * np.exp(log_positive)))
        return SignedTailPrediction(
            threshold=raw_threshold,
            negative=negative,
            positive=positive,
            backend_method=backend.metadata.method,
        )

    def posterior_tail_probabilities(self, threshold: float) -> SignedTailPrediction:
        """Alias emphasizing that signed raw-tail probabilities are mixtures."""
        return self.tail_probabilities(threshold)

    def expected_exceedance_counts(
        self,
        future_size: int,
        threshold: float,
    ) -> ExpectedExceedanceCounts:
        size = _positive_integer("future_size", future_size, maximum=10**12)
        prediction = self.tail_probabilities(threshold)
        return ExpectedExceedanceCounts(
            future_size=size,
            threshold=prediction.threshold,
            negative=size * prediction.negative,
            positive=size * prediction.positive,
        )

    def posterior_predictive(
        self,
        draw_count: int,
        *,
        seed: int,
    ) -> PosteriorPredictiveSample:
        draws = _positive_integer(
            "draw_count", draw_count, maximum=_MAX_PREDICTIVE_DRAWS
        )
        seed_value = _seed(seed)
        generator = np.random.default_rng(seed_value)
        flat_mass = self.posterior.mass.ravel()
        indices = generator.choice(flat_mass.size, size=draws, p=flat_mass)
        alpha = self.posterior.values("alpha").ravel()
        beta = self.posterior.values("beta").ravel()
        values = np.empty(draws, dtype=np.float64)
        backend = self.posterior.prediction_backend()
        for index in np.unique(indices):
            positions = np.flatnonzero(indices == index)
            values[positions] = backend.rvs(
                float(alpha[index]),
                float(beta[index]),
                loc=self.nuisance.loc,
                scale=self.nuisance.scale,
                size=int(positions.size),
                random_state=generator,
            )
        return PosteriorPredictiveSample(
            values=values,
            draw_count=draws,
            seed=seed_value,
            bit_generator=generator.bit_generator.__class__.__name__,
        )

    def predictive_draws(
        self,
        draw_count: int,
        *,
        seed: int,
    ) -> PosteriorPredictiveSample:
        """Return seeded draws from the full stable posterior mixture."""
        return self.posterior_predictive(draw_count, seed=seed)

    def predictive_quantile(
        self,
        probability: float,
        *,
        draws: int = 8_000,
        seed: int = 0,
    ) -> PredictiveQuantileEstimate:
        requested = _probability("probability", probability)
        draw_count = _positive_integer("draws", draws, maximum=_MAX_PREDICTIVE_DRAWS)
        if draw_count < _PREDICTIVE_BATCHES:
            raise ValidationError("draws must provide at least one draw per batch")
        sample = self.posterior_predictive(draw_count, seed=seed)
        batches = np.array_split(sample.values, _PREDICTIVE_BATCHES)
        batch_quantiles = np.array(
            [np.quantile(batch, requested) for batch in batches],
            dtype=np.float64,
        )
        mcse = float(np.std(batch_quantiles, ddof=1) / np.sqrt(_PREDICTIVE_BATCHES))
        return PredictiveQuantileEstimate(
            probability=requested,
            value=float(np.quantile(sample.values, requested)),
            monte_carlo_standard_error=mcse,
            draw_count=draw_count,
            seed=sample.seed,
            bit_generator=sample.bit_generator,
        )

    def predictive_variance(self) -> float:
        alpha = self.posterior.values("alpha")
        if np.any((self.posterior.mass > 0.0) & (alpha < 2.0)):
            raise InfiniteVarianceError(
                "predictive variance is infinite because posterior support "
                "includes alpha < 2"
            )
        raise NumericalProbabilityError(
            "posterior variance status is inconsistent with the Phase 1 prior"
        )


__all__ = [
    "CredibleInterval",
    "ExpectedExceedanceCounts",
    "IdentificationDiagnostics",
    "KnownNuisanceFit",
    "ParameterSummary",
    "PosteriorPredictiveSample",
    "PredictiveQuantileEstimate",
    "SignedTailPrediction",
]
