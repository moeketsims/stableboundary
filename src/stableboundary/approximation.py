"""Explicit compactly truncated signed-Poisson Gamma--Beta approximation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from numbers import Real
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import (  # type: ignore[import-untyped]
    betaln,
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


def _weighted_quantile(
    values: NDArray[np.float64],
    mass: NDArray[np.float64],
    probability: float,
) -> float:
    order = np.argsort(values.ravel(), kind="stable")
    cumulative = np.cumsum(mass.ravel()[order])
    return float(np.interp(probability, cumulative, values.ravel()[order]))


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

    def parameter_summary(self, quantity: str) -> ParameterSummary:
        values = self._values(quantity)
        tail = 0.5 * (1.0 - _INTERVAL_MASS)
        return ParameterSummary(
            mean=float(np.sum(self.mass * values)),
            median=_weighted_quantile(values, self.mass, 0.5),
            credible_interval=CredibleInterval(
                lower=_weighted_quantile(values, self.mass, tail),
                upper=_weighted_quantile(values, self.mass, 1.0 - tail),
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
