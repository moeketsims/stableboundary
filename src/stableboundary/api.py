"""Narrow public fitting workflows."""

from __future__ import annotations

from numpy.typing import ArrayLike

from ._exceptions import ValidationError
from .cells import CellCounts
from .design import KnownNuisance, LocalDesign, LocalPrior
from .posterior import QuadratureConfig, compute_exact_posterior
from .result import KnownNuisanceFit


def fit_known_nuisance(
    x: ArrayLike,
    loc: float,
    scale: float,
    design: LocalDesign,
    prior: LocalPrior | None = None,
    *,
    provenance: str = "caller-declared",
    quadrature: QuadratureConfig | None = None,
) -> KnownNuisanceFit:
    """Fit the exact finite-cell posterior with independently known nuisance."""
    if not isinstance(design, LocalDesign):
        raise ValidationError("design must be a LocalDesign object")
    selected_prior = LocalPrior.default(design) if prior is None else prior
    if not isinstance(selected_prior, LocalPrior) or selected_prior.design != design:
        raise ValidationError("prior must be a LocalPrior on the supplied design")
    nuisance = KnownNuisance.externally_known(
        loc=loc,
        scale=scale,
        provenance=provenance,
    )
    counts = CellCounts.from_observations(x, nuisance=nuisance, design=design)
    posterior = compute_exact_posterior(
        counts,
        design,
        selected_prior,
        quadrature,
    )
    return KnownNuisanceFit._from_components(
        nuisance=nuisance,
        design=design,
        prior=selected_prior,
        counts=counts,
        posterior=posterior,
    )


__all__ = ["fit_known_nuisance"]
