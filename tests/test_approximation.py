"""Explicit limiting Gamma--Beta benchmark contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.integrate import quad  # type: ignore[import-untyped]

import stableboundary.api as exact_api
from stableboundary import (
    LocalDesign,
    LocalPrior,
    fit_limiting_approximation,
)
from stableboundary.cells import CellCounts
from stableboundary.design import KnownNuisance


def _counts(design: LocalDesign, *, negative: int, positive: int) -> CellCounts:
    values = np.zeros(design.n)
    values[:positive] = design.threshold + 1.0
    values[positive : positive + negative] = -design.threshold - 1.0
    return CellCounts.from_observations(
        values,
        nuisance=KnownNuisance.externally_known(
            loc=0.0,
            scale=1.0,
            provenance="test",
        ),
        design=design,
    )


def test_limiting_intensities_and_conjugate_shapes_match_manuscript() -> None:
    design = LocalDesign.from_sample_size(64, c=1.5)
    result = fit_limiting_approximation(_counts(design, negative=2, positive=3), design)
    intensity = result.intensities(1.2, 0.7)
    assert intensity.lambda_plus == pytest.approx(2 * 1.5 * 1.2 * 0.7)
    assert intensity.lambda_minus == pytest.approx(2 * 1.5 * 1.2 * 0.3)
    assert result.gamma_shape == 6.0
    assert result.gamma_rate == 3.0
    assert result.beta_shapes == (4.0, 3.0)
    assert result.approximation is True
    assert result.method == "signed_poisson_gamma_beta_limit"


def test_limiting_posterior_is_compactly_truncated_not_unbounded() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(design=design, h_min=0.25, h_max=0.75, p_min=0.2, p_max=0.8)
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=0), design, prior
    )
    assert np.min(result.h_nodes) > prior.h_min
    assert np.max(result.h_nodes) < prior.h_max
    assert np.min(result.p_nodes) > prior.p_min
    assert np.max(result.p_nodes) < prior.p_max
    assert result.h_truncation_mass < 1.0
    assert result.p_truncation_mass == pytest.approx(0.6, rel=1e-12)
    assert result.parameter_summary("h").mean != pytest.approx(
        result.gamma_shape / result.gamma_rate
    )


def test_limiting_compact_mean_matches_hand_integrated_quadrature() -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=1, positive=2), design)
    shape = result.gamma_shape
    rate = result.gamma_rate
    lower, upper = result.support.h_min, result.support.h_max
    denominator = quad(lambda h: h ** (shape - 1.0) * np.exp(-rate * h), lower, upper)[
        0
    ]
    numerator = quad(lambda h: h**shape * np.exp(-rate * h), lower, upper)[0]
    assert result.parameter_summary("h").mean == pytest.approx(
        numerator / denominator, rel=2e-12
    )


def test_limiting_zero_tail_data_retains_no_sign_information() -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=0, positive=0), design)
    assert result.beta_shapes == (1.0, 1.0)
    assert result.evidence_status == "prior_dominated"
    assert result.parameter_summary("p").mean == pytest.approx(0.5, abs=1e-14)


def test_exact_fitter_has_no_limiting_approximation_dependency() -> None:
    source = inspect.getsource(exact_api)
    assert "approximation" not in source
    assert "fit_limiting_approximation" not in exact_api.__dict__
