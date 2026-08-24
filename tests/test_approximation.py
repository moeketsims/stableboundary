"""Explicit limiting Gamma--Beta benchmark contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.integrate import quad  # type: ignore[import-untyped]
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import (  # type: ignore[import-untyped]
    betainc,
    betaln,
    gammainc,
)

import stableboundary.api as exact_api
import stableboundary.approximation as approximation_module
from stableboundary import (
    LocalDesign,
    LocalPrior,
    ValidationError,
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


def test_limiting_zero_tail_uniform_p_quantiles_are_continuous_and_exact() -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=0, positive=0), design)
    p_summary = result.parameter_summary("p")
    beta_summary = result.parameter_summary("beta")

    assert p_summary.credible_interval.lower == pytest.approx(0.095, abs=2e-14)
    assert p_summary.median == pytest.approx(0.5, abs=2e-14)
    assert p_summary.credible_interval.upper == pytest.approx(0.905, abs=2e-14)
    assert beta_summary.credible_interval.lower == pytest.approx(-0.81, abs=4e-14)
    assert beta_summary.median == pytest.approx(0.0, abs=4e-14)
    assert beta_summary.credible_interval.upper == pytest.approx(0.81, abs=4e-14)


def test_limiting_h_and_alpha_quantiles_invert_the_truncated_gamma_cdf() -> None:
    design = LocalDesign.from_sample_size(64, c=1.25)
    prior = LocalPrior(
        design=design,
        h_min=0.35,
        h_max=2.2,
        p_min=0.1,
        p_max=0.8,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=1, positive=2), design, prior
    )
    shape = result.gamma_shape
    rate = result.gamma_rate
    denominator = gammainc(shape, rate * prior.h_max) - gammainc(
        shape, rate * prior.h_min
    )

    def reference(probability: float) -> float:
        return float(
            brentq(
                lambda h: (
                    (gammainc(shape, rate * h) - gammainc(shape, rate * prior.h_min))
                    / denominator
                    - probability
                ),
                prior.h_min,
                prior.h_max,
            )
        )

    h_summary = result.parameter_summary("h")
    alpha_summary = result.parameter_summary("alpha")
    expected = [reference(probability) for probability in (0.05, 0.5, 0.95)]
    assert (
        h_summary.credible_interval.lower,
        h_summary.median,
        h_summary.credible_interval.upper,
    ) == pytest.approx(expected, rel=2e-12, abs=2e-13)
    assert (
        alpha_summary.credible_interval.lower,
        alpha_summary.median,
        alpha_summary.credible_interval.upper,
    ) == pytest.approx(
        [
            2.0 - design.r * expected[2],
            2.0 - design.r * expected[1],
            2.0 - design.r * expected[0],
        ],
        rel=2e-12,
        abs=2e-13,
    )


def test_limiting_asymmetric_tau_quantiles_match_independent_p_integration() -> None:
    design = LocalDesign.from_sample_size(64, c=1.25)
    prior = LocalPrior(
        design=design,
        h_min=0.35,
        h_max=2.1,
        p_min=0.15,
        p_max=0.75,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=1, positive=4), design, prior
    )
    h_shape = result.gamma_shape
    h_rate = result.gamma_rate
    p_positive, p_negative = result.beta_shapes
    h_normalizer = gammainc(h_shape, h_rate * prior.h_max) - gammainc(
        h_shape, h_rate * prior.h_min
    )
    p_normalizer = betainc(p_positive, p_negative, prior.p_max) - betainc(
        p_positive, p_negative, prior.p_min
    )

    def h_cdf(value: float) -> float:
        if value <= prior.h_min:
            return 0.0
        if value >= prior.h_max:
            return 1.0
        return float(
            (
                gammainc(h_shape, h_rate * value)
                - gammainc(h_shape, h_rate * prior.h_min)
            )
            / h_normalizer
        )

    def p_density(value: float) -> float:
        log_density = (
            (p_positive - 1.0) * np.log(value)
            + (p_negative - 1.0) * np.log1p(-value)
            - betaln(p_positive, p_negative)
        )
        return float(np.exp(log_density) / p_normalizer)

    def reference_quantile(probability: float, *, positive: bool) -> float:
        allocation_lower = prior.p_min if positive else 1.0 - prior.p_max
        allocation_upper = prior.p_max if positive else 1.0 - prior.p_min
        lower = design.r * prior.h_min * allocation_lower
        upper = design.r * prior.h_max * allocation_upper

        def cdf(value: float) -> float:
            breakpoints = []
            for h_bound in (prior.h_min, prior.h_max):
                allocation = value / (design.r * h_bound)
                p_break = allocation if positive else 1.0 - allocation
                if prior.p_min < p_break < prior.p_max:
                    breakpoints.append(p_break)
            integral = quad(
                lambda p: (
                    p_density(p)
                    * h_cdf(value / (design.r * (p if positive else 1.0 - p)))
                ),
                prior.p_min,
                prior.p_max,
                epsabs=2e-13,
                epsrel=2e-13,
                points=sorted(set(breakpoints)) or None,
            )[0]
            return float(integral)

        return float(
            brentq(
                lambda value: cdf(value) - probability,
                lower,
                upper,
                xtol=1e-14,
                rtol=1e-14,
            )
        )

    probabilities = (0.05, 0.5, 0.95)
    expected_positive = [
        reference_quantile(probability, positive=True) for probability in probabilities
    ]
    expected_negative = [
        reference_quantile(probability, positive=False) for probability in probabilities
    ]
    positive_summary = result.parameter_summary("tau_plus")
    negative_summary = result.parameter_summary("tau_minus")

    assert expected_positive != pytest.approx(expected_negative, rel=1e-4)
    assert (
        positive_summary.credible_interval.lower,
        positive_summary.median,
        positive_summary.credible_interval.upper,
    ) == pytest.approx(expected_positive, rel=3e-10, abs=3e-12)
    assert (
        negative_summary.credible_interval.lower,
        negative_summary.median,
        negative_summary.credible_interval.upper,
    ) == pytest.approx(expected_negative, rel=3e-10, abs=3e-12)


def test_limiting_tau_product_quantiles_have_exact_compact_endpoints() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(
        design=design,
        h_min=0.4,
        h_max=1.7,
        p_min=0.2,
        p_max=0.7,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=1, positive=3), design, prior
    )
    h_distribution = result._h_distribution()
    p_distribution = result._p_distribution()

    assert approximation_module._product_quantile(
        h_distribution,
        p_distribution,
        design.r,
        0.0,
        positive=True,
    ) == pytest.approx(design.r * prior.h_min * prior.p_min)
    assert approximation_module._product_quantile(
        h_distribution,
        p_distribution,
        design.r,
        1.0,
        positive=True,
    ) == pytest.approx(design.r * prior.h_max * prior.p_max)
    assert approximation_module._product_quantile(
        h_distribution,
        p_distribution,
        design.r,
        0.0,
        positive=False,
    ) == pytest.approx(design.r * prior.h_min * (1.0 - prior.p_max))
    assert approximation_module._product_quantile(
        h_distribution,
        p_distribution,
        design.r,
        1.0,
        positive=False,
    ) == pytest.approx(design.r * prior.h_max * (1.0 - prior.p_min))


def test_limiting_quantiles_remain_stable_in_a_compact_beta_tail() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(
        design=design,
        h_min=0.4,
        h_max=1.4,
        p_min=0.01,
        p_max=0.02,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=64), design, prior
    )
    shape = result.p_shape_positive

    def reference(probability: float) -> float:
        target = (1.0 - probability) * prior.p_min**shape + (
            probability * prior.p_max**shape
        )
        return target ** (1.0 / shape)

    summary = result.parameter_summary("p")
    expected = [reference(probability) for probability in (0.05, 0.5, 0.95)]
    assert (
        summary.credible_interval.lower,
        summary.median,
        summary.credible_interval.upper,
    ) == pytest.approx(expected, rel=2e-12, abs=2e-14)


def test_limiting_continuous_summaries_are_normalized_and_inside_domains() -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=2, positive=3), design)
    support = result.support
    domains = {
        "h": (support.h_min, support.h_max),
        "p": (support.p_min, support.p_max),
        "alpha": (
            2.0 - design.r * support.h_max,
            2.0 - design.r * support.h_min,
        ),
        "beta": (2.0 * support.p_min - 1.0, 2.0 * support.p_max - 1.0),
        "tau_plus": (
            design.r * support.h_min * support.p_min,
            design.r * support.h_max * support.p_max,
        ),
        "tau_minus": (
            design.r * support.h_min * (1.0 - support.p_max),
            design.r * support.h_max * (1.0 - support.p_min),
        ),
    }
    for quantity, (lower, upper) in domains.items():
        parameter = result.parameter_summary(quantity)
        interval = parameter.credible_interval
        assert lower <= parameter.mean <= upper
        assert lower <= interval.lower <= parameter.median <= interval.upper <= upper
        assert interval.mass == pytest.approx(0.9)

    with pytest.raises(ValidationError, match="unknown approximation quantity"):
        result.parameter_summary("not_a_parameter")


def test_exact_fitter_has_no_limiting_approximation_dependency() -> None:
    source = inspect.getsource(exact_api)
    assert "approximation" not in source
    assert "fit_limiting_approximation" not in exact_api.__dict__
