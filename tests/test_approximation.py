"""Explicit limiting Gamma--Beta benchmark contracts."""

from __future__ import annotations

import inspect
from math import exp, expm1, lgamma, log, log1p
from warnings import warn

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad  # type: ignore[import-untyped]
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
    NumericalProbabilityError,
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


def _log_difference(log_upper: float, log_lower: float) -> float:
    """Return log(exp(log_upper) - exp(log_lower)) without underflow."""
    assert log_lower < log_upper
    return log_upper + log(-expm1(log_lower - log_upper))


def _log_regularized_integer_gamma_lower(shape: int, value: float) -> float:
    """Independent log-domain lower Gamma CDF for integer shape and small x."""
    term = 1.0
    series = 1.0
    for index in range(1, 10_000):
        term *= value / (shape + index)
        series += term
        if term <= np.finfo(np.float64).eps * series:
            break
    else:  # pragma: no cover - the adversarial cases converge in a few terms
        raise AssertionError("reference Gamma series did not converge")
    return -value + shape * log(value) - lgamma(shape + 1.0) + log(series)


def _log_gamma_interval_mass(
    shape: int,
    rate: float,
    lower: float,
    upper: float,
) -> float:
    return _log_difference(
        _log_regularized_integer_gamma_lower(shape, rate * upper),
        _log_regularized_integer_gamma_lower(shape, rate * lower),
    )


def _truncated_gamma_mean(
    shape: int,
    rate: float,
    lower: float,
    upper: float,
) -> float:
    log_mass = _log_gamma_interval_mass(shape, rate, lower, upper)
    log_shifted_mass = _log_gamma_interval_mass(shape + 1, rate, lower, upper)
    return (shape / rate) * exp(log_shifted_mass - log_mass)


def _truncated_gamma_cdf(
    value: float,
    shape: int,
    rate: float,
    lower: float,
    upper: float,
) -> float:
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    return exp(
        _log_gamma_interval_mass(shape, rate, lower, value)
        - _log_gamma_interval_mass(shape, rate, lower, upper)
    )


def _truncated_gamma_survival(
    value: float,
    shape: int,
    rate: float,
    lower: float,
    upper: float,
) -> float:
    if value <= lower:
        return 1.0
    if value >= upper:
        return 0.0
    return exp(
        _log_gamma_interval_mass(shape, rate, value, upper)
        - _log_gamma_interval_mass(shape, rate, lower, upper)
    )


def _truncated_power_mean(shape: int, lower: float, upper: float) -> float:
    log_mass = shape * log(upper) + log1p(-exp(shape * (log(lower) - log(upper))))
    log_first_moment = (shape + 1) * log(upper) + log1p(
        -exp((shape + 1) * (log(lower) - log(upper)))
    )
    return (shape / (shape + 1.0)) * exp(log_first_moment - log_mass)


def _truncated_power_quantile(
    probability: float,
    shape: int,
    lower: float,
    upper: float,
) -> float:
    log_power = float(
        np.logaddexp(
            log1p(-probability) + shape * log(lower),
            log(probability) + shape * log(upper),
        )
    )
    return exp(log_power / shape)


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


def test_limiting_approximation_rejects_counts_bound_to_another_full_design() -> None:
    count_design = LocalDesign.from_sample_size(64, c=1.0)
    requested_design = LocalDesign.from_sample_size(64, c=1.25)
    assert count_design.threshold != requested_design.threshold
    with pytest.raises(ValidationError, match="full supplied design"):
        fit_limiting_approximation(
            _counts(count_design, negative=1, positive=2),
            requested_design,
        )


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


def test_subnormal_gamma_beta_cdfs_share_one_scaled_normalizer() -> None:
    design = LocalDesign.from_sample_size(5_000)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=0.01,
        p_max=0.02,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n), design, prior
    )
    h_distribution = result._h_distribution()
    p_distribution = result._p_distribution()
    h_shape = int(result.h_shape)
    p_shape = int(result.p_shape_positive)

    expected_h_log_mass = _log_gamma_interval_mass(
        h_shape, result.h_rate, prior.h_min, prior.h_max
    )
    expected_p_log_mass = p_shape * log(prior.p_max) + log1p(
        -exp(p_shape * (log(prior.p_min) - log(prior.p_max)))
    )
    assert result.h_truncation_mass == 0.0
    assert result.p_truncation_mass == 0.0
    assert result.h_log_truncation_mass == pytest.approx(expected_h_log_mass, abs=1e-9)
    assert result.p_log_truncation_mass == pytest.approx(expected_p_log_mass, abs=1e-9)

    for probability in (0.05, 0.5, 0.95):
        p_quantile = _truncated_power_quantile(
            probability, p_shape, prior.p_min, prior.p_max
        )
        assert p_distribution.quantile(probability) == pytest.approx(
            p_quantile, abs=1e-10
        )
        assert p_distribution.cdf(p_quantile) == pytest.approx(probability, abs=2e-11)
        assert p_distribution.survival(p_quantile) == pytest.approx(
            1.0 - probability, abs=2e-11
        )
        assert p_distribution.cdf(p_quantile) + p_distribution.survival(
            p_quantile
        ) == pytest.approx(1.0, abs=2e-11)

        h_quantile = h_distribution.quantile(probability)
        assert h_distribution.cdf(h_quantile) == pytest.approx(
            _truncated_gamma_cdf(
                h_quantile,
                h_shape,
                result.h_rate,
                prior.h_min,
                prior.h_max,
            ),
            abs=2e-11,
        )
        assert h_distribution.survival(h_quantile) == pytest.approx(
            _truncated_gamma_survival(
                h_quantile,
                h_shape,
                result.h_rate,
                prior.h_min,
                prior.h_max,
            ),
            abs=2e-11,
        )
        assert h_distribution.cdf(h_quantile) + h_distribution.survival(
            h_quantile
        ) == pytest.approx(1.0, abs=2e-11)


def test_reflected_subnormal_beta_tail_preserves_cdf_survival_symmetry() -> None:
    design = LocalDesign.from_sample_size(5_000)
    positive_prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=0.01,
        p_max=0.02,
    )
    negative_prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=0.98,
        p_max=0.99,
    )
    positive = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n), design, positive_prior
    )
    negative = fit_limiting_approximation(
        _counts(design, negative=design.n, positive=0), design, negative_prior
    )
    positive_distribution = positive._p_distribution()
    negative_distribution = negative._p_distribution()

    assert positive.p_log_truncation_mass == pytest.approx(
        negative.p_log_truncation_mass, abs=1e-9
    )
    assert positive._continuous_mean("p") + negative._continuous_mean(
        "p"
    ) == pytest.approx(1.0, abs=1e-10)
    for probability in (0.05, 0.5, 0.95):
        positive_quantile = positive_distribution.quantile(probability)
        reflected_value = 1.0 - positive_quantile
        assert negative_distribution.cdf(reflected_value) == pytest.approx(
            positive_distribution.survival(positive_quantile), abs=2e-11
        )
        assert negative_distribution.quantile(probability) == pytest.approx(
            1.0 - positive_distribution.quantile(1.0 - probability),
            abs=1e-10,
        )


def test_concentrated_continuous_means_use_gamma_beta_independence() -> None:
    design = LocalDesign.from_sample_size(5_000)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=0.01,
        p_max=0.02,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n), design, prior
    )
    h_mean = _truncated_gamma_mean(
        int(result.h_shape), result.h_rate, prior.h_min, prior.h_max
    )
    p_mean = _truncated_power_mean(
        int(result.p_shape_positive), prior.p_min, prior.p_max
    )
    expected = {
        "h": h_mean,
        "p": p_mean,
        "alpha": 2.0 - design.r * h_mean,
        "beta": 2.0 * p_mean - 1.0,
        "tau_plus": design.r * h_mean * p_mean,
        "tau_minus": design.r * h_mean * (1.0 - p_mean),
    }
    for quantity, value in expected.items():
        assert result._continuous_mean(quantity) == pytest.approx(value, abs=1e-10)

    grid_h_mean = float(np.sum(result.mass * result.h_nodes))
    grid_p_mean = float(np.sum(result.mass * result.p_nodes))
    assert abs(grid_h_mean - h_mean) > 1e-6
    assert abs(grid_p_mean - p_mean) > 1e-10
    assert result.parameter_summary("h").mean == pytest.approx(h_mean, abs=1e-10)
    assert result.parameter_summary("p").mean == pytest.approx(p_mean, abs=1e-10)


def test_concentrated_product_quantiles_match_uniformized_reference() -> None:
    design = LocalDesign.from_sample_size(256)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=0.01,
        p_max=0.02,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n), design, prior
    )
    h_shape = int(result.h_shape)
    p_shape = int(result.p_shape_positive)
    lower_power_ratio = exp(p_shape * (log(prior.p_min) - log(prior.p_max)))

    def p_from_uniform(probability: float) -> float:
        if probability <= 0.0:
            return prior.p_min
        if probability >= 1.0:
            return prior.p_max
        return _truncated_power_quantile(probability, p_shape, prior.p_min, prior.p_max)

    def reference_cdf(value: float) -> float:
        breakpoints = []
        for h_bound in (prior.h_min, prior.h_max):
            p_break = value / (design.r * h_bound)
            if prior.p_min < p_break < prior.p_max:
                transformed = (
                    exp(p_shape * (log(p_break) - log(prior.p_max))) - lower_power_ratio
                ) / (1.0 - lower_power_ratio)
                breakpoints.append(transformed)
        return float(
            quad(
                lambda probability: _truncated_gamma_cdf(
                    value / (design.r * p_from_uniform(probability)),
                    h_shape,
                    result.h_rate,
                    prior.h_min,
                    prior.h_max,
                ),
                0.0,
                1.0,
                epsabs=2e-12,
                epsrel=2e-12,
                limit=300,
                points=sorted(set(breakpoints)) or None,
            )[0]
        )

    lower = design.r * prior.h_min * prior.p_min
    upper = design.r * prior.h_max * prior.p_max
    h_distribution = result._h_distribution()
    p_distribution = result._p_distribution()
    for probability in (0.05, 0.5, 0.95):
        expected = brentq(
            lambda value, probability=probability: reference_cdf(value) - probability,
            lower,
            upper,
            xtol=1e-14,
            rtol=1e-14,
        )
        actual = approximation_module._product_quantile(
            h_distribution,
            p_distribution,
            design.r,
            probability,
            positive=True,
        )
        assert actual == pytest.approx(expected, abs=1e-9)


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


def test_limiting_integrals_fail_on_warning_and_reported_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning_quad(*args: object, **kwargs: object) -> tuple[float, float]:
        del args, kwargs
        warn("forced failure", IntegrationWarning, stacklevel=2)
        return 1.0, 0.0

    monkeypatch.setattr(approximation_module, "quad", warning_quad)
    with pytest.raises(NumericalProbabilityError, match="did not converge"):
        approximation_module._integral(lambda value: value, 0.0, 1.0)

    monkeypatch.setattr(
        approximation_module,
        "quad",
        lambda *args, **kwargs: (1.0, 1e-3),
    )
    with pytest.raises(NumericalProbabilityError, match="did not converge"):
        approximation_module._integral(lambda value: value, 0.0, 1.0)


def test_product_quantile_requires_independent_final_cdf_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=1, positive=2), design)
    monkeypatch.setattr(
        approximation_module,
        "_product_cdf_condition_on_p",
        lambda *args, **kwargs: 0.0,
    )
    with pytest.raises(NumericalProbabilityError, match="independent CDF check"):
        approximation_module._product_quantile(
            result._h_distribution(),
            result._p_distribution(),
            design.r,
            0.5,
            positive=True,
        )


def test_limiting_fit_is_controlled_and_grid_bytes_are_immutable() -> None:
    with pytest.raises(TypeError, match="fit_limiting_approximation"):
        approximation_module.LimitingApproximationFit()

    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=1, positive=2), design)
    for retained in (result.h_nodes, result.p_nodes, result.mass):
        assert not retained.flags.writeable
        assert not retained.flags.owndata
        with pytest.raises(ValueError):
            retained.setflags(write=True)
        with pytest.raises(ValueError):
            retained[0, 0] = 0.0

    summary = result.summary()
    assert result.grid_purpose == "visualization_only"
    assert summary["retained_grid"] == {
        "nodes_per_axis": approximation_module._NODES,
        "purpose": "visualization_only",
        "used_for_summaries": False,
    }

    conflicting_design = LocalDesign.from_sample_size(design.n, c=1.25)
    with pytest.raises(ValidationError, match="full supplied design"):
        approximation_module.LimitingApproximationFit._from_components(
            counts=result.counts,
            design=conflicting_design,
            prior=LocalPrior.default(conflicting_design),
            h_nodes=result.h_nodes,
            p_nodes=result.p_nodes,
            mass=result.mass,
            h_truncation_mass=result.h_truncation_mass,
            p_truncation_mass=result.p_truncation_mass,
            h_log_truncation_mass=result.h_log_truncation_mass,
            p_log_truncation_mass=result.p_log_truncation_mass,
        )


def test_exact_fitter_has_no_limiting_approximation_dependency() -> None:
    source = inspect.getsource(exact_api)
    assert "approximation" not in source
    assert "fit_limiting_approximation" not in exact_api.__dict__
