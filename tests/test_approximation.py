"""Explicit limiting Gamma--Beta benchmark contracts."""

from __future__ import annotations

import inspect
from dataclasses import replace
from fractions import Fraction
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


def test_limiting_intensities_wrap_huge_real_conversion_overflow() -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=1, positive=2), design)
    with pytest.raises(ValidationError, match="h must be a finite real number"):
        result.intensities(Fraction(10**10_000, 1), 0.5)  # type: ignore[arg-type]


def test_limiting_posterior_is_compactly_truncated_not_unbounded() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(design=design, h_min=0.25, h_max=0.75, p_min=0.2, p_max=0.8)
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=0), design, prior
    )
    assert (result._h_distribution().lower, result._h_distribution().upper) == (
        prior.h_min,
        prior.h_max,
    )
    assert (result._p_distribution().lower, result._p_distribution().upper) == (
        prior.p_min,
        prior.p_max,
    )
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


def test_positive_beta_truncation_mass_can_be_subnormal() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=1e-5,
        p_max=1.2e-5,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n), design, prior
    )

    assert 0.0 < result.p_truncation_mass < np.finfo(np.float64).tiny
    assert result.p_truncation_mass == exp(result.p_log_truncation_mass)
    assert result.p_truncation_mass == pytest.approx(1.402e-320, rel=2e-4)
    assert result.p_log_truncation_mass == pytest.approx(-736.4892611636092, abs=2e-12)


def test_reflected_beta_truncation_mass_can_be_subnormal() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=1.0 - 1.2e-5,
        p_max=1.0 - 1e-5,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=design.n, positive=0), design, prior
    )

    assert 0.0 < result.p_truncation_mass < np.finfo(np.float64).tiny
    assert result.p_truncation_mass == exp(result.p_log_truncation_mass)


def test_reflected_subnormal_beta_public_summaries_use_stable_u_coordinate() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=1.0 - 1.2e-5,
        p_max=1.0 - 1e-5,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=design.n, positive=0), design, prior
    )
    distribution = result._p_distribution()
    assert isinstance(
        distribution, approximation_module._ReflectedContinuousDistribution
    )

    u_lower = 1.0 - prior.p_max
    u_upper = 1.0 - prior.p_min
    shape = int(result.p_shape_negative)
    expected_mean = 1.0 - _truncated_power_mean(
        shape,
        u_lower,
        u_upper,
    )
    expected_quantiles = tuple(
        1.0
        - _truncated_power_quantile(
            1.0 - probability,
            shape,
            u_lower,
            u_upper,
        )
        for probability in (0.05, 0.5, 0.95)
    )

    p_summary = result.parameter_summary("p")
    beta_summary = result.parameter_summary("beta")
    p_values = (
        p_summary.credible_interval.lower,
        p_summary.median,
        p_summary.credible_interval.upper,
    )
    assert p_summary.mean == pytest.approx(expected_mean, abs=1e-10)
    assert p_values == pytest.approx(expected_quantiles, abs=1e-10)
    assert beta_summary.mean == pytest.approx(2.0 * expected_mean - 1.0, abs=1e-10)
    assert (
        beta_summary.credible_interval.lower,
        beta_summary.median,
        beta_summary.credible_interval.upper,
    ) == pytest.approx(
        tuple(2.0 * value - 1.0 for value in expected_quantiles), abs=1e-10
    )

    full_summary = result.summary()
    assert full_summary["parameters"]["p"] == p_summary.to_dict()  # type: ignore[index]
    assert full_summary["parameters"]["beta"] == beta_summary.to_dict()  # type: ignore[index]

    positive_prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=u_lower,
        p_max=u_upper,
    )
    positive = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n),
        design,
        positive_prior,
    )._p_distribution()
    assert result.p_log_truncation_mass == positive.log_truncation_mass
    assert distribution.mean() + positive.mean() == pytest.approx(1.0, abs=1e-10)

    for probability, value in zip((0.05, 0.5, 0.95), p_values, strict=True):
        assert distribution.quantile(probability) == value
        u_quantile = distribution.base.quantile(1.0 - probability)
        assert distribution.base.cdf(u_quantile) == pytest.approx(
            1.0 - probability,
            abs=2e-11,
        )
        assert distribution.base.survival(u_quantile) == pytest.approx(
            probability,
            abs=2e-11,
        )
        assert value == pytest.approx(1.0 - positive.quantile(1.0 - probability))

        cdf = distribution.cdf(value)
        survival = distribution.survival(value)
        assert cdf + survival == pytest.approx(1.0, abs=2e-11)
        adjacent_cdfs = (
            distribution.cdf(float(np.nextafter(value, -np.inf))),
            distribution.cdf(float(np.nextafter(value, np.inf))),
        )
        cdf_ulp = max(abs(adjacent - cdf) for adjacent in adjacent_cdfs)
        assert abs(cdf - probability) <= 0.5 * cdf_ulp + 2e-11
        assert abs(survival - (1.0 - probability)) <= 0.5 * cdf_ulp + 2e-11


def test_reflected_beta_quantiles_choose_nearest_representable_p() -> None:
    design = LocalDesign.from_sample_size(64)
    epsilon = 2.0**-53
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=1.0 - 3.0 * epsilon,
        p_max=1.0 - 2.0 * epsilon,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=design.n, positive=0), design, prior
    )
    distribution = result._p_distribution()
    assert isinstance(
        distribution, approximation_module._ReflectedContinuousDistribution
    )

    p_summary = result.parameter_summary("p")
    beta_summary = result.parameter_summary("beta")
    p_values = (
        p_summary.credible_interval.lower,
        p_summary.median,
        p_summary.credible_interval.upper,
    )
    assert p_values == (prior.p_min, prior.p_min, prior.p_max)
    assert p_summary.credible_interval.lower < p_summary.credible_interval.upper
    assert (
        beta_summary.credible_interval.lower,
        beta_summary.median,
        beta_summary.credible_interval.upper,
    ) == tuple(2.0 * value - 1.0 for value in p_values)
    assert beta_summary.credible_interval.lower < beta_summary.credible_interval.upper

    full_summary = result.summary()
    assert full_summary["parameters"]["p"] == p_summary.to_dict()  # type: ignore[index]
    assert full_summary["parameters"]["beta"] == beta_summary.to_dict()  # type: ignore[index]

    for probability, value in zip((0.05, 0.5, 0.95), p_values, strict=True):
        mapped = 1.0 - distribution.base.quantile(1.0 - probability)
        candidates = tuple(
            candidate
            for candidate in (
                mapped,
                float(np.nextafter(mapped, -np.inf)),
                float(np.nextafter(mapped, np.inf)),
            )
            if prior.p_min <= candidate <= prior.p_max
        )
        residual = abs(distribution.cdf(value) - probability)
        assert residual == min(
            abs(distribution.cdf(candidate) - probability) for candidate in candidates
        )
        assert value in candidates
        assert distribution.cdf(value) + distribution.survival(value) == 1.0

    assert 1.0 - distribution.base.quantile(0.05) == prior.p_min
    assert p_summary.credible_interval.upper == prior.p_max


def test_gamma_truncation_mass_can_be_subnormal() -> None:
    design = LocalDesign.from_sample_size(64)
    prior = LocalPrior(
        design=design,
        h_min=0.00014,
        h_max=0.00018,
        p_min=0.05,
        p_max=0.95,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=design.n), design, prior
    )

    assert 0.0 < result.h_truncation_mass < np.finfo(np.float64).tiny
    assert result.h_truncation_mass == exp(result.h_log_truncation_mass)
    assert result.h_truncation_mass == pytest.approx(1.75056517e-315, rel=2e-8)
    assert result.h_log_truncation_mass == pytest.approx(-724.7543656018664, abs=2e-12)


def test_truncation_mass_validation_is_ulp_aware_at_normal_boundary() -> None:
    tiny = float(np.finfo(np.float64).tiny)
    boundary_log = log(tiny)
    for log_mass in (
        float(np.nextafter(boundary_log, -np.inf)),
        boundary_log,
        float(np.nextafter(boundary_log, np.inf)),
    ):
        expected_mass = exp(log_mass)
        approximation_module.LimitingApproximationFit._validate_truncation_mass(
            "boundary", expected_mass, log_mass
        )
        adjacent_mass = float(np.nextafter(expected_mass, np.inf))
        approximation_module.LimitingApproximationFit._validate_truncation_mass(
            "boundary", adjacent_mass, log_mass
        )

    underflow_log = log(float(np.nextafter(0.0, 1.0))) - 1.0
    approximation_module.LimitingApproximationFit._validate_truncation_mass(
        "underflow", 0.0, underflow_log
    )
    with pytest.raises(NumericalProbabilityError, match="disagrees"):
        approximation_module.LimitingApproximationFit._validate_truncation_mass(
            "underflow", float(np.nextafter(0.0, 1.0)), underflow_log
        )
    minimum_subnormal = float(np.nextafter(0.0, 1.0))
    with pytest.raises(NumericalProbabilityError, match="disagrees"):
        approximation_module.LimitingApproximationFit._validate_truncation_mass(
            "representable", 0.0, log(minimum_subnormal)
        )

    ordinary_log_mass = log(0.6)
    with pytest.raises(NumericalProbabilityError, match="disagrees"):
        approximation_module.LimitingApproximationFit._validate_truncation_mass(
            "ordinary", 0.600000000001, ordinary_log_mass
        )


def test_small_positive_truncation_log_mass_is_projected_with_evidence() -> None:
    design = LocalDesign.from_sample_size(200_000)
    prior = LocalPrior(
        design=design,
        h_min=4_000.0,
        h_max=6_000.0,
        p_min=0.9,
        p_max=0.9999,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=10_000), design, prior
    )

    assert result.h_truncation_mass == 1.0
    assert result.h_log_truncation_mass == 0.0
    assert result.h_truncation_mass_projected is True
    assert result.p_truncation_mass_projected is False
    assert result.summary()["truncation_mass_projected"] == {
        "h": True,
        "p": False,
    }


def test_truncation_log_mass_beyond_projection_tolerance_is_rejected() -> None:
    with pytest.raises(NumericalProbabilityError, match="truncation mass is invalid"):
        approximation_module._TruncatedContinuousDistribution(
            lower=0.0,
            upper=1.0,
            peak=0.5,
            log_kernel=lambda value: 4e-11,
        )


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

    assert not hasattr(result, "h_nodes")
    assert not hasattr(result, "p_nodes")
    assert not hasattr(result, "mass")
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


def test_reflected_tau_minus_product_summary_matches_tau_plus_mirror() -> None:
    design = LocalDesign.from_sample_size(128)
    right_prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=1.0 - 2e-6,
        p_max=1.0 - 1e-6,
    )
    mirror_prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=1.0 - right_prior.p_max,
        p_max=1.0 - right_prior.p_min,
    )
    right = fit_limiting_approximation(
        _counts(design, negative=96, positive=16),
        design,
        right_prior,
    )
    mirror = fit_limiting_approximation(
        _counts(design, negative=16, positive=96),
        design,
        mirror_prior,
    )

    right_summary = right.parameter_summary("tau_minus")
    mirror_summary = mirror.parameter_summary("tau_plus")
    assert right_summary == mirror_summary
    assert right_summary.credible_interval.upper == pytest.approx(
        1.0229889796630452e-6,
        abs=1e-15,
    )

    quantile = right_summary.credible_interval.upper
    h_distribution = right._h_distribution()
    p_distribution = right._p_distribution()
    conditioned_on_h = approximation_module._product_cdf_condition_on_h(
        quantile,
        h_distribution,
        p_distribution,
        design.r,
        positive=False,
    )
    conditioned_on_p = approximation_module._product_cdf_condition_on_p(
        quantile,
        h_distribution,
        p_distribution,
        design.r,
        positive=False,
    )
    assert conditioned_on_h == pytest.approx(0.95, abs=1e-9)
    assert conditioned_on_p == pytest.approx(0.95, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("n", [40_000, 75_000, 100_000])
def test_large_concentrated_product_summaries_pass_independent_cdf_checks(
    n: int,
) -> None:
    design = LocalDesign.from_sample_size(n)
    prior = LocalPrior(
        design=design,
        h_min=0.25,
        h_max=4.0,
        p_min=0.01,
        p_max=0.02,
    )
    result = fit_limiting_approximation(
        _counts(design, negative=0, positive=n), design, prior
    )
    h_distribution = result._h_distribution()
    p_distribution = result._p_distribution()

    for quantity, positive in (("tau_plus", True), ("tau_minus", False)):
        parameter = result.parameter_summary(quantity)
        assert (
            parameter.credible_interval.lower
            <= parameter.median
            <= parameter.credible_interval.upper
        )
        direct = approximation_module._product_cdf_condition_on_h(
            parameter.median,
            h_distribution,
            p_distribution,
            design.r,
            positive=positive,
        )
        independent = approximation_module._product_cdf_condition_on_p(
            parameter.median,
            h_distribution,
            p_distribution,
            design.r,
            positive=positive,
        )
        assert abs(direct - 0.5) <= 1e-9
        assert abs(independent - 0.5) <= 1e-9


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


def test_product_quantile_retries_independent_route_for_one_failed_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=1, positive=2), design)
    h_distribution = result._h_distribution()
    p_distribution = result._p_distribution()
    condition_on_h = approximation_module._product_cdf_condition_on_h
    failed_once = False

    def fail_once(*args: object, **kwargs: object) -> float:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise NumericalProbabilityError("forced primary-route failure")
        return condition_on_h(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        approximation_module,
        "_product_cdf_condition_on_h",
        fail_once,
    )
    quantile = approximation_module._product_quantile(
        h_distribution,
        p_distribution,
        design.r,
        0.5,
        positive=True,
    )
    assert failed_once
    assert condition_on_h(
        quantile,
        h_distribution,
        p_distribution,
        design.r,
        positive=True,
    ) == pytest.approx(0.5, abs=1e-9)
    assert approximation_module._product_cdf_condition_on_p(
        quantile,
        h_distribution,
        p_distribution,
        design.r,
        positive=True,
    ) == pytest.approx(0.5, abs=1e-9)


def test_limiting_fit_is_controlled_and_exposes_no_misleading_fixed_grid() -> None:
    with pytest.raises(TypeError, match="fit_limiting_approximation"):
        approximation_module.LimitingApproximationFit()

    design = LocalDesign.from_sample_size(64)
    result = fit_limiting_approximation(_counts(design, negative=1, positive=2), design)
    summary = result.summary()
    assert not hasattr(result, "h_nodes")
    assert not hasattr(result, "p_nodes")
    assert not hasattr(result, "mass")
    assert "retained_grid" not in summary
    assert (
        "_from_components" not in approximation_module.LimitingApproximationFit.__dict__
    )
    assert result.counts.design == result.design
    assert result.prior.design == result.design

    with pytest.raises(TypeError, match="fit_limiting_approximation"):
        approximation_module.LimitingApproximationFit(counts=result.counts)
    with pytest.raises(TypeError, match="fit_limiting_approximation"):
        replace(result)
    with pytest.raises(TypeError, match="fit_limiting_approximation"):
        replace(result, counts=result.counts)


def test_exact_fitter_has_no_limiting_approximation_dependency() -> None:
    source = inspect.getsource(exact_api)
    assert "approximation" not in source
    assert "fit_limiting_approximation" not in exact_api.__dict__
