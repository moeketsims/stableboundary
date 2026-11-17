"""Full-data reference inference checked against independent likelihoods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from math import log, pi

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp, roots_legendre
from scipy.stats import levy_stable, truncnorm

import stableboundary as sb
import stableboundary.posterior as posterior_module
import stableboundary.reference as reference
from stableboundary.backends import BackendMetadata, ScipyS0Backend


class _NormalBackend(ScipyS0Backend):
    """Normalized toy likelihood: X ~ N(loc + scale*beta, scale**2)."""

    @property
    def metadata(self):
        return BackendMetadata(method="normal-reference-test", tolerance=1e-14)

    def logpdf(self, x, alpha, beta, *, loc=0.0, scale=1.0):
        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
        return -0.5 * ((x - loc) / scale - beta) ** 2 - log(scale) - 0.5 * log(2 * pi)

    def logcdf(self, x, alpha, beta, *, loc=0.0, scale=1.0):
        alpha, beta = np.broadcast_arrays(alpha, beta)
        return np.full_like(alpha, log(0.01))

    def logsf(self, x, alpha, beta, *, loc=0.0, scale=1.0):
        return self.logcdf(x, alpha, beta, loc=loc, scale=scale)


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(reference, "ScipyS0Backend", _NormalBackend)
    design = sb.LocalDesign.from_sample_size(32)
    prior = sb.LocalPrior(design, h_min=0.5, h_max=1.5, p_min=0.3, p_max=0.7)
    values = np.tile([-0.4, 0.2, 0.3, 0.7], 8)
    return values, design, prior


def test_full_posterior_matches_truncated_normal_ground_truth(toy):
    values, design, prior = toy
    fit = sb.fit_full_known_nuisance(values, 0.0, 1.0, design, prior)
    mean = float(values.mean())
    sd = 1.0 / np.sqrt(design.n)
    lower, upper = 2 * prior.p_min - 1, 2 * prior.p_max - 1
    law = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
    assert fit.mean("beta") == pytest.approx(law.mean(), abs=1e-10)
    assert fit.median("beta") == pytest.approx(law.ppf(0.5), abs=4e-4)
    interval = fit.credible_interval("beta")
    assert interval.lower == pytest.approx(law.ppf(0.05), abs=4e-4)
    assert interval.upper == pytest.approx(law.ppf(0.95), abs=4e-4)
    assert fit.mean("h") == pytest.approx(1.0, abs=1e-12)
    assert fit.mean("tau_plus") == pytest.approx(design.r * fit.mean("p"), abs=1e-12)
    assert fit.posterior.mass.sum() == pytest.approx(1.0, abs=1e-12)
    assert fit.posterior.likelihood_method == "full_stable_likelihood"
    assert fit.method == "full_stable_likelihood"
    assert fit.status == "research_uncertified"
    assert set(fit.summary()["parameters"]) == {
        "h",
        "p",
        "alpha",
        "beta",
        "tau_plus",
        "tau_minus",
    }
    json.dumps(fit.summary(), allow_nan=False)
    json.dumps(fit.audit_record(), allow_nan=False)


def test_full_likelihood_keeps_magnitudes_that_grouping_discards(toy, monkeypatch):
    values, design, prior = toy
    monkeypatch.setattr(posterior_module, "ScipyS0Backend", _NormalBackend)
    shifted = values + 0.2
    first = sb.fit_full_known_nuisance(values, 0.0, 1.0, design, prior)
    second = sb.fit_full_known_nuisance(shifted, 0.0, 1.0, design, prior)
    assert first.counts == second.counts
    assert second.mean("beta") > first.mean("beta") + 0.09
    reduced_a = sb.fit_known_nuisance(values, 0.0, 1.0, design, prior)
    reduced_b = sb.fit_known_nuisance(shifted, 0.0, 1.0, design, prior)
    np.testing.assert_array_equal(reduced_a.posterior.mass, reduced_b.posterior.mass)


def test_full_affine_equivariance_and_raw_density_jacobian(toy):
    values, design, prior = toy
    fit = sb.fit_full_known_nuisance(
        values, 0, 1, design, prior, observation_chunk_size=1
    )
    shifted = sb.fit_full_known_nuisance(
        7 + 2.5 * values, 7, 2.5, design, prior, observation_chunk_size=3
    )
    np.testing.assert_allclose(fit.posterior.mass, shifted.posterior.mass, atol=1e-14)
    assert shifted.posterior.log_normalizer == pytest.approx(
        fit.posterior.log_normalizer - design.n * log(2.5), abs=1e-11
    )


def test_distances_match_independent_truncated_normal_integrals(toy, monkeypatch):
    values, design, prior = toy
    monkeypatch.setattr(posterior_module, "ScipyS0Backend", _NormalBackend)
    full = sb.fit_full_known_nuisance(values, 0, 1, design, prior)
    reduced = sb.fit_known_nuisance(values, 0, 1, design, prior)
    mean, sd = float(values.mean()), 1 / np.sqrt(design.n)
    lower, upper = 2 * prior.p_min - 1, 2 * prior.p_max - 1
    law = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
    uniform = 1 / (upper - lower)
    tv = 0.5 * quad(lambda b: abs(law.pdf(b) - uniform), lower, upper, epsabs=1e-10)[0]
    h2 = (
        0.5
        * quad(
            lambda b: (np.sqrt(law.pdf(b)) - np.sqrt(uniform)) ** 2,
            lower,
            upper,
            epsabs=1e-10,
        )[0]
    )
    actual = reference._posterior_distances(full.posterior, reduced.posterior, 257)
    np.testing.assert_allclose(actual, [tv, np.sqrt(h2)], atol=0.002, rtol=0)


def test_full_duplicate_multiplicities_match_independent_raw_sum(toy):
    values, design, prior = toy
    fit = sb.fit_full_known_nuisance(values, 0, 1, design, prior)
    beta = fit.posterior.values("beta")
    expected_log = np.sum(
        -0.5 * (values[:, None, None] - beta[None, :, :]) ** 2 - 0.5 * log(2 * pi),
        axis=0,
    ) - log(prior.area)
    _, weights = roots_legendre(fit.posterior.refined_nodes)
    measure = np.outer(weights, weights) * prior.area / 4
    log_mass = expected_log + np.log(measure)
    assert fit.posterior.log_normalizer == pytest.approx(logsumexp(log_mass), abs=1e-12)
    np.testing.assert_allclose(
        fit.posterior.mass, np.exp(log_mass - logsumexp(log_mass)), atol=1e-14
    )
    reversed_fit = sb.fit_full_known_nuisance(values[::-1], 0, 1, design, prior)
    np.testing.assert_array_equal(fit.posterior.mass, reversed_fit.posterior.mass)
    assert fit.posterior.data_sha256 != reversed_fit.posterior.data_sha256


def test_full_result_retains_immutable_evidence_and_data_identity(toy):
    values, design, prior = toy
    expected_hash = hashlib.sha256(
        np.asarray(values, dtype="<f8").tobytes()
    ).hexdigest()
    fit = sb.fit_full_known_nuisance(values, 0, 1, design, prior, provenance="external")
    audit = fit.audit_record()
    values[:] = 0
    assert fit.posterior.data_sha256 == expected_hash
    assert fit.audit_record() == audit
    assert fit.nuisance.provenance == "external"
    assert fit.prior is prior and fit.design is design
    for array in (fit.posterior.mass, fit.posterior.log_density):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        fit.observation_chunk_size = 1
    with pytest.raises(TypeError):
        replace(fit, observation_chunk_size=1)
    with pytest.raises(TypeError):
        sb.FullKnownNuisanceFit()
    with pytest.raises(sb.ValidationError, match="unknown"):
        fit.mean("loc")


@pytest.mark.parametrize(
    "values",
    [[], [1, np.nan], [[1, 2]], [True] * 64, ["1"] * 64, [1 + 2j] * 64, [0] * 63],
)
def test_full_refuses_invalid_observations(values, toy):
    _, design, prior = toy
    with pytest.raises(sb.ValidationError):
        sb.fit_full_known_nuisance(values, 0, 1, design, prior)


@pytest.mark.parametrize("chunk", [True, 0, 1.5, 4097])
def test_full_refuses_invalid_chunk_sizes(chunk, toy):
    values, design, prior = toy
    with pytest.raises(sb.ValidationError):
        sb.fit_full_known_nuisance(
            values, 0, 1, design, prior, observation_chunk_size=chunk
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "wrong-shape"])
def test_full_refuses_invalid_log_densities(bad, toy, monkeypatch):
    values, design, prior = toy

    def broken(self, x, alpha, beta, **kwargs):
        if bad == "wrong-shape":
            return np.array([0.0])
        return np.full(np.broadcast_shapes(x.shape, alpha.shape, beta.shape), bad)

    monkeypatch.setattr(_NormalBackend, "logpdf", broken)
    with pytest.raises(sb.NumericalProbabilityError, match="log densities"):
        sb.fit_full_known_nuisance(values, 0, 1, design, prior)


def test_full_refuses_unconverged_posterior(toy):
    values, design, prior = toy
    with pytest.raises(sb.ConvergenceError, match="refinement failed"):
        sb.fit_full_known_nuisance(
            values,
            0,
            1,
            design,
            prior,
            quadrature=sb.QuadratureConfig(refinement_tolerance=1e-12),
        )


def test_full_refuses_prior_reaching_alpha_near_one_before_scipy(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the alpha guard must run before any SciPy evaluation")

    monkeypatch.setattr(ScipyS0Backend, "_evaluate", fail)
    design = sb.LocalDesign.from_sample_size(40)
    with pytest.raises(sb.ValidationError, match="unreliable near alpha = 1"):
        sb.fit_full_known_nuisance(np.zeros(design.n), 0, 1, design)


def _extreme_sample(design, loc, scale, value):
    values = np.full(design.n, loc)
    values[0] = value
    return values


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_full_refuses_observation_beyond_validated_magnitude_before_scipy(
    monkeypatch, sign
):
    def fail(*args, **kwargs):
        raise AssertionError("the magnitude guard must run before SciPy")

    monkeypatch.setattr(ScipyS0Backend, "_evaluate", fail)
    design = sb.LocalDesign.from_sample_size(64)
    prior = sb.LocalPrior(design, h_min=0.25, h_max=1.0, p_min=0.25, p_max=0.75)
    nuisance = sb.KnownNuisance.externally_known(loc=3.0, scale=2.5, provenance="t")
    limit = ScipyS0Backend.MAX_VALIDATED_ABS_Z
    at_limit = nuisance.loc + sign * nuisance.scale * limit
    inside = float(np.nextafter(at_limit, nuisance.loc))
    outside = float(np.nextafter(at_limit, sign * np.inf))
    assert abs((inside - nuisance.loc) / nuisance.scale) < limit
    assert abs((at_limit - nuisance.loc) / nuisance.scale) == limit
    assert abs((outside - nuisance.loc) / nuisance.scale) > limit

    for accepted in (inside, at_limit):
        reference._require_validated_canonical_magnitude(
            _extreme_sample(design, nuisance.loc, nuisance.scale, accepted), nuisance
        )
    with pytest.raises(sb.ValidationError, match="check loc and scale"):
        sb.fit_full_known_nuisance(
            _extreme_sample(design, nuisance.loc, nuisance.scale, outside),
            nuisance.loc,
            nuisance.scale,
            design,
            prior,
        )
    # Standardization that overflows is refused as well, not passed through.
    with pytest.raises(sb.ValidationError, match="nonfinite"):
        sb.fit_full_known_nuisance(
            _extreme_sample(design, -sign * 1e308, 1.0, sign * 1e308),
            -sign * 1e308,
            1.0,
            design,
            prior,
        )


def test_magnitude_guard_applies_only_to_the_canonical_backend(toy):
    values, design, prior = toy
    extreme = values.copy()
    extreme[0] = 10.0 * ScipyS0Backend.MAX_VALIDATED_ABS_Z
    # With the toy backend the fit may fail to converge, but it must not be
    # refused for magnitude: that limit belongs to the SciPy density.
    try:
        sb.fit_full_known_nuisance(extreme, 0.0, 1.0, design, prior)
    except sb.StableBoundaryError as error:
        assert "standardized observation" not in str(error)


@pytest.fixture(scope="module")
def real_fits():
    design = sb.LocalDesign.from_sample_size(256)
    prior = sb.LocalPrior(design, h_min=0.75, h_max=1.25, p_min=0.25, p_max=0.75)
    values = np.tile([-2.0, -0.5, 0.75, 2.5], 64)
    full = sb.fit_full_known_nuisance(values, 0, 1, design, prior)
    reduced = sb.fit_known_nuisance(values, 0, 1, design, prior)
    return values, full, reduced


def test_real_s0_full_likelihood_matches_direct_scipy_product(real_fits):
    values, full, _ = real_fits
    distribution = type(levy_stable)(name="independent_reference_test")
    distribution.parameterization = "S0"
    distribution.pdf_default_method = "piecewise"
    distribution.quad_eps = 1.2e-14
    # An independent scalar calculation at retained quadrature nodes checks
    # the raw product, prior normalization, and posterior density together.
    for i, j in [(0, 0), (11, 19), (31, 31)]:
        h, p = full.posterior.h_nodes[i, j], full.posterior.p_nodes[i, j]
        log_joint = float(
            np.sum(distribution.logpdf(values, 2 - full.design.r * h, 2 * p - 1))
        ) - log(full.prior.area)
        assert full.posterior.log_density[i + 1, j + 1] == pytest.approx(
            log_joint, abs=2e-10
        )


def test_reference_density_agrees_with_characteristic_function_inversion(real_fits):
    _, full, _ = real_fits
    backend = ScipyS0Backend()
    for x, i, j in [(-2.0, 0, 0), (0.75, 11, 19), (2.5, 31, 31)]:
        alpha = full.posterior.values("alpha")[i, j]
        beta = full.posterior.values("beta")[i, j]

        # Direct inverse Fourier density of S0, independent of Nolan piecewise.
        def integrand(t, alpha=alpha, beta=beta, x=x):
            phase = beta * np.tan(pi * alpha / 2) * (t**alpha - t)
            return np.exp(-(t**alpha)) * np.cos(phase - t * x) / pi

        density, error = quad(integrand, 0, 20, epsabs=1e-12, epsrel=1e-12)
        assert error < 1e-11
        assert np.exp(backend.logpdf(x, alpha, beta)) == pytest.approx(
            density, abs=1e-10
        )


def test_full_reflection_preserves_alpha_and_reverses_beta(real_fits):
    values, full, _ = real_fits
    reflected = sb.fit_full_known_nuisance(-values, 0, 1, full.design, full.prior)
    np.testing.assert_allclose(
        full.posterior.mass[:, ::-1], reflected.posterior.mass, atol=2e-12
    )
    assert full.mean("alpha") == pytest.approx(reflected.mean("alpha"), abs=1e-12)
    assert full.mean("beta") == pytest.approx(-reflected.mean("beta"), abs=1e-12)


def test_comparison_reports_large_disagreement_without_calling_it_safe(real_fits):
    _, full, reduced = real_fits
    comparison = sb.compare_posteriors(full, reduced)
    assert 0 < comparison.hellinger <= 1
    assert comparison.hellinger**2 <= comparison.total_variation
    assert comparison.total_variation <= np.sqrt(2) * comparison.hellinger
    assert comparison.total_variation > comparison.tolerance
    assert comparison.refinement_change <= comparison.tolerance
    assert comparison.refined_grid_points == 129
    assert comparison.summary()["status"] == "research_uncertified"
    json.dumps(comparison.summary(), allow_nan=False)
    with pytest.raises(TypeError):
        replace(comparison, total_variation=0)
    with pytest.raises(TypeError):
        sb.PosteriorComparison()


def test_distances_vanish_for_identical_posteriors_and_are_symmetric(real_fits):
    _, full, reduced = real_fits
    assert reference._posterior_distances(full.posterior, full.posterior, 129) == (0, 0)
    forward = reference._posterior_distances(full.posterior, reduced.posterior, 129)
    reverse = reference._posterior_distances(reduced.posterior, full.posterior, 129)
    np.testing.assert_allclose(forward, reverse, atol=1e-15)


def test_comparison_refuses_mismatched_experiments(real_fits):
    values, full, _ = real_fits
    other = sb.fit_known_nuisance(
        values, 0, 1, full.design, full.prior, provenance="different calibration"
    )
    with pytest.raises(sb.ValidationError, match="matching"):
        sb.compare_posteriors(full, other)


def test_comparison_refuses_custom_backend(toy, monkeypatch):
    values, design, prior = toy
    monkeypatch.setattr(posterior_module, "ScipyS0Backend", _NormalBackend)
    full = sb.fit_full_known_nuisance(values, 0, 1, design, prior)
    reduced = sb.fit_known_nuisance(values, 0, 1, design, prior)
    with pytest.raises(sb.ValidationError, match="canonical"):
        sb.compare_posteriors(full, reduced)


def test_comparison_refuses_unresolved_grid_distance(real_fits):
    _, full, reduced = real_fits
    with pytest.raises(sb.ConvergenceError, match="comparison grid refinement"):
        sb.compare_posteriors(full, reduced, grid_points=3, tolerance=1e-12)


@pytest.mark.parametrize(
    "controls",
    [
        {"grid_points": True},
        {"grid_points": 258},
        {"tolerance": 0},
        {"tolerance": np.nan},
    ],
)
def test_comparison_refuses_invalid_controls(controls, real_fits):
    _, full, reduced = real_fits
    with pytest.raises(sb.ValidationError):
        sb.compare_posteriors(full, reduced, **controls)


def test_retained_density_cannot_be_rebound_to_different_mass(real_fits):
    _, full, _ = real_fits
    altered = object.__new__(sb.PosteriorGrid)
    for field in full.posterior.__dataclass_fields__:
        object.__setattr__(altered, field, getattr(full.posterior, field))
    object.__setattr__(altered, "log_density", full.posterior.log_density + 1)
    with pytest.raises(sb.ConvergenceError, match="disagrees"):
        altered.__post_init__()
