"""Exact finite-posterior integration contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.optimize import brentq  # type: ignore[import-untyped]

import stableboundary.posterior as posterior_module
from stableboundary import KnownNuisanceFit, PosteriorGrid, fit_known_nuisance
from stableboundary._exceptions import ConvergenceError, ValidationError
from stableboundary.backends import BackendMetadata, ScipyS0Backend
from stableboundary.cells import CellCounts
from stableboundary.design import KnownNuisance, LocalDesign, LocalPrior
from stableboundary.posterior import (
    QuadratureConfig,
    _axis_quantile,
    _CommonGrid,
    _joint_total_variation,
    _tau_quantile,
    compute_exact_posterior,
)


class _AnalyticBackend(ScipyS0Backend):
    """Fast broadcast backend for posterior algebra and refinement tests."""

    _test_metadata = BackendMetadata(method="analytic-test-cells", tolerance=1e-14)

    @property
    def metadata(self) -> BackendMetadata:
        return self._test_metadata

    @staticmethod
    def _tail(alpha: ArrayLike, beta: ArrayLike, *, positive: bool) -> object:
        alpha_values, beta_values = np.broadcast_arrays(
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
        )
        allocation = 0.5 * (1.0 + beta_values if positive else 1.0 - beta_values)
        values = 0.002 + 0.05 * (2.0 - alpha_values) * allocation
        result = np.log(values)
        return float(result) if result.ndim == 0 else result

    def logcdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> object:
        del x, loc, scale
        return self._tail(alpha, beta, positive=False)

    def logsf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> object:
        del x, loc, scale
        return self._tail(alpha, beta, positive=True)


class _S1AnalyticBackend(_AnalyticBackend):
    _test_metadata = BackendMetadata(
        method="analytic-test-s1",
        tolerance=1e-14,
        parameterization="S1",
    )


class _FailIfCalledBackend(_AnalyticBackend):
    def logcdf(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("backend must not run for mismatched provenance")

    def logsf(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("backend must not run for mismatched provenance")


class _MetadataMutatingBackend(_AnalyticBackend):
    def __init__(self, final_metadata: BackendMetadata) -> None:
        self._current_metadata = self._test_metadata
        self._final_metadata = final_metadata

    @property
    def metadata(self) -> BackendMetadata:
        return self._current_metadata

    def logsf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> object:
        result = super().logsf(x, alpha, beta, loc=loc, scale=scale)
        self._current_metadata = self._final_metadata
        return result


def _counts(design: LocalDesign) -> CellCounts:
    observations = np.zeros(design.n)
    observations[:2] = design.threshold + 1.0
    observations[2:4] = -design.threshold - 1.0
    return CellCounts.from_observations(
        observations,
        nuisance=KnownNuisance.externally_known(loc=0.0, scale=1.0, provenance="test"),
        design=design,
    )


def test_exact_grid_is_normalized_read_only_and_reproducible() -> None:
    design = LocalDesign.from_sample_size(128)
    prior = LocalPrior.default(design)
    backend = _AnalyticBackend()
    first = compute_exact_posterior(_counts(design), design, prior, backend=backend)
    second = compute_exact_posterior(_counts(design), design, prior, backend=backend)
    assert np.sum(first.mass) == pytest.approx(1.0, abs=1e-12)
    assert np.all(first.mass >= 0.0)
    assert not first.mass.flags.writeable
    assert not first.h_nodes.flags.writeable
    assert np.array_equal(first.mass, second.mass)
    assert first.refinement.converged
    assert first.refinement.common_grid_points == 65
    assert len(first.refinement.summaries) == 6
    assert first.design is design
    assert first.prior is prior
    assert first.counts.design is design
    assert first.counts.nuisance.provenance == "test"
    for retained in (
        first.h_nodes,
        first.p_nodes,
        first.mass,
        first.q_minus,
        first.q_plus,
    ):
        with pytest.raises(ValueError, match="WRITEABLE flag"):
            retained.setflags(write=True)
    derived = first.values("alpha")
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        derived.setflags(write=True)


def test_exact_posterior_rejects_cross_design_counts_before_backend_calls() -> None:
    count_design = LocalDesign.from_sample_size(32, c=1.0)
    requested_design = LocalDesign.from_sample_size(32, c=1.25)
    assert count_design.threshold != requested_design.threshold
    with pytest.raises(ValidationError, match="full design"):
        compute_exact_posterior(
            _counts(count_design),
            requested_design,
            LocalPrior.default(requested_design),
            backend=_FailIfCalledBackend(),
        )


def test_exact_fixed_node_grid_cannot_claim_refinement() -> None:
    design = LocalDesign.from_sample_size(32)
    with pytest.raises(ConvergenceError, match="must exceed"):
        compute_exact_posterior(
            _counts(design),
            design,
            LocalPrior.default(design),
            QuadratureConfig(base_nodes=8, refined_nodes=8),
        )


def test_exact_posterior_rejects_structurally_valid_s1_backend() -> None:
    design = LocalDesign.from_sample_size(32)
    with pytest.raises(ValidationError, match="parameterization.*S0"):
        compute_exact_posterior(
            _counts(design),
            design,
            LocalPrior.default(design),
            backend=_S1AnalyticBackend(),
        )


@pytest.mark.parametrize(
    ("final_metadata", "message"),
    [
        (
            BackendMetadata(method="mutated-method", tolerance=1e-14),
            "metadata changed during posterior inference",
        ),
        (
            BackendMetadata(method="analytic-test-cells", tolerance=2e-14),
            "metadata changed during posterior inference",
        ),
        (
            BackendMetadata(
                method="analytic-test-cells",
                tolerance=1e-14,
                parameterization="S1",
            ),
            "parameterization.*S0",
        ),
    ],
)
def test_exact_posterior_revalidates_backend_metadata_after_all_evaluation(
    final_metadata: BackendMetadata,
    message: str,
) -> None:
    design = LocalDesign.from_sample_size(32)
    with pytest.raises(ValidationError, match=message):
        compute_exact_posterior(
            _counts(design),
            design,
            LocalPrior.default(design),
            backend=_MetadataMutatingBackend(final_metadata),
        )


def test_continuous_uniform_axis_quantiles_include_support_endpoints() -> None:
    design = LocalDesign.from_sample_size(128)
    prior = LocalPrior.default(design)
    tail = 0.05
    h_expected = (
        prior.h_min + tail * (prior.h_max - prior.h_min),
        0.5 * (prior.h_min + prior.h_max),
        prior.h_max - tail * (prior.h_max - prior.h_min),
    )
    p_expected = (
        prior.p_min + tail * (prior.p_max - prior.p_min),
        0.5 * (prior.p_min + prior.p_max),
        prior.p_max - tail * (prior.p_max - prior.p_min),
    )
    h_axis = np.linspace(prior.h_min, prior.h_max, 65)
    p_axis = np.linspace(prior.p_min, prior.p_max, 65)
    h_actual = tuple(
        _axis_quantile(h_axis, np.ones(65), value) for value in (tail, 0.5, 1.0 - tail)
    )
    p_actual = tuple(
        _axis_quantile(p_axis, np.ones(65), value) for value in (tail, 0.5, 1.0 - tail)
    )
    assert h_actual == pytest.approx(h_expected, abs=1e-12)
    assert p_actual == pytest.approx(p_expected, abs=1e-12)


def _uniform_product_cdf(
    value: float,
    h_lower: float,
    h_upper: float,
    p_lower: float,
    p_upper: float,
) -> float:
    full_upper = min(h_upper, value / p_upper)
    full_area = max(0.0, full_upper - h_lower) * (p_upper - p_lower)
    partial_lower = max(h_lower, value / p_upper)
    partial_upper = min(h_upper, value / p_lower)
    partial_area = 0.0
    if partial_upper > partial_lower:
        partial_area = value * np.log(partial_upper / partial_lower) - p_lower * (
            partial_upper - partial_lower
        )
    area = (h_upper - h_lower) * (p_upper - p_lower)
    return min(1.0, max(0.0, (full_area + partial_area) / area))


def test_exact_uniform_product_quantiles_use_continuous_pushforward() -> None:
    design = LocalDesign.from_sample_size(128)
    prior = LocalPrior(design=design, p_min=0.10, p_max=0.70)
    h_axis = np.linspace(prior.h_min, prior.h_max, 65)
    p_axis = np.linspace(prior.p_min, prior.p_max, 65)
    density = np.full(
        (h_axis.size, p_axis.size),
        1.0 / prior.area,
    )
    common = _CommonGrid(
        h_axis=h_axis,
        p_axis=p_axis,
        density=density,
        measure=np.ones_like(density),
    )
    p_widths = np.diff(p_axis)
    prefix = np.zeros_like(density)
    prefix[:, 1:] = np.cumsum(
        0.5 * (density[:, :-1] + density[:, 1:]) * p_widths,
        axis=1,
    )
    row_total = prefix[:, -1]

    def analytic_quantile(probability: float, p_lower: float, p_upper: float) -> float:
        product = brentq(
            lambda value: (
                _uniform_product_cdf(
                    value,
                    prior.h_min,
                    prior.h_max,
                    p_lower,
                    p_upper,
                )
                - probability
            ),
            prior.h_min * p_lower,
            prior.h_max * p_upper,
        )
        return design.r * product

    tail = 0.05
    probabilities = (tail, 0.5, 1.0 - tail)
    expected_positive = tuple(
        analytic_quantile(value, prior.p_min, prior.p_max) for value in probabilities
    )
    expected_negative = tuple(
        analytic_quantile(value, 1.0 - prior.p_max, 1.0 - prior.p_min)
        for value in probabilities
    )
    positive = tuple(
        _tau_quantile(
            common,
            design,
            prefix,
            row_total,
            value,
            positive=True,
        )
        for value in probabilities
    )
    negative = tuple(
        _tau_quantile(
            common,
            design,
            prefix,
            row_total,
            value,
            positive=False,
        )
        for value in probabilities
    )
    for actual, expected, p_lower, p_upper in (
        (positive, expected_positive, prior.p_min, prior.p_max),
        (negative, expected_negative, 1.0 - prior.p_max, 1.0 - prior.p_min),
    ):
        scale = design.r * (prior.h_max * p_upper - prior.h_min * p_lower)
        assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) / scale <= 1e-9
    assert positive[1] != pytest.approx(negative[1])


def test_exact_too_tight_refinement_is_a_structured_failure() -> None:
    design = LocalDesign.from_sample_size(128)
    with pytest.raises(ConvergenceError, match="refinement failed"):
        compute_exact_posterior(
            _counts(design),
            design,
            LocalPrior.default(design),
            QuadratureConfig(
                base_nodes=4,
                refined_nodes=6,
                refinement_tolerance=1e-12,
            ),
            backend=_AnalyticBackend(),
        )


def test_joint_tv_detects_dependence_with_matching_marginals() -> None:
    first = np.array([[1.5, 0.5], [0.5, 1.5]])
    second = np.array([[0.5, 1.5], [1.5, 0.5]])
    measure = np.full((2, 2), 0.25)
    assert np.sum(first * measure, axis=0) == pytest.approx(
        np.sum(second * measure, axis=0)
    )
    assert np.sum(first * measure, axis=1) == pytest.approx(
        np.sum(second * measure, axis=1)
    )
    assert _joint_total_variation(first, second, measure) == pytest.approx(0.5)


def test_exact_quadrature_defaults_are_the_declared_accuracy_policy() -> None:
    config = QuadratureConfig()
    assert (config.base_nodes, config.refined_nodes) == (20, 32)
    assert config.refinement_tolerance == 0.002
    assert config.interval_mass == 0.90
    assert config.common_grid_points == 65


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("refinement_tolerance", True),
        ("refinement_tolerance", "0.1"),
        ("refinement_tolerance", 1 + 0j),
        ("refinement_tolerance", Fraction(10**10_000, 1)),
        ("interval_mass", False),
        ("interval_mass", "0.9"),
        ("interval_mass", 0.9 + 0j),
    ],
)
def test_quadrature_real_controls_reject_bool_and_non_real_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match=field):
        QuadratureConfig(**{field: value})  # type: ignore[arg-type]


def test_guarded_scipy_default_posterior_integration() -> None:
    design = LocalDesign.from_sample_size(5_000)
    posterior = compute_exact_posterior(
        _counts(design),
        design,
        LocalPrior.default(design),
    )
    assert posterior.refinement.converged
    assert posterior.refinement.joint_total_variation <= 0.002
    assert posterior.backend_method == "scipy-piecewise-s0-direct-log-tails"
    assert posterior.backend_parameterization == "S0"
    assert posterior.backend_metadata.library == "scipy"
    assert dict(posterior.backend_metadata.effective_settings)["quad_eps"] == 1.2e-14


def test_fit_known_returns_finite_six_quantity_summary_and_json_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setattr(posterior_module, "ScipyS0Backend", _AnalyticBackend)
    design = LocalDesign.from_sample_size(128)
    observations = np.zeros(design.n)
    observations[0] = design.threshold + 1.0
    observations[1] = -design.threshold - 1.0
    fit = fit_known_nuisance(
        observations,
        loc=0.0,
        scale=1.0,
        design=design,
        provenance="independent calibration",
    )
    assert fit.status == "research_uncertified"
    assert fit.r == design.r
    parameters = fit.summary()["parameters"]
    assert isinstance(parameters, dict)
    assert set(parameters) == {"h", "p", "alpha", "beta", "tau_plus", "tau_minus"}
    for name, value in parameters.items():
        assert isinstance(value, dict)
        assert np.isfinite(value["mean"])
        assert np.isfinite(value["median"])
        retained = fit.posterior.summary_record(name)
        assert value["mean"] == retained.mean
        assert value["median"] == retained.median
        assert value["credible_interval"]["lower"] == retained.interval_lower
        assert value["credible_interval"]["upper"] == retained.interval_upper
    encoded = json.dumps(fit.audit_record(), allow_nan=False)
    assert "research_uncertified" in encoded
    assert "independent calibration" in encoded
    environment = fit.audit_record()["environment"]
    assert isinstance(environment, dict)
    assert set(environment) == {"python", "numpy", "scipy", "stableboundary"}
    assert all(isinstance(value, str) and value for value in environment.values())
    assert fit.audit_record()["backend"]["origin"] == "custom"  # type: ignore[index]

    captured_audit = fit.audit_record()
    monkeypatch.setattr(posterior_module, "python_version", lambda: "post-fit-python")
    monkeypatch.setattr(posterior_module.np, "__version__", "post-fit-numpy")
    monkeypatch.setattr(posterior_module.scipy, "__version__", "post-fit-scipy")
    monkeypatch.setattr(
        posterior_module,
        "_package_version",
        lambda: "post-fit-stableboundary",
    )
    assert fit.audit_record() == captured_audit


def test_posterior_construction_rebinding_and_internal_forgery_are_rejected() -> None:
    with pytest.raises(TypeError, match="compute_exact_posterior"):
        PosteriorGrid()

    design = LocalDesign.from_sample_size(32)
    prior = LocalPrior.default(design)
    counts = _counts(design)
    posterior = compute_exact_posterior(
        counts,
        design,
        prior,
        backend=_AnalyticBackend(),
    )
    values = np.zeros(design.n)
    values[0] = design.threshold + 1.0
    alternate_counts = CellCounts.from_observations(
        values,
        nuisance=counts.nuisance,
        design=design,
    )
    alternate_prior = LocalPrior(design=design, h_max=3.5)
    alternate_backend = BackendMetadata(method="forged", tolerance=1e-9)
    for field_name, alternate in (
        ("counts", alternate_counts),
        ("prior", alternate_prior),
        ("backend_metadata", alternate_backend),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(posterior, field_name, alternate)
        with pytest.raises(TypeError, match="compute_exact_posterior"):
            replace(posterior, **{field_name: alternate})

    components = {
        item.name: getattr(posterior, item.name) for item in fields(posterior)
    }
    forged_nodes = np.array(posterior.h_nodes, copy=True)
    forged_nodes[0, 0] = prior.h_max
    with pytest.raises(ConvergenceError, match="nodes do not match"):
        PosteriorGrid._from_components(
            **(components | {"h_nodes": forged_nodes})  # type: ignore[arg-type]
        )
    forged_mass = np.zeros_like(posterior.mass)
    forged_mass[0, 0] = 1.0
    with pytest.raises(ConvergenceError, match="mean does not match"):
        PosteriorGrid._from_components(
            **(components | {"mass": forged_mass})  # type: ignore[arg-type]
        )


def test_fit_construction_is_package_controlled_and_rejects_false_composition() -> None:
    with pytest.raises(TypeError, match="fit_known_nuisance"):
        KnownNuisanceFit()

    design = LocalDesign.from_sample_size(32)
    prior = LocalPrior.default(design)
    counts = _counts(design)
    posterior = compute_exact_posterior(
        counts,
        design,
        prior,
        backend=_AnalyticBackend(),
    )
    conflicting_nuisance = KnownNuisance.externally_known(
        loc=0.0,
        scale=1.0,
        provenance="different calibration",
    )
    with pytest.raises(ValidationError, match="nuisance provenance"):
        KnownNuisanceFit._from_components(
            nuisance=conflicting_nuisance,
            design=design,
            prior=prior,
            counts=counts,
            posterior=posterior,
        )


def test_fit_known_rejects_observation_count_mismatch() -> None:
    design = LocalDesign.from_sample_size(16)
    with pytest.raises(Exception, match="observation count"):
        fit_known_nuisance(
            np.zeros(15),
            loc=0.0,
            scale=1.0,
            design=design,
        )
