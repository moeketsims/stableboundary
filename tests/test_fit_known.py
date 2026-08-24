"""Exact finite-posterior integration contracts."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

import stableboundary.posterior as posterior_module
from stableboundary import fit_known_nuisance
from stableboundary._exceptions import ConvergenceError
from stableboundary.backends import BackendMetadata, ScipyS0Backend
from stableboundary.cells import CellCounts
from stableboundary.design import KnownNuisance, LocalDesign, LocalPrior
from stableboundary.posterior import (
    QuadratureConfig,
    _joint_total_variation,
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


def _counts(design: LocalDesign) -> CellCounts:
    observations = np.zeros(design.n)
    observations[:2] = design.threshold + 1.0
    observations[2:4] = -design.threshold - 1.0
    return CellCounts.from_observations(
        observations,
        nuisance=KnownNuisance.externally_known(loc=0.0, scale=1.0, provenance="test"),
        design=design,
    )


def _no_data_counts(design: LocalDesign) -> CellCounts:
    counts = object.__new__(CellCounts)
    object.__setattr__(counts, "n_minus", 0)
    object.__setattr__(counts, "n_zero", 0)
    object.__setattr__(counts, "n_plus", 0)
    object.__setattr__(counts, "threshold", design.threshold)
    object.__setattr__(counts, "n", 0)
    return counts


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


def test_exact_fixed_node_grid_cannot_claim_refinement() -> None:
    design = LocalDesign.from_sample_size(32)
    with pytest.raises(ConvergenceError, match="must exceed"):
        compute_exact_posterior(
            _counts(design),
            design,
            LocalPrior.default(design),
            QuadratureConfig(base_nodes=8, refined_nodes=8),
        )


def test_exact_uniform_no_data_posterior_has_analytic_normalization() -> None:
    design = LocalDesign.from_sample_size(128)
    prior = LocalPrior.default(design)
    posterior = compute_exact_posterior(
        _no_data_counts(design),
        design,
        prior,
        backend=_AnalyticBackend(),
    )
    assert posterior.log_normalizer == pytest.approx(0.0, abs=2e-14)
    assert np.sum(posterior.mass * posterior.h_nodes) == pytest.approx(
        0.5 * (prior.h_min + prior.h_max), abs=2e-14
    )
    assert np.sum(posterior.mass * posterior.p_nodes) == pytest.approx(
        0.5 * (prior.p_min + prior.p_max), abs=2e-14
    )


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
    for value in parameters.values():
        assert isinstance(value, dict)
        assert np.isfinite(value["mean"])
        assert np.isfinite(value["median"])
    encoded = json.dumps(fit.audit_record(), allow_nan=False)
    assert "research_uncertified" in encoded
    assert "independent calibration" in encoded


def test_fit_known_rejects_observation_count_mismatch() -> None:
    design = LocalDesign.from_sample_size(16)
    with pytest.raises(Exception, match="observation count"):
        fit_known_nuisance(
            np.zeros(15),
            loc=0.0,
            scale=1.0,
            design=design,
        )
