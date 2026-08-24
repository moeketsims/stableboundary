"""Full-model posterior predictive contracts and infinite-variance refusal."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

import stableboundary.posterior as posterior_module
import stableboundary.result as result_module
from stableboundary import InfiniteVarianceError, LocalDesign, fit_known_nuisance
from stableboundary.backends import BackendMetadata, ScipyS0Backend


class _PredictiveBackend(ScipyS0Backend):
    _test_metadata = BackendMetadata(method="analytic-predictive-test", tolerance=1e-14)

    @property
    def metadata(self) -> BackendMetadata:
        return self._test_metadata

    @staticmethod
    def _tail(
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        positive: bool,
    ) -> object:
        alpha_values, beta_values = np.broadcast_arrays(
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
        )
        allocation = 0.5 * (1.0 + beta_values if positive else 1.0 - beta_values)
        decay = np.exp(-0.2 * abs(float(np.asarray(x))))
        result = np.log(decay * (0.002 + 0.05 * (2.0 - alpha_values) * allocation))
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
        del loc, scale
        return self._tail(x, alpha, beta, positive=False)

    def logsf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> object:
        del loc, scale
        return self._tail(x, alpha, beta, positive=True)

    def rvs(
        self,
        alpha: float,
        beta: float,
        *,
        loc: float,
        scale: float,
        size: int,
        random_state: Generator,
    ) -> NDArray[np.float64]:
        del alpha
        return np.asarray(
            random_state.normal(loc=loc + 0.1 * beta, scale=scale, size=size),
            dtype=np.float64,
        )


@pytest.fixture
def fit(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(posterior_module, "ScipyS0Backend", _PredictiveBackend)
    monkeypatch.setattr(result_module, "ScipyS0Backend", _PredictiveBackend)
    design = LocalDesign.from_sample_size(64)
    values = np.zeros(design.n)
    values[0] = design.threshold + 1.0
    values[1] = -design.threshold - 1.0
    return fit_known_nuisance(values, 0.0, 1.0, design)


def test_expected_counts_are_future_size_times_signed_tail_probabilities(
    fit: object,
) -> None:
    prediction = fit.tail_probabilities(4.0)
    expected = fit.expected_exceedance_counts(250, 4.0)
    assert expected.negative == pytest.approx(250 * prediction.negative)
    assert expected.positive == pytest.approx(250 * prediction.positive)
    assert prediction.backend_method == "analytic-predictive-test"


def test_seeded_prediction_and_quantile_mc_metadata_are_reproducible(
    fit: object,
) -> None:
    first = fit.posterior_predictive(80, seed=20260824)
    second = fit.posterior_predictive(80, seed=20260824)
    assert np.array_equal(first.values, second.values)
    assert not first.values.flags.writeable
    estimate = fit.predictive_quantile(0.9, draws=80, seed=17)
    repeated = fit.predictive_quantile(0.9, draws=80, seed=17)
    assert estimate == repeated
    assert estimate.probability == 0.9
    assert estimate.draw_count == 80
    assert estimate.seed == 17
    assert estimate.bit_generator == "PCG64"
    assert estimate.batches == 8
    assert estimate.monte_carlo_standard_error >= 0.0


def test_predictive_variance_refuses_sub_gaussian_boundary_support(fit: object) -> None:
    with pytest.raises(InfiniteVarianceError, match="alpha < 2"):
        fit.predictive_variance()
