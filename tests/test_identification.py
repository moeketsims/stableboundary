"""Evidence labels and quantitative identification diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

import stableboundary.posterior as posterior_module
from stableboundary import LocalDesign, fit_known_nuisance
from stableboundary.backends import BackendMetadata, ScipyS0Backend


class _AnalyticBackend(ScipyS0Backend):
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
        result = np.log(0.002 + 0.05 * (2.0 - alpha_values) * allocation)
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


def _fit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    negative: int,
    positive: int,
) -> object:
    monkeypatch.setattr(posterior_module, "ScipyS0Backend", _AnalyticBackend)
    design = LocalDesign.from_sample_size(64)
    values = np.zeros(design.n)
    values[:positive] = design.threshold + 1.0
    values[positive : positive + negative] = -design.threshold - 1.0
    return fit_known_nuisance(values, 0.0, 1.0, design)


@pytest.mark.parametrize(
    ("negative", "positive", "evidence", "precision"),
    [
        (0, 0, "prior_dominated", "unidentified"),
        (0, 2, "one_sided_evidence", "not_assessed"),
        (2, 0, "one_sided_evidence", "not_assessed"),
        (1, 1, "two_sided_evidence", "not_assessed"),
    ],
)
def test_identification_statuses_are_evidence_based_without_count_cutoffs(
    monkeypatch: pytest.MonkeyPatch,
    negative: int,
    positive: int,
    evidence: str,
    precision: str,
) -> None:
    fit = _fit(monkeypatch, negative=negative, positive=positive)
    diagnostics = fit.identification
    assert diagnostics.evidence_status == evidence
    assert diagnostics.precision_status == precision
    assert np.isfinite(diagnostics.p_kl_divergence)
    assert diagnostics.p_kl_divergence >= 0.0
    assert np.isfinite(diagnostics.p_interval_width_contraction)
    assert "identified" not in diagnostics.to_dict().values()
