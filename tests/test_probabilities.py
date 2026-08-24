"""Numerical contracts for guarded S0 probabilities and finite cells."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy
from scipy.stats import levy_stable  # type: ignore[import-untyped]

import stableboundary.backends._scipy_s0 as scipy_s0_module
from stableboundary import (
    CellCounts,
    CellProbabilities,
    KnownNuisance,
    LocalCoordinates,
    LocalDesign,
    NumericalProbabilityError,
    ValidationError,
)
from stableboundary.backends import BackendMetadata, ScipyS0Backend, StableBackend
from stableboundary.cells import exact_cell_probabilities

GUARDED_SETTINGS = (
    "parameterization",
    "pdf_default_method",
    "cdf_default_method",
    "quad_eps",
    "piecewise_x_tol_near_zeta",
    "piecewise_alpha_tol_near_one",
    "pdf_fft_grid_spacing",
    "pdf_fft_n_points_two_power",
    "pdf_fft_interpolation_degree",
    "pdf_fft_interpolation_level",
    "pdf_fft_min_points_threshold",
)

HOSTILE_SETTINGS: dict[str, Any] = {
    "parameterization": "S1",
    "pdf_default_method": "dni",
    "cdf_default_method": "fft-simpson",
    "quad_eps": 3.5e-9,
    "piecewise_x_tol_near_zeta": 0.0125,
    "piecewise_alpha_tol_near_one": 0.0175,
    "pdf_fft_grid_spacing": 0.012,
    "pdf_fft_n_points_two_power": 15,
    "pdf_fft_interpolation_degree": 5,
    "pdf_fft_interpolation_level": 4,
    "pdf_fft_min_points_threshold": 73,
}


def _snapshot_scipy_state() -> dict[str, Any]:
    return {name: getattr(levy_stable, name) for name in GUARDED_SETTINGS}


@contextmanager
def _hostile_scipy_state() -> Iterator[dict[str, Any]]:
    original = {
        name: (name in levy_stable.__dict__, getattr(levy_stable, name))
        for name in GUARDED_SETTINGS
    }
    try:
        for name, value in HOSTILE_SETTINGS.items():
            setattr(levy_stable, name, value)
        incoming = _snapshot_scipy_state()
        assert incoming == HOSTILE_SETTINGS
        yield incoming
    finally:
        for name, (was_instance_attribute, value) in original.items():
            if was_instance_attribute:
                setattr(levy_stable, name, value)
            elif name in levy_stable.__dict__:
                delattr(levy_stable, name)


def test_backend_is_runtime_checkable_and_metadata_is_immutable() -> None:
    backend = ScipyS0Backend()
    assert isinstance(backend, StableBackend)
    assert backend.metadata.method == "scipy-piecewise-s0-direct-log-tails"
    assert backend.metadata.tolerance == 1.2e-14
    assert backend.metadata.parameterization == "S0"
    assert backend.metadata.library == "scipy"
    assert backend.metadata.library_version == scipy.__version__
    assert dict(backend.metadata.effective_settings) == {
        "parameterization": "S0",
        "pdf_default_method": "piecewise",
        "cdf_default_method": "piecewise",
        "quad_eps": 1.2e-14,
        "piecewise_x_tol_near_zeta": 0.005,
        "piecewise_alpha_tol_near_one": 0.005,
        "pdf_fft_grid_spacing": 0.001,
        "pdf_fft_n_points_two_power": None,
        "pdf_fft_interpolation_degree": 3,
        "pdf_fft_interpolation_level": 3,
        "pdf_fft_min_points_threshold": None,
    }
    with pytest.raises(FrozenInstanceError):
        backend.metadata.tolerance = 1.0  # type: ignore[misc]


def test_backend_is_independent_of_hostile_public_scipy_state() -> None:
    backend = ScipyS0Backend()
    expected = (
        backend.logpdf(0.25, 1.8, -0.3),
        backend.cdf(-2.0, 1.8, -0.3),
        backend.sf(2.0, 1.8, -0.3),
    )
    with _hostile_scipy_state() as incoming:
        actual = (
            backend.logpdf(0.25, 1.8, -0.3),
            backend.cdf(-2.0, 1.8, -0.3),
            backend.sf(2.0, 1.8, -0.3),
        )
        assert actual == expected
        assert _snapshot_scipy_state() == incoming


def test_backend_never_changes_public_setting_ownership() -> None:
    backend = ScipyS0Backend()
    incoming = {
        name: (name in levy_stable.__dict__, getattr(levy_stable, name))
        for name in GUARDED_SETTINGS
    }
    backend.logpdf(0.25, 1.8, -0.3)
    outgoing = {
        name: (name in levy_stable.__dict__, getattr(levy_stable, name))
        for name in GUARDED_SETTINGS
    }
    assert outgoing == incoming


def test_backend_preserves_public_state_after_private_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ScipyS0Backend()

    def fail(*args: object, **kwargs: object) -> float:
        del args, kwargs
        raise RuntimeError("forced backend failure")

    with _hostile_scipy_state() as incoming:
        monkeypatch.setattr(scipy_s0_module._SCIPY_S0, "logpdf", fail)
        with pytest.raises(NumericalProbabilityError, match=r"logpdf.*alpha_shape"):
            backend.logpdf(0.25, 1.8, -0.3)
        assert _snapshot_scipy_state() == incoming


def test_backend_does_not_call_public_scipy_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ScipyS0Backend()

    def forbidden(*args: object, **kwargs: object) -> float:
        del args, kwargs
        raise AssertionError("public scipy.stats.levy_stable must not be called")

    monkeypatch.setattr(levy_stable, "logpdf", forbidden)
    monkeypatch.setattr(levy_stable, "cdf", forbidden)
    monkeypatch.setattr(levy_stable, "rvs", forbidden)
    assert np.isfinite(backend.logpdf(0.25, 1.8, -0.3))
    assert np.isfinite(backend.logcdf(-2.0, 1.8, -0.3))
    draws = backend.rvs(
        1.8,
        -0.3,
        loc=0.0,
        scale=1.0,
        size=3,
        random_state=np.random.default_rng(17),
    )
    assert draws.shape == (3,)


def test_backend_reforces_every_private_setting_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ScipyS0Backend()
    private = scipy_s0_module._SCIPY_S0
    expected = dict(backend.metadata.effective_settings)
    original_cdf = private.cdf
    observed: dict[str, object] = {}
    for name, value in HOSTILE_SETTINGS.items():
        setattr(private, name, value)

    def recording_cdf(*args: object, **kwargs: object) -> object:
        observed.update({name: getattr(private, name) for name in GUARDED_SETTINGS})
        return original_cdf(*args, **kwargs)

    monkeypatch.setattr(private, "cdf", recording_cdf)
    assert np.isfinite(backend.cdf(-2.0, 1.8, -0.3))
    assert observed == expected


def test_backend_finite_reference_log_values_and_broadcasting() -> None:
    backend = ScipyS0Backend()
    assert np.isfinite(backend.logpdf(0.25, 1.8, -0.3))
    assert np.isfinite(backend.logcdf(-2.0, 1.8, -0.3))
    assert np.isfinite(backend.logsf(2.0, 1.8, -0.3))

    alpha = np.array([1.7, 1.8, 1.9])
    beta = np.array([-0.4, 0.0, 0.4])
    values = backend.logpdf(np.array([-0.5, 0.0, 0.5]), alpha, beta)
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("method_name", ["sf", "logsf"])
def test_positive_tail_backend_uses_direct_reflected_cdf(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    backend = ScipyS0Backend()
    calls: list[tuple[object, object, object, dict[str, object]]] = []

    def direct_cdf(
        x: object,
        alpha: object,
        beta: object,
        **kwargs: object,
    ) -> float:
        calls.append((x, alpha, beta, kwargs))
        return np.exp(-3.0)

    def forbidden_sf(*args: object, **kwargs: object) -> float:
        del args, kwargs
        raise AssertionError("positive tails must not use SciPy's subtractive sf")

    monkeypatch.setattr(scipy_s0_module._SCIPY_S0, "cdf", direct_cdf)
    monkeypatch.setattr(scipy_s0_module._SCIPY_S0, "sf", forbidden_sf)
    result = getattr(backend, method_name)(3.0, 1.8, 0.2, loc=0.5, scale=2.0)
    assert np.isfinite(result)
    assert len(calls) == 1
    x, alpha, beta, kwargs = calls[0]
    assert float(np.asarray(x)) == -3.0
    assert float(np.asarray(alpha)) == 1.8
    assert float(np.asarray(beta)) == -0.2
    assert kwargs == {"loc": -0.5, "scale": 2.0}


def test_logcdf_uses_direct_cdf_then_checked_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ScipyS0Backend()
    calls = {"cdf": 0}

    def direct_cdf(*args: object, **kwargs: object) -> float:
        del args, kwargs
        calls["cdf"] += 1
        return np.exp(-4.0)

    monkeypatch.setattr(scipy_s0_module._SCIPY_S0, "cdf", direct_cdf)
    assert backend.logcdf(-3.0, 1.8, 0.2) == pytest.approx(-4.0)
    assert calls["cdf"] == 1


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(0.0, 0.0), (2.1, 0.0), (1.8, -1.1), (1.8, 1.1)],
)
def test_backend_rejects_invalid_parameter_inputs(alpha: float, beta: float) -> None:
    with pytest.raises(ValidationError):
        ScipyS0Backend().logpdf(0.0, alpha, beta)


class RecordingBackend:
    """Minimal deterministic protocol implementation for failure-path tests."""

    def __init__(self, log_minus: float = -4.0, log_plus: float = -5.0) -> None:
        self.metadata = BackendMetadata(method="recording", tolerance=1e-14)
        self.log_minus = log_minus
        self.log_plus = log_plus
        self.calls: list[tuple[str, object, object, object, float, float]] = []

    def _record(
        self,
        operation: str,
        x: object,
        alpha: object,
        beta: object,
        loc: float,
        scale: float,
    ) -> None:
        self.calls.append((operation, x, alpha, beta, loc, scale))

    def logcdf(
        self,
        x: object,
        alpha: object,
        beta: object,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> float:
        self._record("logcdf", x, alpha, beta, loc, scale)
        return self.log_minus

    def logsf(
        self,
        x: object,
        alpha: object,
        beta: object,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> float:
        self._record("logsf", x, alpha, beta, loc, scale)
        return self.log_plus

    def logpdf(
        self,
        x: object,
        alpha: object,
        beta: object,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> float:
        self._record("logpdf", x, alpha, beta, loc, scale)
        return -1.0

    def cdf(
        self,
        x: object,
        alpha: object,
        beta: object,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> float:
        self._record("cdf", x, alpha, beta, loc, scale)
        raise AssertionError("exact cells must call logcdf directly")

    def sf(
        self,
        x: object,
        alpha: object,
        beta: object,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> float:
        self._record("sf", x, alpha, beta, loc, scale)
        raise AssertionError("exact cells must call logsf directly")

    def rvs(
        self,
        alpha: float,
        beta: float,
        *,
        loc: float,
        scale: float,
        size: int,
        random_state: np.random.Generator,
    ) -> np.ndarray:
        del alpha, beta, loc, scale, random_state
        return np.zeros(size)


def test_exact_cells_use_direct_log_tails_and_standardized_s0(
    local_design: LocalDesign,
) -> None:
    backend = RecordingBackend()
    local = LocalCoordinates(r=local_design.r, h=1.5, p=0.7)
    probabilities = exact_cell_probabilities(local, local_design, backend)
    assert [call[0] for call in backend.calls] == ["logcdf", "logsf"]
    negative_call, positive_call = backend.calls
    assert negative_call[1] == -local_design.threshold
    assert positive_call[1] == local_design.threshold
    assert (
        negative_call[2:4]
        == positive_call[2:4]
        == (
            local.alpha,
            local.beta,
        )
    )
    assert negative_call[4:] == positive_call[4:] == (0.0, 1.0)
    assert probabilities.q_minus == pytest.approx(np.exp(-4.0))
    assert probabilities.q_plus == pytest.approx(np.exp(-5.0))
    assert probabilities.q_minus + probabilities.q_zero + probabilities.q_plus == 1.0


def test_exact_cells_are_finite_normalized_and_reflect(
    local_design: LocalDesign,
) -> None:
    positive = exact_cell_probabilities(
        LocalCoordinates(r=local_design.r, h=1.25, p=0.8),
        local_design,
    )
    reflected = exact_cell_probabilities(
        LocalCoordinates(r=local_design.r, h=1.25, p=0.2),
        local_design,
    )
    assert all(
        np.isfinite(value)
        for value in (
            positive.q_minus,
            positive.q_zero,
            positive.q_plus,
            positive.log_q_minus,
            positive.log_q_plus,
        )
    )
    assert sum((positive.q_minus, positive.q_zero, positive.q_plus)) == pytest.approx(
        1.0,
        abs=CellProbabilities.SIMPLEX_TOLERANCE,
    )
    assert positive.q_minus == pytest.approx(reflected.q_plus, rel=1e-10)
    assert positive.q_plus == pytest.approx(reflected.q_minus, rel=1e-10)
    assert positive.q_zero == pytest.approx(reflected.q_zero, rel=1e-10)


@pytest.mark.parametrize(
    ("log_minus", "log_plus", "message"),
    [
        (-4.0, -1000.0, "underflowed"),
        (-np.inf, -5.0, "invalid log probability"),
        (-4.0, 0.25, "invalid log probability"),
    ],
)
def test_exact_cells_refuse_underflowed_or_invalid_log_tails(
    local_design: LocalDesign,
    log_minus: float,
    log_plus: float,
    message: str,
) -> None:
    local = LocalCoordinates(r=local_design.r, h=1.25, p=0.7)
    with pytest.raises(NumericalProbabilityError, match=message):
        exact_cell_probabilities(
            local,
            local_design,
            RecordingBackend(log_minus=log_minus, log_plus=log_plus),
        )


def test_cell_probability_record_refuses_invalid_simplex_and_log_pair() -> None:
    with pytest.raises(NumericalProbabilityError, match="normalize"):
        CellProbabilities(0.1, 0.7, 0.1, np.log(0.1), np.log(0.1), "fake", 1e-12)
    with pytest.raises(NumericalProbabilityError, match="disagree"):
        CellProbabilities(0.1, 0.8, 0.1, -3.0, np.log(0.1), "fake", 1e-12)


def test_cell_counts_sum_and_threshold_boundaries_are_central(
    local_design: LocalDesign,
) -> None:
    values = np.zeros(local_design.n)
    values[:6] = [
        -local_design.threshold - 1.0,
        -local_design.threshold,
        0.0,
        local_design.threshold,
        local_design.threshold + 1.0,
        0.0,
    ]
    nuisance = KnownNuisance.externally_known(
        loc=0.0,
        scale=1.0,
        provenance="test fixture",
    )
    counts = CellCounts.from_observations(
        values,
        nuisance=nuisance,
        design=local_design,
    )
    assert counts.n_minus == 1
    assert counts.n_plus == 1
    assert counts.n_zero == local_design.n - 2
    assert counts.n_minus + counts.n_zero + counts.n_plus == counts.n
    with pytest.raises(FrozenInstanceError):
        counts.n_zero = 0  # type: ignore[misc]


def test_cell_counts_are_shift_scale_equivariant(local_design: LocalDesign) -> None:
    standardized = np.zeros(local_design.n)
    standardized[:4] = [
        -local_design.threshold - 2.0,
        -local_design.threshold + 0.5,
        local_design.threshold - 0.5,
        local_design.threshold + 2.0,
    ]
    baseline = CellCounts.from_observations(
        standardized,
        nuisance=KnownNuisance.externally_known(
            loc=0.0,
            scale=1.0,
            provenance="standard fixture",
        ),
        design=local_design,
    )
    loc, scale = 17.5, 3.25
    transformed = loc + scale * standardized
    shifted = CellCounts.from_observations(
        transformed,
        nuisance=KnownNuisance.externally_known(
            loc=loc,
            scale=scale,
            provenance="calibration fixture",
        ),
        design=local_design,
    )
    assert shifted == baseline


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 2)),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1, "2"], dtype=object),
    ],
)
def test_cell_counts_reject_invalid_observation_arrays(values: np.ndarray) -> None:
    design = LocalDesign.from_sample_size(max(values.size, 5000))
    nuisance = KnownNuisance.externally_known(
        loc=0.0,
        scale=1.0,
        provenance="test fixture",
    )
    with pytest.raises(ValidationError):
        CellCounts.from_observations(values, nuisance=nuisance, design=design)


def test_cells_source_has_no_hidden_probability_repair() -> None:
    source = Path("src/stableboundary/cells.py").read_text(encoding="utf-8")
    forbidden = ("np.clip", "np.maximum", "clip(", "maximum(", "/= tail_total")
    assert all(token not in source for token in forbidden)
