"""Numerical contracts for guarded S0 probabilities and finite cells."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from scipy.stats import levy_stable  # type: ignore[import-untyped]

from stableboundary import NumericalProbabilityError, ValidationError
from stableboundary.backends import BackendMetadata, ScipyS0Backend, StableBackend

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
    original = _snapshot_scipy_state()
    try:
        for name, value in HOSTILE_SETTINGS.items():
            setattr(levy_stable, name, value)
        incoming = _snapshot_scipy_state()
        assert incoming == HOSTILE_SETTINGS
        yield incoming
    finally:
        for name, value in original.items():
            setattr(levy_stable, name, value)


def test_backend_is_runtime_checkable_and_metadata_is_immutable() -> None:
    backend = ScipyS0Backend()
    assert isinstance(backend, StableBackend)
    assert backend.metadata == BackendMetadata(
        method="scipy-piecewise-s0",
        tolerance=1.2e-14,
        parameterization="S0",
    )
    with pytest.raises(FrozenInstanceError):
        backend.metadata.tolerance = 1.0  # type: ignore[misc]


def test_backend_restores_complete_scipy_state_after_success() -> None:
    backend = ScipyS0Backend()
    with _hostile_scipy_state() as incoming:
        value = backend.logpdf(0.25, 1.8, -0.3)
        assert np.isfinite(value)
        assert _snapshot_scipy_state() == incoming


def test_backend_restores_complete_scipy_state_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ScipyS0Backend()

    def fail(*args: object, **kwargs: object) -> float:
        del args, kwargs
        raise RuntimeError("forced backend failure")

    with _hostile_scipy_state() as incoming:
        monkeypatch.setattr(levy_stable, "logpdf", fail)
        with pytest.raises(NumericalProbabilityError, match=r"logpdf.*alpha_shape"):
            backend.logpdf(0.25, 1.8, -0.3)
        assert _snapshot_scipy_state() == incoming


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


@pytest.mark.parametrize(
    ("method_name", "scipy_name"),
    [("sf", "sf"), ("logsf", "logsf")],
)
def test_positive_tail_backend_calls_direct_survival_method(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    scipy_name: str,
) -> None:
    backend = ScipyS0Backend()
    calls = {"sf": 0, "logsf": 0, "cdf": 0}

    def direct_tail(*args: object, **kwargs: object) -> float:
        del args, kwargs
        calls[scipy_name] += 1
        return -3.0 if scipy_name == "logsf" else np.exp(-3.0)

    def forbidden_cdf(*args: object, **kwargs: object) -> float:
        del args, kwargs
        calls["cdf"] += 1
        raise AssertionError("positive tails must not call cdf")

    monkeypatch.setattr(levy_stable, scipy_name, direct_tail)
    monkeypatch.setattr(levy_stable, "cdf", forbidden_cdf)
    result = getattr(backend, method_name)(3.0, 1.8, 0.2)
    assert np.isfinite(result)
    assert calls[scipy_name] == 1
    assert calls["cdf"] == 0


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(0.0, 0.0), (2.1, 0.0), (1.8, -1.1), (1.8, 1.1)],
)
def test_backend_rejects_invalid_parameter_inputs(alpha: float, beta: float) -> None:
    with pytest.raises(ValidationError):
        ScipyS0Backend().logpdf(0.0, alpha, beta)
