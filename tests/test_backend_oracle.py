"""Independent spot checks of the canonical S0 density and tails.

The oracle here shares no stable-law code with SciPy: it inverts the Nolan S0
characteristic function directly (Fourier inversion for the density,
Gil-Pelaez inversion for the upper tail) on a graded composite Gauss-Legendre
mesh, and uses the large-argument series for one far point. These are spot
checks at supported parameter values, not an accuracy bound.
"""

from __future__ import annotations

from math import gamma, log, pi

import numpy as np
import pytest
from scipy.special import roots_legendre  # type: ignore[import-untyped]

from stableboundary import LocalDesign
from stableboundary.backends import ScipyS0Backend

_NODES, _WEIGHTS = roots_legendre(24)
_ALPHAS = (1.10, 1.5, 1.82, 1.95)
_BETAS = (-0.9, 0.2, 0.9)
_LOG_TOLERANCE = 1e-8
_MINIMUM_DISTANCE_FROM_ZETA = 0.25


def _mesh(x: float, alpha: float, fine: float) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights on [0, T] with exp(-T**alpha) below 1e-20."""
    upper = 46.0 ** (1.0 / alpha)
    width = fine * min(0.25, 1.0 / (1.0 + abs(x)))
    edges = np.unique(
        np.concatenate(
            (
                [0.0],
                width * 1.5 ** np.arange(-90, 1),
                np.arange(width, upper + width, width),
            )
        )
    )
    half = 0.5 * np.diff(edges)
    middle = 0.5 * (edges[1:] + edges[:-1])
    nodes = (middle[:, None] + half[:, None] * _NODES).ravel()
    weights = (half[:, None] * _WEIGHTS).ravel()
    return nodes, weights


def _phase(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    # beta*tan(pi*alpha/2)*(t**alpha - t), written without cancellation.
    return beta * np.tan(pi * alpha / 2.0) * t * np.expm1((alpha - 1.0) * np.log(t))


def _oracle_pdf(x: float, alpha: float, beta: float, fine: float = 1.0) -> float:
    t, w = _mesh(x, alpha, fine)
    integrand = np.exp(-(t**alpha)) * np.cos(_phase(t, alpha, beta) - t * x)
    return float(np.sum(w * integrand) / pi)


def _oracle_sf(x: float, alpha: float, beta: float, fine: float = 1.0) -> float:
    t, w = _mesh(x, alpha, fine)
    integrand = np.exp(-(t**alpha)) * np.sin(_phase(t, alpha, beta) - t * x) / t
    return float(0.5 + np.sum(w * integrand) / pi)


def _resolved(coarse: float, fine: float, relative: float = 1e-10) -> float:
    """Return the finer oracle value after checking its own resolution."""
    assert fine > 0.0
    assert abs(coarse - fine) <= relative * fine
    return fine


def _series_logpdf(x: float, alpha: float, beta: float, terms: int = 4) -> float:
    """Large-|x| series of the S1 density, shifted to S0 by f0(x) = f1(x - zeta)."""
    if x < 0.0:
        return _series_logpdf(-x, alpha, -beta, terms)
    tangent = np.tan(pi * alpha / 2.0)
    argument = x + beta * tangent
    c = complex(1.0, -beta * tangent)
    total, factorial = 0.0, 1.0
    for n in range(1, terms + 1):
        factorial *= n
        coefficient = ((-c) ** n * (-1j) * np.exp(-1j * n * pi * alpha / 2.0)).real
        total += (
            gamma(n * alpha + 1.0)
            / (factorial * pi)
            * coefficient
            * argument ** (-n * alpha - 1.0)
        )
    return log(total)


def test_oracle_reproduces_closed_form_laws() -> None:
    for x in (-3.0, 0.0, 0.7, 5.0):
        gauss = np.exp(-x * x / 4.0) / np.sqrt(4.0 * pi)
        cauchy = 1.0 / (pi * (1.0 + x * x))
        assert _oracle_pdf(x, 2.0, 0.0) == pytest.approx(gauss, abs=1e-14)
        assert _oracle_pdf(x, 1.0, 0.0) == pytest.approx(cauchy, abs=1e-14)
    # S0(0.5, 1) is a Levy law shifted by -1.
    for x in (0.0, 1.0, 10.0):
        y = x + 1.0
        levy = np.sqrt(1.0 / (2.0 * pi)) * y**-1.5 * np.exp(-0.5 / y)
        assert _oracle_pdf(x, 0.5, 1.0) == pytest.approx(levy, abs=1e-14)
    assert _oracle_sf(2.5, 1.0, 0.0) == pytest.approx(
        0.5 - np.arctan(2.5) / pi, abs=1e-14
    )


@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize("beta", _BETAS)
def test_design_tails_agree_with_independent_inversion(
    alpha: float, beta: float
) -> None:
    threshold = LocalDesign.from_sample_size(256).threshold
    zeta = -beta * np.tan(pi * alpha / 2.0)
    assert abs(threshold - zeta) > _MINIMUM_DISTANCE_FROM_ZETA
    assert abs(-threshold - zeta) > _MINIMUM_DISTANCE_FROM_ZETA
    backend = ScipyS0Backend()

    upper = _resolved(
        _oracle_sf(threshold, alpha, beta), _oracle_sf(threshold, alpha, beta, 0.5)
    )
    # Reflection: P(X <= -t) under beta equals P(X >= t) under -beta.
    lower = _resolved(
        _oracle_sf(threshold, alpha, -beta), _oracle_sf(threshold, alpha, -beta, 0.5)
    )
    assert backend.logsf(threshold, alpha, beta) == pytest.approx(
        log(upper), abs=_LOG_TOLERANCE
    )
    assert backend.logcdf(-threshold, alpha, beta) == pytest.approx(
        log(lower), abs=_LOG_TOLERANCE
    )


@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize("beta", _BETAS)
def test_density_beyond_the_design_threshold_agrees_with_independent_inversion(
    alpha: float, beta: float
) -> None:
    threshold = LocalDesign.from_sample_size(256).threshold
    zeta = -beta * np.tan(pi * alpha / 2.0)
    backend = ScipyS0Backend()
    for x in (threshold + 2.0, -(threshold + 3.0)):
        assert abs(x - zeta) > _MINIMUM_DISTANCE_FROM_ZETA
        density = _resolved(
            _oracle_pdf(x, alpha, beta), _oracle_pdf(x, alpha, beta, 0.5)
        )
        assert backend.logpdf(x, alpha, beta) == pytest.approx(
            log(density), abs=_LOG_TOLERANCE
        )


@pytest.mark.parametrize("alpha", (1.5, 1.82, 1.95))
@pytest.mark.parametrize("x", (1.0e3, -1.0e3))
def test_far_density_agrees_with_the_large_argument_series(
    alpha: float, x: float
) -> None:
    beta = 0.5
    # The series is first tied to the inversion oracle where both resolve.
    anchor = 60.0 if x > 0.0 else -60.0
    # Densities near 1e-7 sit close to the inversion's absolute resolution, so
    # the anchor is held to 1e-8 relative, well inside its 1e-6 use below.
    inversion = _resolved(
        _oracle_pdf(anchor, alpha, beta),
        _oracle_pdf(anchor, alpha, beta, 0.5),
        relative=1e-8,
    )
    assert _series_logpdf(anchor, alpha, beta) == pytest.approx(
        log(inversion), abs=1e-6
    )
    assert abs(x) < ScipyS0Backend.MAX_VALIDATED_ABS_Z
    assert ScipyS0Backend().logpdf(x, alpha, beta) == pytest.approx(
        _series_logpdf(x, alpha, beta), abs=_LOG_TOLERANCE
    )
