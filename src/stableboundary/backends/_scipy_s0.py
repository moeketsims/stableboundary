"""Isolated adapter for SciPy's Nolan ``S0`` implementation.

SciPy's public ``levy_stable`` singleton exposes mutable numerical controls.
This module never changes that singleton.  Instead, it owns a private
``levy_stable_gen`` instance whose complete effective configuration is forced
under a package lock before every operation and retained in backend metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from numbers import Integral, Real
from threading import RLock
from typing import Final, cast

import numpy as np
import scipy  # type: ignore[import-untyped]
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy.stats._levy_stable import (  # type: ignore[import-untyped]
    levy_stable_gen,
)

from stableboundary._exceptions import NumericalProbabilityError, ValidationError

from ._protocol import BackendMetadata, BackendResult, BackendSetting

_SCIPY_LOCK = RLock()
_QUAD_EPS: Final = 1.2e-14
_CANONICAL_SETTINGS: Final[tuple[tuple[str, BackendSetting], ...]] = (
    ("parameterization", "S0"),
    ("pdf_default_method", "piecewise"),
    ("cdf_default_method", "piecewise"),
    ("quad_eps", _QUAD_EPS),
    ("piecewise_x_tol_near_zeta", 0.005),
    ("piecewise_alpha_tol_near_one", 0.005),
    ("pdf_fft_grid_spacing", 0.001),
    ("pdf_fft_n_points_two_power", None),
    ("pdf_fft_interpolation_degree", 3),
    ("pdf_fft_interpolation_level", 3),
    ("pdf_fft_min_points_threshold", None),
)
_SCIPY_S0 = levy_stable_gen(name="_stableboundary_levy_stable")


@contextmanager
def _canonical_s0_generator() -> Iterator[levy_stable_gen]:
    """Yield the locked package-owned generator in its canonical state."""
    with _SCIPY_LOCK:
        for name, value in _CANONICAL_SETTINGS:
            setattr(_SCIPY_S0, name, value)
        yield _SCIPY_S0


def _numeric_array(name: str, value: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values")
    return result


def _finite_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _input_context(
    x: NDArray[np.float64],
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
    loc: float,
    scale: float,
) -> str:
    return (
        f"x_shape={x.shape}, alpha_shape={alpha.shape}, beta_shape={beta.shape}, "
        f"loc={loc!r}, scale={scale!r}"
    )


class ScipyS0Backend:
    """Guarded bootstrap backend using SciPy's piecewise Nolan method."""

    _metadata: Final = BackendMetadata(
        method="scipy-piecewise-s0-direct-log-tails",
        tolerance=_QUAD_EPS,
        parameterization="S0",
        library="scipy",
        library_version=scipy.__version__,
        effective_settings=_CANONICAL_SETTINGS,
    )

    @property
    def metadata(self) -> BackendMetadata:
        """Return the immutable effective numerical configuration."""
        return self._metadata

    def _evaluate(
        self,
        operation: str,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float,
        scale: float,
    ) -> BackendResult:
        x_values = _numeric_array("x", x)
        alpha_values = _numeric_array("alpha", alpha)
        beta_values = _numeric_array("beta", beta)
        loc_value = _finite_real("loc", loc)
        scale_value = _finite_real("scale", scale)
        if np.any((alpha_values <= 0.0) | (alpha_values > 2.0)):
            raise ValidationError("alpha must lie in (0, 2]")
        if np.any((beta_values < -1.0) | (beta_values > 1.0)):
            raise ValidationError("beta must lie in [-1, 1]")
        if scale_value <= 0.0:
            raise ValidationError("scale must be strictly positive")
        try:
            np.broadcast_arrays(x_values, alpha_values, beta_values)
        except ValueError as error:
            raise ValidationError("x, alpha, and beta must be broadcastable") from error

        context = _input_context(
            x_values,
            alpha_values,
            beta_values,
            loc_value,
            scale_value,
        )
        try:
            with _canonical_s0_generator() as distribution:
                scipy_operation = getattr(distribution, operation)
                raw = scipy_operation(
                    x_values,
                    alpha_values,
                    beta_values,
                    loc=loc_value,
                    scale=scale_value,
                )
        except NumericalProbabilityError:
            raise
        except Exception as error:
            raise NumericalProbabilityError(
                f"SciPy stable {operation} failed ({context})"
            ) from error

        try:
            values = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise NumericalProbabilityError(
                f"SciPy stable {operation} returned nonnumeric output ({context})"
            ) from error
        if not np.all(np.isfinite(values)):
            raise NumericalProbabilityError(
                f"SciPy stable {operation} returned nonfinite output ({context})"
            )
        if operation in {"cdf", "sf"} and np.any((values < 0.0) | (values > 1.0)):
            raise NumericalProbabilityError(
                f"SciPy stable {operation} returned values outside [0, 1] ({context})"
            )
        if values.ndim == 0:
            return float(values)
        return cast(NDArray[np.float64], values)

    def logpdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        return self._evaluate("logpdf", x, alpha, beta, loc=loc, scale=scale)

    def cdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        return self._evaluate("cdf", x, alpha, beta, loc=loc, scale=scale)

    def sf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        x_values = _numeric_array("x", x)
        beta_values = _numeric_array("beta", beta)
        loc_value = _finite_real("loc", loc)
        # SciPy's generic rv_continuous._sf is implemented as 1 - _cdf.
        # Use exact reflection instead: -X has S0(alpha, -beta, -loc, scale),
        # so the requested upper tail is a directly evaluated lower tail.
        return self._evaluate(
            "cdf",
            -x_values,
            alpha,
            -beta_values,
            loc=-loc_value,
            scale=scale,
        )

    def logcdf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        probabilities = np.asarray(
            self._evaluate("cdf", x, alpha, beta, loc=loc, scale=scale),
            dtype=np.float64,
        )
        if np.any(probabilities <= 0.0):
            raise NumericalProbabilityError(
                "SciPy stable cdf underflowed before direct logarithm"
            )
        values = np.log(probabilities)
        if values.ndim == 0:
            return float(values)
        return cast(NDArray[np.float64], values)

    def logsf(
        self,
        x: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> BackendResult:
        probabilities = np.asarray(
            self.sf(x, alpha, beta, loc=loc, scale=scale), dtype=np.float64
        )
        if np.any(probabilities <= 0.0):
            raise NumericalProbabilityError(
                "SciPy stable sf underflowed before direct logarithm"
            )
        values = np.log(probabilities)
        if values.ndim == 0:
            return float(values)
        return cast(NDArray[np.float64], values)

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
        alpha_value = _finite_real("alpha", alpha)
        beta_value = _finite_real("beta", beta)
        loc_value = _finite_real("loc", loc)
        scale_value = _finite_real("scale", scale)
        if not 0.0 < alpha_value <= 2.0:
            raise ValidationError("alpha must lie in (0, 2]")
        if not -1.0 <= beta_value <= 1.0:
            raise ValidationError("beta must lie in [-1, 1]")
        if scale_value <= 0.0:
            raise ValidationError("scale must be strictly positive")
        if isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0:
            raise ValidationError("size must be a positive integer")
        if not isinstance(random_state, Generator):
            raise ValidationError("random_state must be a numpy.random.Generator")

        context = (
            f"alpha={alpha_value!r}, beta={beta_value!r}, loc={loc_value!r}, "
            f"scale={scale_value!r}, size={int(size)}"
        )
        try:
            with _canonical_s0_generator() as distribution:
                raw = distribution.rvs(
                    alpha_value,
                    beta_value,
                    loc=loc_value,
                    scale=scale_value,
                    size=int(size),
                    random_state=random_state,
                )
        except Exception as error:
            raise NumericalProbabilityError(
                f"SciPy stable rvs failed ({context})"
            ) from error
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (int(size),) or not np.all(np.isfinite(values)):
            raise NumericalProbabilityError(
                f"SciPy stable rvs returned invalid output ({context})"
            )
        return cast(NDArray[np.float64], values)


__all__ = ["ScipyS0Backend"]
