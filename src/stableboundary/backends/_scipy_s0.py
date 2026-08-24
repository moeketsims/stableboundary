"""Scoped adapter for SciPy's process-global Nolan ``S0`` implementation.

SciPy exposes stable-law configuration as mutable class attributes.  Every
call in this module therefore executes while holding one re-entrant process
lock, after taking a complete snapshot and forcing the package's canonical
settings.  The exact incoming values are restored in ``finally``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from numbers import Integral, Real
from threading import RLock
from typing import Final, cast

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy.stats import levy_stable  # type: ignore[import-untyped]

from stableboundary._exceptions import NumericalProbabilityError, ValidationError

from ._protocol import BackendMetadata, BackendResult

_SCIPY_LOCK = RLock()
_QUAD_EPS: Final = 1.2e-14
_GUARDED_SETTINGS: Final = (
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


@contextmanager
def _canonical_s0_state() -> Iterator[None]:
    """Temporarily force canonical SciPy settings and restore every field."""
    with _SCIPY_LOCK:
        snapshot = {
            name: (name in levy_stable.__dict__, getattr(levy_stable, name))
            for name in _GUARDED_SETTINGS
        }
        try:
            levy_stable.parameterization = "S0"
            levy_stable.pdf_default_method = "piecewise"
            levy_stable.cdf_default_method = "piecewise"
            levy_stable.quad_eps = _QUAD_EPS
            yield
        finally:
            for name, (was_instance_attribute, value) in snapshot.items():
                if was_instance_attribute:
                    setattr(levy_stable, name, value)
                elif name in levy_stable.__dict__:
                    delattr(levy_stable, name)


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
            with _canonical_s0_state():
                scipy_operation = getattr(levy_stable, operation)
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
        if operation in {"cdf", "sf"} and np.any(
            (values < 0.0) | (values > 1.0)
        ):
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
        return self._evaluate("sf", x, alpha, beta, loc=loc, scale=scale)

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
            self._evaluate("sf", x, alpha, beta, loc=loc, scale=scale),
            dtype=np.float64,
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
            with _canonical_s0_state():
                raw = levy_stable.rvs(
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
