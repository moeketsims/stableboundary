"""Reproducible public simulation for Nolan ``S0`` stable laws."""

from __future__ import annotations

from numbers import Integral
from typing import cast

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ._exceptions import NumericalProbabilityError, ValidationError
from .backends import ScipyS0Backend
from .parameters import StableParams

_FLOAT64_ITEMSIZE = np.dtype(np.float64).itemsize
_MAX_FLOAT64_VECTOR_LENGTH = np.iinfo(np.intp).max // _FLOAT64_ITEMSIZE


def simulate(
    params: StableParams,
    size: int,
    random_state: int | Generator | None = None,
) -> NDArray[np.float64]:
    """Draw a finite, read-only one-dimensional ``S0`` sample.

    Equal integer seeds reproduce the same sample within a fixed NumPy/SciPy
    dependency environment.  No global random state is read or changed.
    """
    if not isinstance(params, StableParams):
        raise ValidationError("params must be a StableParams object")
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise ValidationError("size must be a positive integer")
    size_value = int(size)
    if size_value <= 0:
        raise ValidationError("size must be a positive integer")
    if size_value > _MAX_FLOAT64_VECTOR_LENGTH:
        raise ValidationError("size exceeds the platform float64 allocation limit")
    if isinstance(random_state, bool):
        raise ValidationError(
            "random_state must be an integer seed, Generator, or None"
        )
    try:
        generator = np.random.default_rng(random_state)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "random_state must be an integer seed, Generator, or None"
        ) from error

    values = ScipyS0Backend().rvs(
        params.alpha,
        params.beta,
        loc=params.loc,
        scale=params.scale,
        size=size_value,
        random_state=generator,
    )
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape != (size_value,) or not np.all(np.isfinite(result)):
        raise NumericalProbabilityError(
            "stable simulation returned a nonfinite or non-vector result"
        )
    result.setflags(write=False)
    return cast(NDArray[np.float64], result)


__all__ = ["simulate"]
