"""Package-owned errors for explicit validation and numerical failures."""

from __future__ import annotations


class StableBoundaryError(Exception):
    """Base class for all errors raised deliberately by :mod:`stableboundary`."""


class ValidationError(StableBoundaryError, ValueError):
    """Raised when caller-supplied values violate a public contract."""


class UnidentifiedParameterError(StableBoundaryError, ValueError):
    """Raised when a requested parameter is not identified by the model."""


class NumericalProbabilityError(StableBoundaryError, ArithmeticError):
    """Raised when a numerical probability is invalid or internally inconsistent."""


class ConvergenceError(StableBoundaryError, ArithmeticError):
    """Raised when a numerical method cannot meet its declared accuracy policy."""


class InfiniteMomentError(StableBoundaryError, ValueError):
    """Raised when a requested stable-law moment does not exist."""


class InfiniteVarianceError(InfiniteMomentError):
    """Raised when posterior support includes stable laws with infinite variance."""
