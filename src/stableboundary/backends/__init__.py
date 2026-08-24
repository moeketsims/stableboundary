"""Private numerical backends used by :mod:`stableboundary`.

The package facade deliberately does not re-export these implementation
details.  They remain importable here so package modules and focused backend
tests can depend on the package-owned protocol rather than on SciPy objects.
"""

from ._protocol import (
    BackendMetadata,
    StableBackend,
    validate_s0_backend,
    validate_s0_metadata,
)
from ._scipy_s0 import ScipyS0Backend

__all__ = [
    "BackendMetadata",
    "ScipyS0Backend",
    "StableBackend",
    "validate_s0_backend",
    "validate_s0_metadata",
]
