"""Shared deterministic fixtures for the stableboundary test suite."""

from __future__ import annotations

import pytest

from stableboundary import LocalCoordinates, StableParams


@pytest.fixture
def package_name() -> str:
    """Return the canonical installed distribution name."""
    return "stableboundary"


@pytest.fixture
def stable_params() -> StableParams:
    """Return a representative near-Gaussian ``S0`` parameter record."""
    return StableParams(alpha=1.97, beta=0.35, loc=0.0, scale=1.0)


@pytest.fixture
def local_coordinates() -> LocalCoordinates:
    """Return representative theorem-interior local coordinates."""
    return LocalCoordinates(r=0.02, h=1.5, p=0.675)
