"""Shared deterministic fixtures for the stableboundary test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def package_name() -> str:
    """Return the canonical installed distribution name."""
    return "stableboundary"
