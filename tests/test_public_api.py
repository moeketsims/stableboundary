"""Smoke tests for the installed package surface and error hierarchy."""

from __future__ import annotations

from importlib import metadata, resources

import stableboundary as sb


def test_version_matches_distribution_metadata(package_name: str) -> None:
    assert sb.__version__ == metadata.version(package_name)


def test_typing_marker_is_packaged(package_name: str) -> None:
    marker = resources.files(package_name).joinpath("py.typed")
    assert marker.is_file()


def test_public_error_hierarchy_is_curated() -> None:
    expected = {
        "ConvergenceError",
        "InfiniteMomentError",
        "NumericalProbabilityError",
        "StableBoundaryError",
        "UnidentifiedParameterError",
        "ValidationError",
        "__version__",
    }
    assert set(sb.__all__) == expected
    for name in expected - {"StableBoundaryError", "__version__"}:
        error_type = getattr(sb, name)
        assert issubclass(error_type, sb.StableBoundaryError)


def test_public_facade_does_not_expose_private_backends() -> None:
    assert all(not name.startswith("_") or name == "__version__" for name in sb.__all__)
    assert not hasattr(sb, "_planning")
