"""Contracts for conventional, local, and signed-gap parameterizations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from math import isclose

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from stableboundary import (
    LocalCoordinates,
    SignedTailGap,
    StableParams,
    UnidentifiedParameterError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 0.0),
        ("alpha", -0.1),
        ("alpha", 2.01),
        ("alpha", float("nan")),
        ("beta", -1.01),
        ("beta", 1.01),
        ("beta", float("inf")),
        ("loc", float("nan")),
        ("scale", 0.0),
        ("scale", -1.0),
        ("scale", float("inf")),
    ],
)
def test_stable_params_reject_invalid_domains(field: str, value: float) -> None:
    values = {"alpha": 1.9, "beta": 0.2, "loc": 0.0, "scale": 1.0}
    values[field] = value
    with pytest.raises(ValidationError):
        StableParams(**values)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"r": 0.0, "h": 1.0, "p": 0.5},
        {"r": float("nan"), "h": 1.0, "p": 0.5},
        {"r": 0.1, "h": 0.0, "p": 0.5},
        {"r": 0.1, "h": 1.0, "p": 0.0},
        {"r": 0.1, "h": 1.0, "p": 1.0},
        {"r": 1.0, "h": 2.0, "p": 0.5},
    ],
)
def test_local_coordinates_reject_unsupported_regions(
    coordinates: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        LocalCoordinates(**coordinates)


def test_exact_identities_and_s0_metadata(
    stable_params: StableParams, local_coordinates: LocalCoordinates
) -> None:
    local = stable_params.to_local(r=0.02)
    assert local.r == local_coordinates.r
    assert isclose(local.h, local_coordinates.h, rel_tol=2e-14)
    assert local.p == local_coordinates.p
    assert isclose(local.alpha, stable_params.alpha, rel_tol=2e-14)
    assert isclose(local.beta, stable_params.beta, rel_tol=2e-14)
    assert local.tau_plus == local.r * local.h * local.p
    assert local.tau_minus == local.r * local.h * (1.0 - local.p)
    assert StableParams.parameterization == "S0"


def test_round_trip_preserves_nuisance_values(stable_params: StableParams) -> None:
    rebuilt = stable_params.to_local(r=0.02).to_stable(
        loc=stable_params.loc,
        scale=stable_params.scale,
    )
    assert isclose(rebuilt.alpha, stable_params.alpha, rel_tol=2e-14)
    assert isclose(rebuilt.beta, stable_params.beta, rel_tol=2e-14)
    assert rebuilt.loc == stable_params.loc
    assert rebuilt.scale == stable_params.scale


def test_signed_gap_round_trip_retains_design_scale(
    local_coordinates: LocalCoordinates,
) -> None:
    gap = local_coordinates.to_signed_tail_gap()
    assert gap.r == local_coordinates.r
    assert gap.to_local() == local_coordinates
    assert gap.alpha == local_coordinates.alpha
    assert gap.beta == local_coordinates.beta


def test_reflection_swaps_signed_tail_gaps() -> None:
    positive = LocalCoordinates(r=0.01, h=1.2, p=0.8).to_signed_tail_gap()
    negative = LocalCoordinates(r=0.01, h=1.2, p=0.2).to_signed_tail_gap()
    assert isclose(positive.tau_plus, negative.tau_minus, rel_tol=2e-14)
    assert isclose(positive.tau_minus, negative.tau_plus, rel_tol=2e-14)
    assert isclose(positive.beta, -negative.beta, rel_tol=2e-14)


def test_exact_gaussian_refuses_local_identification() -> None:
    gaussian = StableParams(alpha=2.0, beta=0.75, loc=3.0, scale=2.0)
    with pytest.raises(UnidentifiedParameterError, match="not identified"):
        gaussian.to_local(r=0.01)


@pytest.mark.parametrize("beta", [-1.0, 1.0])
def test_one_sided_boundaries_refuse_interior_local_mapping(beta: float) -> None:
    with pytest.raises(ValidationError, match="strictly inside"):
        StableParams(alpha=1.9, beta=beta).to_local(r=0.05)


@pytest.mark.parametrize(
    "instance",
    [
        StableParams(alpha=1.9, beta=0.0),
        LocalCoordinates(r=0.1, h=1.0, p=0.5),
        SignedTailGap(r=0.1, tau_plus=0.05, tau_minus=0.05),
    ],
)
def test_public_parameter_objects_are_immutable(instance: object) -> None:
    attribute = "alpha" if isinstance(instance, StableParams) else "r"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(instance, attribute, 2.0)


def test_public_dataclass_fields_never_use_delta() -> None:
    for parameter_type in (StableParams, LocalCoordinates, SignedTailGap):
        assert "delta" not in {field.name for field in fields(parameter_type)}


@given(
    r=st.floats(min_value=1e-6, max_value=0.5, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=1e-4, max_value=3.0, allow_nan=False, allow_infinity=False),
    p=st.floats(
        min_value=1e-5, max_value=0.99999, allow_nan=False, allow_infinity=False
    ),
    loc=st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
def test_hypothesis_round_trips_preserve_alpha_beta_and_r(
    r: float, h: float, p: float, loc: float, scale: float
) -> None:
    assume(r * h < 1.9)
    local = LocalCoordinates(r=r, h=h, p=p)
    conventional = local.to_stable(loc=loc, scale=scale)
    rebuilt = conventional.to_local(r=r)
    assert rebuilt.r == r
    assert isclose(rebuilt.alpha, local.alpha, rel_tol=2e-14, abs_tol=2e-14)
    assert isclose(rebuilt.beta, local.beta, rel_tol=2e-14, abs_tol=2e-14)


@given(
    r=st.floats(min_value=1e-5, max_value=0.4, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=1e-3, max_value=3.0, allow_nan=False, allow_infinity=False),
    p=st.floats(
        min_value=1e-4, max_value=0.9999, allow_nan=False, allow_infinity=False
    ),
)
def test_hypothesis_reflection_swaps_signed_gaps(r: float, h: float, p: float) -> None:
    assume(r * h < 1.9)
    original = LocalCoordinates(r=r, h=h, p=p).to_signed_tail_gap()
    reflected = LocalCoordinates(r=r, h=h, p=1.0 - p).to_signed_tail_gap()
    assert isclose(
        original.tau_plus,
        reflected.tau_minus,
        rel_tol=2e-14,
        abs_tol=1e-15,
    )
    assert isclose(
        original.tau_minus,
        reflected.tau_plus,
        rel_tol=2e-14,
        abs_tol=1e-15,
    )
