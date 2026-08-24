"""Contracts for prespecified designs, compact priors, and provenance."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from inspect import signature
from math import isclose, log, sqrt

import numpy as np
import pytest
from scipy.special import lambertw

from stableboundary import (
    KnownNuisance,
    LocalCoordinates,
    LocalDesign,
    LocalPrior,
    NuisanceMode,
    ValidationError,
)


@pytest.mark.parametrize(
    ("n", "c"),
    [(64, 0.5), (500, 1.0), (5_000, 1.0), (1_000_000, 2.0)],
)
def test_critical_rate_equation_and_closed_form(n: int, c: float) -> None:
    design = LocalDesign.from_sample_size(n, c)
    expected_r = (8.0 * c / n) * float(lambertw(n / (8.0 * c)).real)
    relative_residual = abs(n * design.r / log(1.0 / design.r) - 8.0 * c) / (8.0 * c)
    assert design.r == expected_r
    assert relative_residual < 1e-12
    assert design.critical_rate_relative_residual == relative_residual


def test_threshold_and_formula_identifiers(local_design: LocalDesign) -> None:
    log_inverse_r = log(1.0 / local_design.r)
    expected = 2.0 * sqrt(log_inverse_r + 2.0 * log(log_inverse_r))
    assert local_design.threshold == expected
    assert local_design.formula_id == LocalDesign.FORMULA_ID
    assert local_design.formula_version == LocalDesign.FORMULA_VERSION


@pytest.mark.parametrize("n", [0, -1, 1, True, 12.5])
def test_invalid_sample_sizes_are_rejected(n: object) -> None:
    with pytest.raises(ValidationError):
        LocalDesign.from_sample_size(n)  # type: ignore[arg-type]


@pytest.mark.parametrize("c", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_design_constants_are_rejected(c: object) -> None:
    with pytest.raises(ValidationError):
        LocalDesign.from_sample_size(5_000, c)  # type: ignore[arg-type]


def test_design_is_immutable_and_not_directly_constructible(
    local_design: LocalDesign,
) -> None:
    with pytest.raises((FrozenInstanceError, AttributeError)):
        local_design.r = 0.5  # type: ignore[misc]
    with pytest.raises(TypeError):
        LocalDesign()  # type: ignore[call-arg]


def test_default_prior_is_proper_and_theorem_interior(
    local_design: LocalDesign, local_prior: LocalPrior
) -> None:
    assert (local_prior.h_min, local_prior.h_max) == (0.25, 4.0)
    assert (local_prior.p_min, local_prior.p_max) == (0.05, 0.95)
    assert local_prior.area > 0.0
    for h in (local_prior.h_min, local_prior.h_max):
        for p in (local_prior.p_min, local_prior.p_max):
            coordinates = LocalCoordinates(r=local_design.r, h=h, p=p)
            assert 0.0 < coordinates.alpha < 2.0


@pytest.mark.parametrize(
    "bounds",
    [
        {"h_min": 0.0},
        {"h_min": 1.0, "h_max": 1.0},
        {"h_min": 2.0, "h_max": 1.0},
        {"p_min": 0.0},
        {"p_max": 1.0},
        {"p_min": 0.8, "p_max": 0.2},
        {"h_max": float("inf")},
    ],
)
def test_invalid_prior_support_is_rejected(
    local_design: LocalDesign, bounds: dict[str, float]
) -> None:
    with pytest.raises(ValidationError):
        LocalPrior(design=local_design, **bounds)


def test_prior_rejects_support_outside_stable_region(
    local_design: LocalDesign,
) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        LocalPrior(design=local_design, h_max=2.0 / local_design.r)


def test_uniform_log_density_is_vectorized(local_prior: LocalPrior) -> None:
    expected = -log(local_prior.area)
    assert local_prior.log_density(1.0, 0.5) == expected
    assert local_prior.log_density(0.1, 0.5) == -np.inf
    values = local_prior.log_density(
        np.array([local_prior.h_min, 1.0, local_prior.h_max + 1.0]),
        np.array([local_prior.p_min, 0.5, 0.5]),
    )
    np.testing.assert_array_equal(values, np.array([expected, expected, -np.inf]))


def test_nuisance_enum_is_closed() -> None:
    assert {mode.value for mode in NuisanceMode} == {
        "externally_known",
        "pilot_conditioned",
        "plugin_estimate",
    }


@pytest.mark.parametrize("mode", list(NuisanceMode))
def test_each_nuisance_mode_can_be_recorded(mode: NuisanceMode) -> None:
    nuisance = KnownNuisance(
        loc=0.0,
        scale=1.0,
        mode=mode,
        provenance=f"recorded as {mode.value}",
    )
    assert nuisance.mode is mode


def test_external_constructor_and_phase_one_acceptance() -> None:
    nuisance = KnownNuisance.externally_known(
        loc=1.5,
        scale=2.0,
        provenance="calibration experiment 2026-08-24",
    )
    nuisance.require_externally_known()
    assert nuisance.mode is NuisanceMode.EXTERNALLY_KNOWN


@pytest.mark.parametrize(
    "mode",
    [NuisanceMode.PILOT_CONDITIONED, NuisanceMode.PLUGIN_ESTIMATE],
)
def test_phase_one_refuses_non_external_nuisance_modes(mode: NuisanceMode) -> None:
    nuisance = KnownNuisance(0.0, 1.0, mode, "recorded for a later workflow")
    with pytest.raises(ValidationError, match="externally known"):
        nuisance.require_externally_known()


@pytest.mark.parametrize(
    "values",
    [
        {"loc": float("nan")},
        {"scale": 0.0},
        {"scale": float("inf")},
        {"mode": "externally_known"},
        {"provenance": "   "},
    ],
)
def test_invalid_nuisance_records_are_rejected(values: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "loc": 0.0,
        "scale": 1.0,
        "mode": NuisanceMode.EXTERNALLY_KNOWN,
        "provenance": "independent calibration",
    }
    arguments.update(values)
    with pytest.raises(ValidationError):
        KnownNuisance(**arguments)  # type: ignore[arg-type]


def test_public_design_records_are_immutable(
    local_prior: LocalPrior,
) -> None:
    nuisance = KnownNuisance.externally_known(
        loc=0.0,
        scale=1.0,
        provenance="independent calibration",
    )
    for instance, attribute in ((local_prior, "h_min"), (nuisance, "scale")):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(instance, attribute, 99.0)


def test_design_construction_has_no_observation_input() -> None:
    assert tuple(signature(LocalDesign.from_sample_size).parameters) == ("n", "c")
    assert isclose(LocalDesign.from_sample_size(5_000).n, 5_000)
