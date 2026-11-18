"""Seeded and immutable simulation contracts."""

from __future__ import annotations

import numpy as np
import pytest

import stableboundary as sb


def test_equal_integer_seeds_reproduce_and_changed_seed_differs(
    stable_params: sb.StableParams,
) -> None:
    first = sb.simulate(stable_params, 128, random_state=20260824)
    repeated = sb.simulate(stable_params, 128, random_state=20260824)
    changed = sb.simulate(stable_params, 128, random_state=20260825)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)


def test_simulation_is_finite_one_dimensional_float_and_read_only(
    stable_params: sb.StableParams,
) -> None:
    sample = sb.simulate(stable_params, np.int64(32), random_state=7)
    assert sample.shape == (32,)
    assert sample.dtype == np.float64
    assert np.all(np.isfinite(sample))
    assert not sample.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        sample[0] = 0.0


def test_generator_input_is_reproducible_for_equal_generator_states(
    stable_params: sb.StableParams,
) -> None:
    first = sb.simulate(stable_params, 64, np.random.default_rng(31))
    second = sb.simulate(stable_params, 64, np.random.default_rng(31))
    np.testing.assert_array_equal(first, second)


def test_simulation_does_not_change_numpy_global_random_state(
    stable_params: sb.StableParams,
) -> None:
    incoming = np.random.get_state()
    sb.simulate(stable_params, 16, random_state=11)
    outgoing = np.random.get_state()
    assert incoming[0] == outgoing[0]
    np.testing.assert_array_equal(incoming[1], outgoing[1])
    assert incoming[2:] == outgoing[2:]


@pytest.mark.parametrize("size", [True, 0, -1, 2.5, "10"])
def test_simulation_rejects_invalid_sizes(
    stable_params: sb.StableParams,
    size: object,
) -> None:
    with pytest.raises(sb.ValidationError, match="positive integer"):
        sb.simulate(stable_params, size, random_state=1)  # type: ignore[arg-type]


def test_simulation_rejects_platform_allocation_overflow(
    stable_params: sb.StableParams,
) -> None:
    oversized = np.iinfo(np.intp).max // np.dtype(np.float64).itemsize + 1
    with pytest.raises(sb.ValidationError, match="allocation limit"):
        sb.simulate(stable_params, oversized, random_state=1)


@pytest.mark.parametrize("random_state", [True, -1, "seed"])
def test_simulation_rejects_invalid_random_state(
    stable_params: sb.StableParams,
    random_state: object,
) -> None:
    with pytest.raises(sb.ValidationError, match="random_state"):
        sb.simulate(
            stable_params,
            4,
            random_state=random_state,  # type: ignore[arg-type]
        )


def test_only_planned_simulation_and_cell_types_join_public_facade() -> None:
    assert {"simulate", "CellCounts", "CellProbabilities"} <= set(sb.__all__)
    assert "exact_cell_probabilities" not in sb.__all__
    assert "ScipyS0Backend" not in sb.__all__


@pytest.mark.parametrize("distance", [1e-8, -1e-8, 1e-12, -1e-12, 9e-5, -9e-5])
@pytest.mark.parametrize("beta", [0.9, -0.9])
def test_near_one_skewed_simulation_refuses_before_using_rng(distance, beta):
    rng = np.random.default_rng(412)
    before = rng.bit_generator.state
    with pytest.raises(sb.ValidationError, match="skewed S0 simulation"):
        sb.simulate(sb.StableParams(1 + distance, beta), 100, rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize(
    "alpha,beta", [(1, 0.9), (1 + 1.01e-4, 0.9), (1 - 1.01e-4, -0.9), (1 + 1e-12, 0)]
)
def test_near_one_supported_draws_match_characteristic_function(alpha, beta):
    sample = sb.simulate(sb.StableParams(alpha, beta), 100_000, 123)
    for t in (0.5, 1.0, 2.0):
        phase = (
            -2 / np.pi * t * np.log(t)
            if alpha == 1
            else np.tan(np.pi * alpha / 2) * t * np.expm1((alpha - 1) * np.log(t))
        )
        expected = np.exp(-(t**alpha) + 1j * beta * phase)
        assert abs(np.exp(1j * t * sample).mean() - expected) < 0.015
