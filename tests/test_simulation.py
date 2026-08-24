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
