"""Regenerate the independent numerical reference used by artifact smoke tests.

This script deliberately does not import the package under test.  It evaluates
the declared S0 three-cell likelihood directly with a private SciPy stable-law
generator, compares selected tail cells with Gil--Pelaez inversion of the S0
characteristic function, and checks convergence from tensor orders 48 to 64.

The package's retained 32-node summaries are regression evidence.  This
higher-order calculation is a separate accuracy reference: it is expected to
agree within the package's declared refinement budget, not bit-for-bit.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import gammaln, lambertw, roots_legendre, xlogy
from scipy.stats import levy_stable

SAMPLE_SIZE = 5_000
COUNTS = (1, 4_996, 3)
H_BOUNDS = (0.25, 4.0)
P_BOUNDS = (0.05, 0.95)
REFERENCE_ORDERS = (48, 64)
PROBABILITIES = (0.05, 0.5, 0.95)
# These per-field limits separate the package's 32-node interpolation protocol
# from the independent 64-node PCHIP calculation.  Each limit narrowly clears
# the retained implementation/reference difference and remains below the
# package's 0.002 refinement budget.  Keeping them in this generator makes any
# future broadening visible in both code review and ``--check`` output.
ACCURACY_TOLERANCES: dict[str, Any] = {
    "log_normalizer": 2e-10,
    "posterior_mass": 1e-12,
    "parameters": {
        "h": {"mean": 2e-10, "lower": 5e-4, "median": 1.3e-3, "upper": 5e-4},
        "p": {"mean": 2e-10, "lower": 7e-5, "median": 2e-5, "upper": 4e-5},
        "alpha": {
            "mean": 2e-12,
            "lower": 1e-5,
            "median": 1e-5,
            "upper": 1e-5,
        },
        "beta": {
            "mean": 2e-10,
            "lower": 1.5e-4,
            "median": 3e-5,
            "upper": 7e-5,
        },
        "tau_plus": {
            "mean": 2e-12,
            "lower": 2e-6,
            "median": 7e-6,
            "upper": 6e-6,
        },
        "tau_minus": {
            "mean": 2e-12,
            "lower": 3e-7,
            "median": 4e-6,
            "upper": 3e-6,
        },
    },
    "identification": {
        "p_kl_divergence": 2e-10,
        "p_interval_width_contraction": 5e-5,
    },
}


def _design() -> tuple[float, float]:
    solution = float(lambertw(SAMPLE_SIZE / 8.0).real)
    r_value = 8.0 * solution / SAMPLE_SIZE
    log_inverse = math.log(1.0 / r_value)
    threshold = 2.0 * math.sqrt(log_inverse + 2.0 * math.log(log_inverse))
    return r_value, threshold


def _private_s0() -> Any:
    distribution = type(levy_stable)(name="_stableboundary_independent_oracle")
    settings: dict[str, object] = {
        "parameterization": "S0",
        "pdf_default_method": "piecewise",
        "cdf_default_method": "piecewise",
        "quad_eps": 1.2e-14,
        "piecewise_x_tol_near_zeta": 0.005,
        "piecewise_alpha_tol_near_one": 0.005,
        "pdf_fft_min_points_threshold": None,
        "pdf_fft_grid_spacing": 0.001,
        "pdf_fft_n_points_two_power": None,
        "pdf_fft_interpolation_level": 3,
        "pdf_fft_interpolation_degree": 3,
    }
    for name, value in settings.items():
        setattr(distribution, name, value)
    return distribution


def _axis(lower: float, upper: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(order)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return midpoint + half_width * nodes, half_width * weights


def _log_density(
    h_values: np.ndarray,
    p_values: np.ndarray,
    *,
    distribution: Any,
    r_value: float,
    threshold: float,
) -> np.ndarray:
    alpha = 2.0 - r_value * h_values
    beta = 2.0 * p_values - 1.0
    q_minus = np.asarray(distribution.cdf(-threshold, alpha, beta), dtype=np.float64)
    # Exact S0 reflection computes the upper cell as a lower tail.
    q_plus = np.asarray(distribution.cdf(-threshold, alpha, -beta), dtype=np.float64)
    q_zero = 1.0 - q_minus - q_plus
    if np.any(q_minus <= 0.0) or np.any(q_plus <= 0.0) or np.any(q_zero <= 0.0):
        raise RuntimeError("independent cell probabilities left the simplex")
    n_minus, n_zero, n_plus = COUNTS
    coefficient = float(
        gammaln(SAMPLE_SIZE + 1)
        - gammaln(n_minus + 1)
        - gammaln(n_zero + 1)
        - gammaln(n_plus + 1)
    )
    log_likelihood = coefficient + (
        xlogy(n_minus, q_minus) + xlogy(n_zero, q_zero) + xlogy(n_plus, q_plus)
    )
    prior_area = (H_BOUNDS[1] - H_BOUNDS[0]) * (P_BOUNDS[1] - P_BOUNDS[0])
    return np.asarray(log_likelihood - math.log(prior_area), dtype=np.float64)


def _pchip_quantiles(axis: np.ndarray, density: np.ndarray) -> list[float]:
    interpolator = PchipInterpolator(axis, density, extrapolate=False)
    antiderivative = interpolator.antiderivative()
    lower = float(axis[0])
    upper = float(axis[-1])
    total = float(antiderivative(upper) - antiderivative(lower))
    return [
        float(
            brentq(
                lambda value, target=probability: float(
                    (antiderivative(value) - antiderivative(lower)) / total - target
                ),
                lower,
                upper,
                xtol=5e-15,
                rtol=1e-14,
            )
        )
        for probability in PROBABILITIES
    ]


def _tau_quantiles(
    *,
    h_nodes: np.ndarray,
    h_weights: np.ndarray,
    p_axis: np.ndarray,
    row_density: np.ndarray,
    r_value: float,
    positive: bool,
) -> list[float]:
    antiderivatives = [
        PchipInterpolator(p_axis, row).antiderivative() for row in row_density
    ]
    p_lower = float(p_axis[0])
    p_upper = float(p_axis[-1])
    row_totals = np.array(
        [float(item(p_upper) - item(p_lower)) for item in antiderivatives]
    )
    normalizer = float(np.sum(h_weights * row_totals))

    def probability(value: float) -> float:
        contributions: list[float] = []
        for h_value, antiderivative, row_total in zip(
            h_nodes, antiderivatives, row_totals, strict=True
        ):
            raw_cut = value / (r_value * float(h_value))
            cut = raw_cut if positive else 1.0 - raw_cut
            if positive:
                if cut <= p_lower:
                    contribution = 0.0
                elif cut >= p_upper:
                    contribution = float(row_total)
                else:
                    contribution = float(antiderivative(cut) - antiderivative(p_lower))
            elif cut <= p_lower:
                contribution = float(row_total)
            elif cut >= p_upper:
                contribution = 0.0
            else:
                contribution = float(antiderivative(p_upper) - antiderivative(cut))
            contributions.append(contribution)
        return float(np.sum(h_weights * np.asarray(contributions)) / normalizer)

    allocation_bounds = P_BOUNDS if positive else (1.0 - P_BOUNDS[1], 1.0 - P_BOUNDS[0])
    support = (
        r_value * H_BOUNDS[0] * allocation_bounds[0],
        r_value * H_BOUNDS[1] * allocation_bounds[1],
    )
    return [
        float(
            brentq(
                lambda value, quantile=target: probability(value) - quantile,
                support[0],
                support[1],
                xtol=5e-15,
                rtol=1e-13,
            )
        )
        for target in PROBABILITIES
    ]


def _summary(mean: float, quantiles: list[float]) -> dict[str, float]:
    return {
        "mean": mean,
        "lower": quantiles[0],
        "median": quantiles[1],
        "upper": quantiles[2],
    }


def _tensor_reference(order: int) -> dict[str, Any]:
    r_value, threshold = _design()
    distribution = _private_s0()
    h_nodes, h_weights = _axis(*H_BOUNDS, order)
    p_nodes, p_weights = _axis(*P_BOUNDS, order)
    h_grid, p_grid = np.meshgrid(h_nodes, p_nodes, indexing="ij")
    log_density = _log_density(
        h_grid,
        p_grid,
        distribution=distribution,
        r_value=r_value,
        threshold=threshold,
    )
    shift = float(np.max(log_density))
    density = np.exp(log_density - shift)
    weighted = density * np.multiply.outer(h_weights, p_weights)
    scaled_normalizer = float(np.sum(weighted))
    mass = weighted / scaled_normalizer

    h_axis = np.concatenate(([H_BOUNDS[0]], h_nodes, [H_BOUNDS[1]]))
    p_axis = np.concatenate(([P_BOUNDS[0]], p_nodes, [P_BOUNDS[1]]))
    h_marginal = np.array(
        [
            np.sum(
                np.exp(
                    _log_density(
                        np.full_like(p_nodes, h_value),
                        p_nodes,
                        distribution=distribution,
                        r_value=r_value,
                        threshold=threshold,
                    )
                    - shift
                )
                * p_weights
            )
            for h_value in h_axis
        ],
        dtype=np.float64,
    )
    p_marginal = np.array(
        [
            np.sum(
                np.exp(
                    _log_density(
                        h_nodes,
                        np.full_like(h_nodes, p_value),
                        distribution=distribution,
                        r_value=r_value,
                        threshold=threshold,
                    )
                    - shift
                )
                * h_weights
            )
            for p_value in p_axis
        ],
        dtype=np.float64,
    )
    h_quantiles = _pchip_quantiles(h_axis, h_marginal)
    p_quantiles = _pchip_quantiles(p_axis, p_marginal)

    endpoint_p_grid = np.broadcast_to(p_axis, (order, p_axis.size))
    endpoint_h_grid = np.broadcast_to(h_nodes[:, None], endpoint_p_grid.shape)
    row_density = np.exp(
        _log_density(
            endpoint_h_grid,
            endpoint_p_grid,
            distribution=distribution,
            r_value=r_value,
            threshold=threshold,
        )
        - shift
    )
    tau_plus_quantiles = _tau_quantiles(
        h_nodes=h_nodes,
        h_weights=h_weights,
        p_axis=p_axis,
        row_density=row_density,
        r_value=r_value,
        positive=True,
    )
    tau_minus_quantiles = _tau_quantiles(
        h_nodes=h_nodes,
        h_weights=h_weights,
        p_axis=p_axis,
        row_density=row_density,
        r_value=r_value,
        positive=False,
    )

    h_mean = float(np.sum(mass * h_grid))
    p_mean = float(np.sum(mass * p_grid))
    alpha_mean = float(np.sum(mass * (2.0 - r_value * h_grid)))
    beta_mean = float(np.sum(mass * (2.0 * p_grid - 1.0)))
    tau_plus_mean = float(np.sum(mass * (r_value * h_grid * p_grid)))
    tau_minus_mean = float(np.sum(mass * (r_value * h_grid * (1.0 - p_grid))))
    posterior_p_density = (
        np.sum(density * h_weights[:, None], axis=0) / scaled_normalizer
    )
    prior_p_density = 1.0 / (P_BOUNDS[1] - P_BOUNDS[0])
    p_kl = float(
        np.sum(
            p_weights
            * xlogy(posterior_p_density, posterior_p_density / prior_p_density)
        )
    )
    contraction = 1.0 - (p_quantiles[2] - p_quantiles[0]) / (
        0.9 * (P_BOUNDS[1] - P_BOUNDS[0])
    )
    return {
        "order": order,
        "design": {"r": r_value, "threshold": threshold},
        "log_normalizer": float(math.log(scaled_normalizer) + shift),
        "posterior_mass": float(np.sum(mass)),
        "parameters": {
            "h": _summary(h_mean, h_quantiles),
            "p": _summary(p_mean, p_quantiles),
            "alpha": _summary(
                alpha_mean,
                [
                    2.0 - r_value * h_quantiles[2],
                    2.0 - r_value * h_quantiles[1],
                    2.0 - r_value * h_quantiles[0],
                ],
            ),
            "beta": _summary(beta_mean, [2.0 * value - 1.0 for value in p_quantiles]),
            "tau_plus": _summary(tau_plus_mean, tau_plus_quantiles),
            "tau_minus": _summary(tau_minus_mean, tau_minus_quantiles),
        },
        "identification": {
            "p_kl_divergence": p_kl,
            "p_interval_width_contraction": contraction,
        },
    }


def _s0_characteristic_function(value: float, alpha: float, beta: float) -> complex:
    exponent = -(value**alpha) * (
        1.0
        + 1j * beta * math.tan(math.pi * alpha / 2.0) * (value ** (1.0 - alpha) - 1.0)
    )
    return cmath.exp(exponent)


def _fourier_cdf(x_value: float, alpha: float, beta: float) -> float:
    epsilon = 1e-8

    def integrand(value: float) -> float:
        transformed = cmath.exp(-1j * value * x_value) * _s0_characteristic_function(
            value, alpha, beta
        )
        return transformed.imag / value

    near_zero = epsilon * integrand(0.5 * epsilon)
    integral, _ = quad(
        integrand,
        epsilon,
        12.0,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=500,
        points=(0.01, 0.1, 1.0, 4.0, 8.0),
    )
    return 0.5 - (near_zero + integral) / math.pi


def _fourier_checks() -> dict[str, Any]:
    r_value, threshold = _design()
    distribution = _private_s0()
    anchors = ((0.25, 0.05), (2.125, 0.5), (4.0, 0.95))
    records: list[dict[str, float]] = []
    for h_value, p_value in anchors:
        alpha = 2.0 - r_value * h_value
        beta = 2.0 * p_value - 1.0
        scipy_value = float(distribution.cdf(-threshold, alpha, beta))
        fourier_value = _fourier_cdf(-threshold, alpha, beta)
        records.append(
            {
                "h": h_value,
                "p": p_value,
                "scipy_cdf": scipy_value,
                "fourier_cdf": fourier_value,
                "absolute_difference": abs(scipy_value - fourier_value),
            }
        )
    maximum = max(record["absolute_difference"] for record in records)
    if maximum > 5e-11:
        raise RuntimeError(f"Fourier/SciPy cell disagreement is too large: {maximum}")
    return {"anchors": records, "maximum_absolute_difference": maximum}


def _numeric_paths(value: object, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for name, nested in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            result.update(_numeric_paths(nested, path))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def _convergence(low: dict[str, Any], high: dict[str, Any]) -> dict[str, float]:
    low_values = _numeric_paths(low)
    high_values = _numeric_paths(high)
    ignored = {"order"}
    return {
        name: abs(high_values[name] - low_values[name])
        for name in sorted(high_values.keys() & low_values.keys())
        if name not in ignored
    }


def generate() -> dict[str, Any]:
    low = _tensor_reference(REFERENCE_ORDERS[0])
    high = _tensor_reference(REFERENCE_ORDERS[1])
    differences = _convergence(low, high)
    if max(differences.values()) > 2e-5:
        raise RuntimeError("independent tensor orders did not converge within 2e-5")
    return {
        "method": (
            "Independent SciPy S0 cell evaluation; 48/64 tensor Gauss-Legendre "
            "quadrature; monotone PCHIP marginal and conditional-CDF inversion"
        ),
        "orders": list(REFERENCE_ORDERS),
        "reference": high,
        "order_absolute_differences": differences,
        "fourier_cell_crosscheck": _fourier_checks(),
    }


def _compact_reference(generated: dict[str, Any]) -> dict[str, Any]:
    differences = generated["order_absolute_differences"]
    fourier = generated["fourier_cell_crosscheck"]
    if not isinstance(differences, dict) or not isinstance(fourier, dict):
        raise RuntimeError("generated oracle has an invalid internal schema")
    return {
        "method": generated["method"],
        "orders": generated["orders"],
        "reference": generated["reference"],
        "accuracy_tolerances": ACCURACY_TOLERANCES,
        "maximum_order_absolute_difference": max(differences.values()),
        "maximum_fourier_cell_absolute_difference": fourier[
            "maximum_absolute_difference"
        ],
    }


def _compare(
    expected: object, actual: object, *, tolerance: float, path: str = ""
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise RuntimeError(f"oracle schema mismatch at {path or '<root>'}")
        for name in expected:
            _compare(
                expected[name],
                actual[name],
                tolerance=tolerance,
                path=f"{path}.{name}" if path else name,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise RuntimeError(f"oracle list mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, tolerance=tolerance, path=f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(
            float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance
        ):
            raise RuntimeError(
                f"oracle numeric mismatch at {path}: {expected} != {actual}"
            )
        return
    if expected != actual:
        raise RuntimeError(
            f"oracle value mismatch at {path}: {expected!r} != {actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    arguments = parser.parse_args()
    generated = generate()
    if arguments.check is not None:
        stored = json.loads(arguments.check.read_text(encoding="utf-8"))
        expected = stored.get("independent_reference")
        if not isinstance(expected, dict):
            raise RuntimeError("stored oracle has no independent_reference")
        _compare(expected, _compact_reference(generated), tolerance=5e-12)
    print(json.dumps(generated, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
