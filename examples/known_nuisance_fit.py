"""Run the audited known-nuisance fit and separate seeded simulation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

import stableboundary as sb

SAMPLE_SIZE = 5_000
SEED = 20_260_824
KNOWN_LOC = 0.0
KNOWN_SCALE = 1.0


def _canonical_f8(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")


def _counts(values: np.ndarray, threshold: float) -> dict[str, int]:
    n_minus = int(np.count_nonzero(values <= -threshold))
    n_plus = int(np.count_nonzero(values >= threshold))
    return {
        "n_minus": n_minus,
        "n_zero": int(values.size - n_minus - n_plus),
        "n_plus": n_plus,
    }


def run_example() -> dict[str, Any]:
    """Fit a fixed cell witness and separately audit seeded stable simulation."""
    design = sb.LocalDesign.from_sample_size(SAMPLE_SIZE)
    truth = sb.StableParams(
        alpha=2.0 - design.r * 1.5,
        beta=0.35,
        loc=KNOWN_LOC,
        scale=KNOWN_SCALE,
    )
    observations = np.zeros(SAMPLE_SIZE, dtype=np.float64)
    observations[0] = -(design.threshold + 1.0)
    observations[-3:] = design.threshold + 1.0
    fixture_bytes = _canonical_f8(observations)
    fit = sb.fit_known_nuisance(
        observations,
        loc=KNOWN_LOC,
        scale=KNOWN_SCALE,
        design=design,
        prior=sb.LocalPrior.default(design),
        provenance="fixed cell-count witness derived from the prespecified design",
        quadrature=sb.QuadratureConfig(
            base_nodes=20,
            refined_nodes=32,
            refinement_tolerance=0.002,
            common_grid_points=65,
        ),
    )

    summary = fit.summary()
    audit = fit.audit_record()
    simulated = sb.simulate(truth, size=SAMPLE_SIZE, random_state=SEED)
    simulation_bytes = _canonical_f8(simulated)
    return {
        "schema_version": audit["schema_version"],
        "package_version": audit["package_version"],
        "status": summary["status"],
        "method": summary["method"],
        "parameterization": summary["parameterization"],
        "known_nuisance": audit["known_nuisance"],
        "seed": SEED,
        "truth": {
            "alpha": truth.alpha,
            "beta": truth.beta,
            "loc": truth.loc,
            "scale": truth.scale,
        },
        "inference_fixture": {
            "construction": "[-(threshold+1)] + [0]*4996 + [threshold+1]*3",
            "dtype": "<f8",
            "nbytes": len(fixture_bytes),
            "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        },
        "simulation": {
            "dtype": "<f8",
            "rng_algorithm": (
                f"numpy.random.{np.random.default_rng(SEED).bit_generator.__class__.__name__}"
            ),
            "simulator_algorithm": (
                "scipy.stats.levy_stable.rvs:S0:private-generator:v1"
            ),
            "numpy_version": np.__version__,
            "scipy_version": audit["backend"]["library_version"],
            "sample_sha256": hashlib.sha256(simulation_bytes).hexdigest(),
            "counts": _counts(simulated, design.threshold),
            "minimum": float(np.min(simulated)),
            "maximum": float(np.max(simulated)),
        },
        "design": audit["design"],
        "prior": audit["prior"],
        "counts": audit["counts"],
        "quadrature": audit["quadrature"],
        "parameters": summary["parameters"],
        "posterior_mass": float(fit.posterior.mass.sum()),
        "identification": summary["identification"],
        "refinement": audit["refinement"],
        "backend": audit["backend"],
        "warnings": summary["warnings"],
    }


def main() -> None:
    """Print a JSON-safe summary for users and artifact smoke tests."""
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
