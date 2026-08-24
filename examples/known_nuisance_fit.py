"""Run the fixed-seed, known-nuisance stableboundary example."""

from __future__ import annotations

import json
from typing import Any

import stableboundary as sb

SAMPLE_SIZE = 5_000
SEED = 20_260_824
KNOWN_LOC = 0.0
KNOWN_SCALE = 1.0


def run_example() -> dict[str, Any]:
    """Simulate and fit one exact finite-cell posterior through the public API."""
    design = sb.LocalDesign.from_sample_size(SAMPLE_SIZE)
    truth = sb.StableParams(
        alpha=2.0 - design.r * 1.5,
        beta=0.35,
        loc=KNOWN_LOC,
        scale=KNOWN_SCALE,
    )
    observations = sb.simulate(truth, size=SAMPLE_SIZE, random_state=SEED)
    fit = sb.fit_known_nuisance(
        observations,
        loc=KNOWN_LOC,
        scale=KNOWN_SCALE,
        design=design,
        prior=sb.LocalPrior.default(design),
        provenance="fixed by the simulation design",
        quadrature=sb.QuadratureConfig(
            base_nodes=20,
            refined_nodes=32,
            refinement_tolerance=0.002,
            common_grid_points=65,
        ),
    )

    summary = fit.summary()
    audit = fit.audit_record()
    return {
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
        "design": audit["design"],
        "counts": summary["counts"],
        "parameters": summary["parameters"],
        "posterior_mass": float(fit.posterior.mass.sum()),
        "identification": summary["identification"],
        "refinement": audit["refinement"],
        "warnings": summary["warnings"],
    }


def main() -> None:
    """Print a JSON-safe summary for users and artifact smoke tests."""
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
