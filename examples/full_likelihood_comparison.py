"""Reproduce small full-versus-reduced experiments; failures remain in the report."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

import stableboundary as sb


def run_case(n: int, seed: int) -> dict[str, object]:
    design = sb.LocalDesign.from_sample_size(n)
    prior = sb.LocalPrior(design, h_min=0.25, h_max=1, p_min=0.25, p_max=0.75)
    truth = sb.StableParams(alpha=2 - design.r * 0.6, beta=0.2)
    values = sb.simulate(truth, size=n, random_state=seed)
    record: dict[str, object] = {
        "n": n,
        "seed": seed,
        "truth": {"alpha": truth.alpha, "beta": truth.beta, "loc": 0, "scale": 1},
        "prior": {"h_min": 0.25, "h_max": 1, "p_min": 0.25, "p_max": 0.75},
        "observations": values.tolist(),
        "quadrature": {"base_nodes": 20, "refined_nodes": 32, "tolerance": 0.002},
    }
    fits = {}
    for name, fitter in (
        ("reduced", sb.fit_known_nuisance),
        ("full", sb.fit_full_known_nuisance),
    ):
        started = perf_counter()
        try:
            fit = fitter(values, 0, 1, design, prior, provenance="simulation design")
            fits[name] = fit
            record[name] = {
                "seconds": perf_counter() - started,
                "summary": fit.summary(),
                "audit": fit.audit_record(),
            }
        except sb.StableBoundaryError as error:
            record[name] = {
                "seconds": perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    if len(fits) == 2:
        try:
            record["comparison"] = sb.compare_posteriors(
                fits["full"], fits["reduced"]
            ).summary()
        except sb.StableBoundaryError as error:
            record["comparison"] = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261116])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {
        "scope": "illustrative experiments, not calibration or a performance guarantee",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "stableboundary": sb.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "cases": [],
    }
    cases = []
    for seed in args.seeds:
        cases.append(run_case(args.n, seed))
        report["cases"] = cases
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed n={args.n}, seed={seed}", flush=True)


if __name__ == "__main__":
    main()
