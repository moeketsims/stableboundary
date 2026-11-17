"""Compute the package outputs behind the README figures.

Run from the repository root, with the package installed in the pinned
environment:
  python figures/make_data.py reduced   # seconds
  python figures/make_data.py full      # several minutes

Everything plotted comes from this script. The dataset is one simulated sample
with a seed fixed in advance (the release date). Nothing is tuned afterwards.
The common-grid densities use two private helpers so that the plotted surface is
the one the package itself uses for its summaries and comparisons.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import stableboundary as sb
from stableboundary.backends import ScipyS0Backend
from stableboundary.posterior import _common_grid_density
from stableboundary.reference import _retained_evaluation

HERE = Path(__file__).resolve().parent
N = 64
SEED = 20261117
TRUE_H, TRUE_P = 1.5, 0.65
GRID_POINTS = 129
NODE_CONFIGS = ((20, 32), (32, 48), (48, 64))


def setup():
    design = sb.LocalDesign.from_sample_size(N)
    prior = sb.LocalPrior.default(design)
    truth = sb.StableParams(alpha=2.0 - design.r * TRUE_H, beta=2.0 * TRUE_P - 1.0)
    observations = sb.simulate(truth, size=N, random_state=SEED)
    return design, prior, truth, observations


def fit_with_refinement(fitter, observations, design, prior):
    """Try node settings in a fixed order; keep the first accepted fit."""
    attempts = []
    for base, refined in NODE_CONFIGS:
        started = time.perf_counter()
        try:
            fit = fitter(
                observations,
                0.0,
                1.0,
                design,
                prior,
                provenance="simulation design",
                quadrature=sb.QuadratureConfig(base_nodes=base, refined_nodes=refined),
            )
        except sb.StableBoundaryError as error:
            attempts.append(
                {
                    "nodes": [base, refined],
                    "seconds": time.perf_counter() - started,
                    "refused": f"{type(error).__name__}: {error}"[:200],
                }
            )
            print("refused", base, refined, attempts[-1]["refused"], flush=True)
            continue
        attempts.append(
            {"nodes": [base, refined], "seconds": time.perf_counter() - started}
        )
        print(
            "accepted",
            base,
            refined,
            round(attempts[-1]["seconds"], 1),
            "s",
            flush=True,
        )
        return fit, attempts
    raise SystemExit("no node setting was accepted")


def summaries(fit):
    out = {}
    for name in ("alpha", "beta"):
        interval = fit.credible_interval(name)
        out[name] = {
            "mean": fit.mean(name),
            "median": fit.median(name),
            "lower": interval.lower,
            "upper": interval.upper,
            "mass": interval.mass,
        }
    return out


def common(fit):
    grid = _common_grid_density(
        _retained_evaluation(fit.posterior), fit.prior, GRID_POINTS
    )
    return grid.h_axis, grid.p_axis, grid.density


def reduced() -> None:
    design, prior, truth, observations = setup()
    fit, attempts = fit_with_refinement(
        sb.fit_known_nuisance, observations, design, prior
    )
    h_axis, p_axis, density = common(fit)
    backend = ScipyS0Backend()
    x = np.linspace(-9.0, 9.0, 721)
    np.savez(
        HERE / "demo_reduced.npz",
        observations=observations,
        h_axis=h_axis,
        p_axis=p_axis,
        density=density,
        x=x,
        logpdf_truth=np.asarray(backend.logpdf(x, truth.alpha, truth.beta)),
        logpdf_gauss=np.asarray(backend.logpdf(x, 2.0, 0.0)),
    )
    meta = {
        "n": N,
        "seed": SEED,
        "r": design.r,
        "threshold": design.threshold,
        "truth": {"alpha": truth.alpha, "beta": truth.beta, "h": TRUE_H, "p": TRUE_P},
        "prior": {
            "h_min": prior.h_min,
            "h_max": prior.h_max,
            "p_min": prior.p_min,
            "p_max": prior.p_max,
        },
        "counts": [fit.counts.n_minus, fit.counts.n_zero, fit.counts.n_plus],
        "reduced": {
            "attempts": attempts,
            "summaries": summaries(fit),
            "status": fit.status,
            "method": fit.method,
        },
        "versions": {"stableboundary": sb.__version__, "numpy": np.__version__},
    }
    (HERE / "demo_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["reduced"]["summaries"], indent=1))


def full() -> None:
    design, prior, truth, observations = setup()
    reduced_fit, _ = fit_with_refinement(
        sb.fit_known_nuisance, observations, design, prior
    )
    full_fit, attempts = fit_with_refinement(
        sb.fit_full_known_nuisance, observations, design, prior
    )
    comparison = sb.compare_posteriors(full_fit, reduced_fit)
    h_axis, p_axis, density = common(full_fit)
    np.savez(HERE / "demo_full.npz", h_axis=h_axis, p_axis=p_axis, density=density)
    meta = {
        "full": {
            "attempts": attempts,
            "summaries": summaries(full_fit),
            "status": full_fit.status,
            "method": full_fit.method,
        },
        "comparison": {
            "total_variation": comparison.total_variation,
            "hellinger": comparison.hellinger,
        },
    }
    (HERE / "demo_full_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1)[:1500])


if __name__ == "__main__":
    {"reduced": reduced, "full": full}[sys.argv[1]]()
