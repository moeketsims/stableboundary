# Validation and scientific scope

This alpha provides a reproducible research implementation, not a calibrated
general-purpose four-parameter stable estimator. Supply independently known
location and scale. Estimating them from the same observations and treating
them as fixed is outside the supported model.

## What has been checked

The test suite checks an analytic truncated-normal posterior through the shared
quadrature engine, raw-observation likelihood products, exact duplicate
multiplicities, location-scale transformations including the density Jacobian,
reflection symmetry, normalization, immutable evidence, and numerical refusal.
Selected stable densities are checked by independent characteristic-function
inversion. The grouped-posterior oracle uses independently implemented
48/64-node quadrature and Fourier checks.

Full and grouped fits share the production SciPy backend and posterior engine.
Agreement between them cannot rule out a shared numerical defect. Refinement
diagnostics are empirical convergence evidence, not rigorous error bounds.

## Reproducing the comparison examples

After installation, run from the source archive or repository:

```console
python examples/full_likelihood_comparison.py --n 32 --seeds 20261116 20261117 20261118 --output comparison.json
```

The report retains observations, truth, seeds, priors, runtime versions, audits,
timings, and numerical failures. Each case uses a prespecified compact prior
and the same observations for both fits. It writes after every case so a long
run preserves completed results. Stable log densities are expensive; allow
several minutes per case. Timings cover fitting and its refinement checks,
excluding interpreter startup and simulation, and are machine-specific.

The checked-in `comparison-results.json` is a small illustrative experiment.
It cannot establish repeated-sampling coverage, simulation-based calibration,
uniform approximation accuracy, false-safe rates, or a general speed advantage.
Posterior discrepancy describes that sample under that prior. A large
discrepancy is valid evidence against using the reduction for that case.

## Interpreting results

- Inspect `status`, warnings, and retained refinement diagnostics before using
  summaries. All successful fits currently report `research_uncertified`.
- `ConvergenceError` means no accepted fit was returned. Increase numerical
  resolution and investigate; do not silently discard failed cases in studies.
- The comparison tolerance controls grid-refinement change, not the acceptable
  difference between the statistical models. No automatic safety decision exists.
- The compact prior excludes the exact Gaussian and one-sided shape boundaries.
  This is not model selection between a Gaussian and a stable distribution.
- Full-density and grouped-count normalizers are defined on different data
  spaces and must not be used as a Bayes factor against each other.

## Compatibility

The alpha pins NumPy 2.5.2 and SciPy 1.18.1. This intentionally avoids claiming
validation for every newer release. Use a dedicated virtual environment.
The CI matrix covers Python 3.12, 3.13, and 3.14 on Linux, Windows, and macOS.
Local verification does not substitute for a passing remote matrix on the final
commit. Releases must follow the reviewed-main and tagged-build rules in
`CONTRIBUTING.md`.

## Recorded illustrative results

Local Windows AMD64 run with Python 3.14.7 and the pinned runtime. All three
cases use n=32 and passed the default refinement checks.

| Seed | Reduced fit (s) | Full fit (s) | Total variation | Hellinger |
|---|---:|---:|---:|---:|
| 20261116 | 2.11 | 203.90 | 0.10066 | 0.08840 |
| 20261117 | 5.33 | 232.62 | 0.04001 | 0.03307 |
| 20261118 | 2.50 | 103.48 | 0.19585 | 0.17537 |

These fit-only timings exclude simulation and the posterior comparison. Other
verification processes were running on the same machine, so the timings are
illustrative rather than controlled benchmarks. There is no automatic method
selection or fallback to time. The three selected seeds are reported in full.
