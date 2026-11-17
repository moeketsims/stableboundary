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
Three stable densities are checked by independent characteristic-function
inversion. All three have `alpha` between 1.90 and 1.94 and `|x| <= 2.5`, inside
the central cell of that design. This test does not cover observations beyond
the design threshold, heavier tails, or the limits listed below. The
grouped-posterior oracle uses independently implemented 48/64-node quadrature
and Fourier checks.

Full and grouped fits share the production SciPy backend and posterior engine.
Agreement between them cannot rule out a shared numerical defect. Refinement
diagnostics are empirical convergence evidence, not rigorous error bounds.

## Known backend limits

Before release, the SciPy 1.18.1 piecewise method behind both fitters was
compared with an independent characteristic-function inversion of the S0 law
(Fourier inversion for the density, Gil-Pelaez inversion for the tails). These
were spot checks on finite grids on one Windows AMD64 machine. They are not
error bounds, and they are not part of the test suite. Write
`zeta = -beta * tan(pi * alpha / 2)`.

- **Alpha near 1.** For `|alpha - 1| < 0.005` SciPy evaluates the law at
  `alpha = 1` (the recorded `piecewise_alpha_tol_near_one` setting). Just
  outside that band, for `alpha` of about 0.9944 to 0.9950 and 1.0052 to 1.0056,
  it returned NaN for some inputs and finite but wrong values for others. For
  example, the log density at `x = -2`, `alpha = 1.0055`, `beta = 0.5` came back
  as -36.41 against -3.19.
- **Observations near zeta, alpha below 1.10.** From `alpha` of about 1.006 to
  1.09, the log density was low by about 1 when an observation lay in a window
  around `zeta`. The window was several scale units wide at `alpha = 1.006` and
  under 0.01 wide at `alpha = 1.09`, and wider for larger `|beta|`.
- **The guard.** With the SciPy backend, `fit_known_nuisance()` and
  `fit_full_known_nuisance()` refuse a prior whose support reaches
  `alpha < 1.10`. They raise `ValidationError` before any likelihood
  evaluation, so the outcome does not depend on where quadrature nodes fall.
  The default prior needs `n >= 54`. A smaller `h_max` keeps a smaller sample
  in range. `simulate()` is not guarded and was not checked near `alpha = 1`.
- **Observations near zeta, alpha of 1.10 or more.** SciPy evaluates an
  observation within `0.005 * alpha**(1/alpha)` scale units of `zeta` at `zeta`
  (the recorded `piecewise_x_tol_near_zeta` setting). At `alpha = 1.82`,
  `beta = 0.2` this changed the log density of an affected observation by up to
  2.7e-4. At `alpha` of 1.10, 1.11 and 1.12 with `|beta| >= 0.7`, tail
  probabilities were also held at their value at `zeta` up to about twice that
  distance, a log-tail error of up to 2e-3; this was no longer seen at
  `alpha = 1.15`. Outside these windows, the density and tails agreed with the
  independent inversion to within 1e-6 in the log at every point tested with
  `alpha` from 1.10 to 1.99, `|beta| <= 0.999` and `|x| <= 100`.

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
- `NumericalProbabilityError` means the backend failed or returned an invalid
  value. More quadrature nodes will not fix it. `ValidationError` naming the
  `alpha` limit means the prior reaches the region described under "Known
  backend limits".
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
