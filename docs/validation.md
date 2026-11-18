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
The full-fit reference test checks three stable densities by independent
characteristic-function inversion. All three have `alpha` between 1.90 and 1.94
and `|x| <= 2.5`, inside the central cell of that design; that test does not
cover observations beyond the design threshold or heavier tails. A separate
standalone oracle, `tests/test_backend_oracle.py`, checks log tails at a design
threshold and log densities beyond it for `alpha` in 1.10, 1.5, 1.82 and 1.95
and `beta` in -0.9, 0.2 and 0.9, to 1e-8, and far densities at `|x| = 1e3`
against the large-argument series. These are spot checks at chosen points away
from `zeta`, not an accuracy bound, and they do not cover the limits listed
below. The grouped-posterior oracle uses independently implemented 48/64-node
quadrature and Fourier checks.

Full and grouped fits share the production SciPy backend and posterior engine.
Agreement between them cannot rule out a shared numerical defect. Refinement
diagnostics are empirical convergence evidence, not rigorous error bounds.

## Known backend limits

Before release, the SciPy 1.18.1 piecewise method behind both fitters was
compared with an independent characteristic-function inversion of the S0 law
(Fourier inversion for the density, Gil-Pelaez inversion for the tails). These
were spot checks on finite grids on one Windows AMD64 machine with Python
3.14.7, NumPy 2.5.2 and SciPy 1.18.1. They are not error bounds. The suite
includes a small set of related spot checks, `tests/test_backend_oracle.py`,
intended to run on every CI platform; the rest is not part of the test suite.
Write `zeta = -beta * tan(pi * alpha / 2)`.

- **Alpha near 1.** For `|alpha - 1| < 0.005` SciPy evaluates the law at
  `alpha = 1` (the recorded `piecewise_alpha_tol_near_one` setting). Just
  outside that band, for `alpha` of about 0.9944 to 0.9950 and 1.0052 to 1.0056,
  it returned NaN for some inputs and finite but wrong values for others. For
  example, the log density at `x = -2`, `alpha = 1.0055`, `beta = 0.5` came back
  as -36.41 against -3.19.
- **Observations near zeta, alpha below 1.10.** In spot checks from `alpha`
  1.006 to 1.09, some log densities near `zeta` were approximately 1 too low;
  at `alpha = 1.006` and `beta = 0.7`, some errors were much larger and one
  evaluation was refused. The affected window was several scale units wide in
  some checks at `alpha = 1.006` and under 0.01 wide in checks at
  `alpha = 1.09`; its width depended on `beta`.
- **The alpha guard.** With the SciPy backend, `fit_known_nuisance()`,
  `fit_full_known_nuisance()` and `cells.exact_cell_probabilities()` refuse
  `alpha < 1.10`. For a fit the rule is `2 - design.r * prior.h_max >= 1.10`.
  They raise `ValidationError` before any SciPy evaluation, so the outcome does
  not depend on where quadrature nodes fall. With `c = 1` the default prior
  needs `n >= 54`; `r` depends on `n / c`, so other values of `c` move that
  number. A smaller `h_max` keeps a smaller sample in range. The limit is an
  empirical property of this backend. It is not a statement that the model
  excludes smaller `alpha`, and it is not a certificate of accuracy above 1.10.
- **Simulation.** `simulate()` draws with SciPy's random generator, not with
  the density or tail routines. Version 0.1.0a2 refuses skewed draws when
  0 < |alpha - 1| < 0.0001. At alpha = 1.00000001 and beta = 0.9, the
  empirical characteristic function at unit frequency in 100,000 draws
  was 0.1045 + 0.2658i against the exact exp(-1). Exactly alpha = 1 and
  symmetric draws use distinct cases and remain available. For `alpha` in 0.97, 0.9945,
  0.999, 1.0, 1.001, 1.0055, 1.01, 1.05 and 1.10 and `beta` in 0.5 and -0.9, the
  empirical distribution of 200000 draws differed from the independent
  distribution function by 0.0013 to 0.0018 at most over 11 points. Sampling
  noise at that size is about 0.003, so this check is not designed to resolve
  errors substantially smaller than this sampling variation, and it says
  nothing about far tails.
- **Observations near zeta, alpha of 1.10 or more.** SciPy evaluates an
  observation within `0.005 * alpha**(1/alpha)` scale units of `zeta` at `zeta`
  (the recorded `piecewise_x_tol_near_zeta` setting). At `alpha = 1.82`,
  `beta = 0.2` this changed the log density of an affected observation by up to
  2.7e-4. At `alpha` of 1.10, 1.11 and 1.12 with `|beta| >= 0.7`, tail
  probabilities were also held at their value at `zeta` up to about twice that
  distance, a log-tail error of up to 2e-3; this was no longer seen at
  `alpha = 1.15`. Outside these windows, the density and tails agreed with the
  independent inversion to within 1e-6 in the log at every point tested with
  `alpha` from 1.10 to 1.99, `|beta| <= 0.999` and `|x| <= 100`. In the
  ordinary-grid density check, points at which the independent inversion did
  not resolve its own value to a relative 1e-3 were excluded; in the near-`zeta`
  check with `|beta| <= 0.999` no point was excluded.
- **Extreme observations.** The density loses accuracy as `|x|` grows, sooner
  the closer `alpha` is to 2. The reference was the four-term large-argument
  series, itself within 5e-9 to 2.5e-7 of the Fourier inversion at `|x|` of 30
  to 100 for `alpha` of 1.2 and above. Worst absolute error of the log density over `beta` in -0.9, -0.5, 0,
  0.5 and 0.9 and both signs of `x`:

  | `\|x\|` | alpha 1.5 | 1.82 | 1.95 | 1.99 | 1.998 | 1.9995 |
  |---|---:|---:|---:|---:|---:|---:|
  | 1e3 | 4e-11 | 7e-11 | 8e-11 | 4e-9 | 2e-8 | 2e-7 |
  | 1e4 | 2e-10 | 1e-8 | 7e-9 | 2e-7 | 9e-7 | 3e-6 |
  | 1e5 | 1e-9 | 5e-7 | 4e-6 | 6e-5 | 2e-4 | 5e-3 |
  | 1e6 | 4e-8 | 1e-5 | 1e-4 | 2e-2 | 0.2 | 1 |
  | 1e7 | 1e-6 | 2e-3 | 0.5 | 10 | not run | not run |

  At `|x| = 1e8` the error was 2e-4 at `alpha = 1.5`, 2 at 1.82 and 200 at
  1.95, and some evaluations at 1.99 were refused. At `alpha` 1.1 and 1.2 it
  stayed below 1e-6 from `|x| = 1e3` to `1e8`. No reference point was excluded.
  Refinement checks do not independently validate the density and may fail to
  detect an error of this kind, even when it affects the likelihood.
  `fit_full_known_nuisance()` therefore
  refuses, before any density evaluation, a sample whose largest standardized
  observation exceeds `1e4` scale units. This is an engineering cutoff. It is
  not a proven bound: `alpha` can lie closer to 2 and `|beta|` closer to 1 than
  the grid above, and small per-observation errors add up over `n`. The reduced
  fit uses only cell counts and is not affected.

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
The three recorded cases are prior dominated: each posterior's 90% interval is
89 to 100% as wide as the prior's. They show the machinery running end to end.
They say little about the reduction where the data are informative.

## Interpreting results

- Inspect `status`, warnings, and retained refinement diagnostics before using
  summaries. All successful fits currently report `research_uncertified`.
- `ConvergenceError` means no accepted fit was returned. Increase numerical
  resolution and investigate; do not silently discard failed cases in studies.
- `NumericalProbabilityError` means the calculation could not be completed
  reliably. Investigate the reported input and numerical cause; increasing
  quadrature nodes may change which points are evaluated, but does not establish
  that the underlying backend problem is fixed. A `ValidationError` naming the
  `alpha` limit or the standardized-observation limit means the request reaches
  a region described under "Known backend limits".
- Audit record layouts are method-specific and experimental in this alpha. The
  full-fit and reduced-fit records carry the same `schema_version` but are not
  one common shape.
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

## Predictive checks in 0.1.0a2

For each signed raw threshold, transform to the standardized scale including
location and reflection. Integrate the S0 characteristic function against the
retained posterior masses with 256 and 512 frequency nodes. The alpha accepts
a prediction when both standardized cutoff magnitudes are at most 32, both
Fourier probabilities exceed 1e-9, the two integrations agree within 1e-4
relative, and the SciPy mixture agrees within 1e-3 relative. These are
computational limits for this implementation and empirical checks of the tail
evaluation at the fitted posterior nodes. They do not bound error in posterior
quadrature or establish a uniform relative-error theorem.

For predictive quantiles the smallest of eight batches must have at least five
expected draws in the nearer tail. With tail probability q and batch size m,
the probability of seeing no draw in that tail is (1-q)^m, at most exp(-5)
when mq >= 5. This is a resolution screen. The batch standard error remains
a Monte Carlo estimate.

With zero signed-tail observations, the exact likelihood is the central-cell
probability raised to n. It is symmetric under reversal of beta and can vary
with its magnitude. The exact fit reports no_tail_events and not_assessed.
