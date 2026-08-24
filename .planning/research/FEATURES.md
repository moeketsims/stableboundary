# Feature Research: stableboundary

**Date:** 2026-08-24  
**Scope:** Python package for Bayesian inference for near-Gaussian univariate
alpha-stable laws

## Product Boundary

`stableboundary` is not another general four-parameter stable-law fitter.
Existing libraries already supply densities, simulation, maximum likelihood,
quantile, regression, and moment estimators. The package is differentiated by
one workflow: determine when a near-Gaussian stable posterior can be reduced to
signed cells, fit that finite-sample posterior, and fall back or refuse when
the reduction is unsupported.

## Table Stakes

| Capability | Observable behavior | Complexity | Dependency |
|------------|---------------------|------------|------------|
| Installable package | An isolated wheel install exposes `stableboundary` without path manipulation | Medium | PEP 517/621 and `src/` layout |
| Explicit `S0` parameters | Validated `alpha`, `beta`, `loc`, and `scale` types; parameterization stored in every result | Medium | Parameter module |
| Coordinate conversion | Lossless conversion among stable parameters, `(r,h,p)`, and signed tail gaps | Low | Parameter module |
| Reproducible simulation | Users can generate seeded `S0` samples | Low | Stable probability backend |
| Known-nuisance fit | Exact finite three-cell posterior for known `loc` and `scale` | High | Design, cells, probability engine, quadrature |
| Posterior summaries | Intervals for `h`, `p`, `alpha`, `beta`, and signed gaps | Medium | Posterior and result modules |
| Identification warnings | Zero total tails label beta prior-dominated; one-sided positive counts report directional evidence and information diagnostics without a validated precision claim | Medium | Diagnostics |
| Numerical diagnostics | Invalid mass, normalization, or backend disagreement raises a structured error | High | Backend protocol |
| Full reference fit | Full `S0` posterior under the same prior is available for validation and fallback | High | Stable log-density and quadrature |
| Explicit method state | Results are `reduced`, `full_fallback`, `refused`, or `research_uncertified`; no silent switch | Medium | Result state machine |
| Reproducible audit | Prior, design, thresholds, counts, tolerances, versions, seed, and fallback reason serialize to JSON | Medium | Versioned audit schema |
| Executable documentation | Installation and first-fit examples run in tests | Medium | Documentation toolchain |

## Differentiators

| Capability | Contribution | Admission rule |
|------------|--------------|----------------|
| Finite-sample reduction bound | Quantifies posterior discrepancy on a declared prior/design region | `certified` unavailable until the bound is proved and conservatively enclosed |
| Pre-data reduction planning | Chooses scale and cells without examining main-sample tail outcomes | Must not use truth or the full posterior |
| Automatic full fallback | Uses full likelihood or refuses outside the supported reduction region | Reason and end-to-end cost are audited |
| Signed tail-gap reporting | Reports quantities meaningful as beta loses identification | Always shown with conventional parameters |
| Posterior-distance benchmark | Calculates full-versus-grouped TV/Hellinger on validation problems | Required before release claims |
| Pilot-conditioned joint fit | Fits `alpha`, `beta`, `loc`, and `scale` from pilot full likelihood plus fixed grouped main likelihood | Experimental until nuisance coverage and posterior-distance tests pass |
| Misspecification refusal | Declines certified IID inference under dependence, nonstationarity, censoring, or numerical invalidity | Negative-control tests required |
| Approximation ladder | Separately names exact grouped, finite-mean Poisson, and limiting Poisson fits | Limiting form is never the default finite-sample method |
| Gated multiscale cells | Adds disjoint cells only when they materially improve posterior fidelity | At least 25% mean-TV improvement in two held-out regimes |

## Anti-Features

| Deliberate exclusion | Reason |
|----------------------|--------|
| One-call general stable fitting over all `alpha` | Erases the validated near-boundary scope and duplicates mature libraries |
| Import-time SciPy global configuration | `levy_stable.parameterization` is mutable process state |
| Threshold tuning on observed extremes | Reuses the evidence being summarized and invalidates prespecification |
| Plug-in nuisance estimates labeled as known | Hides uncertainty and falsely extends the theorem |
| A scalar beta estimate at `alpha=2` | Beta is unidentifiable at the Gaussian point |
| Predictive variance when `alpha<2` | Stable variance is infinite; tail probabilities and quantiles are appropriate |
| Clipping negative computed densities | Turns numerical failure into hidden probability mass |
| Certification from Monte Carlo performance | Calibration is not a proved finite-sample upper bound |
| Raw financial-return workflow | Dependence and stochastic volatility violate the initial IID contract |
| R core before Python stabilization | Doubles numerical risk without adding statistical value |

## Minimal Working Fit

```python
import stableboundary as sb

x = sb.simulate(
    sb.StableParams(alpha=1.97, beta=0.35, loc=0.0, scale=1.0),
    size=5000,
    random_state=20260824,
)

fit = sb.fit_known_nuisance(
    x,
    loc=0.0,
    scale=1.0,
    prior=sb.LocalPrior.default(),
    design=sb.LocalDesign.from_sample_size(len(x)),
)

fit.summary()
fit.audit_record()
```

The fit is successful only if it uses exact finite stable cell probabilities,
normalizes a two-dimensional posterior, returns finite posterior summaries,
records the observed cells and design, and labels itself
`research_uncertified`. A fixed-seed automated test must execute the complete
path.

## Dependency Order

```text
package metadata
  -> parameter/coordinate types
    -> stable probability backend
      -> local design and immutable counts
        -> exact grouped likelihood
          -> deterministic posterior quadrature
            -> result, summary, audit, prediction
              -> full-likelihood reference and comparison
                -> accuracy assessment and fallback
                  -> pilot-conditioned four-parameter workflow
```

The certificate interface may be designed early, but a successful certificate
state cannot be implemented before the mathematical and numerical bounds exist.

## Validation Architecture

### Deterministic properties

- Stable/local/signed coordinate round trips and strict input validation.
- Positive/negative probability reflection under beta sign reversal.
- Finite, nonnegative cell probabilities summing to one within tolerance.
- At `alpha=2`, agreement with a Gaussian having standard deviation
  `sqrt(2) * scale`.

### Numerical cross-checks

- Compare selected PDF, CDF, and cell probabilities with an independent engine.
- Refine integration and posterior grids until declared tolerances are met.
- Never repair invalid values by clipping; preserve failures as fixtures.

### Statistical correctness

- Simulation-based calibration for posterior computation.
- Full-versus-grouped posterior comparisons on prespecified grids.
- Coverage and identification diagnostics for zero, one-sided, and balanced
  tail counts.
- Prior and threshold sensitivity checks.

### Failure detection

- Gaussian, Student-t, contaminated Gaussian, tempered-tail, dependent, and
  stochastic-volatility negative controls.
- Explicit refusal/fallback tests with machine-readable reasons.

### Packaging and reproducibility

- Clean wheel and source installation.
- CI on supported Python versions and major operating systems.
- Documentation examples executed as tests.
- Fixed-seed fit and versioned JSON audit snapshot.

## Research Conclusion

The known-nuisance fit is the correct first vertical slice: it exercises the
theorem's coordinates and exact finite-cell likelihood and can be checked
against a full posterior. Joint four-parameter fitting is essential for applied
use, but it must be layered onto this reference implementation rather than
obscuring failures in the boundary reduction.

## RESEARCH COMPLETE
