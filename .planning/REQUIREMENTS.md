# Requirements: stableboundary

**Defined:** 2026-08-24  
**Core Value:** Users can fit a near-Gaussian stable model and trust that fast
reduced inference is never presented without an explicit scope, numerical
status, and fallback decision.

## v1 Requirements

### Package Foundation

- [ ] **PKG-01**: A user can install `stableboundary` from a wheel or source
  distribution into a clean Python 3.12+ environment.
- [ ] **PKG-02**: A user can import a documented, typed public API from the
  `stableboundary` namespace without importing research scripts or private
  numerical modules.
- [ ] **PKG-03**: A maintainer can run formatting, linting, type checks, unit
  tests, build checks, and documentation examples through documented commands.
- [ ] **PKG-04**: Continuous integration tests supported Python versions on
  Windows, Linux, and macOS and verifies both wheel and source installations.

### Parameters and Design

- [ ] **PAR-01**: A user can construct validated Nolan `S0` parameters
  `(alpha, beta, loc, scale)` and receives a structured error for values outside
  their mathematical domains.
- [ ] **PAR-02**: A user can convert between stable parameters, local
  coordinates `(r,h,p)`, and signed tail gaps without naming ambiguity or loss
  of required design context.
- [ ] **PAR-03**: A result at or effectively indistinguishable from
  `alpha = 2` reports beta as unidentified rather than returning a conventional
  skewness estimate.
- [ ] **DES-01**: A user can construct a prespecified `LocalDesign` from sample
  size, local-rate constant, compact prior support, and threshold rule, with
  the derived `r`, threshold, and supported alpha range exposed.
- [ ] **DES-02**: The package refuses designs that include unsupported exact
  Gaussian, one-sided, invalid, or numerically unresolved parameter regions.
- [ ] **DES-03**: The package records whether location and scale are externally
  known, pilot-conditioned, or unsupported plug-in estimates.

### Stable Probability Numerics

- [ ] **NUM-01**: A user can evaluate finite `S0` stable log densities and cell
  probabilities through a package-owned numerical protocol without import-time
  mutation of SciPy global configuration.
- [ ] **NUM-02**: Exact finite-cell probabilities are finite, nonnegative, and
  normalized within a declared tolerance or the computation stops with a
  structured numerical error.
- [ ] **NUM-03**: Selected density and probability values are checked against an
  independent guarded SciPy `S0` backend and disagreements beyond tolerance are
  preserved as failures rather than clipped or renormalized away.
- [ ] **NUM-04**: The package evaluates extreme signed probabilities without
  avoidable `1-cdf` cancellation and records the numerical method and tolerance.
- [ ] **NUM-05**: At `alpha = 2`, backend validation agrees with a Gaussian
  having mean `loc` and standard deviation `sqrt(2) * scale`.

### Theorem-Faithful Known-Nuisance Fit

- [ ] **FIT-01**: A user can simulate reproducible `S0` data from declared
  stable parameters and a random seed.
- [ ] **FIT-02**: A user with independently known location and scale can call
  `fit_known_nuisance()` and obtain the exact finite three-cell multinomial
  posterior for `(h,p)` using deterministic two-dimensional quadrature.
- [ ] **FIT-03**: The known-nuisance result returns normalized posterior weights,
  posterior intervals for `h`, `p`, `alpha`, `beta`, and signed tail gaps, and
  the observed cell counts.
- [ ] **FIT-04**: A fit with zero total exceedances marks beta as
  prior-dominated or unidentified; a one-sided positive count reports
  one-sided evidence and quantitative information diagnostics without claiming
  validated precise identification.
- [ ] **FIT-05**: A user can request posterior predictive draws, quantiles,
  signed tail probabilities, and expected exceedance counts from the full
  stable sampling model.
- [ ] **FIT-06**: A result does not report a finite predictive variance when its
  posterior includes `alpha < 2`.
- [ ] **FIT-07**: The limiting Poisson/Gamma-Beta posterior is available only as
  an explicitly named approximation or initialization method, never as the
  silent finite-sample default.

### Full Reference and Accuracy Decisions

- [ ] **REF-01**: A user can compute a full stable-likelihood posterior under the
  same local prior for validation-sized known-nuisance datasets.
- [ ] **REF-02**: Validation code can compute full-versus-grouped posterior total
  variation and Hellinger diagnostics on a common deterministic parameter grid.
- [ ] **SAFE-01**: Every fit has exactly one machine-readable status:
  `reduced_safe`, `full_fallback`, `refused`, or `research_uncertified`, with a
  structured reason and no silent method switch.
- [ ] **SAFE-02**: An unsupported reduction either invokes the explicitly
  configured full-likelihood fallback or raises a documented refusal error.
- [ ] **SAFE-03**: The accuracy-assessment object states its exact target and
  scope and cannot emit `certified` from simulation calibration alone.
- [ ] **CERT-01**: Before the public certified mode is enabled, the package
  implements a proved finite-sample posterior-discrepancy bound with
  conservative enclosures for analytic and numerical remainder terms.
- [ ] **CERT-02**: A certified result does not require the unknown true parameter
  or prior computation of the full posterior and cannot overstate an expected-TV
  guarantee as a dataset-specific guarantee.

### Four-Parameter Empirical Workflow

- [ ] **NUIS-01**: A user can create a seeded pilot/main split whose indices,
  anchor, raw bin edges, and provenance are immutable and auditable.
- [ ] **NUIS-02**: A user can jointly infer `(alpha,beta,loc,scale)` from the
  pilot full likelihood and fixed main-sample grouped likelihood without
  treating pilot nuisance estimates as known.
- [ ] **NUIS-03**: Central grouped cells carry location-scale information and
  the main observations are not re-binned as candidate parameters change.
- [ ] **NUIS-04**: The four-parameter workflow remains explicitly experimental
  until simulation-based calibration, coverage, numerical, and full-posterior
  comparisons pass prespecified tests.
- [ ] **NUIS-05**: A user receives conventional four-parameter summaries together
  with signed tail gaps and beta-identification diagnostics.

### Validation, Performance, and Empirical Use

- [ ] **VAL-01**: Automated tests cover parameter invariants, reflection,
  probability normalization, location-scale equivariance, Gaussian-boundary
  behavior, posterior normalization, serialization, and fallback/refusal paths.
- [ ] **VAL-02**: A fixed-seed automated example simulates data and completes an
  end-to-end installed-package fit with finite posterior summaries.
- [ ] **VAL-03**: Scheduled validation performs simulation-based calibration,
  coverage, posterior-distance, prior-sensitivity, and threshold-sensitivity
  experiments over prespecified regimes.
- [ ] **VAL-04**: Negative controls include Gaussian, Student-t, contaminated
  Gaussian, tempered-tail, dependent, and stochastic-volatility data and test
  whether unsupported claims are refused.
- [ ] **PERF-01**: Benchmarks include planning, assessment, posterior fitting,
  and fallback costs and do not report kernel-only speedups.
- [ ] **PERF-02**: At least one prespecified nontrivial regime with
  `n <= 250000` achieves at least 90% reduced invocation, conditional mean TV
  at most 0.05, 90th-percentile TV at most 0.10, false-safe rate at most 5%, and
  at least tenfold end-to-end speedup over the full posterior.
- [ ] **EMP-01**: A complete example fits a scientifically qualified empirical
  IID sample or standardized residual series with documented provenance,
  preprocessing, assumptions, and refusal checks.
- [ ] **DOC-01**: Documentation explains installation, first fit, `S0`
  conventions, local coordinates, priors, identification, prediction, audit
  records, fallback, refusal, and the limits of certification.
- [ ] **AUD-01**: Every result exports a versioned non-pickle audit bundle
  containing the analysis plan, versions, seed, data fingerprint, prior,
  thresholds, cells, tolerances, diagnostics, status, and fallback reason.

## v2 Requirements

### Language Interoperability

- **RINT-01**: An R user can call the validated Python numerical core through a
  stable language-neutral result and audit schema.
- **RINT-02**: An R-facing vignette reproduces the canonical simulated fit
  without reimplementing the stable numerical kernel.

### Extended Models

- **EXT-01**: A Gaussian spike-and-slab model can represent exact Gaussianity
  with beta absent under the Gaussian component after separate theoretical and
  computational validation.
- **EXT-02**: One-sided boundary regions can be fitted after theory and
  numerical tests cover vanishing signed intensity.
- **EXT-03**: Regression or time-series adapters can fit stable innovation
  distributions while modeling dependence and heteroskedasticity explicitly.
- **EXT-04**: A gated multiscale partition becomes public only after improving
  held-out mean posterior TV by at least 25% in two prespecified pre-asymptotic
  regimes.

## Out of Scope

| Feature | Reason |
|---------|--------|
| General-purpose stable inference far from `alpha=2` | Mature libraries already cover this and the boundary theory does not justify a global superiority claim |
| Exact-Gaussian model probability in v1 | The manuscript excludes `h=0` and beta is absent there |
| Certified same-sample plug-in standardization | It ignores nuisance uncertainty and invalidates the known-nuisance scope |
| Automatic analysis of raw dependent time series | Initial validity is for IID observations or appropriately modeled residuals |
| Silent clipping, probability repair, method switching, or fallback | These practices conceal numerical or statistical failure |
| Separate initial package paper | The software first accompanies the computational method; a later software paper requires demonstrated use |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKG-01 | Phase 1 | Pending |
| PKG-02 | Phase 1 | Pending |
| PKG-03 | Phase 1 | Pending |
| PKG-04 | Phase 9 | Pending |
| PAR-01 | Phase 1 | Pending |
| PAR-02 | Phase 1 | Pending |
| PAR-03 | Phase 1 | Pending |
| DES-01 | Phase 1 | Pending |
| DES-02 | Phase 1 | Pending |
| DES-03 | Phase 1 | Pending |
| NUM-01 | Phase 1 | Pending |
| NUM-02 | Phase 1 | Pending |
| NUM-03 | Phase 2 | Pending |
| NUM-04 | Phase 1 | Pending |
| NUM-05 | Phase 2 | Pending |
| FIT-01 | Phase 1 | Pending |
| FIT-02 | Phase 1 | Pending |
| FIT-03 | Phase 1 | Pending |
| FIT-04 | Phase 1 | Pending |
| FIT-05 | Phase 1 | Pending |
| FIT-06 | Phase 1 | Pending |
| FIT-07 | Phase 1 | Pending |
| REF-01 | Phase 2 | Pending |
| REF-02 | Phase 2 | Pending |
| SAFE-01 | Phase 3 | Pending |
| SAFE-02 | Phase 3 | Pending |
| SAFE-03 | Phase 3 | Pending |
| CERT-01 | Phase 4 | Pending |
| CERT-02 | Phase 4 | Pending |
| NUIS-01 | Phase 5 | Pending |
| NUIS-02 | Phase 5 | Pending |
| NUIS-03 | Phase 5 | Pending |
| NUIS-04 | Phase 6 | Pending |
| NUIS-05 | Phase 5 | Pending |
| VAL-01 | Phase 7 | Pending |
| VAL-02 | Phase 1 | Pending |
| VAL-03 | Phase 7 | Pending |
| VAL-04 | Phase 7 | Pending |
| PERF-01 | Phase 8 | Pending |
| PERF-02 | Phase 8 | Pending |
| EMP-01 | Phase 9 | Pending |
| DOC-01 | Phase 9 | Pending |
| AUD-01 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 43 total
- Mapped to phases: 43
- Unmapped: 0

---
*Requirements defined: 2026-08-24*
*Last updated: 2026-08-24 after roadmap creation*
