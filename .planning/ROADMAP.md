# Roadmap: stableboundary

## Overview

`stableboundary` progresses from one installable, theorem-faithful known-nuisance fit to independently checked numerics, explicit fallback, a proved finite-sample accuracy decision, joint four-parameter inference, adversarial validation, and a reproducible empirical release. Phase 1 is deliberately a complete vertical slice: a user must be able to install the package, simulate fixed-seed `S0` data, and run the exact finite three-cell posterior before broader claims are attempted. Certification is a mathematical kill gate, not a status label earned by simulation or ordinary floating-point agreement.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): planned milestone work
- Decimal phases (for example, 2.1): urgent insertions created after planning

- [ ] **Phase 1: Working Theorem-Faithful Package** - Install the package and complete a fixed-seed exact finite-cell Bayesian fit with known location and scale.
- [ ] **Phase 2: Independent Numerics and Full Reference** - Check finite-cell numerics independently and compute the full stable-likelihood reference posterior.
- [ ] **Phase 3: Accuracy Decisions, Fallback, and Audit** - Make every inferential choice explicit, recoverable, and machine-auditable.
- [ ] **Phase 4: Finite-Sample Certificate Kill Gate** - Prove and conservatively enclose the prospective posterior-discrepancy bound before certification is enabled.
- [ ] **Phase 5: Pilot-Conditioned Four-Parameter Fit** - Jointly infer `(alpha, beta, loc, scale)` without plug-in standardization or moving bins.
- [ ] **Phase 6: Four-Parameter Qualification Gate** - Keep the joint workflow experimental until calibration and reference-comparison gates pass.
- [ ] **Phase 7: Adversarial Statistical Validation** - Test inferential correctness, robustness, and refusal behavior over prespecified regimes and negative controls.
- [ ] **Phase 8: End-to-End Performance Qualification** - Determine whether the reduced workflow meets the paper's accuracy, safety, and speed criteria.
- [ ] **Phase 9: Empirical Demonstration and Release** - Demonstrate qualified real-data use and ship reproducible cross-platform documentation and artifacts.

## Phase Details

### Phase 1: Working Theorem-Faithful Package
**Goal**: Users can install `stableboundary` and complete a reproducible exact finite three-cell Bayesian fit under the theorem's known-location/scale scope.
**Depends on**: Nothing (first phase)
**Requirements**: PKG-01, PKG-02, PKG-03, PAR-01, PAR-02, PAR-03, DES-01, DES-02, DES-03, NUM-01, NUM-02, NUM-04, FIT-01, FIT-02, FIT-03, FIT-04, FIT-05, FIT-06, FIT-07, VAL-02
**Success Criteria** (what must be TRUE):
  1. A user can install a wheel or source distribution in a clean Python 3.12+ environment and import only the documented, typed `stableboundary` API.
  2. A user can create validated Nolan `S0` parameters and a data-independent local design, inspect every conversion and threshold, and receive structured refusals for theorem-external or ambiguous inputs.
  3. A fixed-seed simulated dataset with externally known `loc` and `scale` completes `fit_known_nuisance()` using finite stable cell probabilities and deterministic two-dimensional quadrature, returning normalized summaries for `h`, `p`, `alpha`, `beta`, and both signed tail gaps.
  4. The result exposes cell counts, predictive tail quantities, identification warnings, and variance limitations; zero or one-sided exceedances never masquerade as learned skewness, and the limiting Gamma-Beta posterior is visibly an approximation.
  5. A maintainer can run the documented format, lint, type, test, build, and installed-package example commands, and the fixed-seed example returns finite reproducible summaries rather than importing the research spike.
**Plans**: TBD

### Phase 2: Independent Numerics and Full Reference
**Goal**: Users and maintainers can distinguish a working reduced calculation from one that agrees with an independently evaluated full stable posterior.
**Depends on**: Phase 1
**Requirements**: NUM-03, NUM-05, REF-01, REF-02
**Success Criteria** (what must be TRUE):
  1. Selected cell probabilities and densities agree within declared tolerances across independent numerical lineages, while unresolved disagreement stops the computation instead of being clipped or renormalized away.
  2. The numerical checks reproduce the `S0` Gaussian boundary as a beta-invariant normal distribution with standard deviation `sqrt(2) * scale`.
  3. A user can compute the full stable-likelihood posterior under the identical compact prior and known-nuisance design used by the grouped posterior.
  4. Validation code reports full-versus-grouped posterior total variation and Hellinger diagnostics on a common refined deterministic grid.
**Plans**: TBD

### Phase 3: Accuracy Decisions, Fallback, and Audit
**Goal**: Users always know which posterior was selected, why it was selected, and what kind of accuracy evidence supports it.
**Depends on**: Phase 2
**Requirements**: SAFE-01, SAFE-02, SAFE-03, AUD-01
**Success Criteria** (what must be TRUE):
  1. Every fit returns exactly one machine-readable state—`reduced_safe`, `full_fallback`, `refused`, or `research_uncertified`—with a structured reason and no silent method change.
  2. When reduced inference is unsupported, the configured full posterior is selected explicitly or the call ends in a documented refusal.
  3. Every accuracy report states its target, scope, numerical evidence, and limitations; simulation calibration or observed-data posterior agreement cannot emit a certified claim.
  4. A user can save, verify, and reload a versioned non-pickle audit bundle containing the design, prior, versions, seed, data fingerprint, tolerances, diagnostics, status, and fallback chain.
**Plans**: TBD

### Phase 4: Finite-Sample Certificate Kill Gate
**Goal**: Establish whether prospective reduced inference can be justified by a proved finite-sample bound and a conservative executable enclosure, without first paying for the full posterior.
**Depends on**: Phase 3
**Requirements**: CERT-01, CERT-02
**Success Criteria** (what must be TRUE):
  1. For a supported prior, compact region, sample size, and fixed design, a user can compute a proved upper bound with separately reported statistical and numerical remainder terms.
  2. The bound can be evaluated without the unknown true parameter, main-sample threshold adaptation, or prior computation of the full posterior.
  3. The result states whether its target is expected prior-predictive posterior TV or an observed-data quantity and never presents the former as a guarantee for every realized dataset.
  4. Every analytic truncation, quadrature, and rounding contribution is conservatively enclosed under a versioned proof/method identifier; any missing or inconclusive term makes certified mode unavailable.
**Plans**: TBD

**Kill gate**: Phase 4 passes only if both the mathematical bound and its conservative numerical enclosure pass independent review. If either fails, `certified` remains constructively unavailable and the computational paper's certification claim is stopped or narrowed; simulation evidence cannot waive this gate.

### Phase 5: Pilot-Conditioned Four-Parameter Fit
**Goal**: Users can jointly estimate `(alpha, beta, loc, scale)` while retaining pilot information and a fixed, valid grouped-data likelihood.
**Depends on**: Phase 4
**Requirements**: NUIS-01, NUIS-02, NUIS-03, NUIS-05
**Success Criteria** (what must be TRUE):
  1. A user can create a reproducible pilot/main split whose indices, nuisance anchor, raw bin edges, and random provenance remain immutable and inspectable.
  2. A joint fit combines the pilot's full stable likelihood with the main sample's fixed grouped likelihood and propagates uncertainty in all four parameters.
  3. Central cells inform location and scale, signed-tail cells inform tail emergence, and observations are never re-binned as parameter proposals change.
  4. Results report conventional four-parameter posterior summaries together with signed tail gaps and an explicit beta-identification diagnostic.
**Plans**: TBD

### Phase 6: Four-Parameter Qualification Gate
**Goal**: Users can tell whether the joint nuisance workflow has earned empirical support or remains experimental.
**Depends on**: Phase 5
**Requirements**: NUIS-04
**Success Criteria** (what must be TRUE):
  1. The four-parameter workflow is visibly labelled experimental until its frozen simulation-based calibration, coverage, numerical-refinement, and full-posterior comparison suite passes.
  2. Passing evidence is tied to declared parameter regimes, priors, bin rules, tolerances, and package versions rather than generalized beyond the tested scope.
  3. A failed or inconclusive qualification run preserves the experimental label and cannot inherit the known-nuisance theorem or certificate.
**Plans**: TBD

### Phase 7: Adversarial Statistical Validation
**Goal**: Users can inspect prespecified evidence that supported fits are calibrated and that unsupported data-generating processes do not receive trustworthy labels.
**Depends on**: Phase 6
**Requirements**: VAL-01, VAL-03, VAL-04
**Success Criteria** (what must be TRUE):
  1. Automated tests exercise coordinate invariants, reflection, normalization, location-scale equivariance, the Gaussian boundary, posterior normalization, serialization, and every fallback/refusal path.
  2. Scheduled, seed-controlled validation reports SBC ranks, interval coverage, posterior distance, predictive calibration, prior sensitivity, and threshold sensitivity with Monte Carlo uncertainty and all failures retained in denominators.
  3. Gaussian, Student-t, contaminated Gaussian, tempered-tail, dependent, and stochastic-volatility controls trigger the declared warning, fallback, or refusal behavior rather than unsupported success claims.
  4. Validation artifacts identify the data generator, independent oracle, regime manifest, code version, and numerical tolerances needed to reproduce every reported result.
**Plans**: TBD

### Phase 8: End-to-End Performance Qualification
**Goal**: Determine whether certified reduced inference is materially faster than the full posterior without sacrificing the prespecified accuracy and safety criteria.
**Depends on**: Phase 7
**Requirements**: PERF-01, PERF-02
**Success Criteria** (what must be TRUE):
  1. Users can inspect benchmarks that include planning, accuracy assessment, posterior fitting, numerical failures, and fallback costs under matched tolerances and declared hardware.
  2. At least one prespecified nontrivial regime with `n <= 250000` attains at least 90% reduced invocation, conditional mean TV at most 0.05, 90th-percentile TV at most 0.10, false-safe rate at most 5%, and at least tenfold end-to-end speed-up.
  3. Qualification reports computational failure below 1% and Monte Carlo uncertainty small enough to resolve the declared accuracy and false-safe thresholds.
**Plans**: TBD

**Kill gate**: If no regime meets every criterion, the package may remain a reference/fallback implementation, but the computational paper cannot claim a practically successful certified reduction.

### Phase 9: Empirical Demonstration and Release
**Goal**: Applied users can reproduce a scientifically defensible empirical analysis from a clean installed artifact on supported platforms.
**Depends on**: Phase 8
**Requirements**: PKG-04, EMP-01, DOC-01
**Success Criteria** (what must be TRUE):
  1. A documented empirical example fits an IID sample or appropriately standardized residual series with data provenance, preprocessing, assumptions, sensitivity checks, and visible refusal diagnostics.
  2. A new user can follow the documentation from installation through first fit, coordinate and prior interpretation, prediction, identification, audit inspection, fallback, refusal, and the exact limits of certification.
  3. Continuous integration verifies wheel and source installations plus the documented quickstart on supported Python versions across Windows, Linux, and macOS.
  4. The empirical and documentation builds use the released package artifact and reproduce their recorded results without repository-path imports.
**Plans**: TBD

## Requirement Coverage

| Phase | Requirement count |
|-------|------------------:|
| 1. Working Theorem-Faithful Package | 20 |
| 2. Independent Numerics and Full Reference | 4 |
| 3. Accuracy Decisions, Fallback, and Audit | 4 |
| 4. Finite-Sample Certificate Kill Gate | 2 |
| 5. Pilot-Conditioned Four-Parameter Fit | 4 |
| 6. Four-Parameter Qualification Gate | 1 |
| 7. Adversarial Statistical Validation | 3 |
| 8. End-to-End Performance Qualification | 2 |
| 9. Empirical Demonstration and Release | 3 |
| **Total** | **43** |

Every v1 requirement is assigned once. There are no orphaned or duplicate mappings.

## Progress

**Execution Order:** Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8 -> Phase 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Working Theorem-Faithful Package | 0/TBD | Not started | - |
| 2. Independent Numerics and Full Reference | 0/TBD | Not started | - |
| 3. Accuracy Decisions, Fallback, and Audit | 0/TBD | Not started | - |
| 4. Finite-Sample Certificate Kill Gate | 0/TBD | Not started | - |
| 5. Pilot-Conditioned Four-Parameter Fit | 0/TBD | Not started | - |
| 6. Four-Parameter Qualification Gate | 0/TBD | Not started | - |
| 7. Adversarial Statistical Validation | 0/TBD | Not started | - |
| 8. End-to-End Performance Qualification | 0/TBD | Not started | - |
| 9. Empirical Demonstration and Release | 0/TBD | Not started | - |

---
*Roadmap created: 2026-08-24*  
*Granularity: fine*  
*Coverage: 43/43 v1 requirements mapped exactly once*
