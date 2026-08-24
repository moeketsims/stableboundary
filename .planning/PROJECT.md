# stableboundary

## What This Is

`stableboundary` is a Python package for Bayesian inference for univariate
alpha-stable laws close to the Gaussian boundary. It translates the asymptotic
signed-Poisson reduction in the accompanying theoretical manuscript into an
auditable finite-sample workflow: exact finite-cell inference when the
reduction is reliable, full stable-likelihood fallback when it is not, and
honest reporting of weak or absent skewness identification.

The first usable release targets standardized observations or observations
with independently known location and scale. A subsequent joint workflow will
estimate the conventional four parameters `(alpha, beta, loc, scale)` using a
pilot likelihood and fixed grouped main-sample likelihood while propagating
nuisance uncertainty.

## Core Value

Users can fit a near-Gaussian stable model and trust that the package never
presents a fast reduced posterior as reliable without quantifying its scope,
numerical status, and fallback decision.

## Requirements

### Validated

- ✓ The manuscript's deterministic Python spike evaluates standardized
  `S0` stable densities, signed-cell probabilities, Hellinger information, and
  rare-count approximations over prespecified parameter grids — existing
  research evidence, not production package functionality.
- ✓ Independent SciPy piecewise-density and lower-order quadrature checks
  reproduce selected deterministic calculations — existing numerical audit.

### Active

- [ ] Provide a standards-compliant, installable Python package using a
  `src/` layout, `pyproject.toml`, typed public API, tests, documentation, and
  reproducible examples.
- [ ] Represent conventional `S0` stable parameters, local boundary
  coordinates, signed tail gaps, and their conversions without parameter-name
  ambiguity.
- [ ] Derive a prespecified local design from sample size and expose the
  analysis scale, threshold, compact prior region, and all assumptions.
- [ ] Fit the exact finite three-cell multinomial posterior for `(h, p)` with
  deterministic two-dimensional quadrature when location and scale are known.
- [ ] Return posterior summaries for `alpha`, `beta`, signed tail gaps, and
  predictive tail probabilities, with explicit warnings when beta is
  prior-dominated or unidentified.
- [ ] Implement an independently checked full `S0` stable-likelihood reference
  posterior and explicit fallback contract.
- [ ] Demonstrate an end-to-end fit on simulated data and reproduce it through
  both a Python API example and automated test.
- [ ] Validate coordinate identities, probability normalization, reflection,
  Gaussian-boundary behavior, posterior computation, and numerical backend
  agreement.
- [ ] Define a versioned accuracy-assessment interface that cannot label output
  `certified` until the finite-sample bound and conservative numerical
  enclosure have been proved and implemented.
- [ ] Add a pilot-conditioned grouped likelihood for joint inference on
  `(alpha, beta, loc, scale)` only after the standardized reference workflow
  passes posterior-distance and coverage tests.
- [ ] Add at least one scientifically qualified empirical example using IID
  observations or appropriately standardized residuals.

### Out of Scope

- A generic stable-distribution package for the full range of `alpha` — SciPy,
  `stabledist`, and `StableEstim` already serve that broader problem; this
  package is specialized to the near-Gaussian boundary.
- Claiming a posterior probability of exact Gaussianity — the present theorem
  excludes `alpha = 2`, where beta is not identified.
- Certified inference with plug-in location or scale estimates — nuisance
  uncertainty must be propagated under a separately validated workflow.
- Automatic inference for raw dependent, nonstationary, heteroskedastic,
  censored, or contaminated series — users must supply suitable observations
  or model residuals, and diagnostics may refuse inference.
- An independent R numerical implementation in the initial milestone — an R
  wrapper may follow after the Python numerical contract stabilizes.
- A separate software paper before the computational method and package have
  demonstrated genuine use — the package initially accompanies the
  computational paper.

## Context

- The theoretical source is
  `gaussian_boundary_stable_manuscript.tex`, titled *Bayesian Posterior
  Reduction at the Gaussian Boundary of Standardized Stable Laws: Hellinger
  Geometry and Signed-Poisson Equivalence*.
- The theorem assumes known location and scale, compact interior local
  parameter sets, and excludes the exact Gaussian and one-sided boundaries.
- The local coordinates are `2 - alpha = r*h` and
  `p = (1 + beta)/2`; the signed weights are `w_plus = h*p` and
  `w_minus = h*(1-p)`.
- The empirical-scale signed gaps
  `(2-alpha)*(1+beta)/2` and `(2-alpha)*(1-beta)/2` remain meaningful as tail
  emergence vanishes and should be primary reported quantities.
- The current research code is
  `.planning/spikes/001-gaussian-boundary-stable/boundary_spike.py`. It is a
  deterministic falsification audit with fixed grids, truncation, interpolation,
  and first-tail continuation; it must remain reference material rather than be
  repackaged as production code.
- Existing calculations show slow pre-asymptotic convergence: exact finite-cell
  probabilities are therefore the practical baseline, while the limiting
  Gamma-Beta posterior is an approximation and benchmark.
- The user must be able to run the package and see a successful fit before the
  theoretical manuscript is submitted for review.

## Constraints

- **Statistical scope**: Version 1 begins with standardized data or independently
  known location and scale because this is the scope justified by the theorem.
- **Parameterization**: Nolan's continuous `S0` parameterization is canonical;
  every result and serialized artifact records it explicitly.
- **Language**: Python 3.11+ is the primary implementation language, using
  NumPy and SciPy behind package-owned numerical protocols.
- **Package structure**: Use a `src/stableboundary/` layout, PEP 621 metadata,
  immutable result objects, a narrow public API, and no import-time mutation of
  SciPy's global stable-distribution configuration.
- **Numerics**: Ordinary floating-point agreement is validation evidence, not a
  proof certificate. Negative densities, invalid probabilities, or backend
  disagreements beyond tolerance must raise structured errors.
- **Inference**: Exact finite-cell probabilities are the default reduced
  likelihood. Limiting Poisson/Gamma-Beta calculations cannot be silently
  substituted.
- **Safety**: Reduction, fallback, and refusal states are explicit and
  machine-readable; the package never silently changes inferential methods.
- **Reproducibility**: Seeds, versions, priors, design scale, thresholds, counts,
  numerical tolerances, and fallback reasons are retained in an audit record.
- **Performance claims**: Runtime comparisons include planning, accuracy
  assessment, posterior computation, and fallback costs.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build Python first | Existing research code and SciPy stable-law support make one trustworthy core faster than duplicate Python/R implementations | — Pending |
| Use provisional package name `stableboundary` | Names the specialized inferential regime without claiming a general stable-law solution | — Pending |
| Make exact finite-cell inference the reduced default | The manuscript documents slow convergence of limiting cell means | — Pending |
| Keep limiting Gamma-Beta inference as an explicit approximation | It is useful for initialization, pedagogy, and benchmarking but is not exact at finite sample size | — Pending |
| Report signed tail gaps alongside conventional parameters | They remain interpretable when beta becomes weakly identified near the Gaussian boundary | — Pending |
| Build known-nuisance reference workflow before joint fitting | It provides a two-dimensional ground truth for posterior-distance validation | — Pending |
| Require pilot likelihood plus fixed main-sample bins for unknown nuisance parameters | Two signed counts cannot identify four parameters, and plug-in standardization ignores uncertainty | — Pending |
| Reserve `certified` for a proved and conservatively enclosed finite-sample bound | Simulation calibration or arbitrary precision alone does not establish certification | — Pending |
| Ship software with the computational methods paper | Avoids an initial third publication that could be viewed as slicing the same contribution | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Move implemented and verified requirements to Validated.
2. Move invalidated requirements to Out of Scope with the reason.
3. Add requirements that emerge from numerical or empirical evidence.
4. Record consequential implementation and statistical decisions.
5. Update the project description if the supported inferential scope changes.

**After each milestone**:
1. Recheck the Core Value.
2. Audit whether every public claim is supported by tests and evidence.
3. Revisit deferred nuisance, empirical, and R-interface work.
4. Update the context with package users, datasets, benchmarks, and failures.

---
*Last updated: 2026-08-24 after project initialization*
