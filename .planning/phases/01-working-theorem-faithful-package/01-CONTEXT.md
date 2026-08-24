# Phase 1: Working Theorem-Faithful Package - Context

**Gathered:** 2026-08-24  
**Status:** Ready for planning  
**Source:** User-directed package planning plus adversarial package research

<domain>
## Phase Boundary

Build the first complete and installable `stableboundary` vertical slice. A
user must be able to install the package, simulate a fixed-seed Nolan `S0`
sample, provide independently known location and scale, construct a
data-independent local design, fit the exact finite three-cell posterior for
`(h,p)` with deterministic quadrature, and inspect conventional and signed-tail
posterior summaries.

This phase establishes a trustworthy computational reference for the theorem.
It does not implement the full stable posterior comparison, an automatic
safe/fallback decision, a mathematical certificate, or joint unknown
location/scale inference; those have dedicated later phases.

</domain>

<decisions>
## Implementation Decisions

### D-01 Package form

- Use an installable Python 3.12+ package named `stableboundary`.
- Use PEP 621 metadata, Hatchling, and a `src/stableboundary/` layout.
- Runtime dependencies are NumPy and SciPy only.
- Research scripts under `.planning/spikes/` are references and reproduction
  artifacts; production imports must not depend on them.

### D-02 Public statistical contract

- Nolan's continuous `S0` parameterization is canonical and explicit.
- Public stable parameters are `alpha`, `beta`, `loc`, and `scale`; do not use
  `delta` for the local Gaussian gap because it conflicts with stable location
  notation.
- Local coordinates are `(r,h,p)` with `alpha = 2-r*h` and
  `beta = 2*p-1`.
- Primary boundary summaries include the signed gaps
  `tau_plus=(2-alpha)*(1+beta)/2` and
  `tau_minus=(2-alpha)*(1-beta)/2`.
- `h` is never serialized or reported without its design scale `r`.
- At the exact Gaussian point, beta is not identified; unsupported inputs are
  refused rather than assigned an arbitrary value.

### D-03 Local design

- `LocalDesign.from_sample_size()` derives `r` from a prespecified positive
  constant `c` using the critical-rate equation
  `n*r/log(1/r)=8*c` and records all inputs and derived values.
- The theoretical moving threshold
  `u=2*sqrt(log(1/r)+2*log(log(1/r)))` is the initial rule.
- Prior support is a proper compact rectangle strictly inside `h>0` and
  `0<p<1`; it must map inside `0<alpha<2`.
- Thresholds and the design are fixed before main-sample cell counts are
  calculated.

### D-04 Finite-cell likelihood

- The finite-sample default is the exact-model three-cell multinomial
  likelihood with probabilities evaluated from the finite `S0` distribution.
- Positive probability uses a direct survival calculation or direct tail
  evaluation, never unguarded `1-cdf` subtraction.
- Invalid, nonfinite, materially negative, or non-normalizing probabilities
  produce a structured numerical failure. Production code cannot clip density
  or probability values to conceal failure.
- The limiting signed-Poisson/Gamma-Beta posterior is an explicitly named
  approximation and regression target, not a hidden replacement.

### D-05 Posterior engine

- The known-nuisance posterior is a compact two-dimensional problem and uses
  deterministic tensor Gauss-Legendre quadrature in transformed coordinates.
- Normalize in the log domain and retain the weighted grid.
- Sampling frameworks and MCMC are not part of Phase 1.
- Numerical refinement policy and normalization diagnostics are part of the
  result.

### D-06 Result and identification behavior

- `fit_known_nuisance()` returns an immutable package-owned result.
- The result exposes the design, prior, cells, posterior grid/weights,
  intervals for `h`, `p`, `alpha`, `beta`, `tau_plus`, and `tau_minus`,
  predictive tail quantities, warnings, and a versioned audit dictionary.
- Phase 1 results are marked `research_uncertified`; no object can claim
  `certified`, `reduced_safe`, or superiority to a full posterior yet.
- Zero total tail counts update total intensity through the no-event
  probability but create no sign-allocation information, so beta/p is
  prior-dominated. One-sided positive counts contribute a nonconstant
  conditional-binomial likelihood for p and must be reported as one-sided
  evidence with quantitative information diagnostics, not categorically
  dismissed as prior-dominated or promoted to validated precise identification.
- Posterior prediction uses the full stable sampling model. Predictive
  variance is unavailable whenever posterior support includes `alpha<2`.

### D-07 First proof of work

- A documented fixed-seed example simulates data with the installed package
  and completes the full public fitting path.
- The same example executes as an automated test and returns finite normalized
  summaries.
- Phase success is not achieved by a package scaffold, coordinate utilities,
  or a likelihood function alone.

### D-08 Quality contract

- Public functions and dataclasses are typed; ship `py.typed`.
- Use immutable dataclasses and package-owned exception types.
- Unit tests cover domains, coordinate identities, reflection, probability
  normalization, count construction, posterior normalization, zero/one-sided
  identification behavior, predictive variance refusal, and the installed
  fixed-seed fit.
- Provide documented maintainer commands for format, lint, type check, tests,
  build, and wheel smoke installation.

### the agent's Discretion

- Exact class/function decomposition beneath the locked public behavior.
- Initial quadrature node counts and refinement increments, provided tolerances
  are explicit and failure is possible.
- Default compact prior hyperparameters and `c`, provided defaults are
  documented and the design remains theorem-interior.
- Whether the package-owned finite-cell backend initially uses direct Fourier
  inversion, a guarded SciPy bootstrap implementation behind the protocol, or
  both, provided the public API and tests do not claim independent validation
  before Phase 2.
- Documentation framework details beyond an executable README/quickstart in
  this phase.

</decisions>

<canonical_refs>
## Canonical References

### Statistical source of truth

- `gaussian_boundary_stable_manuscript.tex` — `S0` definition, local
  coordinates, critical rate, threshold, exact cell experiment, limiting
  posterior, estimators, scope, and pre-asymptotic cautions.
- `.planning/spikes/001-gaussian-boundary-stable/CONVENTIONS.md` — numerical
  parameter and Hellinger conventions.
- `.planning/spikes/001-gaussian-boundary-stable/boundary_spike.py` — formulas
  and reproducibility reference only; production anti-patterns must not be
  copied blindly.
- `.planning/spikes/001-gaussian-boundary-stable/results.json` — deterministic
  regression evidence.

### Product and architecture source of truth

- `.planning/PROJECT.md` — product scope, core value, constraints, and settled
  decisions.
- `.planning/REQUIREMENTS.md` — the 20 Phase 1 requirement contracts.
- `.planning/research/SUMMARY.md` — reconciled implementation research.
- `.planning/research/STACK.md` — supported stack and numerical tooling.
- `.planning/research/ARCHITECTURE.md` — public API, private modules, and data
  flow.
- `.planning/research/PITFALLS.md` — release-blocking risks and validation
  architecture.

</canonical_refs>

<specifics>
## Specific Ideas

The intended first user experience is:

```python
import stableboundary as sb

truth = sb.StableParams(alpha=1.97, beta=0.35, loc=0.0, scale=1.0)
x = sb.simulate(truth, size=5000, random_state=20260824)
design = sb.LocalDesign.from_sample_size(len(x))

fit = sb.fit_known_nuisance(
    x,
    loc=0.0,
    scale=1.0,
    design=design,
    prior=sb.LocalPrior.default(design),
)

print(fit.summary())
print(fit.audit_record())
```

The exact names may be refined only if the resulting public workflow remains
equally direct and explicit.

</specifics>

<deferred>
## Deferred Ideas

- Independent full likelihood and posterior-distance comparison — Phase 2.
- Automatic safe/fallback decision and persisted audit bundle — Phase 3.
- Proved and conservatively enclosed certificate — Phase 4.
- Pilot-conditioned four-parameter inference — Phases 5-6.
- Full adversarial calibration and performance qualification — Phases 7-8.
- Empirical dataset and cross-platform release — Phase 9.
- Gaussian spike, one-sided boundaries, R wrapper, regression/time-series
  adapters, and gated multiscale cells — v2 or later.

</deferred>

---
*Phase: 01-working-theorem-faithful-package*  
*Context gathered: 2026-08-24 from user direction and project research*
