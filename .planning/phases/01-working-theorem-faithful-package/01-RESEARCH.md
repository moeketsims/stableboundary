# Phase 1 Research: Working Theorem-Faithful Package

**Phase:** 01 — Working Theorem-Faithful Package  
**Researched:** 2026-08-24  
**Status:** Ready for planning  
**Confidence:** High for this scoped vertical slice

## Research Question

What is the smallest installable implementation that faithfully turns the
Gaussian-boundary stable-law result into a usable Bayesian analysis, while
making no claim that has not yet been established by independent numerical
validation or a finite-sample certificate?

## Answer

Build one complete known-location/scale workflow around the **exact finite
three-cell multinomial likelihood**. Use the manuscript's local coordinates
only to define the compact parameter region and interpret results. Evaluate
finite cell probabilities from the Nolan `S0` stable law, integrate the
two-dimensional posterior deterministically on a Gauss--Legendre product grid,
and retain the normalized grid as the canonical posterior representation.

SciPy's stable-law implementation is acceptable as a guarded bootstrap backend
for this first working phase because the result will be labelled
`research_uncertified`. It is not acceptable as both implementation and oracle
for a reliability claim. Independent finite-cell numerics and a full-likelihood
reference posterior are Phase 2 requirements; the constructive certificate is
a later kill gate.

The output of Phase 1 is therefore scientifically useful but deliberately
narrow: a reproducible exact-model posterior conditional on independently known
location and scale, with transparent identification warnings and no automatic
safe-reduction claim.

## Statistical Specification

### Canonical parameterization

Use Nolan's continuous `S0` parameterization throughout. Public conventional
parameters are

```text
(alpha, beta, loc, scale),
0 < alpha <= 2, -1 <= beta <= 1, scale > 0.
```

The local coordinates are

```text
alpha = 2 - r h,
beta  = 2 p - 1,
```

with a prespecified `r > 0`, compact `h` support strictly inside `h > 0`, and
compact `p` support strictly inside `(0,1)`. The signed boundary gaps are

```text
tau_plus  = r h p,
tau_minus = r h (1-p).
```

These equal `(2-alpha)(1+beta)/2` and
`(2-alpha)(1-beta)/2`, respectively. Every local-coordinate result must carry
its design scale `r`; reporting `h` without `r` is not meaningful.

At `alpha=2`, `beta` is unidentifiable. Conversion from a Gaussian parameter
object into `(h,p)` must therefore refuse rather than invent a value for `p`.

### Prespecified local design

For sample size `n` and positive design constant `c`, solve

```text
n r / log(1/r) = 8 c
```

using the closed form

```text
r = (8 c / n) W(n / (8 c)),
```

where `W` is the principal Lambert-W branch. Define the standardized threshold

```text
u = 2 sqrt(log(1/r) + 2 log(log(1/r))).
```

Validate all intermediate quantities. In particular, the square-root argument
must be positive and the entire prior rectangle must map into `0 < alpha < 2`.
The design is constructed from `n`, `c`, and prior support before inspecting the
observations.

### Exact finite-cell experiment

Given known `loc` and `scale`, standardize each observation as

```text
z = (x-loc)/scale.
```

With fixed threshold `u`, form immutable counts

```text
N_minus  = count(z < -u)
N_zero   = count(-u <= z <= u)
N_plus   = count(z > u).
```

For each `(h,p)` quadrature node, map to `(alpha,beta)` and evaluate

```text
q_minus = P(Z < -u),
q_plus  = P(Z >  u),
q_zero  = 1 - q_minus - q_plus.
```

The positive tail must come from a survival-function call, not `1-cdf(u)`.
The central probability may be calculated as the residual only after the two
direct tail evaluations have passed finite, nonnegative, and normalization
checks. Materially invalid probabilities raise a package-owned numerical
exception; production code must not clip them into the simplex.

The log likelihood is the multinomial kernel

```text
N_minus log(q_minus) + N_zero log(q_zero) + N_plus log(q_plus),
```

with zero-count terms handled by `scipy.special.xlogy` or an equivalent safe
operation.

### Prior and posterior

The first public prior should be a proper uniform density on a compact
rectangle in `(h,p)`. This is enough to exercise the theorem without hiding
extra inferential structure. The class should retain its bounds and expose a
`default(design)` constructor. Future prior families can implement the same
log-density contract.

Use Legendre nodes and weights transformed independently onto the `h` and `p`
intervals. Add log prior, log likelihood, and log quadrature measure; normalize
with `logsumexp`. Store read-only node arrays and normalized probability masses.
Refinement must be explicit: compare a base grid with a larger grid and record
the largest change among selected posterior means and central credible interval
endpoints. If normalization fails or refinement exceeds the declared tolerance,
raise a structured convergence error rather than returning a nominal fit.

The normalized weighted grid is sufficient for posterior means, quantiles,
credible intervals, signed-gap summaries, and posterior predictive mixtures.
MCMC would make the first reference less deterministic and is not justified for
a compact two-dimensional integral.

### Identification behavior

Cell counts contain two kinds of information: total tail occurrence informs the
gap/intensity direction, while allocation between positive and negative tails
informs skewness. Therefore:

- With zero total tail counts, update the total intensity through the no-event
  probability but label `beta`/`p` prior-dominated: conditional on zero total
  events, the limiting sign-allocation likelihood contains no p information.
- With positive tail counts all on one side, report `one_sided_evidence` and
  quantitative prior-to-posterior information. The conditional likelihood is
  proportional to `p**N` or `(1-p)**N`, so it can be strongly informative when
  N is large; nevertheless, do not promote it to validated precise
  identification before calibration.
- If posterior support reaches `alpha=2`, `beta` is unidentified. Phase 1 prior
  support is strictly below two, but the public parameter conversion still
  needs the exact-boundary refusal.

The diagnostic rule should be deterministic and included in the audit record.
It is a warning about information in this experiment, not a convergence test.

### Prediction

For a requested raw threshold `t`, transform it with the known nuisance values,
evaluate direct stable-law CDF/SF values at every posterior node, and integrate
with the stored posterior masses. Posterior predictive draws first sample a
posterior grid node, then draw from the full `S0` stable sampling model.

Do not report a finite predictive variance when any posterior mass lies on
`alpha < 2`; the stable model has infinite variance there. Raise a dedicated
`InfiniteVarianceError`. Predictive quantiles remain valid and are estimated
from reproducible mixture draws in Phase 1.

### Limiting approximation

Expose the signed-Poisson/Gamma--Beta limit only through an explicitly named
function or result type containing `approximation=True`. It must not be called
inside `fit_known_nuisance()`. Its role in this phase is a transparent
regression/comparison target, not the default finite-sample likelihood.

## Package Architecture

Use a `src/` layout with a deliberately small public facade:

```text
src/stableboundary/
  __init__.py          curated public API and version
  py.typed             typing marker
  _exceptions.py       package-owned validation/numerical errors
  parameters.py        StableParams and coordinate/signed-gap objects
  design.py            LocalDesign, LocalPrior, known-nuisance provenance
  cells.py             immutable counts and exact finite probabilities
  posterior.py         deterministic weighted-grid integration
  result.py            immutable fit, summaries, diagnostics, prediction
  approximation.py     explicitly named limiting approximation
  api.py               simulate and fit_known_nuisance entry points
  backends/
    _protocol.py       package-owned stable probability protocol
    _scipy_s0.py       guarded SciPy bootstrap adapter
```

No module outside `backends/_scipy_s0.py` may import
`scipy.stats.levy_stable`. This makes independent numerical replacement
possible without altering the statistical API.

SciPy exposes mutable process-global stable-distribution settings. The adapter
must snapshot relevant settings, acquire a re-entrant process lock, set `S0`
only for the duration of a call, and restore the snapshot in `finally`. It must
never mutate the parameterization at import time. Tests should deliberately set
SciPy to `S1`, call the adapter, and verify restoration.

Use frozen dataclasses for public data structures. NumPy arrays inside frozen
objects must additionally be copied and marked non-writeable. Validation errors,
probability-evaluation errors, convergence failures, unsupported-identification
errors, and infinite-moment errors should have distinct package-owned types.

## Public Workflow

The first complete workflow remains:

```python
import stableboundary as sb

design = sb.LocalDesign.from_sample_size(5_000)
truth = sb.StableParams(
    alpha=2.0 - design.r * 1.5,
    beta=0.35,
    loc=0.0,
    scale=1.0,
)
x = sb.simulate(truth, size=5_000, random_state=20260824)
fit = sb.fit_known_nuisance(
    x,
    loc=0.0,
    scale=1.0,
    design=design,
    prior=sb.LocalPrior.default(design),
)
print(fit.summary())
```

Using a design-derived truth prevents the quickstart from accidentally placing
the true alpha outside the local prior region when defaults change.

Every result must expose `status="research_uncertified"`. There is no code path
in Phase 1 that can manufacture `certified` or `reduced_safe`.

## Verification Strategy

The phase is accepted only as an installed, exercised package:

1. Unit and property tests cover parameter domains, coordinate identities,
   design equations, reflection, cell normalization, immutable arrays, global
   SciPy-state restoration, and refusal cases.
2. Integration tests run exact finite-cell fits, check posterior normalization
   and quadrature refinement, exercise zero/one-sided tail diagnostics, and
   verify predictive behavior.
3. A fixed-seed example is shared by documentation and an automated test.
4. Build both wheel and sdist, inspect their contents, install the wheel in a
   temporary virtual environment, change outside the repository source tree,
   and execute the example there.
5. Run Ruff, strict mypy, and pytest before accepting the phase.

Phase 1 tests establish software and internal numerical correctness only. They
must not describe agreement with the guarded SciPy backend as independent
scientific validation.

## Risks and Required Responses

| Risk | Required response in Phase 1 |
|---|---|
| `S0`/`S1` mismatch | Canonical parameter object plus guarded adapter and restoration test |
| Stable scale mistaken for Gaussian SD | Documentation says `alpha=2` implies variance `2*scale**2` |
| Post-data threshold selection | Design object is required and constructed before counts; audit stores inputs |
| Rare-tail cancellation | Direct survival evaluation; structured failure on invalid simplex |
| Hidden numerical repair | No probability clipping; record normalization/refinement diagnostics |
| False beta precision | Deterministic identification labels and visible warnings |
| Approximation silently replacing finite model | Separate named API; exact finite likelihood is hard-coded default |
| Self-validation | Keep `research_uncertified`; add independent lineage in Phase 2 |
| Reproducibility overclaim | Record seed and bit-generator class; promise seeded workflow, not eternal bitwise identity across every future dependency version |

## Planning Consequences

Phase 1 should be split into four dependency-ordered implementation plans:

1. **Package contracts and design:** packaging, exception hierarchy, immutable
   parameters, coordinate maps, local design/prior, and their tests.
2. **Stable backend and finite experiment:** guarded `S0` protocol/adapter,
   simulation, cell counts/probabilities, and numerical-state/reflection tests.
3. **Posterior and result:** deterministic exact finite posterior, diagnostics,
   summaries, prediction, explicit limiting approximation, and integration
   tests.
4. **Installed proof of work:** executable quickstart, maintenance commands,
   build artifacts, isolated wheel smoke test, and final verification.

Plans 1 and 2 may start in parallel only after the package metadata and public
contracts are fixed. Plan 3 depends on both. Plan 4 depends on the complete
public path.

## Sources Consulted

- `gaussian_boundary_stable_manuscript.tex` — theorem, local coordinates,
  finite experiment, and scope.
- `.planning/spikes/001-gaussian-boundary-stable/` — executable formula and
  regression evidence, treated as a spike rather than production design.
- `.planning/research/{STACK,FEATURES,ARCHITECTURE,PITFALLS,SUMMARY}.md` —
  package ecosystem, API, architecture, and adversarial risks.
- SciPy `levy_stable` documentation — `S0` behavior and mutable configuration.
- NumPy polynomial Legendre and random APIs — deterministic quadrature and RNG.
- PyPA packaging guidance — PEP 621, build isolation, and `src/` layout.

## Research Complete

Phase 1 has no unresolved implementation question that should block planning.
The unresolved scientific claims—independent numerical agreement, full
posterior closeness, prospective safety, and certification—are explicitly
outside this phase and cannot be inferred from its success.
