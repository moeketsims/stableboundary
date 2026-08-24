# Architecture Patterns

**Project:** `stableboundary`  
**Domain:** Auditable Bayesian inference for standardized alpha-stable laws near the Gaussian boundary  
**Researched:** 2026-08-24  
**Confidence:** HIGH for the first standardized vertical slice; MEDIUM for the later certificate and four-parameter extension because their mathematics is not yet complete

## Architectural Verdict

Build a small statistical core around the theorem's actual experiment, not a generic stable-distribution framework. Version 1 should expose one theorem-faithful fitting path for observations with independently known location and scale:

1. freeze an `S0` local design before inspecting the observations;
2. standardize using the declared known nuisance values;
3. form the exact finite three-cell multinomial likelihood;
4. compute its two-dimensional posterior deterministically;
5. independently compute the full stable-likelihood posterior on the same parameter grid;
6. compare the two posteriors and either select the reduced posterior or fall back explicitly to the full posterior; and
7. return an immutable result with a complete audit record.

This first slice demonstrates that the package works but is **not a computational certificate**. Its posterior comparison is conditional on the observed dataset and on ordinary floating-point calculations. The theorem proves prior-predictive mean posterior total-variation convergence on a compact interior local parameter set; it does not supply a finite-sample, dataset-specific error bound. A later `PROVED_BOUND` state may be enabled only after a finite-sample reconstruction bound and conservative numerical enclosure are both implemented.

The architecture must therefore separate four things that are easy to conflate:

- the statistical model (full stable versus exact finite cells versus limiting Poisson);
- the numerical engine used to evaluate a model;
- the accuracy evidence available for a particular computation; and
- the policy that chooses reduced inference, full fallback, or refusal.

## Recommended Package Shape

Use a `src/` layout and make the package import surface much smaller than its internal module tree. The Python Packaging User Guide recommends the `src` layout because it prevents repository files from being imported accidentally instead of the installed distribution. This is particularly important here: the research spike must not silently become production code.

```text
stableboundary/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   └── stableboundary/
│       ├── __init__.py             # curated public exports only
│       ├── py.typed
│       ├── api.py                  # plan(), fit()
│       ├── parameters.py           # public parameter/coordinate values
│       ├── design.py               # public immutable analysis design
│       ├── priors.py               # supported compact local priors
│       ├── results.py              # immutable result and summaries
│       ├── exceptions.py           # structured failures
│       ├── _validation.py          # data and contract gates
│       ├── _partition.py           # fixed signed-cell labelling
│       ├── _likelihood/
│       │   ├── cells.py            # exact finite multinomial likelihood
│       │   ├── full.py             # full S0 stable reference likelihood
│       │   └── poisson.py          # explicit limiting benchmark only
│       ├── _posterior/
│       │   ├── grid.py             # deterministic 2-D posterior integration
│       │   ├── summaries.py
│       │   └── predictive.py
│       ├── _numerics/
│       │   ├── protocols.py        # FAST, REFERENCE, future ENCLOSED
│       │   ├── probability.py      # backend protocol and checked outputs
│       │   ├── scipy_s0.py         # guarded SciPy adapter
│       │   ├── fourier_s0.py       # package-owned independent reference
│       │   ├── quadrature.py
│       │   └── enclosures.py       # unavailable until mathematically valid
│       ├── _accuracy/
│       │   ├── reports.py          # evidence and scope types
│       │   ├── reference_audit.py  # observed posterior comparison
│       │   └── proved_bound.py     # future finite-sample certificate
│       ├── _decision.py            # reduced/fallback/refusal state machine
│       ├── _serialization/
│       │   ├── schema.py
│       │   └── bundle.py
│       └── _pilot/                 # later four-parameter workflow
│           ├── split.py
│           ├── bins.py
│           ├── likelihood.py
│           └── inference.py
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── statistical/
│   ├── regression/
│   ├── integration/
│   └── packaging/
├── docs/
│   ├── quickstart.md
│   ├── mathematical-contract.md
│   ├── numerical-contract.md
│   └── audit-record.md
└── examples/
    └── standardized_fit.py
```

Do not copy `.planning/spikes/001-gaussian-boundary-stable/boundary_spike.py` into `src/`. It is a fixed-grid falsification audit with hard-coded cutoffs, interpolation, and first-tail continuation. Extract formulas only after each formula has a focused production test; preserve the spike and its JSON as an external regression oracle.

## Public API and Private Boundaries

### Public surface

`stableboundary.__init__` should initially export only:

```python
from stableboundary import (
    BoundaryDesign,
    BoundaryRegion,
    FitOptions,
    KnownLocationScale,
    LocalGammaBetaPrior,
    StableBoundaryResult,
    fit,
    plan,
)
```

The intended call is explicit about the theorem's scope:

```python
region = BoundaryRegion(h=(0.5, 2.0), p=(0.10, 0.90))
design = plan(n=x.size, r=0.01, region=region)

result = fit(
    x,
    design=design,
    nuisance=KnownLocationScale(loc=0.0, scale=1.0, source="external"),
    prior=LocalGammaBetaPrior(A=1.0, B=1.0, a=1.0, b=1.0),
    options=FitOptions(accuracy="reference_audit", max_posterior_tv=0.05),
)
```

All arguments after `x` should be keyword-only. There should be no bare `alpha, beta, loc, scale` positional interface and no implicit parameterization choice.

### Public value types

| Type | Required invariants | Purpose |
|---|---|---|
| `BoundaryRegion` | `0 < h_min < h_max`, `0 < p_min < p_max < 1` | Represents the theorem's compact interior set `K`; it cannot represent the exact Gaussian or one-sided boundaries. |
| `BoundaryDesign` | fixed `n`, `r`, `L=log(1/r)`, `u`, `K`, rule identifier, and implied `c_n = nr/(8L)` | Freezes the local experiment and threshold before data analysis. |
| `KnownLocationScale` | finite location, strictly positive scale, provenance string | Prevents same-sample plug-in standardization from masquerading as known nuisance inference. |
| `LocalGammaBetaPrior` | proper density restricted and renormalized on `K` | Supplies a supported deterministic prior; arbitrary callables are deferred until normalization and serialization contracts exist. |
| `FitOptions` | registered accuracy policy and numerical protocol names, tolerances, refinement limits | Makes behavior reproducible without exposing raw backend objects. |
| `StableBoundaryResult` | internally consistent selected posterior, candidate posterior(s), decision, diagnostics, and audit | Is the only normal return type from `fit`. |

Use frozen, slotted, keyword-only dataclasses for these values. Frozen dataclasses are not deeply immutable, so array-bearing result objects must defensively copy arrays and mark them non-writeable before exposure.

### Internal-only extension points

`ProbabilityBackend`, `PosteriorIntegrator`, `AccuracyAssessor`, and `ReductionPolicy` should be private protocols. Users select package-owned, versioned names such as `"reference"`; they do not inject an arbitrary object into a supposedly reproducible fit. Backend injection remains available in internal tests. If third-party extension becomes necessary later, it should require a declared capability record and must never inherit certification status automatically.

## Parameter and Design Contract

The canonical distribution is Nolan's continuous `S0` parameterization. For standardized data, `loc=0` and `scale=1`; at `alpha=2` this is `N(0, 2)`, not a standard-normal variance convention. The software should use these transformations in one authoritative module:

```text
delta      = 2 - alpha
delta      = r * h
p          = (1 + beta) / 2
w_plus     = h * p
w_minus    = h * (1 - p)
tau_plus   = delta * p       = r * w_plus
tau_minus  = delta * (1-p)   = r * w_minus
alpha      = 2 - r * h
beta       = 2 * p - 1
```

`tau_plus` and `tau_minus` are the empirical signed tail gaps and should be primary result quantities. `alpha` and `beta` remain available in conventional summaries. With known nuisance values, `loc` and `scale` are reported with role `"fixed"`, not as posterior estimates. The package must not tell users that it estimated four parameters in this workflow.

`plan()` must be deterministic and data-free apart from `n`. The initial threshold rule is the manuscript rule

```text
L = log(1/r)
u = 2 * sqrt(L + 2*log(L))
```

and raw thresholds are `loc ± scale*u`. `plan()` rejects a design if the expression is not real, if any `alpha=2-rh` in `K` leaves the stable parameter space, or if the declared local region includes `h=0` or `p` in `{0,1}`. It records rather than hides the fact that `nr/L` may be far from the critical regime. A convenience constructor may solve `nr/log(1/r)=8c` for `r`, but it must record the root bracket and solver residual.

Never estimate `r`, choose `K`, or tune `u` using the same observations passed to `fit()` in the first workflow. Doing so changes the experiment and invalidates the fixed-partition reasoning on which the reduction rests.

## Component Boundaries

| Component | Responsibility | Must not do | Communicates with |
|---|---|---|---|
| `api` | Orchestrate a request and return one result | Evaluate densities directly or mutate library state | validation, design, likelihoods, accuracy, decision, result |
| `parameters` | Validate and convert `S0`, local, and signed-gap coordinates | Guess parameterization from numbers | design, likelihoods, summaries, serialization |
| `design` | Freeze `n`, `r`, `K`, threshold rule, and scope | Inspect sample values | partition, likelihoods, audit |
| `partition` | Standardize and assign `-`, `0`, `+` labels | Re-estimate nuisance values or move thresholds | cell likelihood, audit |
| `cells` | Evaluate finite `q_-`, `q_0`, `q_+` and multinomial log likelihood | Substitute asymptotic Poisson means | numerical probability backend, posterior grid |
| `full` | Evaluate the full finite-sample `S0` log likelihood | Share formulas/code with the independent cell check unnecessarily | density backend, posterior grid |
| `poisson` | Implement the limiting Gamma-Beta benchmark | Act as default finite-sample inference | summaries and research benchmarks only |
| `posterior.grid` | Normalize a deterministic posterior measure and transform it | Clip invalid likelihoods or omit quadrature weights | priors, likelihoods, summaries |
| `numerics` | Return values plus status, tolerances, error evidence, and backend identity | Label ordinary error estimates as proof enclosures | likelihoods, accuracy report |
| `accuracy` | State what accuracy evidence means and its mathematical scope | Turn simulation or observed-data agreement into a uniform guarantee | candidate posteriors, numerical evidence, decision |
| `decision` | Select reduced, full fallback, or refusal using a fixed policy | Silently change method or tolerance | accuracy report, result |
| `serialization` | Write and validate versioned audit bundles | Pickle executable Python objects or raw data by default | all immutable result records |
| `_pilot` | Later joint nuisance workflow with randomized split and fixed main-sample bins | Re-label bins at each posterior proposal | four-parameter likelihood and inference |

## End-to-End Data Flow

```text
 raw observations x
        │
        ▼
 InputGate ──reject──► RefusalReport
        │ finite float64, one-dimensional, n matches design
        ▼
 KnownLocationScale ──► z=(x-loc)/scale
        │
        ├──────────────► Full S0 log likelihood ─► full posterior ─┐
        │                                                         │
        ▼                                                         │
 Fixed Threshold u ─► SignedCounts(n-,n0,n+)                     │
                              │                                   │
                              ▼                                   │
                 finite cell probabilities on (h,p)              │
                              │                                   │
                              ▼                                   │
                 multinomial log likelihood                      │
                              │                                   │
                              ▼                                   │
                    reduced posterior ────────────────────────────┤
                                                                  ▼
                         AccuracyAssessor ─► ReductionPolicy
                                                  │
                           ┌──────────────────────┼───────────────┐
                           ▼                      ▼               ▼
                    reduced selected       full fallback       refused
                           └──────────────────────┼───────────────┘
                                                  ▼
                                StableBoundaryResult + audit bundle
```

### Cell likelihood

At each quadrature point `(h,p)`, convert to `(alpha,beta)`, then calculate

```text
q_minus = F_S0(-u; alpha, beta)
q_plus  = S_S0(+u; alpha, beta)
q_zero  = 1 - q_minus - q_plus
```

Use the survival function for `q_plus`; do not compute it as `1-cdf(u)`. Evaluate

```text
ell_cell = n_minus*log(q_minus) + n_zero*log(q_zero) + n_plus*log(q_plus)
```

with zero-count-safe `xlogy` semantics. A probability triple must arrive with numerical evidence. Negative probabilities, nonfinite values, disagreement beyond tolerance, or mass error beyond a declared roundoff allowance raise a structured numerical error. Tiny renormalization is allowed only under a documented tolerance and the correction is retained in the audit; silent clipping is forbidden.

“Exact finite-cell likelihood” means the finite multinomial statistical model rather than the limiting Poisson model. It does **not** mean that its floating-point probability evaluation is mathematically exact.

### Deterministic posterior integration

The first workflow is only two-dimensional, so deterministic quadrature is preferable to MCMC. Build a nested/refinable tensor rule on compact `K`, evaluate log prior plus log likelihood, include the quadrature weights in normalization, and use log-sum-exp arithmetic. Represent the numerical posterior as nodes and normalized probability masses on the common refined grid. Transform those masses to `alpha`, `beta`, `w_±`, and `tau_±` for summaries and predictive quantities.

The full and reduced posterior comparison must use the same nodes, prior evaluations, and integration weights. The reported approximate total variation is

```text
0.5 * sum(abs(full_mass - reduced_mass))
```

after both posterior grids pass independent refinement checks. Reusing a common integration grid makes the posterior difference interpretable; it does not eliminate density or quadrature error.

### Result selection

The first release should support these machine-readable decisions:

| Decision | Meaning | Selected posterior |
|---|---|---|
| `REDUCED_REFERENCE_AGREEMENT` | Observed-data posterior TV is below the configured tolerance and both numerical refinements passed | finite-cell posterior |
| `FULL_FALLBACK` | Reference posterior was available but agreement or reduced numerical checks failed | full posterior |
| `REFUSED` | Neither supported posterior completed reliably, the design was outside scope, or inputs violated assumptions | none |
| `REDUCED_BOUND_PASSED` | Reserved for a future proved, conservatively enclosed finite-sample bound | finite-cell posterior |

Do not emit `certified=True` in version 1. A user-requested “reduced only” research mode may return a reduced posterior, but its decision must be `UNASSESSED_REDUCED`, display a prominent warning, and never be the default.

## Numerical Architecture: Fast, Reference, and Enclosed Are Different

| Protocol | Backend strategy | Evidence supplied | Permitted claim |
|---|---|---|---|
| `FAST` | Guarded SciPy `S0` piecewise CDF/SF/PDF with vectorization and basic identities | convergence/status checks and recorded SciPy configuration | ordinary approximate computation |
| `REFERENCE` | Package-owned deterministic Fourier inversion/integration plus a guarded SciPy piecewise comparison and refinement | two implementation paths, mass/reflection/refinement discrepancies | independently checked numerical reference |
| `ENCLOSED` (future) | Analytic tail bounds plus interval- or directed-rounding quadrature with a proved error budget | lower/upper probability and posterior-function bounds | may feed a mathematical certificate |

SciPy's current official `levy_stable` API exposes `parameterization`, `pdf_default_method`, `cdf_default_method`, and quadrature controls as mutable class variables; the documented default parameterization is `S1`, while this project requires `S0`. Consequently:

- never change SciPy stable-law configuration at import time;
- place every SciPy call behind `_numerics/scipy_s0.py`;
- acquire a process-local re-entrant lock, snapshot every setting that will be touched, set all required values explicitly, call SciPy, and restore settings in `finally`;
- test restoration on success, warnings, and exceptions;
- do not claim thread isolation from unrelated code that calls SciPy directly; use process-based parallelism for package-controlled parallel work; and
- record the effective SciPy settings with every numerical result.

The guarded adapter limits accidental leakage but cannot make process-wide mutable state suitable for rigorous enclosure. `ENCLOSED` must use a package-owned implementation whose assumptions and rounding behavior can be audited. SciPy's own integration documentation warns that adaptive algorithms and their error estimates cannot guarantee correctness for arbitrary integrands; therefore a `quad` error estimate alone cannot be promoted to the certificate's numerical remainder.

The spike's `FourierStable` implementation is useful as a second algorithmic lineage, but production `fourier_s0.py` needs explicit domain decomposition, truncation bounds, vectorized convergence status, and tests before use. It should not inherit `X_CUT=30`, `T_MAX=10`, or `QUAD_N=900` as unexplained package constants.

## Accuracy and Certificate Scope

Accuracy evidence should be a typed object, not a Boolean:

```text
AccuracyReport
├── level: UNASSESSED | REFERENCE_COMPARED | EMPIRICALLY_CALIBRATED | PROVED_BOUND
├── target: POSTERIOR_TV | CELL_PROBABILITY | PREDICTIVE_FUNCTIONAL | ...
├── scope: OBSERVED_DATA | PRIOR_PREDICTIVE_MEAN_OVER_K
├── value / upper_bound
├── numerical_error_budget
├── statistical_remainder
├── assumptions
├── method_id and proof_id
└── passed configured policy: bool
```

The future theorem-derived `B_n` belongs in a separate concrete type, for example `ExpectedPriorPredictiveTVBound`. It must state all of:

- the prior and compact interior `K` over which it applies;
- the fixed `n`, `r`, threshold/partition, and nuisance scope;
- that its target is expected posterior TV under the full prior predictive distribution;
- the analytic reconstruction/statistical term;
- the conservatively enclosed numerical term;
- the proof/method version; and
- whether the requested tolerance is met.

An expected-TV bound is not a guarantee that the posterior for every observed dataset lies within that number. The first release's full-versus-reduced posterior TV is a useful observed-data audit but does not prove the false-safe probability or the prior-predictive mean guarantee. Keep these two report types impossible to confuse in both Python types and serialized field names.

## Serialization and Audit Schema

`StableBoundaryResult.save(path)` should write a non-executable bundle:

```text
fit-result/
├── manifest.json
├── posterior.npz
└── SHA256SUMS
```

`manifest.json` uses a versioned JSON Schema identifier such as `stableboundary.fit/v1`. `posterior.npz` contains numeric arrays only and is always loaded with `allow_pickle=False`; NumPy explicitly recommends disabling pickle for security and portability. Do not serialize callables, SciPy frozen-distribution instances, or Python class objects.

Minimum manifest topology:

```json
{
  "schema": "stableboundary.fit/v1",
  "package": {"name": "stableboundary", "version": "...", "git": "..."},
  "environment": {"python": "...", "numpy": "...", "scipy": "...", "platform": "..."},
  "model": {"family": "stable", "parameterization": "nolan_s0"},
  "input": {"n": 3740, "dtype": "float64", "sha256": "...", "raw_data_saved": false},
  "nuisance": {"mode": "known", "loc": 0.0, "scale": 1.0, "source": "external"},
  "design": {"r": 0.03, "L": 3.5066, "u": 4.9054, "region": {}, "rule": "theorem-loglog-v1"},
  "counts": {"minus": 0, "zero": 3734, "plus": 6},
  "prior": {},
  "numerics": {"protocols": [], "tolerances": {}, "refinements": []},
  "accuracy": {"level": "reference_compared", "scope": "observed_data", "target": "posterior_tv"},
  "decision": {"status": "...", "selected": "...", "reasons": []},
  "summaries": {},
  "warnings": [],
  "timings": {}
}
```

Do not save raw observations by default. Fingerprint a canonical little-endian float64 copy after documenting how signed zero and missing values are handled. The digest establishes identity, not privacy or provenance. Store seeds and randomized indices once pilot splitting exists. Validate the schema when saving and loading, reject unknown future major schema versions, and migrate old versions through explicit pure functions.

## First Theorem-Faithful Vertical Slice

The first end-to-end slice should be one fixed, reproducible standardized experiment rather than a broad API with placeholders:

1. Use a prespecified compact interior region, proper truncated Gamma-Beta prior, known `loc=0`, `scale=1`, and a design from the manuscript's threshold rule. A useful slow integration case is the spike's `r=.03`, `n=3740`, approximately `c=4`; add a smaller smoke fixture for routine CI.
2. Generate or freeze one sample from a package-controlled `S0` simulator. Simulation itself passes through the same parameterization guard and records its seed.
3. Run `plan`, standardization, signed partition, and exact finite-cell probability evaluation.
4. Compute the finite-cell `(h,p)` posterior by deterministic quadrature.
5. Compute an independent full stable posterior on the identical posterior nodes.
6. Refine both numerical routes, calculate observed-data posterior TV, and execute the explicit selection/fallback policy.
7. Report posterior summaries for `alpha`, `beta`, `tau_plus`, `tau_minus`, predictive positive/negative exceedance probabilities, tail event counts, and a plainly named skewness-information diagnostic.
8. Save, reload, and compare the audit bundle byte-for-byte except for explicitly volatile timing fields.
9. Expose the run in `examples/standardized_fit.py` and in a slow integration test. The example passes only if it produces a valid posterior and an explicit decision; it does not require the reduced method to win.

The slice should deliberately include a fallback fixture. A package that only demonstrates its favorable path has not tested its central safety claim.

## Later Pilot-Conditioned Four-Parameter Path

Do not estimate location and scale by standardizing all observations once and then pretending the resulting thresholds are fixed. The later workflow should preserve likelihood validity as follows:

1. Before inspecting values, draw a randomized index split `(A,B)` from a recorded seed.
2. Use pilot observations `x_A` to construct raw, fixed bin boundaries. Include multiple central bins, not only two tails, because two signed counts cannot identify four continuous parameters.
3. Retain the full pilot likelihood for `(alpha,beta,loc,scale)`.
4. Conditional on the realized pilot, group `x_B` once. At any parameter proposal, evaluate bin probabilities using the same raw boundaries:

   ```text
   q_j(theta) = F_S0((b_j-loc)/scale; alpha,beta)
              - F_S0((a_j-loc)/scale; alpha,beta)
   ```

5. Use the joint likelihood

   ```text
   L(theta) = L_full(x_A | theta) * Mult(counts_B | q(theta; boundaries_A)).
   ```

6. Propagate uncertainty in all four parameters and label `loc`/`scale` as estimated only in this workflow.

The bins are conditional design objects and cannot be recomputed at each posterior proposal. The pilot likelihood cannot be discarded: doing so treats data-derived boundaries as external constants while losing the information and uncertainty in the pilot. The split indices, raw boundaries, pilot construction rule, and all seeds become part of the audit schema.

The four-dimensional posterior should initially use a compact deterministic validation implementation (sparse or nested quadrature on restricted test regions), then a separately validated adaptive importance-sampling/NPMC implementation for practical fits. Do not jump directly to a black-box MCMC result: the known-nuisance two-dimensional workflow is the numerical ground truth needed to test marginal slices, transformations, and grouped-likelihood calculations first.

The existing theorem does not establish a four-parameter reduction or a certificate for this workflow. Its result status must remain `REFERENCE_COMPARED` or `EMPIRICALLY_CALIBRATED` until a nuisance-aware theorem and bound exist.

## Validation Architecture

Validation is part of the design, not an after-the-fact test folder.

### Layer 1: algebraic and contract tests

- Round-trip `(alpha,beta) ↔ (h,p)` for fixed `r` and verify `tau_+ + tau_- = 2-alpha` and `(tau_+-tau_-)/(tau_++tau_-)=beta` away from zero.
- Verify reflection: `q_plus(alpha,beta,u) = q_minus(alpha,-beta,u)`.
- Verify `S0(alpha=2, beta, loc=0, scale=1)` is beta-invariant and matches `N(0,2)`.
- Reject exact `h=0`, one-sided `p`, inconsistent `n`, invalid scales, nonfinite observations, invalid thresholds, and ambiguous parameterization strings.
- Verify design construction never reads observation values.

### Layer 2: numerical cross-validation

- Compare guarded SciPy piecewise values against package-owned Fourier values over a prespecified `(alpha,beta,x)` grid, including extreme tails and values close to two.
- Compare CDF/SF cell probabilities with independently integrated density values; test tail probabilities without subtracting from one.
- Require nonnegative probabilities and normalized mass within tolerance; inject corrupted backends and verify structured failure rather than clipping.
- Refine spatial/Fourier quadrature, posterior nodes, and tolerances separately so agreement is not produced by a shared discretization.
- Reproduce selected immutable values from `results.json`, but never make the spike module an imported dependency.
- Run concurrency tests that alter SciPy settings before a fit, execute successful and failing calls, and verify exact restoration. Run package parallelism in processes and compare results with serial execution.

### Layer 3: inferential correctness

- Test cell log likelihoods against hand-calculated multinomial cases, including zero counts.
- Use analytically tractable limiting Gamma-Beta cases to test posterior normalization and transformations, while keeping that likelihood explicitly marked `LIMITING`.
- Verify that full and reduced posteriors share prior values and quadrature weights before computing TV.
- Under simulation, run simulation-based calibration for posterior ranks, frequentist coverage, predictive tail calibration, and prior-to-posterior information diagnostics.
- Prespecify parameter grids, replications, random seeds, Monte Carlo uncertainty, and failure accounting. Failed computations count as failures rather than disappearing from summaries.

### Layer 4: reduction and fallback behavior

- Include regimes known from the spike to retain only about 58--78% of finite-`r` Hellinger information and regimes closer to the limit.
- Test both `REDUCED_REFERENCE_AGREEMENT` and `FULL_FALLBACK` outcomes.
- Verify that changing the TV tolerance changes only the policy decision, never the computed posterior objects.
- When a future bound exists, test coverage of the bound, false-safe frequency, deliberate bound violations, and every assumption boundary. An empirical false-safe study validates implementation behavior but does not prove the bound.

### Layer 5: package and artifact validation

- Build wheel and source distribution and test the installed wheel in a clean environment; do not run release tests against repository imports.
- Type-check the public API and ship `py.typed`.
- Save and reload every decision state with `allow_pickle=False`; validate content hashes and reject altered bundles.
- Run the documented quickstart as a test.
- Benchmark end-to-end time including planning, accuracy assessment, and fallback. A fast reduced likelihood is not a package speed-up if the default policy always pays the full-likelihood cost.

## Dependency-Ordered Build Sequence

1. **Freeze mathematical and serialization contracts.** Implement coordinate types, `BoundaryRegion`, `BoundaryDesign`, result enums, exceptions, audit schema, and package scaffold. No density code yet.
2. **Build and falsify numerical primitives.** Implement the guarded SciPy adapter, independent Fourier reference, checked probability triples, and refinement records. Reproduce selected spike values.
3. **Implement the exact finite-cell posterior.** Add partitioning, multinomial likelihood, proper compact priors, deterministic two-dimensional posterior quadrature, transformations, and predictive signed-tail probabilities.
4. **Implement the full reference posterior.** Reuse posterior nodes and priors but use an independent full density path. Add observed-data posterior-TV comparison.
5. **Complete the first vertical slice.** Add the decision state machine, immutable result, save/reload bundle, one favorable example, one fallback example, and clean-wheel tests. This is the first point at which the user should be asked to fit data and inspect results.
6. **Run the prespecified validation study.** Simulation-based calibration, coverage, posterior TV, predictive performance, runtime, memory, and numerical failure rates. Freeze benchmarks and kill criteria before looking at results.
7. **Develop the finite-sample certificate as a separate research workstream.** Derive the reconstruction bound, implement numerical enclosures, validate the bound, and only then activate `REDUCED_BOUND_PASSED`. Do not block the reference-audited package prototype on an unproved interface, but do block any certification claim.
8. **Add the pilot-conditioned joint workflow.** Fixed randomized split, raw bins, pilot likelihood, grouped main likelihood, four-dimensional reference validation, and then practical adaptive importance sampling.
9. **Add an empirical example and release engineering.** Use only scientifically defensible IID data or appropriately modeled residuals; document preprocessing as part of the analysis rather than hide it in the package.

## Anti-Patterns to Avoid

### Packaging the asymptotic posterior as the default

The spike shows that multinomial-to-Poisson error can be tiny while finite cell means remain materially different from limiting means. Use finite stable cell probabilities by default; the Gamma-Beta posterior is a named benchmark.

### Silent `S1`/`S0` switching

SciPy defaults and user-modified class state can silently change the modeled distribution. Every backend call must force and audit `S0` through an isolated adapter; every result records `nolan_s0`.

### Calling ordinary quadrature “certified”

Backend agreement and adaptive error estimates are evidence, not conservative mathematical enclosures. Keep `REFERENCE_COMPARED` and `PROVED_BOUND` different in type, status, and documentation.

### Same-sample plug-in standardization

Estimating location/scale from all data and then treating them as fixed changes thresholds and understates uncertainty. Use independently known nuisance values in version 1 and the joint pilot likelihood later.

### Re-binning inside the parameter loop

Bins are observed-data statistics. Reassigning observations as `loc` and `scale` change produces a parameter-dependent statistic rather than the declared grouped-data likelihood. Fix raw boundaries once after the pilot.

### A backend-driven public API

Letting users inject arbitrary density functions makes results unauditable and any reliability label meaningless. Keep numerical backends private and versioned.

### Pickled result objects

Pickle couples artifacts to Python code and permits code execution when loading. Use JSON plus numeric NPZ arrays with pickle disabled.

### Treating zero tail events as evidence for symmetric skewness

No signed events primarily mean that sign allocation was not learned. Report event counts and prior/posterior information directly; do not turn posterior mean `beta≈0` into a claim of empirical symmetry.

## Scalability Considerations

| Concern | First vertical slice | Validated computational paper | Later four-parameter workflow |
|---|---|---|---|
| Likelihood dimension | 2-D compact `(h,p)` | many simulation replicates over fixed 2-D grids | 4-D `(alpha,beta,loc,scale)` |
| Data cost | counts are `O(n)` once; full audit is roughly `O(n × grid)` | cache parameter-grid cell probabilities; parallelize replicates by process | pilot full likelihood plus cheap grouped likelihood |
| Memory | stream observations into counts; retain data only for full audit | store summaries and seeds, not every posterior by default | retain pilot and grouped counts |
| Parallelism | serial default | process pools, never threads around mutable SciPy stable state | process-based adaptive populations |
| Reproducibility | deterministic quadrature and one seed | hierarchical seed sequences per replicate | persist split indices, particles, and resampling seeds |
| Performance claim | none until full end-to-end benchmark | include assessment and fallback costs | compare complete workflows, not likelihood kernels alone |

## Confidence and Open Research Dependencies

| Area | Confidence | Reason |
|---|---|---|
| Public/private package boundaries | HIGH | Standard `src` package pattern and a narrow theorem-defined use case. |
| Standardized finite-cell posterior | HIGH | Direct finite multinomial likelihood on the manuscript's two-dimensional compact parameter set. |
| Full reference audit | MEDIUM-HIGH | Feasible with existing density algorithms, but tail probability and posterior-grid tolerances require empirical validation. |
| Runtime advantage in the first default policy | LOW | Computing the full posterior for every reference audit removes most speed gain; this release establishes correctness, not speed. |
| Finite-sample certificate | MEDIUM-LOW | Architecture can reserve the interface, but the bound and conservative numerical enclosure remain research deliverables. |
| Pilot-conditioned four-parameter inference | MEDIUM | The conditional likelihood construction is sound, but a nuisance-aware reduction theorem and production posterior engine remain unvalidated. |

The largest architecture risk is marketing ahead of mathematics: the package can be useful as a reference-audited Bayesian fitter before it is certified, but it must describe that status exactly. The second risk is numerical dependence between “independent” checks. The SciPy and package-owned Fourier paths must not share tail corrections, grids, or silent probability repairs in a way that makes their agreement circular.

## Sources

- [Theoretical manuscript](../../gaussian_boundary_stable_manuscript.tex) — model, compact local parameter set, threshold, signed-cell equivalence, posterior scope, and nuisance limitations. **HIGH confidence; primary project source.**
- [Executed spike verdict](../spikes/001-gaussian-boundary-stable/README.md), [frozen conventions](../spikes/001-gaussian-boundary-stable/CONVENTIONS.md), and [machine-readable results](../spikes/001-gaussian-boundary-stable/results.json) — pre-asymptotic behavior and existing numerical checks. **HIGH confidence for reported computations; not production evidence.**
- [SciPy `levy_stable` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html) — current `S0`/`S1` controls, mutable class settings, density/CDF methods, and numerical controls. **HIGH confidence; official documentation.**
- [SciPy integration documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html) and [integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) — returned error estimates and limitations of adaptive sampling. **HIGH confidence; official documentation.**
- [Python Packaging User Guide: `src` layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) and [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) — package isolation and PEP 621 metadata. **HIGH confidence; official PyPA guidance.**
- [NumPy file I/O guidance](https://numpy.org/doc/stable/user/how-to-io.html) — NPZ storage and the security/portability reason to disable pickle. **HIGH confidence; official documentation.**

## RESEARCH COMPLETE
