# Project Research Summary

**Project:** `stableboundary`  
**Domain:** Auditable Bayesian inference and scientific software for univariate alpha-stable laws near the Gaussian boundary  
**Researched:** 2026-08-24  
**Confidence:** HIGH for the package, standardized inference, and validation architecture; MEDIUM-LOW for the unproved certificate and later four-parameter workflow

## Executive Summary

`stableboundary` should be built as a narrow scientific Python package that makes the theoretical signed-cell reduction executable without overstating what the theorem proves. The package is not a generic stable-law fitter. Its first useful product is a reproducible, known-location/scale analysis that plans a data-independent local experiment, forms the exact finite three-cell multinomial likelihood, computes the two-dimensional posterior deterministically, compares it with an independently evaluated full stable posterior, and then returns an explicit reduced, fallback, or refusal decision with a complete audit record. The limiting Poisson/Gamma--Beta posterior remains a named approximation and benchmark, never the finite-sample default.

The implementation should use CPython 3.12+, NumPy 2.2+, SciPy 1.18+, a `src/` layout, PEP 621 metadata, and Hatchling. The package owns the mathematical contracts, backend protocols, error/status objects, decision policy, and result schema. SciPy is accessed only through a guarded `S0` adapter and supplies a fast/bootstrap route and the full-likelihood reference; it is not the sole scientific authority. A package-owned Fourier/inversion implementation must provide an independent lineage and become the primary finite-cell probability route before reliability claims are released. Deterministic quadrature is the correct first posterior engine because the validated problem is two-dimensional and compact; probabilistic-programming frameworks and SMC would add uncertainty before there is a trustworthy reference solution.

The dominant risks are scientific, not syntactic: confusing `S0` and `S1`, treating stable scale as Gaussian standard deviation, adapting thresholds to the observations, hiding weak identification of beta, losing rare probabilities numerically, silently clipping failures, or presenting simulated agreement as a mathematical certificate. The build order must therefore make contracts, numerical oracles, refusal behavior, and independent validation prerequisites to inference features. The later four-parameter goal is retained, but it is a distinct pilot-conditioned workflow: include the pilot full likelihood, freeze raw main-sample bins after the pilot, and propagate uncertainty in `(alpha, beta, loc, scale)`. It cannot be reached by plugging sample location and scale into the two-count theorem.

## Resolved Decisions

Research files contained three decisions that the roadmap must not leave ambiguous:

| Conflict | Decision | Consequence |
|---|---|---|
| `PROJECT.md` says Python 3.11+; stack research recommends 3.12+ | **Require CPython 3.12+ and test 3.12--3.14.** Current NumPy, SciPy, ArviZ, and Scientific Python support windows make 3.11 a separate legacy dependency branch with no inferential benefit. | Update project metadata when scaffolding; use `numpy>=2.2` and `scipy>=1.18` without speculative upper bounds. |
| Package-owned numerics versus SciPy backend | **Own the protocol and eventually the finite-cell kernel; use SciPy only through a snapshot/lock/restore `S0` adapter.** SciPy may bootstrap the first working slice and remain the full-likelihood reference, but a package-owned Fourier path must independently check it and become the release finite-cell route. | No direct `scipy.stats.levy_stable` calls outside the adapter; no import-time global mutation; no reliability claim based on one engine agreeing with itself. |
| Known nuisance in version 1 versus the desired four estimates | **Ship and validate known/independently calibrated location and scale first; estimate all four parameters only in the later pilot-conditioned workflow.** | The first result estimates `alpha`, `beta`, and signed tail gaps while reporting `loc`/`scale` as fixed. It must never advertise a four-parameter estimate. The joint workflow is a later release gate, not abandoned scope. |
| Working package versus mathematical certificate | **A reference-audited package may work before certification, but cannot use `certified`.** | Version 0.1 may return observed-data reference agreement, full fallback, refusal, or unassessed research output. Certification stays constructively impossible until a finite-sample theorem and conservative numerical enclosure exist. |

## Key Findings

### Recommended Stack

Use a deliberately small, pure-Python scientific core. Keep NumPy and SciPy as the only mandatory third-party runtime dependencies through the known-nuisance release. Optional plotting, ArviZ interoperability, ball arithmetic, and later stochastic inference must not enlarge or destabilize the core contract.

**Core technologies:**

- **CPython 3.12+**: supported runtime; test 3.12, 3.13, and 3.14.
- **NumPy 2.2+**: arrays, deterministic quadrature nodes, immutable numeric payloads, and explicit `Generator`-based random streams.
- **SciPy 1.18+**: guarded Nolan piecewise stable calculations, special functions, log-domain algebra, and independent quadrature checks.
- **Hatchling + PEP 517/621**: standards-based pure-Python wheel and source builds from a `src/` layout.
- **uv**: committed development/reproduction lock only; it must not replace standards-compatible `pip` and `python -m build` workflows.
- **pytest, Hypothesis, Ruff, strict mypy, coverage.py, ASV**: contract/property tests, code quality, typing, and controlled performance history.
- **Sphinx, MyST-NB, numpydoc**: executable public-API documentation; Matplotlib and ArviZ remain optional extras.
- **Python-FLINT, later and optional**: ball arithmetic for finite-path audits; it does not confer certification without analytic truncation and error-propagation proofs.

Do not begin with PyMC, Stan, NumPyro, JAX, native extensions, or SMC. The first posterior is a two-dimensional compact integral with a deterministic reference. Add a package-owned adaptive-tempering SMC engine only when the four-dimensional nuisance workflow requires it and only after validation against deterministic ground truth.

### Expected Features and Requirements Implications

**Must have for the first usable package:**

- Clean wheel and source installation with `src/stableboundary/`, PEP 621 metadata, `py.typed`, and a narrow curated import surface.
- Frozen typed objects for Nolan `S0` parameters, boundary coordinates, compact local region, prior, prespecified design, known nuisance provenance, numerical evidence, decision state, and audit schema.
- Lossless, tested mappings among `alpha`, `beta`, `delta=2-alpha`, `(r,h,p)`, signed weights, and `tau_plus/tau_minus`.
- A deterministic, data-free `plan(n, r, region)` that fixes the threshold and rejects invalid or theorem-external designs.
- Seeded `S0` simulation with the bit generator and environment recorded.
- Exact-model finite three-cell multinomial posterior for known location and scale, calculated with finite stable probabilities and deterministic two-dimensional quadrature.
- Summaries for `alpha`, `beta`, `h`, `p`, `tau_plus`, `tau_minus`, and predictive signed-tail probabilities; location and scale explicitly marked fixed.
- Identification diagnostics that label beta as identified, weak, prior-dominated, or unidentified rather than turning a near-zero posterior mean into evidence of symmetry.
- A separately evaluated full `S0` posterior on a common refined grid, observed-data posterior-TV comparison, explicit reduction/fallback/refusal policy, and at least one deliberately triggered fallback example.
- Immutable result and versioned JSON-plus-NPZ audit bundle with checksums and `allow_pickle=False`; raw observations are not persisted by default.
- Executable quickstart, fixed-seed end-to-end fit, clean-wheel integration test, and machine-readable numerical/fallback reports.

**Differentiators that require evidence before admission:**

- A finite-sample reduction bound with accurately stated scope and conservatively enclosed numerical remainder.
- Pre-data scale/partition planning that does not use the truth, main-sample extremes, or a full posterior.
- Automatic reduction that becomes computationally useful because the proved assessment can avoid routinely computing the full posterior.
- Pilot-conditioned joint inference for all four conventional parameters.
- Misspecification warnings/refusals, posterior-distance validation, and a transparent approximation ladder.
- Disjoint multiscale cells only if held-out mean TV improves by at least 25% in two prespecified pre-asymptotic regimes.

**Explicit non-requirements for the initial milestone:**

- General stable fitting across the full alpha range.
- Exact-Gaussian model selection or a Gaussian mixture spike.
- Plug-in nuisance estimates presented as known or certified.
- Raw financial/time-series automation, arbitrary censoring/weights, or automatic preprocessing.
- An independent R numerical core, a standalone software paper, or a public SMC engine.

### Architecture Approach

Use a small public API over private, replaceable components. Public users create immutable designs and priors and call `plan()` and `fit()`; they do not inject arbitrary numerical backends. Internally, statistical models, numerical evaluation, accuracy evidence, and policy decisions are separate layers. This separation prevents a backend fallback from silently changing the inferential meaning of a result and prevents ordinary numerical agreement from being promoted into a proof claim.

**Major components:**

1. **Parameters and design** — define `S0`, coordinate transformations, compact theorem-supported regions, proper local priors, threshold rule, and known-nuisance provenance.
2. **Input gate and partition** — validate one-dimensional finite IID input, standardize only with declared nuisance values, and form immutable negative/central/positive counts under a prespecified threshold.
3. **Numerical protocols** — package-owned value-plus-status/error interfaces; guarded SciPy `S0` adapter; independent Fourier probability/density implementation; deterministic quadrature and future enclosure path.
4. **Likelihood ladder** — exact finite-cell multinomial as default; full stable likelihood as reference/fallback; finite-mean Poisson and limiting Gamma--Beta as explicitly labelled benchmarks only.
5. **Posterior engine** — refinable tensor Gauss--Legendre integration in log space over compact `(h,p)`, independently cross-checked with adaptive cubature.
6. **Accuracy and decision** — typed evidence with target and scope, followed by a closed state machine selecting reduced, full fallback, or refusal without modifying computed posteriors.
7. **Results and serialization** — immutable posterior grid/weights, summaries, diagnostics, full fallback chain, environment record, and JSON/NPZ bundle.
8. **Pilot workflow, later** — randomized split, full pilot likelihood, frozen raw main bins, grouped main likelihood, and a separately validated four-dimensional engine.

### First Vertical Slice

The first implementation target is one complete, reproducible standardized experiment—not a broad collection of stubs:

```text
prespecified n, r, K, prior, known loc/scale
                  |
                  v
             plan design
                  |
raw x -> validate/standardize -> fixed signed counts
                  |                         |
                  |                         v
                  |            finite cell probabilities
                  |                         |
                  |                 multinomial posterior
                  |                         |
                  +----> full stable posterior
                                    |
                          common-grid TV audit
                                    |
                     reduced / fallback / refused
                                    |
                    immutable result + audit bundle
```

The slice passes only when it can simulate a fixed-seed `S0` sample, plan without inspecting sample values, compute and refine both posterior routes, produce finite summaries and predictive tail probabilities, serialize/reload the result, and make its method decision explicit. It must include both a favorable fixture and a fallback fixture. Its honest status is `REFERENCE_COMPARED` or `RESEARCH_UNCERTIFIED`, not `CERTIFIED`.

### Validation Architecture

Validation is part of the product architecture and must use independent evidence paths. A large test suite sharing one tail routine is not independent validation.

| Layer | Required evidence | Exit gate |
|---|---|---|
| **V0 Contract** | Coordinate round trips, prior Jacobian, immutable/data-free design, `S0` and `N(0,2)` semantics, closed decision truth table, no global mutation | Invalid states refuse deterministically; certified construction is impossible |
| **V1 Numerical kernel** | Normal/Cauchy/converted Levy anchors, characteristic-function and reflection identities, positivity, normalization, tail monotonicity, convergence, hostile SciPy settings, independent oracle corpus | Under-resolution or backend disagreement raises; every supported probability carries usable error evidence |
| **V2 Posterior engine** | Hand-computed multinomial cases, zero/rare/high counts, log-domain normalization, boundary/mode detection, quadrature refinement and independent cubature | Refined-grid posterior TV changes by at most `.002` in the benchmark region |
| **V3 Bayesian calibration** | SBC, repeated-sampling coverage, predictive calibration, beta-identification behavior, prior/threshold sensitivity | Empirical 95% coverage in 92--98%, with Monte Carlo half-width at most `.015`, unless superseded by a stricter frozen specification |
| **V4 Reduction accuracy** | Full-versus-grouped TV/Hellinger/Wasserstein, action regret, false-safe and false-refusal behavior over frozen regimes | Mean TV at most `.05`, 90th-percentile TV at most `.10`, action regret at most 5%, false-safe at most 5% in declared regimes |
| **V5 Numerical independence** | Production finite-cell backend, high-precision/Fourier oracle, and full-posterior reference do not share critical tail code | Backend swap changes posterior TV by at most `.005`; unexplained discrepancy blocks claims |
| **V6 Misspecification stress** | Gaussian, Student-t, tempered, contaminated, dependent, stochastic-volatility, censored, rounded, and shifted/scaled negative controls | Unsupported cases warn, fall back, or refuse; none receive a certified label |
| **V7 Performance** | Cold/warm end-to-end runtime, memory, assessment and fallback cost, matched tolerances and controlled hardware | At least 10x end-to-end speed-up in a nontrivial qualifying regime; computational failure below 1% |
| **V8 Distribution** | Clean wheel/sdist install, minimum/latest dependency matrix, Windows/macOS/Linux smoke runs, docs, serialization and public-symbol tests | Installed artifacts reproduce the quickstart without repository-path imports |
| **V9 Paper reproduction** | Frozen validation manifest, exact release, data provenance, one-command tables/figures, clean-checkout reproduction | Independent reproduction before manuscript submission |

Monte Carlo validation continues until the standard error of mean TV is at most `.005` and binomial half-widths for coverage/false-safe rates are at most `.015`. Failures remain in denominators. The package must never regenerate its numerical oracle fixtures from production code during ordinary tests.

### Critical Pitfalls

1. **Design-dependent `h` presented as an ordinary parameter** — make `delta` and signed gaps the scientific quantities, carry `r`, `K`, threshold and prior everywhere, and never infer `r` from main-sample extremes.
2. **False beta identification** — report tail counts and prior-to-posterior information; suppress estimate-like beta language when signed events contain no allocation information.
3. **Two counts presented as four-parameter inference** — restrict the first workflow to independently known nuisance values; later use pilot likelihood plus fixed grouped main likelihood.
4. **Silent approximation or method switching** — exact finite-cell likelihood is the default; Poisson variants, fallbacks, tolerance changes, and refusal reasons are named and serialized.
5. **Tail numerical corruption** — never use `1-cdf`, arbitrary epsilons, clipped negative densities, or mass normalization to hide failure; use direct survival/tail computation, `log1p`, `xlogy`, and structured error propagation.
6. **Circular validation** — keep production cells, numerical oracle, full posterior, generators, and future certificate evaluation independent enough for a defect in one path to be observable.
7. **Certification by vocabulary** — simulation, full-posterior agreement, higher precision, or a SciPy error estimate cannot create `PROVED_BOUND`; proof scope and conservative numerical enclosure are both necessary.
8. **Empirical overreach** — the initial model is IID and stable; diagnostics can identify reasons for concern but cannot prove those assumptions. Omit a weak application rather than force raw returns or unsuitable data into the package.

## Implications for Roadmap

### Phase 1: Mathematical Contract and Package Scaffold

**Rationale:** Every later numerical result depends on unambiguous coordinates, scale semantics, design scope, state vocabulary, and artifact format.  
**Delivers:** Python 3.12+ `src/` package; `pyproject.toml`; typed frozen parameter/design/prior/nuisance types; result states and exceptions; audit schema; curated public API; wheel smoke test.  
**Addresses:** Installability, `S0` parameterization, coordinate conversion, reproducibility foundations.  
**Avoids:** `r/h` ambiguity, S0/S1 drift, scale/SD confusion, uncertified status leakage, local-source imports.  
**Research flag:** Standard packaging patterns; skip additional phase research.

### Phase 2: Stable Numerical Kernels and Oracle Corpus

**Rationale:** Inference cannot be trusted until rare signed probabilities and full densities fail visibly when unresolved.  
**Delivers:** Guarded SciPy adapter; checked numerical return types; independent Fourier/inversion prototype; special-law and high-precision fixtures; reflection/normalization/tail tests; hostile-global-state tests.  
**Addresses:** Reproducible simulation, exact-model cell probabilities, full density primitive, numerical diagnostics.  
**Avoids:** Cancellation, clipping, shared-code agreement, import-time SciPy mutation.  
**Research flag:** Deep phase research required for Fourier tail continuation, truncation control, and supported delta floor.

### Phase 3: Exact Finite-Cell Posterior

**Rationale:** This is the theorem-faithful inferential core and the shortest route to a visible working fit.  
**Delivers:** Immutable signed counts; finite multinomial log likelihood; compact Gamma--Beta prior; refinable deterministic posterior grid; transformations, summaries, beta diagnostics, and predictive tails.  
**Addresses:** Known-nuisance fit and meaningful estimates of `alpha`, `beta`, and `tau_plus/tau_minus`.  
**Avoids:** Limiting-likelihood substitution, zero-count log failures, coarse normalized but inaccurate posterior grids.  
**Research flag:** Standard deterministic integration patterns after Phase 2; no separate broad research phase needed.

### Phase 4: Full Reference, Decision Policy, and Working Vertical Slice

**Rationale:** The user must be able to run and inspect the package before paper submission; a result without a tested fallback path does not meet the core value.  
**Delivers:** Full stable posterior on common nodes; observed-data posterior-TV audit; reduced/reference-agreement, full-fallback, refused, and unassessed states; immutable result; save/reload bundle; favorable and fallback examples; installed-wheel quickstart.  
**Addresses:** Full reference fit, explicit method state, reproducible audit, executable documentation.  
**Avoids:** Silent switches, favorable-case-only demos, claims that observed agreement is certification.  
**Research flag:** Targeted numerical planning may be needed for full-likelihood cost and grid reuse; architecture is otherwise settled.

### Phase 5: Prespecified Statistical and Computational Validation

**Rationale:** A working demonstration tests plumbing; it does not establish calibration, reduction fidelity, robustness, or practical value.  
**Delivers:** Frozen validation manifest; SBC and coverage study; posterior-distance ladder; adversarial and misspecification fixtures; ASV/end-to-end benchmarks; failure and fallback accounting; manuscript-ready reproducible artifacts.  
**Addresses:** Posterior correctness, identification behavior, model-scope warnings, performance evidence.  
**Avoids:** Post hoc regime choice, dropping failed replications, unfair kernel-only speed comparisons, package self-validation.  
**Research flag:** No open-ended ecosystem research; statistical design must be reviewed before confirmatory runs.

### Phase 6: Finite-Sample Accuracy Bound and Enclosed Numerics

**Rationale:** This is the differentiating computational-method contribution and the only route from reference-audited correctness to fast pre-data reduction.  
**Delivers:** Proved bound with declared target/scope; decomposition of reconstruction and numerical terms; optional Python-FLINT enclosure path plus analytic tail/truncation bounds; validated false-safe behavior; activation of the reserved bound-passed state only after all gates pass.  
**Addresses:** Genuine certification, prospective safe/unsafe planning, computational speed without routinely evaluating the full posterior.  
**Avoids:** Simulation-labelled certificates, dataset-specific wording for expected-TV results, incomplete error budgets.  
**Research flag:** Highest-risk research phase; derive the mathematics before scheduling implementation as routine engineering.

### Phase 7: Pilot-Conditioned Four-Parameter Inference

**Rationale:** Applied users ultimately need `(alpha, beta, loc, scale)`, but this must be built on the validated two-dimensional reference rather than hiding nuisance-induced failures.  
**Delivers:** Recorded randomized split; full pilot likelihood; fixed raw main-sample bins with central and signed-tail cells; joint grouped likelihood; deterministic restricted reference; then validated adaptive importance sampling/SMC; four-parameter summaries and uncertainty.  
**Addresses:** The stated applied four-estimate goal.  
**Avoids:** Plug-in standardization, re-binning within proposals, discarding pilot information, two-count non-identifiability.  
**Research flag:** Deep phase research required for bin design, nuisance-aware theory, practical posterior engine, and coverage study.

### Phase 8: Qualified Empirical Example and Release

**Rationale:** Empirical usefulness and packaging reproducibility are final qualifications, not substitutes for numerical or statistical validation.  
**Delivers:** Scientifically justified IID observations or defensibly standardized residuals; alternative-model and sensitivity checks; visible refusal example; Sphinx site; clean release archives, citation metadata, checksums, validation manifest, and paper reproduction command.  
**Addresses:** Real-data usability and submission-ready reproducibility.  
**Avoids:** Raw-return misuse, cherry-picked application, code/manuscript drift, premature software-paper slicing.  
**Research flag:** Dataset/domain research required; ordinary release engineering uses standard patterns.

### Phase Ordering Rationale

- Contracts precede kernels because parameterization, status, and refusal semantics determine what a numerical result means.
- Kernels precede posterior code because posterior agreement is meaningless when both routes inherit the same tail error.
- The exact finite-cell posterior precedes the full comparison so there is one minimal inferential path to test in isolation.
- Full-reference comparison and fallback complete the first user-visible product; performance is deliberately not claimed yet because this path routinely pays full-likelihood cost.
- Validation precedes certification and four-parameter expansion so failures cannot be blamed on an unvalidated nuisance engine.
- The certificate is a separate mathematical workstream; it should not block a reference-audited prototype, but it blocks the paper's certified-reduction claim.
- Four-parameter inference follows only after the known-nuisance ground truth is stable; empirical work follows only after the package knows when to refuse.

## Deferred Work

- Independent R implementation; consider a thin wrapper only after the Python numerical and serialization contracts stabilize.
- Public SMC, arbitrary priors, third-party backends, multivariate stable laws, regression, and general alpha-range fitting.
- Exact-Gaussian spike/model selection and one-sided boundary theory.
- Native acceleration or Numba until profiling identifies a release-relevant bottleneck.
- Multiscale cells unless the frozen 25% held-out TV-improvement admission rule is met.
- Separate software paper until later adoption and functionality supply a contribution beyond the computational-methods paper.

## Non-Negotiable Kill and Refusal Rules

**Kill or narrow the computational-paper claim if:**

1. The finite-sample accuracy quantity cannot be proved for its advertised scope or its numerical error cannot be conservatively enclosed.
2. The accuracy decision requires the unknown truth, a threshold selected from unmodelled main-sample evidence, or first computing the full posterior.
3. Known-nuisance inference fails independent-oracle, posterior-refinement, or SBC/coverage gates.
4. No prespecified regime with `n <= 250,000` simultaneously attains mean TV at most `.05`, 90th-percentile TV at most `.10`, false-safe probability at most 5%, computational failure below 1%, and at least 10x end-to-end speed-up.
5. Joint nuisance inference fails coverage after including the pilot likelihood and fixing main-sample bins.
6. The contribution reduces to coding the limiting Gamma--Beta posterior or reproducing the theoretical manuscript's tables.

**The package must refuse rather than return a successful fit when:**

- inputs are nonfinite, not one-dimensional, or incompatible with the frozen design;
- the local region reaches unsupported Gaussian/one-sided boundaries or maps outside the stable parameter space;
- independently known nuisance provenance is absent in the first workflow, or plug-in estimates are presented as known;
- a threshold or partition was selected from unsupported main-sample adaptation;
- probabilities/densities are materially negative, nonfinite, fail normalization, or disagree beyond tolerance;
- posterior refinement fails, both reduced and full routes fail, or a required backend silently changes semantics;
- the requested `certified` status is unavailable because the proved bound/enclosure is absent or inconclusive.

An unsuitable empirical dataset is not a package kill condition: omit the application rather than weaken the model contract. A failed certificate is also an admissible outcome: report an inconclusive assessment, use the explicit full fallback when reliable, or refuse.

## Confidence Assessment

| Area | Confidence | Notes |
|---|---|---|
| Stack | HIGH | Current official Scientific Python, NumPy, SciPy, PyPA, and tooling sources support the chosen pure-Python stack and Python 3.12 floor. |
| Feature boundary | HIGH | The theorem and project evidence clearly separate the near-boundary workflow from mature generic stable fitting. |
| Standardized architecture | HIGH | The finite multinomial likelihood and compact two-dimensional posterior produce clear component and validation boundaries. |
| Numerical implementation | MEDIUM-HIGH | Independent routes are feasible, but supported tail tolerances and the smallest reliable gap must be established experimentally. |
| First working vertical slice | HIGH | It is implementable with current tools and has explicit completion and failure behavior. |
| Runtime advantage | LOW before validation | Computing the full posterior for every reference audit removes most speed benefit; the prospective bound must change this before performance claims. |
| Finite-sample certificate | MEDIUM-LOW | Interface and error decomposition are clear, but the mathematical bound and analytic truncation controls remain research deliverables. |
| Four-parameter workflow | MEDIUM | The conditional likelihood construction is sound; nuisance-aware theory, bin design, scalable inference, and coverage remain unvalidated. |
| Empirical application | MEDIUM-LOW | No qualifying dataset or diagnostic-power study has yet been selected. |
| Pitfalls and refusal rules | HIGH | They follow directly from the theorem, inspected spike behavior, official SciPy semantics, and established Bayesian validation practice. |

**Overall confidence:** HIGH that the proposed build order will produce an honest working package; MEDIUM that the certificate and four-parameter extension will meet publication-level accuracy and speed gates.

### Gaps to Address

- **Finite-sample bound:** derive its exact target and scope before implementing `PROVED_BOUND`; distinguish expected prior-predictive TV from an observed-dataset guarantee.
- **Numerical tail enclosure:** prove Fourier truncation/tail bounds and propagate them through probabilities and posterior quantities; ball arithmetic alone is insufficient.
- **Supported boundary floor:** determine where floating-point `alpha=2-rh` collapses and define/refuse values below a tested gap.
- **Independent simulator/oracle:** establish genuinely distinct data-generation and probability-evaluation paths for SBC and held-out validation.
- **Speed feasibility:** profile end-to-end cost only after the reference workflow is correct; the certificate must avoid making full posterior computation routine.
- **Nuisance workflow:** prespecify pilot size, raw bin construction, central-bin count, and four-dimensional reference/production engines.
- **Empirical domain:** identify a scientifically credible IID dataset or residual analysis with provenance and alternative-model checks.
- **Repository/release:** confirm package-name availability, license, archive route, CI resources, and controlled benchmark hardware before public release.

## Sources

### Project Sources (HIGH confidence)

- [`PROJECT.md`](../PROJECT.md) — intended product scope, active requirements, constraints, and exclusions.
- [`STACK.md`](STACK.md) — ecosystem, version, backend, CI, release, and numerical-tool decisions.
- [`FEATURES.md`](FEATURES.md) — table stakes, differentiators, anti-features, and minimal working fit.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — component boundaries, data flow, result states, serialization, first slice, and dependency order.
- [`PITFALLS.md`](PITFALLS.md) — scientific/numerical failure modes, validation gates, kill criteria, and refusal fixtures.
- [Theoretical manuscript](../../gaussian_boundary_stable_manuscript.tex) — theorem, local experiment, known-nuisance scope, and boundary exclusions.
- [Executed spike](../spikes/001-gaussian-boundary-stable/boundary_spike.py) and [machine-readable results](../spikes/001-gaussian-boundary-stable/results.json) — existing deterministic falsification evidence and pre-asymptotic behavior; not production code or sufficient validation.

### Authoritative External Sources (HIGH confidence)

- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) — interpreter and core-dependency support policy.
- [NumPy downstream guidance](https://numpy.org/doc/2.3/dev/depending_on_numpy.html) and [random compatibility policy](https://numpy.org/doc/2.0/reference/random/compatibility.html) — dependency bounds and reproducibility limits.
- [SciPy `levy_stable` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html) — `S0`/`S1`, stable scale, piecewise methods, mutable settings, and FFT warning.
- [PyPA `pyproject.toml` guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) and [`src` layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) — standards-based packaging and installed-copy isolation.
- [Stan simulation-based calibration guidance](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html) and [Talts et al.](https://arxiv.org/abs/1804.06788) — posterior-computation validation and its interpretive limits.
- [Ament and O'Neil (2018)](https://doi.org/10.1007/s11222-017-9725-y) and [Nolan (1998)](https://doi.org/10.1016/S0167-7152(98)00010-8) — stable-density quadrature/asymptotics and continuous parameterization.
- [Python-FLINT](https://pypi.org/project/python-flint/) and [FLINT integration documentation](https://flintlib.org/doc/acb_calc.html) — ball arithmetic and the need to handle improper-integral truncation separately.

---
*Research completed: 2026-08-24*  
*Ready for roadmap: yes*

## SYNTHESIS COMPLETE
