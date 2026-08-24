# Domain Pitfalls: `stableboundary`

**Domain:** Certified Bayesian inference for univariate alpha-stable laws near the Gaussian boundary  
**Researched:** 2026-08-24  
**Overall confidence:** HIGH for mathematical and numerical failure modes; MEDIUM for empirical and performance claims until a prototype and a qualifying dataset exist

## Executive Pre-Mortem

The package fails scientifically if it turns a local, standardized, compact-interior asymptotic theorem into a generic four-parameter stable-law estimator. The theorem concerns `S0` laws with known location and scale, a prespecified local design `delta = r*h`, and parameters bounded away from the exact Gaussian and one-sided boundaries. Two signed counts can identify two emerging tail intensities in that experiment; they cannot, by themselves, identify `(alpha, beta, loc, scale)`. Version 1 must therefore make standardized inference genuinely reliable before any joint nuisance workflow is exposed.

The package fails computationally if a plausible-looking posterior can survive tail cancellation, clipped negative densities, a changed SciPy global parameterization, or an unreported method fallback. These are not hypothetical risks. The research spike mutates `levy_stable.parameterization`, clips small negative Fourier densities, uses fixed truncation and a first-term tail continuation, and notes that SciPy CDF/SF calculations lose the small stable-tail contribution in the regime of interest. Those choices were acceptable in a deterministic falsification audit with independent checks; they are unacceptable as silent production behavior.

The package fails as a publication artifact if simulation is allowed to rename an unproved diagnostic a “certificate,” if the empirical example is used to validate the model that selected it, or if the software merely republishes the theoretical paper's Gamma--Beta algebra. The computational contribution must be the finite-sample method: an auditable exact finite-cell posterior, a separately validated full-likelihood reference, a mathematically justified accuracy assessment, and explicit reduction/fallback/refusal behavior. If that cannot be delivered with useful accuracy and speed, the computational-paper claim should be killed rather than diluted.

## Roadmap Phases Used Below

| Phase | Purpose | Exit condition relevant to this pre-mortem |
|---|---|---|
| **P0 — Mathematical and API contract** | Freeze coordinates, parameterization, scope, status semantics, priors, and threshold-selection rules | Every public quantity and state has an unambiguous mathematical definition |
| **P1 — Probability and density kernels** | Implement finite-cell probabilities and independent reference evaluations | Numerical error is detected, bounded, and cross-checked in the supported region |
| **P2 — Standardized posterior** | Exact finite three-cell inference for known location and scale | Posterior normalization, summaries, and identification diagnostics pass oracle and SBC tests |
| **P3 — Full reference and accuracy assessment** | Full stable posterior, reconstruction bound, safe/fallback logic | No `certified` state exists until the theorem and conservative numerical enclosure both exist |
| **P4 — Validation and performance** | Prespecified simulation, posterior-distance, calibration, stress, and benchmark study | Accuracy, false-safe, failure-rate, and end-to-end speed gates all pass |
| **P5 — Joint nuisance workflow** | Pilot-conditioned inference for `(alpha, beta, loc, scale)` | Pilot likelihood is included, bins are fixed after the pilot, and nuisance uncertainty is propagated |
| **P6 — Empirical and release qualification** | Qualified application, package hardening, archival release, paper artifact | Model scope is defensible, installations reproduce, and manuscript claims match the released version |

## Critical Pitfalls

### 1. Treating `h` as an ordinary parameter and hiding the design scale `r`

**Confidence:** HIGH — direct consequence of the manuscript's local coordinates and induced prior.

**What goes wrong:** The identity `delta = 2 - alpha = r*h` does not determine `r` and `h` separately. At a fixed finite sample, different analyst-chosen `r` values produce different `h`, thresholds, compact sets, and induced priors while representing the same `alpha`. A package that reports `h` without its design, or silently estimates `r` from the same data, makes an arbitrary analysis choice look like a scientific estimate.

**Warning signs:** `fit(x)` produces `h` without a `BoundaryDesign`; two design scales give materially different `alpha` posteriors; the posterior piles up at an `h` boundary; documentation calls `r` an estimated parameter; thresholds change during posterior evaluation.

**Prevention:** Make `delta` the canonical shape gap and `r` an immutable, prespecified analysis design. Require every `h` result to carry `r`, `n`, `K`, `c_n`, and the induced prior transformation. Select `r` from sample size and a declared prior/design region, never from observed extremes. Report conventional `alpha` and signed gaps as the scientific outputs; keep `h` explicitly local and design-dependent.

**Required tests:** Coordinate round trips over the supported domain; property test `delta == r*h` within the declared floating-point contract; prior-Jacobian tests; design-sensitivity tests; boundary-contact tests that force fallback/refusal; serialization round trip preserving the complete design.

**Roadmap phase:** P0; blocking gate for all inference.

### 2. Reporting an identified `beta` when the data contain no sign-allocation information

**Confidence:** HIGH — at `alpha = 2` the `S0` distribution is `N(0,2)` for every `beta`, and the limiting beta posterior equals its prior when the total signed count is zero.

**What goes wrong:** A narrow beta credible interval can be produced entirely by its prior. Near the Gaussian boundary, beta is an allocation parameter for vanishing tail mass; at the exact boundary it is not a parameter of the sampling law at all. Reporting four ordinary estimates conceals the quotient geometry and gives users false certainty.

**Warning signs:** beta is summarized without an identification status; beta intervals remain narrow under zero or nearly zero signed counts; the software returns different fitted distributions at `alpha=2` for different beta values; documentation describes beta as skewness at the Gaussian point.

**Prevention:** Make the primary boundary outputs `tau_plus = delta*(1+beta)/2` and `tau_minus = delta*(1-beta)/2`. Return beta only with `identified`, `weak`, or `unidentified` status and a prior-to-posterior information diagnostic. At exact Gaussianity, serialize beta as not identified rather than as an estimate. Keep exact Gaussian and `beta = +/-1` outside theorem-backed certification unless a separate boundary theory is implemented.

**Required tests:** Gaussian data under several beta priors; zero-count cases reproducing the beta prior; reflection `beta -> -beta`; `tau_plus + tau_minus = delta`; endpoint refusal; summaries must suppress estimate-like beta output when status is `unidentified`.

**Roadmap phase:** P0–P2; blocking public-API gate.

### 3. Pretending two signed counts estimate all four stable parameters

**Confidence:** HIGH — structural identifiability failure, explicitly outside the theorem.

**What goes wrong:** Positive and negative exceedance counts provide at most two independent cell probabilities. Unknown location changes signs and thresholds; unknown scale changes standardized distances. Plug-in centering/scaling treats estimated nuisance parameters as known and can turn ordinary nuisance error into apparent tail asymmetry or tail thickness.

**Warning signs:** a four-parameter `fit` is implemented by standardizing with sample median/scale and then applying the two-count posterior; location and scale have no likelihood contribution; uncertainty intervals ignore the pilot; the same observations select bins and populate them without conditioning.

**Prevention:** Version 1 accepts only already standardized data or independently known `loc` and `scale`. P5 may add a randomized pilot plus its full stable likelihood and a fixed grouped likelihood for the remaining observations. The raw-unit bin boundaries must be frozen after the pilot, and all four parameters must enter every finite-cell probability. Do not market the package as a general four-parameter estimator before this workflow passes its own validation.

**Required tests:** Rank/sensitivity checks for the grouped likelihood; simulations varying nuisance parameters; comparison with and without propagated pilot uncertainty; split-accounting tests proving every observation has exactly one likelihood role; coverage for all four parameters; adversarial shifted/scaled samples that make the plug-in shortcut fail.

**Roadmap phase:** P0 scope refusal, then P5 as a separate gate.

### 4. Selecting thresholds from the analysis data and then using an unconditional cell likelihood

**Confidence:** HIGH — this invalidates the stated multinomial sampling model unless selection is included.

**What goes wrong:** Optimizing a threshold after inspecting extremes double-uses the observations. Counts conditional on the selected threshold no longer have the simple prespecified multinomial likelihood, and simulation coverage can be optimistic if the same data tune and assess the rule.

**Warning signs:** thresholds depend on fitted alpha, observed order statistics, posterior draws, or whichever candidate gives the best apparent certificate; repeated calls on a permutation change the design; the audit record lacks the selection rule and split.

**Prevention:** Choose thresholds before seeing the main sample using `n`, the declared prior region, and numerical constraints; or use a randomized pilot and include its likelihood. Freeze thresholds before the main counts are computed. A user-supplied data-dependent threshold must force an `unsupported_design` state unless its selection likelihood is implemented.

**Required tests:** Permutation invariance; deterministic design from identical metadata; split reproducibility; simulation comparing valid prespecified and invalid optimized thresholds; a test that the threshold object cannot mutate inside likelihood/posterior calls; audit reconstruction of the selected bins.

**Roadmap phase:** P0 and P2; blocking validity gate.

### 5. Silently substituting the limiting Poisson/Gamma--Beta likelihood for exact finite-cell probabilities

**Confidence:** HIGH — direct numerical evidence shows slow convergence.

**What goes wrong:** Poissonizing rare finite cells can be accurate while replacing their finite means by limiting intensities remains poor. In the current audit, signed cells retain only about 58–78% of full Hellinger information at `r=.03` and about 87–88% even at `r=1e-4`. For `beta=.5`, `c=4`, and approximately 2.95 million observations, finite means are about `(4.864, 1.792)`, not the limiting `(6, 2)`, although the multinomial-to-Poisson bound is about `3e-5`.

**Warning signs:** an “exact” result invokes Gamma and Beta updates; expected counts are computed from asymptotic coefficients rather than finite probabilities; no distinction is made among exact multinomial, finite-mean Poisson, and limiting Poisson errors.

**Prevention:** Make exact finite-cell multinomial inference the reduced default. Expose finite-mean and limiting Poisson analyses only as named approximations/benchmarks. Decompose accuracy into full-to-cell information loss, multinomial-to-finite-Poisson error, and finite-to-limit mean error.

**Required tests:** Exact likelihood equality against direct multinomial evaluation; limiting conjugacy tests; posterior-distance grids for every approximation layer; regression fixtures reproducing the manuscript audit; zero-count and high-count edge cases.

**Roadmap phase:** P1–P4; blocking release gate.

### 6. Calling a simulation-calibrated diagnostic a mathematical certificate

**Confidence:** HIGH for the semantic risk; MEDIUM for the eventual computability of the proposed bound.

**What goes wrong:** A small observed discrepancy on a simulation grid does not upper-bound an unseen discrepancy. Likewise, the manuscript proves convergence in prior-predictive mean total variation; that is not automatically a dataset-specific guarantee. A quantity called `B_n` is a certificate only if a theorem proves the inequality over its stated parameter/design class and every numerical term is conservatively enclosed.

**Warning signs:** `certified=True` appears before a finite-sample theorem; a fitted dataset is declared within TV 0.05 using only an average simulation result; interpolation error, tail truncation, or parameter-grid gaps are absent from `B_n`; the same backend computes both the bound and its validation target.

**Prevention:** Reserve `certified_*` names and statuses until proof and numerical enclosure are complete. Until then use `experimental_accuracy_assessment`. State whether a bound is uniform, prior-predictive expected, high-probability, or dataset-specific. Include reconstruction, product amplification, prior support, quadrature, interpolation, truncation, and floating-point budgets. A certificate must be allowed to be inconclusive.

**Required tests:** Prove-and-test each inequality component; compare the bound with independently computed discrepancies on held-out and adversarial grids; force each error component to dominate in a fixture; verify no false strengthening in user text; prespecified false-safe study with upper confidence bound at or below 5%.

**Roadmap phase:** P3; hard prohibition before proof.

### 7. Hiding method switches, fallbacks, or refusals

**Confidence:** HIGH — silent adaptation makes results unauditable and can select a method using numerical luck.

**What goes wrong:** An `auto` method that silently changes from reduced to full likelihood, changes numerical backend, or returns an approximation after failure produces nonreproducible semantics. Warnings are often missed in batch workflows.

**Warning signs:** output objects do not contain a machine-readable method state; a warning is the only evidence of fallback; failed tail evaluation returns a full-likelihood result under the original method label; exceptions are caught broadly and ignored.

**Prevention:** Use a closed status set such as `reduced_assessed`, `reduced_certified`, `full_fallback`, and `refused`. Return the requested method, executed method, reason code, numerical backend, tolerance, and complete fallback chain. Never convert refusal to success. Require explicit opt-in for approximate modes.

**Required tests:** Fault injection for each numerical backend; assertions on status and reason codes; warning-as-error CI; serialization of fallback history; tests that `method="reduced"` fails rather than silently changes while `method="auto"` records the change.

**Roadmap phase:** P0, P3, and P6.

### 8. Mixing `S0` and `S1`, or confusing stable scale with Gaussian standard deviation

**Confidence:** HIGH — SciPy documents `S1` as its default, uses mutable class attributes to change parameterization/method, and gives `alpha=2, scale=1` as a normal law with standard deviation `sqrt(2)`.

**What goes wrong:** The same numeric `(alpha, beta, loc, scale)` can refer to different location conventions. A user may interpret `scale=1` as Gaussian standard deviation one, although this project's standardized endpoint is `N(0,2)`. The spike sets `levy_stable.parameterization = "S0"` globally; production code doing this at import or fit time can change another library's behavior and race across threads.

**Warning signs:** serialized results omit parameterization; package outputs disagree after another module changes SciPy settings; unit-normal data are called standardized without conversion; tests pass only when run in a particular order.

**Prevention:** `S0` is a value in every parameter/result object, not ambient state. Own the numerical protocol or use a locked, scoped adapter that restores every SciPy class attribute even on failure. Do not mutate SciPy at import. Name stable `scale` separately from Gaussian `sd`; document that the endpoint SD is `sqrt(2)*scale`. Any S1 conversion must be explicit and tested.

**Required tests:** Characteristic-function identities; `alpha=2` against `Normal(loc, sqrt(2)*scale)`; reflection; S0/S1 conversion round trips; test-order randomization; concurrent calls under hostile external SciPy settings; assertion that imports and fits leave all SciPy globals unchanged.

**Roadmap phase:** P0–P1; blocking kernel gate.

### 9. Losing the rare probability through cancellation or underflow

**Confidence:** HIGH — the spike explicitly avoids SciPy CDF/SF for this regime after observing loss of the small stable-tail contribution.

**What goes wrong:** Computing `1 - cdf(u)`, subtracting two nearly equal CDFs, or exponentiating very negative log probabilities can turn a meaningful tail cell into zero. The lighter signed tail is especially vulnerable as `|beta|` approaches one. Near `alpha=2`, direct evaluation of `sin(pi*alpha/2)` and `2-alpha` can also lose relative precision.

**Warning signs:** one signed probability is exactly zero inside the supported interior; reflection fails; probabilities are nonmonotone in the threshold; a small change in precision changes posterior beta materially; `log(0)` is handled by adding an arbitrary epsilon.

**Prevention:** Compute in `delta` coordinates and stable log form; use direct signed-tail integration plus controlled asymptotic continuation, not `1-cdf`; use `log1p` for the central cell; derive posterior error tolerances from the total-TV budget. Detect when `alpha` rounds to two and refuse rather than fabricate precision.

**Required tests:** High-precision comparisons across both tails, deep deltas, and skewness values; monotonicity in threshold; positivity and normalization enclosures; reflection; overlap tests where direct integration and tail series must agree; posterior sensitivity to probability perturbations at the numerical bound.

**Roadmap phase:** P1; blocking numerical gate.

### 10. Clipping negative densities or renormalizing away a failed quadrature

**Confidence:** HIGH — the spike uses `np.maximum(f, 0)` and `np.maximum(f, 1e-300)` after a coarse material-negativity check.

**What goes wrong:** Clipping makes plots and square roots run, but changes probability mass, Hellinger distance, scores, and likelihoods. A global mass near one does not prove each tiny signed cell is accurate; the central mass can hide a 100% relative tail error. Fixed Fourier truncation, spatial interpolation, and a first tail term can agree accidentally over the audit grid.

**Warning signs:** production code contains `maximum(pdf, 0)` or a density floor in likelihood evaluation; results improve when tolerances are loosened; cell probabilities are accepted because total mass is close to one; no per-cell error estimate exists.

**Prevention:** Negative values beyond a rigorously justified roundoff enclosure raise a structured numerical error. If a tiny enclosed negative is corrected, record the correction and propagate its mass into the error budget. Use adaptive truncation/refinement and per-cell absolute and relative error, with an independently derived tail continuation.

**Required tests:** Deliberately under-resolved quadrature must fail; convergence under nodes/cutoff/grid refinement; mass and score-integral identities; per-cell rather than only total-mass checks; high-precision oracle points in centre, crossover, and both tails; no unreported clipping found by source-level test.

**Roadmap phase:** P1; blocking numerical gate.

### 11. Validating an implementation with itself and calling the full likelihood “ground truth” without audit

**Confidence:** HIGH — stable densities lack an elementary generic form, so shared-code agreement is weak evidence.

**What goes wrong:** If the simulator, finite-cell likelihood, full likelihood, and expected fixtures share the same density routine or tail coefficient helper, one error can make all tests pass. SciPy's full likelihood is a useful reference, not an infallible oracle; its stable implementation has configurable algorithms and documented experimental FFT paths.

**Warning signs:** golden fixtures are regenerated by the production package; backend comparison is just two tolerance settings of the same routine; posterior TV is measured against a reference using the same cached probabilities; special-law tests are absent.

**Prevention:** Maintain genuinely independent paths: production finite-cell kernel, high-precision/Fourier oracle, and a separately configured piecewise stable reference. Use analytic special laws and characteristic-function/reflection identities. Freeze oracle fixtures with provenance; do not regenerate them automatically in ordinary tests.

**Required tests:** `alpha=2` normal and `alpha=1, beta=0` Cauchy identities; carefully parameterized Levy cases; independent density-engine comparisons; backend-swap posterior TV; mutation tests that deliberately corrupt one tail coefficient; fixture provenance/hash checks.

**Roadmap phase:** P1–P3.

### 12. Applying the posterior theorem outside its prior and parameter scope

**Confidence:** HIGH — the theorem uses proper priors on compact interior local sets; its generalized prior is used only for a separate count-rule risk calculation.

**What goes wrong:** A broad fixed prior on alpha is not the induced local prior on `h`; an improper boundary-singular prior is not covered by the full-posterior theorem. Letting `h -> 0`, `p -> 0/1`, or prior mass escape the certified rectangle invalidates probability-ratio and reconstruction bounds.

**Warning signs:** posterior mass touches the local rectangle boundary; docs claim the generalized Bayes rule is the exact finite full-likelihood Bayes action; certification ignores prior support; users may change priors after assessment without recomputing it.

**Prevention:** Version the prior and compact set as part of the analysis design. The accuracy assessment is invalidated by any prior or support change. Separate proper-posterior results from the generalized decision rule in API and documentation. Boundary contact triggers design expansion plus reassessment or full fallback, never truncation presented as certainty.

**Required tests:** Prior-transform/Jacobian checks; boundary-mass triggers; prior-sensitivity suite; improper-prior rejection in posterior APIs; audit hash changes whenever prior or support changes.

**Roadmap phase:** P0–P3.

### 13. Certifying an IID stable calculation for dependent or misspecified empirical data

**Confidence:** HIGH that the theorem is model-conditional; MEDIUM on which diagnostics will have useful finite-sample power.

**What goes wrong:** Tail clustering under serial dependence invalidates the independent multinomial/Poisson count law. Heteroskedasticity, nonstationarity, mixtures, tempering, truncation, contamination, rounding, or fitted residuals can mimic or suppress stable tails. A numerically perfect posterior can therefore answer the wrong data-generating question.

**Warning signs:** raw financial returns or sensor series are passed directly; exceedances cluster; results change across time blocks; data are winsorized/censored; a few observations determine the fit; the package labels model-conditional numerical validity as empirical truth.

**Prevention:** V1 accepts IID finite observations only, with known standardization. Diagnostics can refuse or flag, but cannot prove IID/stability. For time series, require a separately specified residual model and state that uncertainty from that model is not propagated unless implemented. Use posterior predictive checks for central mass, signed counts, tail magnitudes, and held-out blocks; compare scientifically plausible alternatives.

**Required tests:** Stress simulations with AR dependence, volatility clustering, mixtures, Student-t/tempered tails, contamination, censoring, and rounding; refusal/flag tests; sensitivity to deleting top observations and changing blocks; held-out predictive assessment. An empirical example must fail visibly under at least one deliberately violated condition.

**Roadmap phase:** P4 and P6; blocks empirical claims, not the standardized kernel.

### 14. Assuming a seed alone guarantees reproducibility

**Confidence:** HIGH — NumPy documents that exact stream compatibility depends on the bit generator, call sequence, build, environment, and machine, and that `default_rng` may change its default bit generator.

**What goes wrong:** Parallel execution, dependency updates, BLAS differences, changed draw shapes, or an unspecified bit generator can change a pilot split, simulations, or Monte Carlo summaries. Mutable global configuration and caches make failures order-dependent.

**Warning signs:** audit records contain only an integer seed; tests rely on exact floating-point equality across platforms; parallel and serial simulations differ without explanation; cache keys omit tolerances or parameterization.

**Prevention:** Use an explicit `Generator(BitGenerator(seed))`, deterministic child streams, and record bit-generator type/state, dependency/build versions, platform, backend, and call-level design. Treat numerical reproducibility as tolerance-based unless a bitwise guarantee is actually supported. Cache keys include all parameters, thresholds, parameterization, backend, tolerance, and package version; immutable results prevent mutation after fit.

**Required tests:** Serial/parallel statistical equivalence and deterministic stream allocation; repeated audit replay in a locked environment; hostile cache-invalidation tests; randomized test order; environment and config snapshots; paper artifact reproduced from a clean checkout.

**Roadmap phase:** P0, P4, and P6.

### 15. Claiming a speed-up that disappears after certification and fallback

**Confidence:** MEDIUM — the bottleneck and attainable speed-up require implementation evidence.

**What goes wrong:** Grouped likelihood can be cheap while threshold planning, probability evaluation, the accuracy bound, full-reference checks, and frequent fallbacks dominate total cost. Comparing optimized reduced code with an intentionally weak or untuned full likelihood overstates utility.

**Warning signs:** benchmarks exclude planning/certification; caches are warm only for the proposed method; fallback time is omitted; speed is reported only for millions of observations although the method fails its accuracy gate there; hardware and tolerances are absent.

**Prevention:** Benchmark end-to-end wall time, peak memory, failures, and fallback rate from raw input to auditable result. Use the same accuracy target, hardware, process state, and cache policy for competitors. Report scaling with `n`, grid size, prior region, and required tolerance. Optimize only after correctness profiles identify the bottleneck.

**Required tests:** Cold/warm benchmarks; scaling and memory tests; certificate/fallback cost accounting; competitor parity review; regression budgets. The proposed publication kill gate is no qualifying regime with `n <= 250,000` that simultaneously achieves mean posterior TV <= .05, 90th-percentile TV <= .10, false-safe <= 5%, and at least 10x end-to-end speed-up.

**Roadmap phase:** P4; blocks performance and practical-value claims.

### 16. Shipping a repository that works locally but not as an installed package

**Confidence:** HIGH — standard Python packaging failures are well understood, and PyPA specifically notes that `src/` layout prevents accidentally importing the working-tree copy.

**What goes wrong:** Tests can import local files that are absent from wheels, omit package data, depend on undeclared extras, or behave differently under editable installation. An overly broad API locks in experimental terminology and status behavior before the mathematics stabilizes.

**Warning signs:** tests run only from the repository root; no wheel/sdist install test exists; examples import private modules; generated data are missing from distributions; minimum supported NumPy/SciPy versions are untested; import changes global state.

**Prevention:** Use PEP 621 metadata, `pyproject.toml`, and `src/stableboundary/`. Keep one narrow typed public API and immutable result/config objects. Separate core, optional diagnostics, docs, and development dependencies. Build wheel and sdist and test each in a clean environment without the source tree on `sys.path`. Define a serialization schema and deprecation policy before public release.

**Required tests:** Clean wheel/sdist install on supported Python/OS/dependency matrix; editable-versus-wheel behavior; import smoke test from a temporary directory; docs examples as tests; public-symbol snapshot; package-data and license checks; no-network test run.

**Roadmap phase:** P0 structure, enforced continuously through P6.

### 17. Turning one contribution into three thin papers, or letting the package validate its own premise

**Confidence:** HIGH for the slicing risk; MEDIUM for eventual journal judgment.

**What goes wrong:** A theoretical paper, a simulation paper that merely confirms it, and a separate software paper can look like artificial slicing. Conversely, drafting strong claims before the certificate and package pass creates pressure to reinterpret failures as successes. A convenient real dataset cannot establish numerical validity or novelty.

**Warning signs:** the computational manuscript's contribution is described as “implementation of Paper 1”; simulation regimes are changed after results are seen; only weak baselines are compared; package documentation claims certification before the theorem; a software-paper plan appears before users and evidence.

**Prevention:** Treat the package as the executable artifact of one distinct computational-methods paper. Its new claim is finite-sample assessed/certified reduction with honest fallback—not Gamma--Beta updating. Freeze validation regimes, baselines, metrics, Monte Carlo precision, and kill criteria before confirmatory runs. Archive the exact code/data version cited by the paper. Do not pursue a separate software paper unless later adoption, functionality, and user evidence justify it.

**Required tests/review:** Hostile novelty review before manuscript drafting; immutable validation manifest; comparison against full likelihood and mature stable estimators; trace every paper table to an automated artifact; independent reproduction from the release archive; claim-to-test matrix signed off before submission.

**Roadmap phase:** P3–P6; publication gate.

## Moderate Pitfalls

| Pitfall | What goes wrong / warning sign | Prevention and test | Phase |
|---|---|---|---|
| **`alpha`/`delta` floating-point collapse** (HIGH) | `2 - delta` rounds to exactly two, destroying beta-sensitive calculations | Store `delta` canonically, define the smallest supported gap, compare tail constants with high precision, and refuse below support | P0–P1 |
| **Hellinger convention drift** (HIGH) | Halved and unhalved squared distances introduce factors of two in thresholds, `c`, and acceptance criteria | Public name `hellinger_unhalved_sq`; analytic Bernoulli/Poisson fixtures; serialize convention | P0–P1 |
| **Scale interpreted as standard deviation** (HIGH) | Users standardize by SD one while the canonical `scale=1` endpoint has variance two | Explicit conversion helpers and normal-boundary tests; never label stable scale `sd` | P0–P2 |
| **Tail gaps described as literal probabilities** (HIGH) | `tau_+/-` are shape-gap coordinates, not threshold-free probability masses; scale also affects raw tails | Name and document them precisely; provide separately defined predictive tail probabilities at user thresholds; scale-equivariance tests | P0–P2 |
| **Posterior quadrature looks normalized but misses a mode/boundary** (HIGH) | Coarse 2-D grids hide posterior structure | Adaptive deterministic quadrature, log-sum-exp, refinement TV test <= .002, mode/boundary search, independent dense-grid fixtures | P2 |
| **Central-cell subtraction loses precision** (HIGH) | `q0 = 1-q+-q-` or `n*log(q0)` loses accuracy | Use checked sums and `log1p(-(q+ + q-))`; compare with direct central integration and propagate error | P1–P2 |
| **Data coercion changes observations** (MEDIUM) | Object arrays, missing values, infinities, weights, censoring, or float32 silently alter likelihood | Accept finite one-dimensional float data only in V1; explicit copy/coercion report; reject unsupported weights/censoring | P0–P2 |
| **Warnings are invisible in pipelines** (HIGH) | Prior dominance or model concerns are printed but not queryable | Structured diagnostics in immutable result and audit record; warnings mirror, not replace, state | P0–P2 |
| **Multiscale cells are treated as independent when nested** (HIGH) | Double-counted observations create a false likelihood | Use a disjoint partition; verify counts sum to `n`; nested thresholds require their joint multinomial law | P2–P4 |
| **Adaptive multiscale complexity without demonstrated gain** (MEDIUM) | Extra cells increase numerical and explanatory burden but add no material accuracy | Keep three cells as baseline; retain multiscale only if held-out mean-TV improves by at least 25% in two prespecified pre-asymptotic regimes | P4 |
| **Sensitive empirical data leak through audit records** (MEDIUM) | Full data or revealing raw extremes are serialized unintentionally | Store configuration and optional cryptographic fingerprint, not raw data, by default; explicit export controls and privacy tests | P6 |
| **Package name or artifact cannot be archived** (MEDIUM) | PyPI collision, missing license, or mutable paper dependency harms reuse | Check name before release, choose license, include CITATION metadata, archive versioned release with DOI and hashes | P6 |

## Validation Architecture

Validation must be designed as independent evidence, not as a large collection of tests sharing one implementation.

### Independent evidence paths

| Path | Responsibility | Must not share |
|---|---|---|
| **Production kernel A** | Fast finite-cell probabilities and log likelihood | Oracle fixtures or an unchecked first-tail helper |
| **Numerical oracle B** | High-precision Fourier/direct integration and controlled tails on a finite benchmark corpus | Production integration/caching code |
| **Full-posterior reference C** | Deterministic known-nuisance full stable posterior | Reduced cell probabilities or certificate decision code |
| **Data generators D1/D2** | Stable simulation for SBC and operating-characteristic studies | A single common implementation; use two cross-checked generators/parameterization routes where feasible |
| **Certificate evaluator E** | Computes the proved bound/enclosure and status | The held-out discrepancy calculation used to estimate false-safe behavior |

Analytic anchors include `alpha=2 -> Normal(loc, sqrt(2)*scale)`, the symmetric Cauchy case, carefully converted Levy special cases, the `S0` characteristic function, reflection under `x -> -x, beta -> -beta`, exact multinomial likelihoods, and the limiting Gamma--Beta posterior. A golden fixture generated by production code is not an oracle.

### Validation layers and gates

| Layer | Required evidence | Gate |
|---|---|---|
| **V0 — Contract** | Coordinate/domain properties, prior Jacobian, immutable design, status truth table, no SciPy global mutation | All invalid states refuse deterministically |
| **V1 — Numerical kernel** | Special laws, normalization, positivity, reflection, tail monotonicity, high-precision crossover/tail corpus, refinement, backend comparison, error-budget propagation | Every supported cell probability has a usable error assessment; under-resolution raises |
| **V2 — Posterior engine** | Exact multinomial fixtures, adaptive-quadrature convergence, zero/rare/high counts, prior sensitivity, posterior summaries | Refinement changes posterior TV by <= .002 in the benchmark region |
| **V3 — Bayesian calibration** | Simulation-based calibration for `h`, `p`, `alpha`, beta-status behavior, `tau_+/-`, and predictive quantities; frequentist coverage over fixed grids | Empirical 95% coverage in 92–98% band, with Monte Carlo half-width <= .015, unless a stricter prespecified criterion replaces it |
| **V4 — Reduction accuracy** | Full-versus-cell posterior TV, Hellinger/Wasserstein, action regret, certificate conservativeness, held-out false-safe study | Mean TV <= .05, 90th-percentile TV <= .10, action regret <= 5%, false-safe <= 5% in declared operating regimes |
| **V5 — Numerical independence** | Full reference backend swap and oracle comparisons | Backend swap changes posterior TV by <= .005; unexplained discrepancy blocks claims |
| **V6 — Misspecification stress** | Dependence, volatility, mixture, tempered-tail, contamination, censoring, rounding, first-stage residual scenarios | Failures produce warnings/refusal/fallback rather than a certified label |
| **V7 — Performance** | Cold/warm end-to-end time, memory, fallback cost/rate, scaling, matched-accuracy competitors | >=10x speed-up in at least one nontrivial qualifying regime; computational failures <1% |
| **V8 — Distribution artifact** | Clean wheel/sdist installs, Python/OS/min-latest dependency matrix, docs execution, public API and serialization checks | Release artifacts reproduce examples without repository-path imports |
| **V9 — Paper reproduction** | Frozen simulation manifest, exact package version, data provenance, one-command table/figure rebuild, environment record | Independent clean-checkout reproduction before submission |

SBC is necessary for checking posterior computation under the declared prior and likelihood; it does not establish model adequacy or the finite-sample reduction theorem. Monte Carlo runs continue until the standard error of mean TV is at most `.005` and the binomial half-width for coverage/false-safe rates is at most `.015`. Parameter grids, random streams, exclusions, and stopping rules are frozen before confirmatory runs.

### Certificate-specific validation

The accuracy object must decompose, rather than hide, its budget:

```text
full experiment -> exact finite partition -> reconstructed experiment
                -> finite multinomial/Poisson (if requested)
                -> limiting signed-Poisson (if requested)

total reported bound = mathematical reduction term
                     + probability-kernel numerical enclosure
                     + posterior integration enclosure
                     + interpolation/truncation/floating-point allowances
```

Each arrow has its own named quantity, scope, and test. If only the first arrow is relevant to the production reduced posterior, Poisson approximation errors must not be added or invoked. If the theorem controls prior-predictive expected TV, the UI must say exactly that; it must not promise a TV bound for the observed posterior unless an additional result supports it.

### Mandatory adversarial fixtures

1. Zero signed counts with multiple beta priors.
2. One light-tail count under high skewness.
3. Posterior mass at every edge of `K`.
4. `alpha` so close to two that float conversion collapses.
5. Hostile external SciPy settings and concurrent calls.
6. Under-resolved Fourier quadrature that previously would have been clipped.
7. Tiny right/left probabilities where `1-cdf` returns zero or loses digits.
8. A data-chosen threshold that must be rejected.
9. Shifted/scaled data passed to standardized mode.
10. Clustered exceedances, GARCH-like volatility, contamination, and censoring.
11. Forced certificate failure followed by explicit full fallback.
12. Forced failure of both reduced and full backends, ending in `refused`.

## Phase-Specific Warnings

| Phase topic | Likely pitfall | Mitigation before advancing |
|---|---|---|
| P0 contracts | Ambiguous `r/h`, beta status, S0/scale semantics | Freeze typed mathematical objects and refusal matrix before coding inference |
| P1 numerics | Cancellation, clipping, same-engine agreement | Build oracle corpus first; error-producing states are first-class outputs/exceptions |
| P2 standardized fit | Approximation substitution and quadrature bias | Exact finite multinomial default; limiting posterior explicitly named; SBC and refinement gates |
| P3 certificate | False assurance and hidden theorem scope | Prove scope first, conservatively enclose numerics, prohibit dataset-specific wording without a dataset-specific result |
| P4 validation | Post hoc regime selection and unfair speed comparisons | Freeze manifest/kill rules; held-out confirmatory grid; include all overhead/fallbacks |
| P5 nuisance | Two counts masquerading as four-parameter inference | Pilot likelihood plus fixed main bins; propagate nuisance; separate validation programme |
| P6 empirical | IID/stable misspecification and cherry-picking | Qualifying-domain rationale, held-out checks, alternative models, visible refusal/failure example |
| P6 publication | Package-paper slicing and code/manuscript drift | One computational contribution, archived release, claim-to-test traceability, no software paper yet |

## Release-Blocking Kill Conditions

Stop or narrow the package/paper claim if any of the following remains true:

1. The finite-sample accuracy quantity cannot be proved for its advertised scope or cannot conservatively include numerical error.
2. The accuracy decision requires the unknown true parameter, a data-selected threshold not included in the likelihood, or first computing the full posterior.
3. Negative densities, invalid probabilities, or backend disagreements can be converted silently into a successful fit.
4. Known-nuisance inference fails SBC, refinement, or independent-oracle tests.
5. No prespecified regime with `n <= 250,000` meets the joint accuracy, false-safe, failure-rate, and end-to-end speed gates.
6. Joint nuisance inference cannot attain coverage after including the pilot likelihood and fixing the main-sample bins.
7. A qualified empirical dataset cannot satisfy the model-use conditions; omit the empirical illustration rather than use an indefensible one.
8. The computational manuscript's contribution reduces to coding the asymptotic Gamma--Beta posterior or reproducing the theoretical paper's tables.

## Evidence and Sources

### Project evidence

- [`PROJECT.md`](../PROJECT.md) — intended scope, package contract, and explicit exclusions.
- [`gaussian_boundary_stable_manuscript.tex`](../../gaussian_boundary_stable_manuscript.tex) — theorem assumptions, local prior, exact Gaussian/non-one-sided exclusions, slow convergence, and posterior/risk distinctions.
- [`boundary_spike.py`](../spikes/001-gaussian-boundary-stable/boundary_spike.py) — deterministic audit implementation; evidence for global SciPy mutation, clipping, fixed truncation, interpolation, and CDF/SF tail concerns.
- [`results.json`](../spikes/001-gaussian-boundary-stable/results.json) — 24-row information-retention audit, finite cell means, numerical checks, and runtime evidence.

### Authoritative external sources

- [SciPy `levy_stable` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html) — `S1` default, `S0` option, mutable class settings, numerical methods, experimental FFT warning, and `alpha=2` normal scale semantics. **Confidence: HIGH.**
- [Python Packaging User Guide: source-distribution format](https://packaging.python.org/en/latest/specifications/source-distribution-format/) and [`src` versus flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) — `pyproject.toml`, standard artifacts, and installed-copy isolation. **Confidence: HIGH.**
- [NumPy random compatibility policy](https://numpy.org/doc/2.0/reference/random/compatibility.html) — limits of seed-only and cross-environment stream reproducibility. **Confidence: HIGH.**
- [Stan User's Guide: simulation-based calibration](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html) and [Talts et al.](https://arxiv.org/abs/1804.06788) — SBC design and interpretation. **Confidence: HIGH for algorithm validation; it does not certify model adequacy.**
- [Ament and O'Neil, optimized quadrature and asymptotics](https://doi.org/10.1007/s11222-017-9725-y) and [Nolan, continuous parameterizations](https://doi.org/10.1016/S0167-7152(98)00010-8) — independent numerical and parameterization foundations. **Confidence: HIGH.**

## Confidence Assessment

| Area | Confidence | Reason |
|---|---|---|
| Mathematical/statistical scope | HIGH | Directly stated and proved/excluded in the manuscript |
| Numerical risks | HIGH | Direct code inspection, machine-readable audit, SciPy documentation, and stable-law numerical literature |
| API and packaging risks | HIGH | Project constraints plus current PyPA/SciPy behavior |
| Reproducibility risks | HIGH | Direct mutable-state/code evidence and NumPy's documented compatibility limits |
| Empirical failure modes | MEDIUM | Model-conditional logic is certain, but no candidate dataset or diagnostic power study exists yet |
| Performance risk | MEDIUM | Cost components are clear, but no production implementation or fair benchmark exists |
| Publication risk | MEDIUM | Contribution boundaries are clear; editorial novelty judgment cannot be guaranteed |

## RESEARCH COMPLETE
