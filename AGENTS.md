<!-- GSD:project-start source:PROJECT.md -->
## Project

**stableboundary**

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

**Core Value:** Users can fit a near-Gaussian stable model and trust that the package never
presents a fast reduced posterior as reliable without quantifying its scope,
numerical status, and fallback decision.

### Constraints

- **Statistical scope**: Version 1 begins with standardized data or independently
  known location and scale because this is the scope justified by the theorem.
- **Parameterization**: Nolan's continuous `S0` parameterization is canonical;
  every result and serialized artifact records it explicitly.
- **Language**: Python 3.12+ is the primary implementation language, using
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
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommendation in One Sentence
## Compatibility Decision
### Revise the provisional Python floor to 3.12
- Scientific Python SPEC 0's support table moved to Python 3.12+ by April 2026
- Current NumPy 2.5.2 and SciPy 1.18.0 both require Python 3.12+.
- Current ArviZ 1.2.0 also requires Python 3.12+.
- The manuscript's deterministic audit was already run on CPython 3.13.14.
### Runtime dependency policy
## Recommended Stack
### Core runtime
| Technology | Supported/minimum | Purpose | Why this choice |
|---|---:|---|---|
| CPython | 3.12; test 3.12--3.14 | Runtime | Matches the current scientific stack and removes a needless compatibility fork. |
| NumPy | `>=2.2` (current 2.5.2) | Arrays, typed numerical data, Gauss--Legendre nodes, RNG | The package is array-oriented and deterministic quadrature is naturally vectorized. Use `numpy.random.Generator`, never global RNG state. |
| SciPy | `>=1.18` (current 1.18.0) | Stable-law reference calculations, quadrature, special functions, log-domain algebra, distributions | It provides Nolan's piecewise stable density/CDF implementation, `integrate.cubature`, `quad_vec`, `special.logsumexp`, Gamma/Beta functions, and probability utilities. |
| Python standard library | 3.12+ | Public models and audit records | Use frozen `dataclasses`, `Enum`, `Protocol`, `pathlib`, `json`, and `importlib.metadata`; no runtime model-framework dependency is needed. |
### Packaging and development environment
| Technology | Minimum/current baseline | Purpose | Decision |
|---|---:|---|---|
| `pyproject.toml` / PEP 621 | current PyPA specification | Metadata, dependencies, tool configuration | Mandatory; no `setup.py` or `setup.cfg`. |
| Hatchling | `>=1.32,<2` (current 1.32.0) | PEP 517 wheel/sdist backend | Use because the first package is pure Python, it supports reproducible builds, and it handles a `src/` layout without custom build code. |
| uv | `>=0.12` (current 0.12.3) | Developer environments and lockfile | Use as a frontend only. Commit `uv.lock`; keep Hatchling as the build backend so ordinary `pip` and `python -m build` users remain supported. |
| PyPA `build` | `>=1.5,<2` (1.5.0; 1.5.1 was yanked) | Standards-based release build | CI must build both sdist and wheel with `python -m build`; never invoke `setup.py`. |
| `check-wheel-contents` | `>=0.6.3,<1` | Wheel-content audit | Ensures tests, notebooks, spike artifacts, and private modules are not accidentally shipped. |
| Twine | `>=7,<8` | Metadata validation only | Use `twine check dist/*`; publish through PyPI Trusted Publishing, not a stored token. |
### Quality tooling
| Tool | Minimum/current baseline | Use |
|---|---:|---|
| pytest | `>=9.1,<10` (current 9.1.1) | Test runner and markers for `unit`, `numerical`, `statistical`, `validation`, `slow`, and `enclosure`. |
| Hypothesis | `>=6.161` | Property-based tests for coordinate maps, reflection, normalization, and valid parameter domains. |
| coverage.py | `>=7.15,<8` | Branch coverage; target 100% for parameter conversions/status transitions and high branch coverage overall. Coverage is not a substitute for numerical validation. |
| Ruff | `>=0.16,<0.17` | Formatter and linter; replaces Black, isort, Flake8, and common plugins. Pin the minor line in the development lock because its rule set evolves quickly. |
| mypy | `>=2.3,<3` | Strict checking of the public API and core numerical protocols. Ship `py.typed` as required by PEP 561. |
| ASV | `>=0.6.6,<0.7` | Historical performance benchmarks across dependency versions. Run claims on controlled hardware, not a noisy shared runner. |
### Documentation
| Tool | Version line | Purpose |
|---|---:|---|
| Sphinx | `>=9.1,<10` | API and narrative documentation. |
| MyST-NB | `>=1.4,<2` | Executable notebook examples integrated into Sphinx. |
| numpydoc | `>=1.10,<2` | NumPy-style scientific docstrings. |
| PyData Sphinx Theme | `>=0.20,<0.21` | Familiar scientific-Python documentation UI. |
| Matplotlib | `>=3.11,<4` | Plotting in docs and the optional `plot` extra. |
## Proposed `pyproject.toml` Dependency Surface
## Numerical Backend Architecture
### One public protocol, independent implementations
### Backend responsibilities
| Operation | Production/default method | Independent check | Future rigorous path |
|---|---|---|---|
| Exact-model signed-cell probabilities | Package-owned `S0` characteristic-function/Fourier inversion with adaptive order/cutoff and explicit analytic tail continuation | SciPy Nolan piecewise CDF/SF plus direct density integration at audit points | Python-FLINT `acb` ball integration on the finite path plus a proved analytic truncation/tail bound |
| Full stable log likelihood | SciPy Nolan piecewise implementation through a controlled `ScipyS0Backend` | Package-owned Fourier density on prespecified grids and difficult points | Selected-point ball enclosures first; a complete full-likelihood enclosure is not a version-1 requirement |
| Two-dimensional posterior normalization | Tensor Gauss--Legendre rules in transformed compact coordinates, evaluated in the log domain | `scipy.integrate.cubature` with Gauss--Kronrod rule and independently refined node orders | Ball summation/integration after likelihood enclosures exist |
| Limiting Gamma--Beta posterior | Closed-form SciPy special/distribution functions | The same deterministic grid engine | Arb Gamma/Beta evaluations if a certificate uses them |
| Posterior predictive tail probabilities | Integrate exact finite-cell probabilities against weighted posterior grid | Direct posterior draws and backend swap | Propagate cell-probability balls through positive weighted sums |
### SciPy-specific guardrail
### Stable log-likelihood algebra
- validate each cell probability as finite and within `[0,1]`;
- form the central log probability with `log1p(-(q_plus + q_minus))`;
- use `scipy.special.xlogy` for zero counts;
- normalize posterior weights with `scipy.special.logsumexp`;
- reject materially negative densities/probabilities instead of clipping them;
- allow only a documented, tolerance-sized projection to the simplex, and
- refine quadrature until a package-owned `AccuracyPolicy` passes; otherwise
## Posterior Result Standard
### Canonical result remains package-owned
- the weighted posterior quadrature grid and log normalizing constant;
- summaries for `alpha`, `beta`, `h`, `p`, `tau_plus`, and `tau_minus`;
- predictive quantities and their numerical status;
- the prior and local design;
- backend and refinement reports;
- identification warnings;
- a machine-readable reduction state;
- a versioned audit record.
### ArviZ is an optional interoperability layer
- retain the exact weighted grid in a custom, versioned group;
- if conventional posterior draws are requested, resample explicitly and
- store observed counts, design constants, numerical diagnostics, and package
- never discard weights and then compute “exact quadrature” summaries from
## Deterministic Posterior Engine
## Optional Sequential Monte Carlo
## Rigorous Interval and Enclosure Option
- a uniform Fourier-tail truncation bound;
- handling of oscillatory integration and any branch/domain conditions;
- a stable-density/probability enclosure that remains useful near
- outward propagation through Hellinger/reconstruction and posterior bounds.
## Validation Architecture
### Layer 0 — API and algebraic invariants (every pull request)
- parameter-domain validation;
- round trips among `(alpha,beta)`, `(r,h,p)`, and signed-gap coordinates;
- `w_plus + w_minus = h` and reflection under `beta -> -beta`;
- immutable result and audit objects;
- explicit status transitions and impossible `CERTIFIED` construction;
- no import-time mutation of SciPy or random global state.
### Layer 1 — Deterministic numerical contracts (every pull request)
- density and probability finiteness/nonnegativity;
- cell probabilities sum to one within the returned error budget;
- direct survival probability rather than `1-cdf` cancellation;
- convergence under Fourier order, cutoff, spatial grid, and quadrature order;
- exact Gaussian limit `N(0,2)` where applicable;
- limiting Gamma--Beta posterior reproduced by numerical quadrature;
- failure-path tests with deliberately impossible tolerances.
### Layer 2 — Independent backend agreement (pull requests plus nightly grid)
- package Fourier density versus SciPy piecewise density;
- integrated signed cells versus SciPy CDF/SF;
- tensor posterior quadrature versus SciPy adaptive cubature;
- ordinary precision versus high precision/ball arithmetic on the enclosure
- reflection pairs and values very close to the Gaussian boundary.
### Layer 3 — Posterior ground truth (scheduled and release gate)
### Layer 4 — Statistical calibration (scheduled and release gate)
- prior-predictive simulation-based calibration where sampling is used;
- repeated-sampling coverage over prespecified `(n,r,h,p)` regimes;
- beta-identification diagnostics as tail intensity vanishes;
- false-safe and false-refusal rates for any reduction decision;
- Monte Carlo replication until the prespecified standard-error targets are
### Layer 5 — Certificate tests (blocked until theorem and bounds exist)
- every reported interval encloses independent high-precision values;
- truncation, quadrature, interpolation, and roundoff budgets compose in the
- the empirical false-safe rate is at most the paper's prespecified bound;
- adversarial cases widen the bound or trigger fallback/refusal rather than
- removing any error term causes a dedicated mutation test to fail.
## CI, Release, and Benchmarking
### Pull-request CI
### Scheduled CI
- full adversarial numerical grid and statistical validation;
- Python-FLINT enclosure subset;
- weekly NumPy/SciPy nightly job following Scientific Python SPEC 4;
- ASV comparison against the main branch;
- warnings-as-errors job to detect SciPy deprecations before a release.
### Release CI
- rerun all validation layers supported by the release;
- reproduce manuscript tables from versioned scripts and emit machine-readable
- build from a clean tag, then install and test the wheel artifact;
- publish with PyPI Trusted Publishing/OIDC and a protected GitHub environment;
- retain the PEP 740 attestations produced by the official PyPA publish action;
- attach checksums, validation manifest, and environment/SBOM export to the
## What Not to Use
| Rejected choice | Reason | Use instead |
|---|---|---|
| Packaging the spike as-is | It mutates SciPy global state, clips numerical densities, uses fixed grids/cutoffs, and writes directly to a research path. | Port formulas behind protocols and tests; retain the spike as an independent reproduction artifact. |
| PyMC/Stan/NumPyro as the first posterior engine | The validated problem is a compact two-dimensional integral; a sampler adds diagnostics and approximation error without adding inferential capability. | Deterministic quadrature first; add SMC only for the four-parameter workflow. |
| JAX/BlackJAX for SMC | SciPy's stable likelihood is not JAX-native or differentiable, producing two numerical stacks and awkward callbacks. | NumPy/SciPy adaptive-tempering SMC behind a later optional engine. |
| SciPy FFT stable density/CDF | SciPy documents this route as experimental, and the boundary problem is especially sensitive to small tail probabilities. | Nolan piecewise plus package Fourier inversion. |
| `levy_stable.fit` as the Bayesian package core | It is a generic point-estimation routine and does not implement the local prior, exact cell posterior, identification diagnostics, or reduction decision. | Package-owned likelihood and posterior engines. |
| `1 - cdf(x)` for the positive tail | Catastrophic cancellation can erase the rare stable contribution. | Direct survival probability or direct tail integration with an error report. |
| mpmath precision as “certification” | More digits do not prove a bound; its interval support is documented as experimental. | Python-FLINT ball arithmetic plus proved analytic truncation bounds. |
| An exact-Gaussian mixture/spike in version 1 | The theorem excludes `alpha=2`, where beta is unidentified; a spike changes the model-selection problem. | Refuse or clearly label boundary-extrapolative inputs until separate theory exists. |
| Plug-in location/scale in a certified result | It ignores nuisance uncertainty and is outside the proved experiment. | Known/independently calibrated nuisance first; later pilot likelihood plus fixed grouped main-sample likelihood. |
| pandas/xarray/ArviZ as mandatory runtime dependencies | They add a broad stack to a small numerical core and are not needed for fitting. | Optional adapters and plotting extras. |
| Pickled result objects | Unsafe and unstable across versions. | Versioned JSON metadata plus arrays, and optional ArViZ DataTree export. |
| Native extension or Numba in milestone 1 | It complicates wheels, reproducibility, and error inspection before a performance bottleneck is established. | Vectorized NumPy/SciPy and ASV profiling first. |
## Compatibility Risks and Mitigations
| Risk | Severity | Mitigation |
|---|---|---|
| SciPy stable-law settings are mutable global class variables | HIGH | No import-time writes; snapshot/lock/restore adapter; process parallelism; record effective settings; independent Fourier checks. |
| Stable tail CDF/SF can lose rare mass near `alpha=2` | HIGH | Direct Fourier/tail integration as production cell backend; compare CDF/SF and integrated density; reject disagreement. |
| “Exact finite-cell” is confused with exact arithmetic | HIGH | API/documentation says exact-model; every numerical value carries status/error; reserve `CERTIFIED`. |
| NumPy/SciPy changes alter low-level numerical values | MEDIUM | Minimum/latest/nightly matrices, reference intervals rather than exact floats, versioned audit records. |
| ArviZ 1.x changed from the older monolithic `InferenceData` ecosystem toward modular DataTree APIs | MEDIUM | Optional adapter pinned to `>=1.2,<2`, tested against official conversion API; package result remains canonical. |
| Python-FLINT wheels or global precision behavior differ by platform | MEDIUM | Optional extra, wheel smoke matrix, context-managed precision, process isolation, no certification claim until all supported platforms pass. |
| Free-threaded CPython exposes dependency/thread-safety edge cases | MEDIUM | Do not claim free-threaded support in the first release; add only after NumPy, SciPy, and Python-FLINT validation. |
| A universal lockfile hides the supported dependency range | MEDIUM | Lock the research environment, but separately test declared minima, latest releases, and nightlies. |
| Statistical validation becomes flaky CI | MEDIUM | Separate deterministic PR tests from seeded scheduled validation; use MC standard-error targets and versioned reports. |
## Installation and Maintainer Commands
## Source Assessment
| Source | What it supports | Confidence |
|---|---|---|
| [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) | Python and core-dependency support-window policy | HIGH |
| [NumPy 2.5.2 on PyPI](https://pypi.org/project/numpy/2.5.2/) and [NumPy downstream guidance](https://numpy.org/doc/2.3/dev/depending_on_numpy.html) | Current release/Python floor and dependency-bound guidance | HIGH |
| [SciPy 1.18 release notes](https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html) | Python/NumPy requirements | HIGH |
| [SciPy `levy_stable` API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html) | S0/S1 definitions, piecewise/DNI/FFT methods, global settings, and FFT warning | HIGH |
| [SciPy `integrate.cubature`](https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.integrate.cubature.html) | Adaptive multidimensional quadrature and non-guaranteed convergence | HIGH |
| [PyPA `pyproject.toml` guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) and [src-layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) | Standards-based metadata/build and source layout | HIGH |
| [Hatchling on PyPI](https://pypi.org/project/hatchling/) and [Hatch sdist documentation](https://hatch.pypa.io/dev/plugins/builder/sdist/) | Current backend and reproducible build behavior | HIGH |
| [uv project layout and lockfile](https://docs.astral.sh/uv/concepts/projects/layout/) | Cross-platform developer lockfile; distinction from standardized `pylock.toml` | HIGH |
| [Typing library guidance](https://typing.python.org/en/latest/guides/libraries.html) | Inline typing and the `py.typed` marker | HIGH |
| [pytest](https://pypi.org/project/pytest/), [Hypothesis](https://pypi.org/project/hypothesis/), [coverage.py](https://pypi.org/project/coverage/), [Ruff](https://pypi.org/project/ruff/), and [mypy](https://pypi.org/project/mypy/) | Current quality-tool versions and Python support | HIGH |
| [SciPy toolchain roadmap](https://docs.scipy.org/doc/scipy-1.16.0/dev/toolchain.html) | Scientific-project precedent for pytest, Hypothesis, ASV, Sphinx, MyST-NB, and numpydoc | HIGH |
| [ASV on PyPI](https://pypi.org/project/asv/) | Current benchmark tool | HIGH |
| [ArviZ 1.2 on PyPI](https://pypi.org/project/arviz/) and [`arviz.from_dict`](https://python.arviz.org/en/stable/api/generated/arviz.from_dict.html) | Current modular ArviZ line and DataTree conversion | HIGH |
| [Python-FLINT 0.9 on PyPI](https://pypi.org/project/python-flint/) and [FLINT integration documentation](https://flintlib.org/doc/acb_calc.html) | Ball arithmetic, platform wheels, rigorous finite-path integration, and improper-integral limitation | HIGH |
| [mpmath interval documentation](https://mpmath.org/doc/current/contexts.html) | Experimental status of its interval layer | HIGH |
| [Scientific Python SPEC 4](https://scientific-python.org/specs/spec-0004/) and [SPEC 7](https://scientific-python.org/specs/spec-0007/) | Nightly-wheel CI and RNG API recommendations | HIGH |
| [PyPA GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) | OIDC Trusted Publishing and PEP 740 attestations | HIGH |
| [Ament and O'Neil (2018)](https://doi.org/10.1007/s11222-017-9725-y) | Independent literature basis for optimized quadrature/asymptotics for stable densities | HIGH |
| `gaussian_boundary_stable_manuscript.tex` and `.planning/spikes/001-gaussian-boundary-stable/boundary_spike.py` | Required S0 convention, known-nuisance scope, slow pre-asymptotics, and current research implementation risks | HIGH (local primary artifacts) |
## Final Stack Decision
- **Runtime:** CPython 3.12+, NumPy 2.2+, SciPy 1.18+.
- **Build:** `src/` layout, PEP 621, Hatchling; uv only for the locked developer
- **Inference:** exact-model finite-cell likelihood plus deterministic
- **Numerics:** package-owned `S0` Fourier backend and guarded SciPy piecewise
- **Results:** immutable package result and audit record; optional ArviZ DataTree
- **Rigor:** Python-FLINT optional, with no `certified` label until analytic
- **Scaling:** a later NumPy/SciPy adaptive-tempering SMC engine behind a stable
## RESEARCH COMPLETE
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
