# Technology Stack

**Project:** `stableboundary`  
**Researched:** 2026-08-24  
**Mode:** ecosystem and feasibility  
**Overall confidence:** HIGH for the packaging and floating-point stack; MEDIUM for the future rigorous-enclosure layer because the mathematical truncation bounds still have to be derived

## Recommendation in One Sentence

Build `stableboundary` as a pure-Python, typed package for CPython 3.12+ with a
two-dependency runtime (`numpy>=2.2`, `scipy>=1.18`), Hatchling as the PEP 517
build backend, deterministic two-dimensional quadrature as the first posterior
engine, a package-owned `S0` numerical-backend protocol, SciPy's Nolan
piecewise implementation as an independent/reference backend, and optional
ArviZ, plotting, Python-FLINT, and later SMC layers.

This is intentionally not a PyMC, Stan, JAX, or generic probabilistic-programming
package. The first validated inferential problem is two-dimensional on a compact
rectangle and has a deterministic ground truth. Adding a large sampling framework
before that ground truth works would make the package harder to validate without
solving a scientific problem.

## Compatibility Decision

### Revise the provisional Python floor to 3.12

The current `PROJECT.md` says Python 3.11+. The recommended package metadata is
instead:

```toml
requires-python = ">=3.12"
```

This is a deliberate change, not an oversight:

- Scientific Python SPEC 0's support table moved to Python 3.12+ by April 2026
  and NumPy 2.2+ by August 2026.
- Current NumPy 2.5.2 and SciPy 1.18.0 both require Python 3.12+.
- Current ArviZ 1.2.0 also requires Python 3.12+.
- The manuscript's deterministic audit was already run on CPython 3.13.14.

Supporting 3.11 would force the project to maintain an older SciPy branch and a
different ArviZ generation for no inferential gain. Support CPython 3.12, 3.13,
and 3.14 at the first release. Do not set an upper Python bound in package
metadata; test a new Python release before adding its classifier.

Follow SPEC 0 for future support drops: retain Python feature releases for about
three years and core dependencies for about two years. Announce support-window
changes in release notes rather than letting dependency resolution fail
silently.

### Runtime dependency policy

Use lower bounds and no speculative upper bounds for NumPy/SciPy:

```toml
dependencies = [
  "numpy>=2.2",
  "scipy>=1.18",
]
```

NumPy's downstream guidance says most pure-Python libraries should not impose
an upper bound. Compatibility is established by testing both the declared
minimum set and the latest released set. A weekly allowed-to-fail job should
test NumPy and SciPy nightly wheels initially; make it blocking after the first
stable release.

## Recommended Stack

### Core runtime

| Technology | Supported/minimum | Purpose | Why this choice |
|---|---:|---|---|
| CPython | 3.12; test 3.12--3.14 | Runtime | Matches the current scientific stack and removes a needless compatibility fork. |
| NumPy | `>=2.2` (current 2.5.2) | Arrays, typed numerical data, Gauss--Legendre nodes, RNG | The package is array-oriented and deterministic quadrature is naturally vectorized. Use `numpy.random.Generator`, never global RNG state. |
| SciPy | `>=1.18` (current 1.18.0) | Stable-law reference calculations, quadrature, special functions, log-domain algebra, distributions | It provides Nolan's piecewise stable density/CDF implementation, `integrate.cubature`, `quad_vec`, `special.logsumexp`, Gamma/Beta functions, and probability utilities. |
| Python standard library | 3.12+ | Public models and audit records | Use frozen `dataclasses`, `Enum`, `Protocol`, `pathlib`, `json`, and `importlib.metadata`; no runtime model-framework dependency is needed. |

**Core runtime rule:** keep NumPy and SciPy as the only mandatory third-party
dependencies through the known-location/scale release. In particular, do not
make pandas, Matplotlib, xarray, ArviZ, JAX, or an MCMC framework mandatory.

### Packaging and development environment

| Technology | Minimum/current baseline | Purpose | Decision |
|---|---:|---|---|
| `pyproject.toml` / PEP 621 | current PyPA specification | Metadata, dependencies, tool configuration | Mandatory; no `setup.py` or `setup.cfg`. |
| Hatchling | `>=1.32,<2` (current 1.32.0) | PEP 517 wheel/sdist backend | Use because the first package is pure Python, it supports reproducible builds, and it handles a `src/` layout without custom build code. |
| uv | `>=0.12` (current 0.12.3) | Developer environments and lockfile | Use as a frontend only. Commit `uv.lock`; keep Hatchling as the build backend so ordinary `pip` and `python -m build` users remain supported. |
| PyPA `build` | `>=1.5,<2` (1.5.0; 1.5.1 was yanked) | Standards-based release build | CI must build both sdist and wheel with `python -m build`; never invoke `setup.py`. |
| `check-wheel-contents` | `>=0.6.3,<1` | Wheel-content audit | Ensures tests, notebooks, spike artifacts, and private modules are not accidentally shipped. |
| Twine | `>=7,<8` | Metadata validation only | Use `twine check dist/*`; publish through PyPI Trusted Publishing, not a stored token. |

Recommended repository layout:

```text
stableboundary/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── src/
│   └── stableboundary/
│       ├── __init__.py
│       ├── py.typed
│       ├── api.py
│       ├── parameters.py
│       ├── design.py
│       ├── likelihoods.py
│       ├── posterior.py
│       ├── results.py
│       ├── diagnostics.py
│       ├── exceptions.py
│       └── numerics/
│           ├── protocols.py
│           ├── fourier_s0.py
│           ├── scipy_s0.py
│           ├── quadrature.py
│           └── enclosures.py
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── statistical/
│   ├── validation/
│   └── data/
├── benchmarks/
├── docs/
└── examples/
```

The old spike stays under `.planning/spikes/` and is treated as an external
reference artifact. Do not copy it wholesale into `src/`; port each formula
behind a tested protocol.

### Quality tooling

| Tool | Minimum/current baseline | Use |
|---|---:|---|
| pytest | `>=9.1,<10` (current 9.1.1) | Test runner and markers for `unit`, `numerical`, `statistical`, `validation`, `slow`, and `enclosure`. |
| Hypothesis | `>=6.161` | Property-based tests for coordinate maps, reflection, normalization, and valid parameter domains. |
| coverage.py | `>=7.15,<8` | Branch coverage; target 100% for parameter conversions/status transitions and high branch coverage overall. Coverage is not a substitute for numerical validation. |
| Ruff | `>=0.16,<0.17` | Formatter and linter; replaces Black, isort, Flake8, and common plugins. Pin the minor line in the development lock because its rule set evolves quickly. |
| mypy | `>=2.3,<3` | Strict checking of the public API and core numerical protocols. Ship `py.typed` as required by PEP 561. |
| ASV | `>=0.6.6,<0.7` | Historical performance benchmarks across dependency versions. Run claims on controlled hardware, not a noisy shared runner. |

Use inline annotations throughout and `mypy --strict` for `src/stableboundary`.
Do not type away numerical distinctions: define separate types/dataclasses for
conventional `S0` parameters, local `(r,h,p)` coordinates, signed gaps, cell
probabilities, posterior grids, numerical error reports, and reduction states.

### Documentation

| Tool | Version line | Purpose |
|---|---:|---|
| Sphinx | `>=9.1,<10` | API and narrative documentation. |
| MyST-NB | `>=1.4,<2` | Executable notebook examples integrated into Sphinx. |
| numpydoc | `>=1.10,<2` | NumPy-style scientific docstrings. |
| PyData Sphinx Theme | `>=0.20,<0.21` | Familiar scientific-Python documentation UI. |
| Matplotlib | `>=3.11,<4` | Plotting in docs and the optional `plot` extra. |

Documentation examples must call only the public API and run in CI. Keep one
small deterministic smoke example in the ordinary test job; execute longer
simulation and validation notebooks only in the scheduled/release workflow.
Every fit page must explain Nolan `S0`, the `N(0,2)` boundary convention, the
difference between finite-cell and limiting likelihoods, and the meaning of
`reduced`, `fallback`, `refused`, and `certified` status words.

## Proposed `pyproject.toml` Dependency Surface

The exact metadata will be created during scaffolding, but the dependency
contract should begin as follows:

```toml
[build-system]
requires = ["hatchling>=1.32,<2"]
build-backend = "hatchling.build"

[project]
name = "stableboundary"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.2",
  "scipy>=1.18",
]

[project.optional-dependencies]
plot = ["matplotlib>=3.11,<4"]
arviz = ["arviz>=1.2,<2"]
certify = ["python-flint>=0.9,<1"]

[dependency-groups]
test = [
  "pytest>=9.1,<10",
  "hypothesis>=6.161",
  "coverage[toml]>=7.15,<8",
]
quality = [
  "ruff>=0.16,<0.17",
  "mypy>=2.3,<3",
]
docs = [
  "sphinx>=9.1,<10",
  "myst-nb>=1.4,<2",
  "numpydoc>=1.10,<2",
  "pydata-sphinx-theme>=0.20,<0.21",
  "matplotlib>=3.11,<4",
]
benchmark = ["asv>=0.6.6,<0.7"]
release = [
  "build==1.5.0",
  "check-wheel-contents>=0.6.3,<1",
  "twine>=7,<8",
]
```

Library runtime ranges are deliberately broad. The committed `uv.lock` records
the exact developer and manuscript-validation environments. Each published
validation bundle should additionally record `python --version`, `numpy`,
`scipy`, platform, BLAS information where relevant, backend configuration,
and the package commit/tag.

## Numerical Backend Architecture

### One public protocol, independent implementations

Do not allow inference code to call `scipy.stats.levy_stable` directly. Define
a package-owned protocol along these lines:

```python
class StableS0Backend(Protocol):
    def logpdf(self, x: FloatArray, params: S0Parameters,
               policy: AccuracyPolicy) -> NumericalArray: ...
    def signed_cell_probabilities(self, threshold: float,
                                  params: S0Parameters,
                                  policy: AccuracyPolicy) -> CellProbabilities: ...
```

Every numerical return should include the value, a convergence/status code,
the method and tolerance, an error estimate or enclosure when available, and
diagnostics. A bare floating-point array is insufficient for a method whose
scientific claim depends on numerical reliability.

### Backend responsibilities

| Operation | Production/default method | Independent check | Future rigorous path |
|---|---|---|---|
| Exact-model signed-cell probabilities | Package-owned `S0` characteristic-function/Fourier inversion with adaptive order/cutoff and explicit analytic tail continuation | SciPy Nolan piecewise CDF/SF plus direct density integration at audit points | Python-FLINT `acb` ball integration on the finite path plus a proved analytic truncation/tail bound |
| Full stable log likelihood | SciPy Nolan piecewise implementation through a controlled `ScipyS0Backend` | Package-owned Fourier density on prespecified grids and difficult points | Selected-point ball enclosures first; a complete full-likelihood enclosure is not a version-1 requirement |
| Two-dimensional posterior normalization | Tensor Gauss--Legendre rules in transformed compact coordinates, evaluated in the log domain | `scipy.integrate.cubature` with Gauss--Kronrod rule and independently refined node orders | Ball summation/integration after likelihood enclosures exist |
| Limiting Gamma--Beta posterior | Closed-form SciPy special/distribution functions | The same deterministic grid engine | Arb Gamma/Beta evaluations if a certificate uses them |
| Posterior predictive tail probabilities | Integrate exact finite-cell probabilities against weighted posterior grid | Direct posterior draws and backend swap | Propagate cell-probability balls through positive weighted sums |

“Exact finite-cell likelihood” means the exact multinomial likelihood of the
stable model rather than the limiting Poisson likelihood. Its probabilities
are still numerical and must never be described as mathematically exact unless
an enclosure has been established.

### SciPy-specific guardrail

SciPy 1.18 documents `levy_stable.parameterization`,
`pdf_default_method`, and numerical tolerances as mutable class variables. The
default parameterization is `S1`; the project requires `S0`. Therefore:

1. Never change these variables at import time.
2. The adapter snapshots all relevant settings, acquires a package lock, sets
   `S0` and `piecewise`, performs the call, and restores settings in a
   `finally` block.
3. Mark the SciPy adapter as not safe against foreign threads that mutate the
   same singleton. Use process-level parallelism, not threads, for full
   likelihood evaluation.
4. Record the effective SciPy settings in every audit record.
5. Never use SciPy's experimental `fft-simpson` density/CDF path for a release
   result. It may be a benchmark only.

The existing spike's unconditional
`levy_stable.parameterization = "S0"` is acceptable in an isolated research
script, but not in production library import or public API code.

### Stable log-likelihood algebra

The reduced multinomial likelihood must be computed without avoidable
cancellation:

- validate each cell probability as finite and within `[0,1]`;
- form the central log probability with `log1p(-(q_plus + q_minus))`;
- use `scipy.special.xlogy` for zero counts;
- normalize posterior weights with `scipy.special.logsumexp`;
- reject materially negative densities/probabilities instead of clipping them;
- allow only a documented, tolerance-sized projection to the simplex, and
  record when it occurs;
- refine quadrature until a package-owned `AccuracyPolicy` passes; otherwise
  return a numerical failure, not a posterior.

## Posterior Result Standard

### Canonical result remains package-owned

Use a frozen `StableBoundaryFit` dataclass as the canonical return type. It
should contain:

- the weighted posterior quadrature grid and log normalizing constant;
- summaries for `alpha`, `beta`, `h`, `p`, `tau_plus`, and `tau_minus`;
- predictive quantities and their numerical status;
- the prior and local design;
- backend and refinement reports;
- identification warnings;
- a machine-readable reduction state;
- a versioned audit record.

The result state should be an enum, not free text. During the first milestones,
allowed states should distinguish at least `EXACT_CELL`, `LIMIT_APPROXIMATION`,
`FULL_REFERENCE`, `FALLBACK`, `REFUSED`, and `EXPERIMENTAL_UNCERTIFIED`.
Reserve `CERTIFIED` in the enum but make construction impossible until both the
finite-sample theorem and conservative numerical enclosure have been
implemented and tested.

### ArviZ is an optional interoperability layer

Current ArviZ 1.2 is modular and its conversion functions produce an
`xarray.DataTree` following ArviZ conventions. Provide `fit.to_arviz()` behind
the `arviz` extra, but do not make ArviZ the source of truth:

- retain the exact weighted grid in a custom, versioned group;
- if conventional posterior draws are requested, resample explicitly and
  record the resampling algorithm, draw count, and `rng` state/seed;
- store observed counts, design constants, numerical diagnostics, and package
  version in standard/custom groups as appropriate;
- never discard weights and then compute “exact quadrature” summaries from
  resampled draws.

For package-native persistence, use a documented JSON metadata file plus NPZ
arrays (or a documented directory container), with an explicit schema version
and checksums. Do not pickle result objects.

## Deterministic Posterior Engine

The known-location/scale problem is two-dimensional on compact
`K = [h_min,h_max] x [p_min,p_max]`. Use deterministic quadrature, not MCMC:

1. transform the rectangle to `[-1,1]^2`;
2. evaluate tensor Gauss--Legendre rules at increasing orders, starting, for
   example, at `32 x 32` and refining to `64 x 64`, then `96 x 96` or
   `128 x 128` only as needed;
3. calculate all weights in the log domain;
4. compare normalization, posterior means, tail-gap quantiles, and selected
   predictive probabilities between successive rules;
5. cross-check difficult cases with `scipy.integrate.cubature(rule="gk21")`;
6. fail with a structured convergence report if the requested tolerance is
   not met.

SciPy explicitly states that adaptive cubature convergence is not guaranteed;
its returned status and error estimate therefore support a check, not a proof.
The tensor rule remains valuable because it is simple, deterministic, and easy
to refine independently.

## Optional Sequential Monte Carlo

Do not ship SMC in version 0.1. Design a `PosteriorEngine` protocol now so it
can be added without changing `fit()` or the result schema.

When unknown location and scale make the posterior four-dimensional, the
recommended first stochastic engine is a package-owned adaptive-tempering SMC
using NumPy `Generator`, log-sum-exp weights, systematic or stratified
resampling, and random-walk rejuvenation. This choice can evaluate the existing
NumPy/SciPy likelihood without automatic differentiation.

Before it becomes public, validate SMC against the deterministic two-dimensional
posterior, including total variation or Wasserstein discrepancies, evidence
error, repeated-run Monte Carlo standard errors, ESS, resampling counts, and
failure rates. The `particles` package may be used as a research comparator
because it implements adaptive and waste-free tempering, but its latest PyPI
release is old enough that it should not become a runtime dependency without a
fresh compatibility audit. Do not use BlackJAX for this backend: it requires
JAX/XLA and the SciPy stable likelihood is neither JAX-traceable nor
automatically differentiable.

Adopt Scientific Python SPEC 7 from the first stochastic API: accept a
keyword-only `rng`, normalize it with `numpy.random.default_rng`, and never use
`numpy.random.seed` or implicit global state.

## Rigorous Interval and Enclosure Option

Use `python-flint>=0.9,<1` as the optional rigorous arithmetic dependency.
Python-FLINT wraps FLINT/Arb real and complex ball arithmetic and supplies
binary wheels for the main CPython platforms. Its `acb.integral` can produce a
rigorous enclosure of a finite-path integral.

That is necessary but not sufficient for a stable-law certificate. FLINT's
integration documentation requires improper integrals to be truncated or
regularized manually. Consequently the package must also prove and implement:

- a uniform Fourier-tail truncation bound;
- handling of oscillatory integration and any branch/domain conditions;
- a stable-density/probability enclosure that remains useful near
  `alpha = 2`;
- outward propagation through Hellinger/reconstruction and posterior bounds.

Until these pieces exist, Python-FLINT results are “ball-arithmetic audits,”
not a certified posterior. Ordinary `mpmath` may be used for high-precision
cross-checks but not for certification: its own documentation calls interval
support experimental and says many functions do not properly support it.

Python-FLINT also has a global precision context. Wrap precision changes in its
context manager and serialize enclosure calls or isolate them by process. Do
not expose mutable global precision as package configuration.

## Validation Architecture

The validation architecture is part of the scientific method, not a collection
of generic package tests.

### Layer 0 — API and algebraic invariants (every pull request)

- parameter-domain validation;
- round trips among `(alpha,beta)`, `(r,h,p)`, and signed-gap coordinates;
- `w_plus + w_minus = h` and reflection under `beta -> -beta`;
- immutable result and audit objects;
- explicit status transitions and impossible `CERTIFIED` construction;
- no import-time mutation of SciPy or random global state.

Use pytest and Hypothesis here. Generate parameters within prespecified compact
interior sets; tests at the exact Gaussian and one-sided boundaries should
assert refusal or the separately specified behavior rather than pretending the
current theorem covers them.

### Layer 1 — Deterministic numerical contracts (every pull request)

- density and probability finiteness/nonnegativity;
- cell probabilities sum to one within the returned error budget;
- direct survival probability rather than `1-cdf` cancellation;
- convergence under Fourier order, cutoff, spatial grid, and quadrature order;
- exact Gaussian limit `N(0,2)` where applicable;
- limiting Gamma--Beta posterior reproduced by numerical quadrature;
- failure-path tests with deliberately impossible tolerances.

Do not use bit-for-bit golden floats across platforms. Store input fixtures and
reference intervals/error budgets, and compare scientific quantities at their
documented scales.

### Layer 2 — Independent backend agreement (pull requests plus nightly grid)

At representative and adversarial points, compare:

- package Fourier density versus SciPy piecewise density;
- integrated signed cells versus SciPy CDF/SF;
- tensor posterior quadrature versus SciPy adaptive cubature;
- ordinary precision versus high precision/ball arithmetic on the enclosure
  subset;
- reflection pairs and values very close to the Gaussian boundary.

Backend disagreement beyond the prespecified tolerance is an error with both
reports attached. Do not average disagreeing backends.

### Layer 3 — Posterior ground truth (scheduled and release gate)

For known location/scale, compute a high-resolution two-dimensional full stable
posterior and compare it with:

1. the exact finite three-cell posterior;
2. the finite-mean Poisson posterior;
3. the limiting Gamma--Beta posterior;
4. any multiscale grouped posterior.

Measure posterior total variation on a common refined grid, Hellinger distance,
Wasserstein distance for primary quantities, interval coverage, predictive
tail probabilities, decision risk/regret, runtime, memory, and failures. The
current 24-row spike is a seed for fixtures, not sufficient ground truth.

### Layer 4 — Statistical calibration (scheduled and release gate)

- prior-predictive simulation-based calibration where sampling is used;
- repeated-sampling coverage over prespecified `(n,r,h,p)` regimes;
- beta-identification diagnostics as tail intensity vanishes;
- false-safe and false-refusal rates for any reduction decision;
- Monte Carlo replication until the prespecified standard-error targets are
  met.

Statistical acceptance bands belong in a versioned validation specification,
not hidden as convenient pytest tolerances. Expensive stochastic tests use
fixed master `SeedSequence` values and independent child streams; they should
not be ordinary per-commit tests.

### Layer 5 — Certificate tests (blocked until theorem and bounds exist)

- every reported interval encloses independent high-precision values;
- truncation, quadrature, interpolation, and roundoff budgets compose in the
  declared direction;
- the empirical false-safe rate is at most the paper's prespecified bound;
- adversarial cases widen the bound or trigger fallback/refusal rather than
  silently passing;
- removing any error term causes a dedicated mutation test to fail.

No amount of simulation or floating-point backend agreement can activate the
`CERTIFIED` state without this layer.

## CI, Release, and Benchmarking

### Pull-request CI

Use GitHub Actions with actions pinned to full commit SHAs:

1. Linux matrix on Python 3.12, 3.13, and 3.14 using current dependencies.
2. A minimum-dependency Linux job using NumPy 2.2.x and SciPy 1.18.x.
3. Windows and macOS smoke/numerical jobs on one current Python version.
4. Ruff format/lint and strict mypy.
5. Sphinx build with warnings as errors and a small executed example.
6. Build sdist and wheel; run `twine check` and `check-wheel-contents`.
7. Install the wheel into a clean environment and run the public-API smoke fit.

The package is pure Python in the first release, so no platform wheel build
matrix or `cibuildwheel` is needed. Add native code only after profiling proves
it is necessary; that decision would require a new packaging review.

### Scheduled CI

- full adversarial numerical grid and statistical validation;
- Python-FLINT enclosure subset;
- weekly NumPy/SciPy nightly job following Scientific Python SPEC 4;
- ASV comparison against the main branch;
- warnings-as-errors job to detect SciPy deprecations before a release.

### Release CI

- rerun all validation layers supported by the release;
- reproduce manuscript tables from versioned scripts and emit machine-readable
  manifests;
- build from a clean tag, then install and test the wheel artifact;
- publish with PyPI Trusted Publishing/OIDC and a protected GitHub environment;
- retain the PEP 740 attestations produced by the official PyPA publish action;
- attach checksums, validation manifest, and environment/SBOM export to the
  GitHub release.

Do not store a long-lived PyPI token. Do not make performance claims from
GitHub's shared runners. ASV benchmarks supporting the paper must run on
identified, controlled hardware and include the full workflow cost: design,
probability evaluation, accuracy assessment, posterior computation, and
fallback.

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

User installations:

```bash
python -m pip install stableboundary
python -m pip install "stableboundary[plot,arviz]"
python -m pip install "stableboundary[certify]"
```

Maintainer workflow:

```bash
uv sync --all-groups --all-extras
uv run pytest -m "not slow and not validation and not enclosure"
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/stableboundary
uv run sphinx-build -W docs docs/_build/html
uv run python -m build
uv run twine check dist/*
uv run check-wheel-contents dist/*.whl
```

## Source Assessment

All critical stack claims were checked against primary project documentation or
official PyPI metadata as of 2026-08-24.

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

Use a small scientific core and make validation architecture first-class:

- **Runtime:** CPython 3.12+, NumPy 2.2+, SciPy 1.18+.
- **Build:** `src/` layout, PEP 621, Hatchling; uv only for the locked developer
  environment.
- **Inference:** exact-model finite-cell likelihood plus deterministic
  two-dimensional quadrature first.
- **Numerics:** package-owned `S0` Fourier backend and guarded SciPy piecewise
  backend, never an unexamined singleton call.
- **Results:** immutable package result and audit record; optional ArviZ DataTree
  export.
- **Rigor:** Python-FLINT optional, with no `certified` label until analytic
  truncation and propagation bounds exist.
- **Scaling:** a later NumPy/SciPy adaptive-tempering SMC engine behind a stable
  protocol, validated against quadrature before it handles unknown nuisance
  parameters.

## RESEARCH COMPLETE
