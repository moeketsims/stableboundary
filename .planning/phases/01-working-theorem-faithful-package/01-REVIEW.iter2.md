---
phase: 01-working-theorem-faithful-package
status: issues_found
depth: standard
files_reviewed: 30
files_reviewed_list:
  - .github/workflows/ci.yml
  - README.md
  - pyproject.toml
  - examples/known_nuisance_fit.py
  - scripts/smoke_wheel.py
  - src/stableboundary/__init__.py
  - src/stableboundary/_exceptions.py
  - src/stableboundary/api.py
  - src/stableboundary/approximation.py
  - src/stableboundary/backends/__init__.py
  - src/stableboundary/backends/_protocol.py
  - src/stableboundary/backends/_scipy_s0.py
  - src/stableboundary/cells.py
  - src/stableboundary/design.py
  - src/stableboundary/parameters.py
  - src/stableboundary/posterior.py
  - src/stableboundary/result.py
  - src/stableboundary/simulation.py
  - tests/conftest.py
  - tests/test_approximation.py
  - tests/test_design.py
  - tests/test_fit_known.py
  - tests/test_identification.py
  - tests/test_installed_package.py
  - tests/test_parameters.py
  - tests/test_prediction.py
  - tests/test_probabilities.py
  - tests/test_public_api.py
  - tests/test_simulation.py
  - src/stableboundary/py.typed
findings:
  critical: 5
  warning: 5
  info: 0
  total: 10
reviewed: 2026-08-24
---

# Phase 01 Code Review

## Verdict

The package is not ready for the Phase 01 verification gate. The implementation is coherent and its existing tests pass, but the adversarial numerical and package reviews identified five correctness or safety blockers and five reliability defects. All ten findings are in scope for repair before the next push.

## Critical findings

### CR-01: Posterior quantiles treat quadrature masses as pointwise CDF values

- **Files:** `src/stableboundary/posterior.py`, `src/stableboundary/result.py`
- **Evidence:** Marginal quantiles are obtained by applying `numpy.interp` directly to cumulative Gauss-node weights. Gaussian quadrature weights represent integrated cell mass, not CDF values located at the nodes. Even a uniform posterior therefore exhibits a half-cell displacement. The refinement test compares the same biased construction at two resolutions and can certify the wrong summaries.
- **Required fix:** Construct marginal continuous CDFs on supports that include their endpoints, integrate normalized marginal densities, invert those CDFs for all reported quantiles, and make convergence checks assess the exact summaries returned to users. Add a no-data/uniform-posterior regression test with analytically known median and interval endpoints.

### CR-02: Structurally compatible non-S0 backends are silently accepted

- **Files:** `src/stableboundary/cells.py`, `src/stableboundary/posterior.py`, `src/stableboundary/result.py`
- **Evidence:** Backend calls are accepted through the protocol without checking `metadata.parameterization`. Result audit metadata then reports `S0` unconditionally. An S1 backend can consequently be used while the public result claims S0 semantics.
- **Required fix:** Validate S0 parameterization at every public inference/probability boundary, preserve the actual validated backend metadata in results, and add rejection tests for an otherwise conforming S1 test double.

### CR-03: Ambient SciPy settings can alter supposedly canonical inference

- **File:** `src/stableboundary/backends/_scipy_s0.py`
- **Evidence:** The configuration context snapshots and restores some SciPy settings but does not force and record every result-affecting piecewise tolerance/method setting. Prior mutations of the public SciPy singleton can change computed probabilities while package metadata continues to report package defaults.
- **Required fix:** Use a fully configured package-owned stable-distribution generator, explicitly set all result-affecting options, record their effective values, and test that hostile ambient SciPy settings do not change package output.

### CR-04: A package-private lock does not isolate SciPy process-global mutation

- **File:** `src/stableboundary/backends/_scipy_s0.py`
- **Evidence:** The lock coordinates only calls made through `stableboundary`. Direct concurrent use of `scipy.stats.levy_stable` can observe temporary package mutations or race with them.
- **Required fix:** Stop mutating SciPy's public singleton. Configure and guard a package-owned generator instance (or an equivalently isolated implementation), and test that the public SciPy singleton remains unchanged during package calls.

### CR-05: Artifact inspection can normalize unsafe archive members into safe-looking paths

- **File:** `scripts/smoke_wheel.py`
- **Evidence:** Member paths are normalized with `lstrip("./")`, which erases absolute or traversal prefixes before validation. Tar symbolic- and hard-link targets are not validated. A malicious path such as `../../payload.py` can therefore become `payload.py` and the artifact may subsequently be passed to `pip`.
- **Required fix:** Reject absolute paths, parent traversal, drive-qualified paths, and unsafe tar link targets before installation. Add hostile wheel/sdist member tests.

## Warnings

### WR-01: Prediction silently replaces the fitted backend

- **File:** `src/stableboundary/result.py`
- **Evidence:** Predictive methods instantiate `ScipyS0Backend` even when inference used an injected, validated backend. Prediction can therefore use different numerical semantics from fitting.
- **Required fix:** Retain and reuse the fitted backend, or require an explicitly supplied backend whose metadata is checked for compatibility. Add a backend-continuity test.

### WR-02: ZIP sdists are accepted by discovery but opened as tar archives

- **File:** `scripts/smoke_wheel.py`
- **Evidence:** Artifact discovery accepts `.zip`, while archive inspection always calls the tar reader.
- **Required fix:** Restrict the smoke contract to the built `.tar.gz` sdist or implement a separate ZIP inspection path. Test the selected contract.

### WR-03: Installed smoke does not certify the inference method

- **File:** `scripts/smoke_wheel.py`
- **Evidence:** The subprocess oracle checks status and numerical summaries but does not require `method == "exact_finite_three_cell"`.
- **Required fix:** Assert the exact method identifier in installed wheel and sdist runs.

### WR-04: Malformed scientific output can pass installed-smoke validation

- **File:** `scripts/smoke_wheel.py`
- **Evidence:** Counts are coerced through `int`, allowing fractional or string values; non-negativity, count totals, interval ordering, and parameter-domain constraints are not comprehensively checked.
- **Required fix:** Validate strict JSON types, count keys and total, finite values, interval ordering, and all alpha/beta/local-parameter domains without lossy coercion. Add malformed-payload tests.

### WR-05: Build and installed-smoke subprocesses have no time bounds

- **Files:** `scripts/smoke_wheel.py`, `tests/test_installed_package.py`
- **Evidence:** Dependency installation and proof-of-work subprocesses can hang indefinitely, including in CI.
- **Required fix:** Add explicit, stage-appropriate subprocess timeouts, surface timeout context cleanly, and cover timeout handling in tests.

## Required re-review

Re-run numerical correctness, backend-provenance, archive-safety, lint, strict typing, ordinary tests, build metadata, wheel-content, and clean wheel/sdist installation checks after repairs. The review status may become `clean` only when no Critical or Warning findings remain.

