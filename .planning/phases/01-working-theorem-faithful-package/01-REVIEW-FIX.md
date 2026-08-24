---
phase: 01-working-theorem-faithful-package
fixed_at: 2026-08-24T14:59:49+02:00
review_path: .planning/phases/01-working-theorem-faithful-package/01-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 10
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-08-24T14:59:49+02:00  
**Source review:** `.planning/phases/01-working-theorem-faithful-package/01-REVIEW.md`  
**Iteration:** 1

**Summary:**

- Findings in scope: 10
- Fixed: 10
- Skipped: 0

## Fixed Issues

### CR-01: Posterior quantiles treat quadrature masses as pointwise CDF values

**Files modified:** `src/stableboundary/posterior.py`, `src/stableboundary/result.py`, `tests/test_fit_known.py`  
**Commit:** f52067a  
**Applied fix:** Replaced atomic Gauss-node quantiles with endpoint-aware continuous marginal CDFs for `h`, `p`, `alpha`, and `beta`, plus split-cell bilinear push-forward integration for `tau_plus` and `tau_minus`. The exact refined summaries returned to users are now retained and assessed by refinement, including median and common-grid remeshing error. Uniform and asymmetric-product analytic regressions verify the corrected quantiles.  
**Status:** fixed: requires human verification (numerical logic change)

### CR-02: Structurally compatible non-S0 backends are silently accepted

**Files modified:** `src/stableboundary/backends/_protocol.py`, `src/stableboundary/backends/__init__.py`, `src/stableboundary/cells.py`, `src/stableboundary/posterior.py`, `src/stableboundary/result.py`, `tests/test_probabilities.py`, `tests/test_fit_known.py`, `tests/test_prediction.py`  
**Commit:** 76007d7  
**Applied fix:** Added centralized backend and metadata validation that rejects every non-S0 backend before numerical evaluation. Posterior and cell results retain the validated immutable metadata snapshot, and summaries and audit records report that snapshot instead of a hard-coded parameterization. S1 test doubles are rejected at both probability and posterior boundaries.  
**Status:** fixed: requires human verification (backend-selection logic change)

### CR-03: Ambient SciPy settings can alter supposedly canonical inference

**Files modified:** `src/stableboundary/backends/_protocol.py`, `src/stableboundary/backends/_scipy_s0.py`, `tests/test_probabilities.py`  
**Commit:** a05ff8a  
**Applied fix:** Introduced a package-owned `levy_stable_gen`, explicitly forces all eleven result-affecting piecewise and FFT controls before every operation, and records the complete effective settings plus SciPy version in immutable backend metadata. Hostile public settings no longer alter package results.  
**Status:** fixed: requires human verification (numerical backend logic change)

### CR-04: A package-private lock does not isolate SciPy process-global mutation

**Files modified:** `src/stableboundary/backends/_scipy_s0.py`, `tests/test_probabilities.py`  
**Commit:** a05ff8a  
**Applied fix:** Removed all snapshot/mutate/restore operations on `scipy.stats.levy_stable`. Calls are serialized only around the package-owned generator, while direct public SciPy use remains independent and unchanged on success and failure paths.  
**Status:** fixed: requires human verification (concurrency and state-isolation logic change)

### CR-05: Artifact inspection can normalize unsafe archive members into safe-looking paths

**Files modified:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`  
**Commit:** 2f9bb38  
**Applied fix:** Validates raw archive member names before normalization and rejects absolute, parent-traversal, drive-qualified, and backslash-ambiguous paths. Tar symbolic-link and hard-link targets receive the same containment checks before either artifact is passed to `pip`; hostile wheel and sdist regressions cover the rejection paths.  
**Status:** fixed: requires human verification (artifact security logic change)

### WR-01: Prediction silently replaces the fitted backend

**Files modified:** `src/stableboundary/posterior.py`, `src/stableboundary/result.py`, `tests/test_prediction.py`  
**Commit:** 76007d7  
**Applied fix:** Retains the exact fitted backend behind the immutable posterior result, checks that its live metadata still matches the fitted snapshot, and reuses it for tail probabilities and posterior-predictive sampling. The continuity regression patches only the fitting backend, so constructing a replacement in prediction would fail.  
**Status:** fixed: requires human verification (prediction-path logic change)

### WR-02: ZIP sdists are accepted by discovery but opened as tar archives

**Files modified:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`  
**Commit:** 2f9bb38  
**Applied fix:** Restricted artifact discovery to exactly one wheel and one `.tar.gz` sdist, matching the build contract and tar-only inspection implementation. ZIP-only and mixed-artifact cases are rejected explicitly.

### WR-03: Installed smoke does not certify the inference method

**Files modified:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`  
**Commit:** 7596736  
**Applied fix:** The installed-artifact oracle now requires `method == "exact_finite_three_cell"` for wheel and sdist executions, with a regression for incorrect method identifiers.

### WR-04: Malformed scientific output can pass installed-smoke validation

**Files modified:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`  
**Commit:** 7596736  
**Applied fix:** Added strict JSON type checks without lossy integer coercion, exact count-key and total checks, non-negativity, finite-number checks, interval ordering, credible-mass validation, and quantity-specific domains for `alpha`, `beta`, `h`, `p`, `tau_plus`, and `tau_minus`. Malformed payload regressions exercise each contract.

### WR-05: Build and installed-smoke subprocesses have no time bounds

**Files modified:** `scripts/smoke_wheel.py`, `tests/test_installed_package.py`, `tests/test_smoke_wheel.py`  
**Commit:** 7596736  
**Applied fix:** Added explicit stage-specific timeouts for environment creation, installation, import checks, example execution, and the outer installed-artifact test. Timeout failures identify the stage and command context, and focused tests cover timeout propagation.

## Verification

- 169 ordinary tests passed; 1 installed test was deselected in the ordinary run.
- 31 focused artifact-smoke tests passed; 1 installed test was deselected.
- The fixed-seed real package example passed with `method="exact_finite_three_cell"`, `parameterization="S0"`, posterior mass `0.9999999999999999`, and joint total variation `0.0016834020163450326` below the `0.002` policy.
- Ruff lint and format checks passed.
- Strict mypy passed for all 13 source files.
- Wheel and sdist builds, Twine metadata validation, and wheel-content inspection passed.

---

_Fixed: 2026-08-24T14:59:49+02:00_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 1_

