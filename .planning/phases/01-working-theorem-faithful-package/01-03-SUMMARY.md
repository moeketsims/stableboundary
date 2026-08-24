---
phase: 01-working-theorem-faithful-package
plan: "03"
subsystem: bayesian-inference
tags: [python, scipy, gauss-legendre, stable-law, posterior, prediction]

requires:
  - phase: 01-working-theorem-faithful-package-01
    provides: Immutable local coordinates, design, prior, and nuisance contracts
  - phase: 01-working-theorem-faithful-package-02
    provides: Guarded S0 backend, simulation, counts, and exact finite cells
provides:
  - Exact finite three-cell posterior with common-grid joint-TV convergence evidence
  - Immutable known-nuisance fit with six summaries and identification diagnostics
  - Full stable predictive tails, expected counts, draws, quantiles, and variance refusal
  - Explicit compactly truncated signed-Poisson Gamma-Beta approximation
affects: [01-04-installed-proof-of-work, phase-02-independent-numerical-lineage]

tech-stack:
  added: []
  patterns:
    - Centered log-domain tensor quadrature with read-only retained mass
    - Joint-posterior convergence on a shared bilinear/trapezoidal grid
    - Exact S0 upper tails through reflected direct CDF evaluation

key-files:
  created:
    - src/stableboundary/posterior.py
    - src/stableboundary/result.py
    - src/stableboundary/api.py
    - src/stableboundary/approximation.py
    - tests/test_fit_known.py
    - tests/test_identification.py
    - tests/test_prediction.py
    - tests/test_approximation.py
  modified:
    - src/stableboundary/__init__.py
    - src/stableboundary/_exceptions.py
    - src/stableboundary/backends/_scipy_s0.py
    - tests/test_probabilities.py

key-decisions:
  - "Use 20/32 Gauss-Legendre nodes because the planned 16/24 pair failed the unchanged 0.002 joint-TV gate on a real finite-cell fixture."
  - "Define p information gain as KL(posterior || prior) and interval contraction against the prior's equal-tail interval at the same mass."
  - "Evaluate S0 survival probabilities as reflected direct CDF values because SciPy's generic survival implementation subtracts from one."
  - "Keep the Gamma-Beta limit compactly truncated and reachable only through fit_limiting_approximation."

patterns-established:
  - "Research status is a literal: every exact Phase 1 fit is research_uncertified."
  - "Evidence and precision are separate: positive counts never imply calibrated precision."

requirements-completed: [FIT-02, FIT-03, FIT-04, FIT-05, FIT-06, FIT-07]

duration: 40 min
completed: 2026-08-24
---

# Phase 1 Plan 03: Exact Posterior, Results, and Prediction Summary

**Exact finite S0 cell inference with joint-TV-refined deterministic quadrature, honest skewness diagnostics, full-model prediction, and a separately truncated Gamma-Beta benchmark**

## Performance

- **Duration:** 40 min
- **Started:** 2026-08-24T10:58:39Z
- **Completed:** 2026-08-24T11:38:19Z
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments

- Built a centered log-domain exact finite-cell posterior whose read-only 20-by-20 and 32-by-32 Gauss-Legendre evaluations must agree in joint TV, evidence, six summaries, and predictive tails within 0.002.
- Added the public known-nuisance workflow, immutable six-quantity summaries, JSON-safe audit record, zero/one/two-sided evidence states, KL and interval-contraction diagnostics, and a dedicated infinite-variance refusal.
- Added direct posterior stable-tail prediction, expected signed exceedance counts, reproducible mixture draws, and quantiles with seed, bit generator, draw count, requested probability, and eight-batch MCSE.
- Added a distinct compactly truncated signed-Poisson Gamma-Beta limit with explicit assumptions, intensities, conjugate parameters, support, and approximation metadata.

## Task Commits

1. **Task 1: Exact finite posterior on a refinable grid** - `dab9fa0` (feat)
2. **Task 2: Immutable results, diagnostics, audit, and prediction** - `71809ad` (feat)
3. **Task 3: Explicit limiting Gamma-Beta benchmark** - `668e19e` (feat)

Supporting numerical corrections: `54e5313`, `cf9295f`.

## Files Created/Modified

- `src/stableboundary/posterior.py` - Exact tensor posterior and structured common-grid refinement evidence.
- `src/stableboundary/result.py` - Immutable summaries, evidence diagnostics, audit, prediction, and refusal behavior.
- `src/stableboundary/api.py` - Direct `fit_known_nuisance` workflow with no approximation dependency.
- `src/stableboundary/approximation.py` - Compact signed-Poisson Gamma-Beta benchmark.
- `src/stableboundary/backends/_scipy_s0.py` - Fast checked logarithms and cancellation-free reflected survival evaluation.
- `src/stableboundary/_exceptions.py` - Dedicated `InfiniteVarianceError`.
- `src/stableboundary/__init__.py` - Curated exact and long-form approximation exports.
- `tests/test_fit_known.py`, `tests/test_identification.py`, `tests/test_prediction.py`, `tests/test_approximation.py`, `tests/test_probabilities.py` - Numerical, inference, prediction, separation, and backend regressions.

## Decisions Made

- Increased the quadrature pair from the planned 16/24 to 20/32 while retaining the 65-by-65 common grid and 0.002 tolerance. On the real benchmark fixture, 16/24 refused with joint TV 0.00294887; 20/32 passed with joint TV 0.00182789178 and maximum other component 0.000335855395.
- Kept evidence strength distinct from calibrated precision: zero tails are `prior_dominated/unidentified`; one-sided and two-sided positive evidence remain `not_assessed` for precision.
- Used exact S0 reflection, `SF(x; beta, loc) = CDF(-x; -beta, -loc)`, because inspection showed SciPy's generic `_sf` computes `1 - _cdf`.
- Defined the limiting result on the same finite prior rectangle; no unbounded conjugate law is substituted.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Numerical bug] Increased the quadrature pair to satisfy the declared joint-TV gate**
- **Found during:** Task 1
- **Issue:** The specified 16/24 pair produced joint TV 0.00294887 on a representative exact finite posterior and therefore could not honestly pass tolerance 0.002.
- **Fix:** Raised the default pair to 20/32 without weakening the tolerance or removing any refinement component.
- **Files modified:** `src/stableboundary/posterior.py`, `tests/test_fit_known.py`
- **Verification:** Real benchmark passed in 1.712 seconds with TV 0.00182789178 and all other changes at most 0.000335855395.
- **Committed in:** `dab9fa0`

**2. [Rule 1 - Performance bug] Removed SciPy generic log-tail wrapper overhead**
- **Found during:** Task 1
- **Issue:** Generic `levy_stable.logcdf/logsf` made the default grid take over two minutes.
- **Fix:** Evaluate guarded finite probabilities first, reject underflow, then take checked logarithms.
- **Files modified:** `src/stableboundary/backends/_scipy_s0.py`, `tests/test_probabilities.py`
- **Verification:** Backend tests pass and a real default grid completes in practical time.
- **Committed in:** `54e5313`

**3. [Rule 1 - Scientific correctness] Replaced subtractive SciPy survival with exact reflection**
- **Found during:** Final threat/numerical scan
- **Issue:** SciPy's generic `_sf` is implemented as `1.0 - _cdf`, contrary to the direct-tail contract.
- **Fix:** Evaluate the reflected S0 lower tail directly and forbid the subtractive survival path in regression tests.
- **Files modified:** `src/stableboundary/backends/_scipy_s0.py`, `tests/test_probabilities.py`
- **Verification:** 25 backend tests, the real default posterior, and the 134-test suite pass.
- **Committed in:** `cf9295f`

**4. [Rule 2 - Missing critical refusal] Added a dedicated infinite-variance error**
- **Found during:** Task 2
- **Issue:** The existing exception hierarchy had only a generic infinite-moment error, while the public contract requires an explicit variance refusal.
- **Fix:** Added and exported `InfiniteVarianceError` and tested refusal whenever positive mass has alpha below two.
- **Files modified:** `src/stableboundary/_exceptions.py`, `src/stableboundary/result.py`, `src/stableboundary/__init__.py`, `tests/test_prediction.py`
- **Verification:** Prediction and full suites pass.
- **Committed in:** `71809ad`

---

**Total deviations:** 4 auto-fixed (3 Rule 1, 1 Rule 2).  
**Impact on plan:** Accuracy, runtime, and scientific refusal behavior were strengthened; the exact likelihood, 0.002 gate, and public scope were unchanged.

## Issues Encountered

- The real fixed-seed public workflow completed with counts `(1, 4993, 6)`, six finite posterior means, `two_sided_evidence/not_assessed`, and maximum refinement component 0.00180275304.

## User Setup Required

None - no external service configuration required.

## Verification

- `python -m pytest -q tests/test_fit_known.py tests/test_identification.py tests/test_prediction.py tests/test_approximation.py` - 21 passed.
- `python -m pytest -q -m "not slow and not installed"` - 134 passed in 17.10 seconds.
- `python -m ruff check src tests` - passed.
- `python -m mypy src` - passed with strict configuration.
- Source inspection confirmed `api.py` has no limiting-approximation import and SciPy stable access remains isolated to the guarded backend.
- Stub scan found no goal-blocking stubs; optional `None` controls and local empty accumulator lists are intentional.

## Next Phase Readiness

- Plan 01-04 can document, build, install, and execute the fixed-seed public workflow.
- Phase 2 can replace bootstrap numerical self-agreement with an independent finite-cell and full-posterior lineage.

## Self-Check: PASSED

- All created files exist and all task/supporting commits are present in git history.
- Every task acceptance criterion and plan-level verification command passes.
- No untracked manuscript, spike, or temporary workspace artifact was modified or staged.

---
*Phase: 01-working-theorem-faithful-package*
*Completed: 2026-08-24*
