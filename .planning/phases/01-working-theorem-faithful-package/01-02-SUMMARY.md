---
phase: 01-working-theorem-faithful-package
plan: "02"
subsystem: numerical-backend
tags: [python, scipy, levy-stable, s0, numpy, finite-cells]

# Dependency graph
requires:
  - phase: 01-working-theorem-faithful-package-01
    provides: Immutable stable parameters, local design, and known-nuisance contracts
provides:
  - Runtime-checkable package-owned stable numerical backend protocol
  - Locked snapshot/set/restore SciPy S0 piecewise adapter
  - Seeded read-only stable simulation and immutable three-cell counts
  - Direct-log-tail exact finite S0 cell probabilities with visible refusal states
affects:
  - 01-03-posterior-and-result
  - 01-04-installed-proof-of-work
  - phase-02-independent-numerical-lineage

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Scoped ownership-preserving restoration of mutable third-party global settings
    - Direct log-tail evaluation followed by strict finite-simplex validation

key-files:
  created:
    - src/stableboundary/backends/__init__.py
    - src/stableboundary/backends/_protocol.py
    - src/stableboundary/backends/_scipy_s0.py
    - src/stableboundary/cells.py
    - src/stableboundary/simulation.py
    - tests/test_probabilities.py
    - tests/test_simulation.py
  modified:
    - src/stableboundary/__init__.py

key-decisions:
  - "Guard all eleven relevant levy_stable settings and restore both their exact values and inherited-versus-instance ownership."
  - "Evaluate finite cells at explicit standardized S0 parameters loc=0 and scale=1 through logcdf(-u) and logsf(u), refusing underflow or invalid simplexes without repair."
  - "Use numpy.random.default_rng for public simulation and return copied non-writeable float64 vectors."

patterns-established:
  - "SciPy isolation: levy_stable is imported only by the guarded adapter and every call holds one module RLock."
  - "Finite cells: validate direct signed log tails before exponentiation, derive the center once, and never repair probabilities silently."

requirements-completed: [NUM-01, NUM-02, NUM-04, FIT-01]

# Metrics
duration: 23 min
completed: 2026-08-24
---

# Phase 1 Plan 02: Stable Backend and Finite Experiment Summary

**Ownership-preserving guarded SciPy S0 calls, seeded immutable simulation, and exact finite three-cell probabilities from direct log tails without clipping or renormalization**

## Performance

- **Duration:** 23 min
- **Started:** 2026-08-24T10:29:00Z
- **Completed:** 2026-08-24T10:52:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Defined a runtime-checkable stable backend protocol with immutable method, tolerance, and S0 metadata plus scalar/array probability operations.
- Isolated SciPy stable-law access behind an RLock and a complete eleven-setting snapshot, forcing S0 piecewise PDF/CDF behavior and restoring state on success and exceptions.
- Added deterministic generator-based stable simulation with size/allocation guards and finite, one-dimensional, non-writeable output.
- Added immutable known-nuisance cell counts and exact-model finite S0 cell probabilities using direct `logcdf(-u)` and `logsf(u)` calls with strict underflow/simplex refusal.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build the package-owned backend protocol and guarded SciPy S0 adapter** - `3336d97` (feat)
2. **Task 2: Implement seeded S0 simulation and immutable three-cell counting** - `d77d46a` (feat)
3. **Final numerical-state audit fix** - `cf35bc0` (fix)

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `src/stableboundary/backends/__init__.py` - Internal backend protocol and implementation exports.
- `src/stableboundary/backends/_protocol.py` - Runtime-checkable numerical protocol and immutable backend metadata.
- `src/stableboundary/backends/_scipy_s0.py` - Locked, ownership-preserving SciPy S0 snapshot/set/restore adapter.
- `src/stableboundary/simulation.py` - Seeded public S0 simulation with allocation and output validation.
- `src/stableboundary/cells.py` - Immutable cell counts and validated exact finite probabilities.
- `src/stableboundary/__init__.py` - Curated exports for `simulate`, `CellCounts`, and `CellProbabilities`.
- `tests/test_probabilities.py` - State restoration, direct-tail, reflection, invalid-simplex, counting, and source-policy tests.
- `tests/test_simulation.py` - Seed, allocation, global-RNG isolation, shape, finiteness, and immutability tests.

## Decisions Made

- Preserved not only each incoming SciPy setting value but also whether that value was inherited from the distribution class or owned by the singleton; this avoids persistent instance-level shadows after a package call.
- Kept SciPy as the explicitly named Phase 1 bootstrap backend. Deep-tail zero or nonfinite output is a visible `NumericalProbabilityError`, not a repaired probability or an independent-validation claim.
- Required exact cell evaluation to use local coordinates on the design's identical `r` scale and explicitly constructed standardized `StableParams(alpha, beta, loc=0, scale=1)`.
- Exposed only the three planned additions on the package facade; the backend and `exact_cell_probabilities` function remain internal implementation contracts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored SciPy setting ownership as well as values**
- **Found during:** Final plan-level numerical-state audit
- **Issue:** Restoring an originally inherited SciPy setting with `setattr` preserved its value but created a permanent singleton attribute, preventing later class-level changes from propagating.
- **Fix:** Snapshot whether every guarded field is instance-owned, restore owned values with `setattr`, and remove temporary shadows with `delattr` for originally inherited fields.
- **Files modified:** `src/stableboundary/backends/_scipy_s0.py`, `tests/test_probabilities.py`
- **Verification:** Ownership regression test, focused state tests, Ruff, strict mypy, focused plan suite, and full suite all pass.
- **Committed in:** `cf35bc0`

---

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** The fix strengthens the required no-global-state-leak guarantee without changing the public or statistical API.

## Issues Encountered

None. SciPy may report zero or nonfinite deep-tail values in difficult regimes; this planned Phase 1 limitation is handled by structured refusal and remains subject to the independent numerical lineage in Phase 2.

## User Setup Required

None - no external service configuration required.

## Verification

- `python -m pytest -q tests/test_probabilities.py tests/test_simulation.py` - 38 passed.
- `python -m pytest -q` - 112 passed.
- `python -m ruff check src tests` - passed.
- `python -m mypy src` - passed with strict configuration.
- Source inspection confirmed `scipy.stats.levy_stable` appears only in `_scipy_s0.py` under `src/stableboundary`.
- Source inspection confirmed no CDF subtraction, clipping, maximum-based repair, or hidden renormalization in `cells.py`.
- Tests confirmed incoming hostile S1 state and all ten additional guarded settings survive both successful and exceptional backend calls.

## Next Phase Readiness

- Plan 01-03 can consume immutable cell counts and exact finite probabilities through the package-owned backend protocol.
- Phase 2 still owns independent tail numerics and scientific backend agreement; this plan makes SciPy underflow explicit rather than masking it.

## Self-Check: PASSED

- All seven created files, the modified public facade, and this summary exist.
- Task commits `3336d97`, `d77d46a`, and `cf35bc0` exist in git history.
- Focused tests, the full suite, Ruff, strict mypy, and every task acceptance criterion pass.

---
*Phase: 01-working-theorem-faithful-package*
*Completed: 2026-08-24*
