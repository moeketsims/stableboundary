---
phase: 01-working-theorem-faithful-package
plan: "01"
subsystem: package-contracts
tags: [python, hatchling, scipy, lambertw, hypothesis, typed-dataclasses]

# Dependency graph
requires: []
provides:
  - Standards-based installable stableboundary package with a typed public facade
  - Immutable S0, local-coordinate, and signed-tail-gap parameter contracts
  - Auditable critical-rate design, compact prior, and nuisance provenance records
affects:
  - 01-02-stable-backend-and-finite-experiment
  - 01-03-posterior-and-result
  - 01-04-installed-proof-of-work

# Tech tracking
tech-stack:
  added: [numpy>=2.2, scipy>=1.18, hatchling, pytest, hypothesis, ruff, mypy]
  patterns:
    - Frozen slotted dataclasses for public statistical records
    - Package-owned validation and refusal errors at trust boundaries
    - Validated class constructors for derived auditable state

key-files:
  created:
    - pyproject.toml
    - src/stableboundary/_exceptions.py
    - src/stableboundary/parameters.py
    - src/stableboundary/design.py
    - tests/test_public_api.py
    - tests/test_parameters.py
    - tests/test_design.py
  modified: []

key-decisions:
  - "LocalDesign is constructible only through from_sample_size so callers cannot supply inconsistent derived values."
  - "LocalPrior retains its LocalDesign and validates the entire compact support on that exact design scale."
  - "Nuisance provenance uses a closed enum without string coercion, and Phase 1 explicitly accepts only externally_known records."
  - "Exact alpha=2 objects remain representable in S0 form but refuse conversion to unidentified local allocation coordinates."

patterns-established:
  - "Boundary records retain r: neither local h nor signed gaps lose their design scale."
  - "Derived formulas validate finiteness, domain, and an independently recomputed residual before returning an object."

requirements-completed: [PKG-02, PAR-01, PAR-02, PAR-03, DES-01, DES-02, DES-03]

# Metrics
duration: 20 min
completed: 2026-08-24
---

# Phase 1 Plan 01: Package Contracts and Design Summary

**Installable Hatchling package with immutable Nolan S0 coordinate contracts, a residual-checked Lambert-W local design, compact prior support, and explicit known-nuisance provenance**

## Performance

- **Duration:** 20 min
- **Started:** 2026-08-24T10:12:00Z
- **Completed:** 2026-08-24T10:32:00Z
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments

- Created a Python 3.12+ `src/` package with NumPy/SciPy runtime dependencies, a PEP 561 marker, strict quality tooling, and package-owned error types.
- Made conventional S0, local `(r,h,p)`, and signed-gap coordinates executable and immutable, including reflection properties and exact-Gaussian non-identification refusal.
- Derived `r` on the principal Lambert-W branch, verified the critical-rate residual below `1e-12`, and fixed the theoretical log-log threshold before any observations can enter.
- Added a normalized vectorized compact-uniform prior and closed nuisance provenance modes with explicit Phase 1 refusal semantics.

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold the standards-based package and test toolchain** - `661034e` (chore)
2. **Task 2: Implement immutable S0 and boundary-coordinate objects** - `ce13589` (feat)
3. **Task 3: Implement the prespecified local design, compact prior, and nuisance provenance** - `eed00e7` (feat)

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `.gitignore` - Excludes Python/build/cache and LaTeX auxiliary output while retaining manuscript sources and PDFs.
- `pyproject.toml` - PEP 621 metadata, Hatchling build configuration, dependencies, and test/lint/type settings.
- `LICENSE` - MIT license.
- `src/stableboundary/__init__.py` - Curated versioned public facade.
- `src/stableboundary/py.typed` - PEP 561 marker included in wheel and sdist.
- `src/stableboundary/_exceptions.py` - Package-owned validation, identification, numerical, convergence, and moment errors.
- `src/stableboundary/parameters.py` - Frozen S0, local-coordinate, and signed-tail-gap contracts and conversions.
- `src/stableboundary/design.py` - Critical-rate design, compact prior, and nuisance provenance contracts.
- `tests/conftest.py` - Deterministic shared parameter, design, and prior fixtures.
- `tests/test_public_api.py` - Distribution metadata, typing marker, facade, and exception hierarchy tests.
- `tests/test_parameters.py` - Domain, identity, round-trip, reflection, Gaussian refusal, and immutability tests.
- `tests/test_design.py` - Critical equation, threshold, prior, provenance, refusal, and immutability tests.

## Decisions Made

- Used the principal Lambert-W branch exactly as specified and independently recomputed the critical-rate equation before constructing a `LocalDesign`.
- Prevented direct `LocalDesign()` construction so an object cannot exist without validated `r`, threshold, formula identifiers, and residual.
- Included `LocalDesign` in `LocalPrior`, making prior validity inseparable from the scale on which its alpha support is interpreted.
- Required actual `NuisanceMode` members rather than coercing strings, preserving a closed and auditable provenance contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Allowed the public facade to grow across planned tasks**
- **Found during:** Task 2 (immutable S0 and boundary-coordinate objects)
- **Issue:** The Task 1 smoke test asserted an exact initial `__all__` set, which incorrectly rejected the parameter and design exports required later in the same plan.
- **Fix:** Changed the assertion to require the complete exception/version subset while permitting explicitly curated planned exports.
- **Files modified:** `tests/test_public_api.py`
- **Verification:** `python -m pytest -q tests/test_public_api.py tests/test_parameters.py`
- **Committed in:** `ce13589`

**2. [Rule 1 - Bug] Prevented uninitialized LocalDesign records**
- **Found during:** Task 3 (prespecified design and provenance)
- **Issue:** A frozen dataclass with generated initialization disabled still inherited a no-argument `object.__init__`, allowing `LocalDesign()` to create an object without derived fields.
- **Fix:** Added a refusing initializer and retained `from_sample_size()` as the sole validated constructor.
- **Files modified:** `src/stableboundary/design.py`, `tests/test_design.py`
- **Verification:** `python -m pytest -q tests/test_design.py tests/test_parameters.py`
- **Committed in:** `eed00e7`

---

**Total deviations:** 2 auto-fixed bugs.  
**Impact on plan:** Both fixes enforce the intended extensible public facade and constructor-integrity threat mitigation; no scope was added.

## Issues Encountered

- Context7 indexed SciPy but returned no matching Lambert-W excerpt, so the implementation was checked against the official SciPy 1.18 `scipy.special.lambertw` documentation instead.
- Twine validation passed with non-blocking long-description warnings; user-facing README work remains in the installed proof-of-work plan.

## User Setup Required

None - no external service configuration required.

## Verification

- `python -m pytest -q tests/test_public_api.py tests/test_parameters.py tests/test_design.py` - 74 passed.
- `python -m ruff check src tests` - passed.
- `python -m mypy src` - passed with strict configuration.
- `python -m build` - built `stableboundary-0.1.0.tar.gz` and `stableboundary-0.1.0-py3-none-any.whl`.
- `python -m check_wheel_contents dist/*.whl` - passed.
- `python -m twine check dist/*` - passed.
- Wheel and sdist inspection confirmed `stableboundary/py.typed` is present.

## Next Phase Readiness

- Plan 01-02 can build the guarded S0 backend and exact finite three-cell experiment on stable, tested parameter/design contracts.
- No blockers remain for the next plan.

## Self-Check: PASSED

- All key files listed above exist.
- Task commits `661034e`, `ce13589`, and `eed00e7` exist in git history.
- All task acceptance criteria and plan-level verification commands pass.

---
*Phase: 01-working-theorem-faithful-package*
*Completed: 2026-08-24*
