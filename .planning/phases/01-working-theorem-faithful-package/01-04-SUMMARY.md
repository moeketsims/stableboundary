---
phase: 01-working-theorem-faithful-package
plan: "04"
subsystem: packaging
tags: [python, hatchling, wheel, sdist, ci, installed-smoke, documentation]

requires:
  - phase: 01-03
    provides: Immutable exact finite-cell known-nuisance fit and public result API
provides:
  - Fixed-seed top-level public example with an auditable exact finite-cell fit
  - Scope-honest README for known nuisance, S0, and research_uncertified status
  - Member-inspected wheel and sdist installation smoke in separate clean environments
  - Python 3.12-3.14 CI across Linux, Windows, and macOS
affects: [02-independent-numerics, release-validation, documentation]

tech-stack:
  added: []
  patterns:
    - Repository-owned example copied outside the checkout before installed execution
    - Archive member allow/deny inspection before installing built artifacts
    - Separate temporary virtual environment for each wheel and sdist smoke

key-files:
  created:
    - README.md
    - examples/known_nuisance_fit.py
    - scripts/smoke_wheel.py
    - tests/test_installed_package.py
    - .github/workflows/ci.yml
  modified:
    - pyproject.toml
    - src/stableboundary/backends/_scipy_s0.py
    - src/stableboundary/result.py
    - tests/test_probabilities.py

key-decisions:
  - "Use one repository-owned top-level-API example for documentation and installed-artifact verification."
  - "Inspect and install both the wheel and sdist in distinct temporary virtual environments outside the checkout."
  - "Run ordinary tests before building artifacts in every CI matrix job."

requirements-completed: [PKG-01, PKG-03, VAL-02]

duration: 21 min
completed: 2026-08-24
---

# Phase 1 Plan 4: Installed Proof and Packaging Summary

**A real fixed-seed S0 fit now runs through the top-level API from both clean wheel and sdist installations, with exact finite-cell scope and research_uncertified status made explicit.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-08-24T11:43:33Z
- **Completed:** 2026-08-24T12:05:02Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Added a fixed-seed `n=5000` example that creates the design before deriving
  `alpha`, simulates through `stableboundary`, and completes the exact
  known-location/scale posterior using only the top-level public API.
- Documented installation, S0 scale semantics, infinite-variance behavior,
  signed-gap interpretation, identification limits, and the deliberately
  `research_uncertified` Phase 1 status without implying four-parameter use.
- Added explicit zip/tar member inspection, `py.typed` verification, and fresh
  external wheel and sdist installation tests that reject checkout imports.
- Added a 3-OS by 3-Python-version CI matrix in the required lint, type,
  ordinary-test, build, metadata, content, and installed-smoke order.

## Actual Fixed-Seed Result

| Field | Result |
|---|---|
| Status | `research_uncertified` |
| Method | `exact_finite_three_cell` |
| Parameterization | `S0` |
| Fixed nuisance | `loc=0.0`, `scale=1.0`, externally known |
| Counts | `n_minus=1`, `n_zero=4996`, `n_plus=3` |
| Identification | `two_sided_evidence`; precision `not_assessed` |
| Posterior mass | `0.9999999999999999` |
| Refinement | converged; joint TV/max component `0.0016834020163450326` at tolerance `0.002` |
| Direct example wall time | 19.46 seconds on CPython 3.13.14/Windows |

Both the wheel and sdist returned the same status, counts, normalized mass, and
refinement result while importing from different temporary-environment
`site-packages` directories outside the repository.

## Task Commits

1. **Task 1: Write and test the fixed-seed installed-package quickstart** — `eea96d7` (feat)
2. **Task 2: Add isolated wheel smoke verification and CI** — `24457d8` (chore)
3. **Verification correction: align repository quality checks** — `ca7be14` (style)

## Files Created/Modified

- `examples/known_nuisance_fit.py` — Public, fixed-seed simulation and exact fit.
- `README.md` — Scope-honest user quickstart and maintainer commands.
- `scripts/smoke_wheel.py` — Archive inspection plus separate wheel/sdist venv execution.
- `tests/test_installed_package.py` — Installed-marker delegation to the artifact runner.
- `.github/workflows/ci.yml` — Cross-platform, cross-version package verification.
- `pyproject.toml` — README metadata, controlled build contents, and tooling scope.
- `src/stableboundary/backends/_scipy_s0.py` — Ruff-only formatting correction.
- `src/stableboundary/result.py` — Ruff-only formatting correction.
- `tests/test_probabilities.py` — Ruff-only formatting correction.

## Decisions Made

- The documentation and installed smoke share the exact same example entry
  point, preventing a lightweight test-only path from masquerading as proof.
- Each archive gets a separate clean environment; the sdist must build and run
  independently rather than inheriting the wheel environment.
- The smoke runner removes `PYTHONPATH`/`PYTHONHOME`, uses isolated interpreter
  mode for imports and execution, and requires the installed origin to reside
  inside the active temporary venv.
- Ruff excludes preserved planning research and generated directories, while
  production source and tests remain subject to repository-wide checks.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Scoped repository-wide Ruff checks away from preserved research artifacts**

- **Found during:** Overall verification
- **Issue:** `ruff check .` traversed the intentionally preserved untracked
  `.planning/spikes/` research implementation, which is not production code and
  contains independent historical formatting conventions.
- **Fix:** Added narrow `.planning`, `build`, `dist`, and `tmp` Ruff exclusions.
- **Files modified:** `pyproject.toml`
- **Verification:** `python -m ruff check .` passes.
- **Commit:** `ca7be14`

**2. [Rule 3 - Blocking] Corrected pre-existing formatter drift in three prior-plan files**

- **Found during:** Overall verification
- **Issue:** `ruff format --check .` reported three previously committed files.
- **Fix:** With orchestrator authorization, applied Ruff formatting to exactly
  those files and no other prior-plan or user files.
- **Files modified:** `src/stableboundary/backends/_scipy_s0.py`,
  `src/stableboundary/result.py`, `tests/test_probabilities.py`
- **Verification:** `python -m ruff format --check .` reports 28 files already formatted.
- **Commit:** `ca7be14`

**Total deviations:** 2 auto-fixed blocking verification issues. **Impact:**
Tooling scope and formatting only; no statistical or runtime behavior changed.

## Verification

- `python -m ruff check .` — passed.
- `python -m ruff format --check .` — passed (28 files formatted).
- `python -m mypy src` — passed (13 source files).
- `python -m pytest -q -m "not installed"` — 134 passed, 1 deselected.
- `python -m build` — built one wheel and one sdist.
- `python -m twine check dist/*` — both archives passed.
- `python -m check_wheel_contents dist/*.whl` — wheel OK.
- `python -m pytest -q tests/test_installed_package.py -m installed -s` —
  1 passed in 500.04 seconds; both archives installed and ran the full example.
- Manual README claim review — known nuisance, S0, infinite variance,
  identification limits, and `research_uncertified` status are explicit; no
  certificate, automatic safety decision, or four-parameter fit is claimed.

## Known Stubs

None. Empty lists found by the stub scan are test call collectors and do not
flow to package output or documentation.

## Issues Encountered

Fresh Windows environments incurred substantial dependency installation and
first-import latency. The process tree confirmed that duplicate-looking Python
processes were the normal Windows venv redirector parent/child pair, not smoke
recursion. Both archive runs completed successfully.

## Next Phase Readiness

Phase 1 now has an installed proof of work. Phase 2 can build independent
numerical checks and the full stable-likelihood reference posterior on top of a
verified public artifact and reproducible example.

## Self-Check: PASSED

All declared key files exist and commits `eea96d7`, `24457d8`, and `ca7be14`
are present in repository history.
