---
phase: 1
slug: working-theorem-faithful-package
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-24
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for the first installable, theorem-faithful
> known-location/scale fit.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x with Hypothesis properties |
| **Config file** | `pyproject.toml` — Wave 0 creates it |
| **Quick run command** | `python -m pytest -q -m "not slow and not installed"` |
| **Full suite command** | `python -m pytest -q` |
| **Build command** | `python -m build` |
| **Installed smoke command** | Build wheel, install it into a temporary virtual environment, then run `python examples/known_nuisance_fit.py` outside the repository root |
| **Estimated quick runtime** | Under 30 seconds after Wave 0 |
| **Estimated full runtime** | Under 120 seconds for Phase 1 |

## Sampling Rate

- **After every task commit:** Run the smallest named pytest module covering
  the modified behavior plus
  `python -m pytest -q -m "not slow and not installed"` when the change crosses
  module boundaries.
- **After every plan wave:** Run `python -m pytest -q`.
- **After metadata or packaging changes:** Run `python -m build` and inspect the
  wheel/sdist contents.
- **Before phase verification:** Run the complete test, type, lint, build, and
  installed-wheel smoke sequence.
- **Max ordinary feedback latency:** 30 seconds.

## Requirement Verification Map

| Requirement | Expected test or artifact | Test type | Automated command | Status |
|-------------|---------------------------|-----------|-------------------|--------|
| PKG-01 | Clean wheel and sdist installation | packaging | `python -m build && python -m pytest -q -m installed` | ⬜ pending |
| PKG-02 | Public import surface and `py.typed` | unit/typing | `python -m pytest -q tests/test_public_api.py && python -m mypy src` | ⬜ pending |
| PKG-03 | Documented maintainer commands execute | tooling | `python -m ruff check . && python -m ruff format --check . && python -m mypy src && python -m pytest -q && python -m build` | ⬜ pending |
| PAR-01 | Stable parameter domains and immutability | unit/property | `python -m pytest -q tests/test_parameters.py` | ⬜ pending |
| PAR-02 | Stable/local/signed round trips retain `r` | unit/property | `python -m pytest -q tests/test_parameters.py` | ⬜ pending |
| PAR-03 | Exact Gaussian conversion refuses arbitrary beta identification | unit | `python -m pytest -q tests/test_parameters.py -k gaussian` | ⬜ pending |
| DES-01 | Critical-rate `r`, threshold, and supported alpha region | unit/property | `python -m pytest -q tests/test_design.py` | ⬜ pending |
| DES-02 | Unsupported compact regions are refused | unit/property | `python -m pytest -q tests/test_design.py -k invalid` | ⬜ pending |
| DES-03 | Known-nuisance provenance is explicit | unit | `python -m pytest -q tests/test_design.py -k nuisance` | ⬜ pending |
| NUM-01 | Package-owned probability protocol and guarded `S0` call | numerical/unit | `python -m pytest -q tests/test_probabilities.py` | ⬜ pending |
| NUM-02 | Cell probabilities are finite, nonnegative, and normalized or fail | numerical/property | `python -m pytest -q tests/test_probabilities.py -k cells` | ⬜ pending |
| NUM-04 | Positive extreme probability avoids unguarded `1-cdf` | numerical/regression | `python -m pytest -q tests/test_probabilities.py -k tail` | ⬜ pending |
| FIT-01 | Fixed-seed `S0` simulation reproducibility | unit | `python -m pytest -q tests/test_simulation.py` | ⬜ pending |
| FIT-02 | Exact finite multinomial posterior through deterministic 2-D quadrature | integration | `python -m pytest -q tests/test_fit_known.py -k exact` | ⬜ pending |
| FIT-03 | Posterior grid normalizes and all required summaries are finite | integration | `python -m pytest -q tests/test_fit_known.py -k summary` | ⬜ pending |
| FIT-04 | Zero counts are prior-dominated; one-sided positive counts report evidence and quantitative information without a precision claim | unit/integration | `python -m pytest -q tests/test_identification.py` | ⬜ pending |
| FIT-05 | Predictive draws, quantiles, and tail probabilities are reproducible | integration | `python -m pytest -q tests/test_prediction.py` | ⬜ pending |
| FIT-06 | Predictive variance is refused for posterior support below alpha two | unit | `python -m pytest -q tests/test_prediction.py -k variance` | ⬜ pending |
| FIT-07 | Limiting Gamma-Beta method is explicit and never default | unit/regression | `python -m pytest -q tests/test_approximation.py` | ⬜ pending |
| VAL-02 | Installed-package fixed-seed example completes with finite summaries | end-to-end | `python -m pytest -q -m installed` | ⬜ pending |

## Wave 0 Requirements

- [ ] `pyproject.toml` — build metadata plus pytest, Ruff, and mypy configuration.
- [ ] `tests/conftest.py` — fixed designs, priors, stable parameters, and small
  deterministic fixtures.
- [ ] `tests/test_public_api.py` — import and type-marker smoke tests.
- [ ] `tests/test_parameters.py` — parameter-domain and coordinate properties.
- [ ] `tests/test_design.py` — critical-rate and threshold contracts.
- [ ] `tests/test_probabilities.py` — finite cell probability contracts.
- [ ] `tests/test_fit_known.py` — posterior normalization and summary contract.
- [ ] `tests/test_identification.py` — zero/one-sided count behavior.
- [ ] `tests/test_prediction.py` — predictive quantities and variance refusal.
- [ ] `tests/test_approximation.py` — explicit limiting approximation contract.
- [ ] `tests/test_simulation.py` — seeded simulation behavior.
- [ ] `tests/test_installed_package.py` — wheel installation and external-path
  example execution.
- [ ] Development dependencies for pytest, Hypothesis, Ruff, mypy, build, and
  wheel-content checks.

## Independent Oracles

Phase 1 tests may use SciPy's guarded `S0` distribution calculations as a
bootstrap reference, but neither SciPy nor the package implementation alone is
an oracle for scientific claims. Phase 2 adds the independent numerical lineage
and full posterior. Phase 1 therefore verifies internal mathematical contracts,
refinement, explicit limitations, and an end-to-end working fit while labeling
results `research_uncertified`.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README communicates the known-nuisance and uncertified scope without implying a general four-parameter solution | PKG-02, FIT-02 | Claim language requires scientific reading | Read the rendered README and confirm every quickstart result shows `research_uncertified`, known `loc/scale`, and `S0` |
| First-fit summary is understandable to an applied user | FIT-03, FIT-04 | Presentation quality is not fully machine-checkable | Run the example and confirm alpha/beta and signed-gap intervals, cell counts, and identification warnings are visible |

## Validation Sign-Off

- [ ] Every implementation task names at least one automated command.
- [ ] No three consecutive tasks lack automated feedback.
- [ ] Wave 0 creates all missing test and tool infrastructure before dependent
  functionality is accepted.
- [ ] No test command uses watch mode.
- [ ] Quick feedback latency remains under 30 seconds.
- [ ] The installed-package example runs outside the repository source tree.
- [ ] No Phase 1 result can construct a certified or reduced-safe state.
- [ ] Full suite, type, lint, build, and installed smoke checks are green.

**Approval:** approved for Phase 1 planning on 2026-08-24
