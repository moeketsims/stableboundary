---
phase: 01-working-theorem-faithful-package
status: issues_found
depth: standard
iteration: 2
reviewed_head: 1bdb863
files_reviewed: 23
files_reviewed_list:
  - src/stableboundary/api.py
  - src/stableboundary/approximation.py
  - src/stableboundary/backends/_protocol.py
  - src/stableboundary/backends/_scipy_s0.py
  - src/stableboundary/cells.py
  - src/stableboundary/design.py
  - src/stableboundary/posterior.py
  - src/stableboundary/result.py
  - scripts/smoke_wheel.py
  - examples/known_nuisance_fit.py
  - tests/test_approximation.py
  - tests/test_fit_known.py
  - tests/test_prediction.py
  - tests/test_probabilities.py
  - tests/test_smoke_wheel.py
  - tests/test_installed_package.py
  - README.md
  - pyproject.toml
  - .github/workflows/ci.yml
  - AGENTS.md
  - .planning/research/STACK.md
  - .planning/phases/01-working-theorem-faithful-package/01-REVIEW-FIX.md
  - uv.lock
findings:
  critical: 7
  warning: 9
  info: 1
  total: 17
reviewed: 2026-08-24
---

# Phase 01 Code Re-review — Iteration 2

## Verdict

The first repair cycle fixed its ten recorded findings, but three independent adversarial reviewers found seven new correctness or proof-of-work blockers and nine warnings at head 1bdb863. PR #1 must remain a draft. The original review is preserved in 01-REVIEW.iter2.md, and its fix accounting remains in 01-REVIEW-FIX.md.

## Critical findings

### CR-01: Exact posterior accepts counts from a different experiment

- **Files:** src/stableboundary/cells.py, src/stableboundary/posterior.py
- **Evidence:** Valid CellCounts produced under one LocalDesign were accepted with another design of the same n but a different c and threshold. The likelihood was silently evaluated at the second threshold.
- **Required fix:** Bind counts to immutable full design and nuisance provenance and reject any mismatch before backend evaluation.

### CR-02: Public fit composition can produce a false audit record

- **Files:** src/stableboundary/posterior.py, src/stableboundary/result.py, src/stableboundary/api.py
- **Evidence:** A posterior computed under one compact prior can be paired with KnownNuisanceFit carrying a different prior; the audit then reports the false prior. Counts can likewise be paired with different nuisance provenance.
- **Required fix:** Retain full counts/design/prior provenance in PosteriorGrid and make fit construction package-controlled with exact component equality checks.

### CR-03: Limiting-approximation quantiles repeat the Gauss-mass CDF error

- **File:** src/stableboundary/approximation.py
- **Evidence:** For a limiting Uniform[0.05, 0.95] p posterior, expected (q05, median, q95) is (0.095, 0.5, 0.905), while the public result returns (0.0943427141, 0.4926754648, 0.9056572859).
- **Required fix:** Use continuous truncated Gamma/Beta inversion, exact monotone transforms, and continuous product-distribution integration for tau_plus and tau_minus, with analytic regressions.

### CR-04: Windows path canonicalization bypasses artifact leakage checks

- **Files:** scripts/smoke_wheel.py, tests/test_smoke_wheel.py
- **Evidence:** tests./payload.py and paper.tex. pass lexical inspection but extract on Windows as tests/payload.py and paper.tex. Reserved devices and alternate-data-stream names also reach pip.
- **Required fix:** Reject trailing dot/space, colon/control characters, reserved basenames, noncanonical components/separators, and portable canonical collisions for members and link targets; add extraction regressions.

### CR-05: The artifact smoke can install and test the wrong distribution

- **File:** scripts/smoke_wheel.py
- **Evidence:** Fabricated evil-9 wheel and sdist archives with Name: evil passed discovery and inspection. A transitive dependency named stableboundary can also satisfy the origin check.
- **Required fix:** Validate canonical filenames plus embedded wheel METADATA and sdist PKG-INFO name/version, install dependencies separately then the exact artifact with --no-deps, and validate installed version/direct_url against that artifact.

### CR-06: Installed proof ignores required fixed-example evidence

- **Files:** scripts/smoke_wheel.py, examples/known_nuisance_fit.py, tests/test_smoke_wheel.py
- **Evidence:** Payloads without seed, truth, design, identification, or warnings pass, as does n=10 instead of the fixed n=5000 proof case.
- **Required fix:** Require and validate the complete fixed design/truth/identification/warning schema, exact seed and sample size, strict prior-induced parameter domains, and common 0.90 credible mass; compare wheel and sdist evidence.

### CR-07: converged=true can contradict reported refinement evidence

- **File:** scripts/smoke_wheel.py
- **Evidence:** Payloads pass with log-normalizer change 999, tolerance 2, common grid 999, and missing summary or predictive changes.
- **Required fix:** Validate the complete refinement schema and recompute convergence from all joint, normalizer, six-summary, and predictive components against the exact configured tolerance and grid.

## Warnings

### WR-01: Retained custom backend behavior may mutate after fitting

- **Files:** src/stableboundary/posterior.py, src/stableboundary/result.py
- **Evidence:** A conforming backend changed prediction after fitting while retaining identical metadata.
- **Required fix:** Reconstruct only the canonical immutable package backend from exact metadata and explicitly refuse prediction for injected research backends.

### WR-02: QuadratureConfig accepts booleans/strings and leaks native exceptions

- **File:** src/stableboundary/posterior.py
- **Required fix:** Require non-boolean Real values and raise ValidationError for all malformed controls.

### WR-03: The non-S0 regression catches Exception too broadly

- **File:** tests/test_fit_known.py
- **Required fix:** Require ValidationError specifically.

### WR-04: SciPy compatibility relies on a direct private-module import

- **Files:** src/stableboundary/backends/_scipy_s0.py, pyproject.toml
- **Required fix:** Construct the isolated generator through a guarded compatibility layer based on the public levy_stable object and exercise supported dependency lines in CI.

### WR-05: Required scientific environment lock and minimum-dependency checks are absent

- **Files:** uv.lock, .github/workflows/ci.yml, pyproject.toml
- **Required fix:** Commit uv.lock and add locked plus NumPy 2.2.0/SciPy 1.18.0 jobs while retaining the latest matrix.

### WR-06: Audit records omit Python and NumPy versions

- **File:** src/stableboundary/result.py
- **Required fix:** Record a structured environment block for Python, NumPy, SciPy, and stableboundary versions.

### WR-07: Coverage policy is configured but unenforced

- **Files:** pyproject.toml, .github/workflows/ci.yml
- **Required fix:** Enforce branch coverage at a declared threshold; current measured overall coverage is approximately 82%.

### WR-08: The documented executable example is absent from distributions

- **Files:** README.md, pyproject.toml
- **Required fix:** Include the example in the sdist and provide a complete installed-user workflow in the README.

### WR-09: GitHub Actions use mutable, deprecated major tags

- **File:** .github/workflows/ci.yml
- **Required fix:** Use current official action releases pinned to immutable full commit SHAs.

## Repository setting

### IN-01: Git best practice is not server-enforced

- **Evidence:** main has no protection/ruleset, and two repair commits were pushed directly before the branch workflow was corrected.
- **Required action:** After the repair PR is clean, require pull requests and successful CI for main, block force-push/deletion including administrators, enable merged-branch cleanup, and use one consistent merge strategy.

## Confirmed repairs

- Exact-posterior endpoint-aware and asymmetric tau quantiles passed independent adaptive checks.
- Private S0 settings remained isolated under hostile global/class state and concurrency.
- The POSIX virtual-environment launcher defect was fixed in 1bdb863 and all nine OS/Python CI jobs passed.
- The first-cycle archive traversal, strict basic payload, and timeout fixes remain effective but are not sufficient for a clean verdict.

## Required re-review

Re-run all three adversarial scopes on the combined second-cycle branch. A clean verdict requires no Critical or Warning findings, a green latest/minimum/locked CI suite, clean wheel and sdist proof runs, and server-enforced main-branch policy.
