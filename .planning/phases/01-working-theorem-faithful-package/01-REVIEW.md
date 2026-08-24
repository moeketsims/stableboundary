---
phase: 01-working-theorem-faithful-package
status: clean
depth: deep
files_reviewed: 10
files_reviewed_list:
  - .github/workflows/ci.yml
  - README.md
  - pyproject.toml
  - uv.lock
  - scripts/artifact_oracle.json
  - scripts/generate_artifact_oracle.py
  - scripts/smoke_wheel.py
  - tests/test_installed_package.py
  - tests/test_smoke_wheel.py
  - src/stableboundary/approximation.py
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
reviewed: 2026-08-24
reviewed_head: 5ee43cbaf6ec180c9d48ff52368a3f162d353f60
review_iterations: 3
ci_run: 32779570642
---

# Phase 01 Artifact and Installation Final Review

## Verdict

Clean at exact code head `5ee43cbaf6ec180c9d48ff52368a3f162d353f60`.
Independent security, numerical-science, and package-maintenance reviewers found
no remaining blocker or warning in the reviewed scope. The closeout documentation
commit may merge only if the same 11 protected checks pass again.

## Final evidence

- All 11 protected jobs passed on Linux, macOS, and Windows with Python
  3.12--3.14, minimum runtime dependencies, and the locked environment in CI
  run `32779570642`.
- The ordinary suite passed with `493 passed, 1 skipped, 1 deselected`.
- The focused artifact suite passed with `284 passed, 1 skipped`.
- Strict mypy passed for all 15 configured package and verifier files; Ruff lint
  and formatting passed.
- Branch coverage was 83% for package source and 80% for the two verifier
  scripts, with separate 80% enforcement gates.
- Independent 48/64-node oracle regeneration and the three Fourier anchors
  passed; the maximum order discrepancy remained
  `5.5808775573946284e-06` and the maximum Fourier/SciPy discrepancy remained
  `2.2815083156046967e-14`.
- The locked Linux job installed the actual sdist through ordinary isolated pip,
  ran `pip check`, and imported from outside the checkout. The authenticated
  verifier separately inspected both the wheel and sdist-derived wheel.

## Adversarial review convergence

The initial deep review found three critical and three warning findings. The
first fix pass closed pre-import execution, hostile child configuration,
snapshot TOCTOU, incomplete installed inventory, incorrect affine noise
requirements, and missing hostile/independent-oracle gates.

The second security pass demonstrated and then closed Windows junction escapes,
the final-proof/import mutation window, and permissive nested direct-URL
provenance. Cross-platform CI additionally exposed bytecode creation during the
post-proof import; every proof-boundary interpreter now disables bytecode writes
while the exact post-runtime reproof remains active.

The third maintenance pass added a real ordinary sdist install, placed both
critical verifier scripts under strict typing and branch coverage, pinned the
build backend, and derived the release version from the trusted repository
metadata. The complete repair record is in `01-REVIEW-FIX.md`.

## Scientific limitation retained honestly

The independent oracle is strong reproducibility and regression evidence, not a
mathematical certificate. Any finite black-box gate can be special-cased; source
byte authentication, reflection, 19 substantive reference-bound components,
broader tests, and source inspection jointly mitigate that risk. The package
continues to report `research_uncertified` and the README makes no certification
claim.
