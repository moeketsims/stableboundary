---
phase: 01-working-theorem-faithful-package
fixed_at: 2026-08-24T21:33:10Z
review_path: .planning/phases/01-working-theorem-faithful-package/01-REVIEW.md
iterations: 3
findings_in_scope: 12
fixed: 12
skipped: 0
status: all_fixed
final_code_head: 5ee43cbaf6ec180c9d48ff52368a3f162d353f60
---

# Phase 01 Code Review Fix Report

All 12 findings raised across three adversarial review iterations were repaired.
No finding was waived, downgraded, or deferred.

## Iteration 1: Original deep-review findings

### Installed code executed before raw validation

Commit `74279a0` split inert parent-side verification from runtime execution and
authenticated the complete environment delta, package, dist-info, metadata,
direct URL, and `RECORD` before import. Hostile marker tests prove substituted
package code cannot execute first.

### Hostile installer and interpreter configuration

Commit `5450878` added the `-I -S` outer boundary, an allowlisted child
environment, isolated noninteractive pip, disabled compilation, and regressions
for hostile pip/Python variables and startup hooks.

### Archive snapshot TOCTOU and incomplete installed inventory

Commit `74279a0` moved dependency/build setup before the proof boundary, added
content-addressed snapshots with entry/exit authentication, explicitly built
and inspected the sdist-derived wheel, and required the virtual-environment
delta to equal the approved distribution plus documented pip-generated files.

### Incorrect refinement-noise contract

Commit `c9efed5` enforces exact `alpha == h` and `beta == p` mean identities,
uses the inclusive `[0, 5e-14]` band only for the seven true noise-scale terms,
and retains all 19 substantive reference comparisons.

### Missing hostile and independent-oracle gates

Commits `5450878`, `74279a0`, and `5b69a85` added the demonstrated attacks and
made oracle regeneration an explicit protected-CI step.

## Iteration 2: Security re-review findings

### Windows junction/reparse escape

Commit `a282daa` rejects junctions/reparse points and requires every installed
path to resolve within the authenticated environment/site-packages root. Live
Windows attacks cover both package and dist-info junctions.

### Final-proof/import mutation window

Commit `6a7f9bb` moves `pip check` before the final raw proof, makes proof and
first import adjacent, and repeats the exact raw proof after runtime. Regressions
prove check-time mutation prevents import and persistent import-time mutation is
rejected.

### Permissive nested direct-URL provenance

Commit `dccd025` requires the exact nested `archive_info` and SHA-256 maps.

### Cross-platform bytecode mutation found by CI

Commit `2e2ae7f` adds explicit `-B` to every runtime probe. This prevents the
verified import from creating `.pyc` files without weakening the post-runtime
inventory.

## Iteration 3: Maintenance re-review findings

### Ordinary sdist installation was not directly tested

Commit `9105d70`, tightened by `17a2d88`, adds one locked-Linux fresh-environment
install through ordinary isolated pip with build isolation and dependency
resolution, followed by `pip check` and an outside-checkout import. The README
distinguishes this compatibility test from the authenticated sdist-to-wheel
proof.

### Verifier scripts were outside typing and coverage gates

Commits `17a2d88` and `5ee43cb` place both scripts under strict mypy and a
separate 80% branch-coverage gate without weakening the package's 80% gate.

### Release and build identity was duplicated

Commit `5590814` pins Hatchling 1.32.0 consistently, refreshes the lock, retains
the hard-coded trusted project-name anchor, and derives the version from
repository `pyproject.toml`. Tests cover alternate, wrong, missing, malformed,
and unsafe metadata.

## Final verification

- Ordinary suite: `493 passed, 1 skipped, 1 deselected`.
- Focused artifact suite: `284 passed, 1 skipped`.
- Strict mypy: 15 configured files passed.
- Ruff lint and format: passed.
- Package branch coverage: 83% (80% required).
- Verifier-script branch coverage: 80% (80% required).
- Build and lock validation: passed.
- Independent oracle regeneration: passed.
- Protected CI run `32779570642`: 11/11 jobs passed.
- Final security reviewer: clean at `5ee43cb` after 14 hostile/version probes.
- Final maintenance reviewer: clean at `5ee43cb`.
- Final numerical-science reviewer: clean at `5ee43cb`; 112 focused
  oracle/reflection/refinement tests and five affine/default-SciPy tests passed.
