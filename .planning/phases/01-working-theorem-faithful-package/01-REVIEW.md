---
phase: 01-working-theorem-faithful-package
status: issues_found
depth: deep
files_reviewed: 6
files_reviewed_list:
  - .github/workflows/ci.yml
  - scripts/artifact_oracle.json
  - scripts/generate_artifact_oracle.py
  - scripts/smoke_wheel.py
  - tests/test_installed_package.py
  - tests/test_smoke_wheel.py
findings:
  critical: 3
  warning: 3
  info: 0
  total: 6
reviewed: 2026-08-24
reviewed_head: 5d82a8a
---

# Phase 01 Artifact and Installation Security Re-review

## Verdict

The earlier phase findings were repaired in focused pull requests. The final artifact/install branch is still blocked by three demonstrated execution-integrity defects and three scientific or coverage defects. This report supersedes the earlier review state while Git history preserves the prior findings.

## Critical findings

### CR-01: Installed package code executes before its bytes are validated

- **Files:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`
- **Evidence:** The installed probe imports `stableboundary` before it enumerates and hashes the installed package. The parent validates the returned hashes only after arbitrary package code has run. Substituted code can therefore execute first and forge the probe output.
- **Required fix:** Split inert parent-side filesystem verification from runtime execution. Before any package import, validate the complete installed package tree, distribution metadata, and `RECORD` from raw paths. Launch a separate interpreter for runtime and scientific checks only after that verification succeeds. Add a hostile pre-import execution regression.

### CR-02: Installer and verifier processes inherit hostile configuration

- **Files:** `scripts/smoke_wheel.py`, `tests/test_installed_package.py`, `tests/test_smoke_wheel.py`
- **Evidence:** `_run` retains `PIP_CONFIG_FILE`, `PIP_INDEX_URL`, `PIP_TARGET`, and related ambient variables; pip calls omit `--isolated`. `python -I` still imports `site` and can execute `.pth` or `sitecustomize.py`. The outer smoke runner is launched without `-I -S`, so startup hooks or script-directory shadowing can run before the verifier sanitizes its environment.
- **Required fix:** Launch the stdlib-only outer verifier with `-I -S`; use an allowlisted subprocess environment; use pip `--isolated --no-input` and disable bytecode compilation; complete dependency/build setup before the final package proof boundary; and add hostile environment/startup-hook tests. Keep compatibility installation distinct from the claim that package bytes were authenticated.

### CR-03: Snapshot authentication has a demonstrated TOCTOU window

- **Files:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`
- **Evidence:** The archive is authenticated, then dependency installation and a Python precheck run before pip consumes the artifact path. A hostile mutation during the first subprocess caused pip to receive replacement bytes. For an sdist, replacement bytes may execute during the build before the post-install mismatch is detected.
- **Required fix:** Finish dependency and build-backend setup first. Create and inspect the private content-addressed snapshot at the final boundary, authenticate immediately before consumption, install with `--no-deps`, and reauthenticate afterward. Explicitly build an sdist wheel, inspect that exact wheel with the wheel contract, and install only the inspected result. Add the demonstrated between-check-and-use mutation regression.

## Warnings

### WR-01: Installed distribution inventory is incomplete

- **Files:** `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`
- **Evidence:** The probe inventories only `.py` and `py.typed` below the imported package directory. It does not prevent top-level `.pth` or `sitecustomize.py`, extra modules/native libraries/scripts, or altered dist-info metadata and `RECORD`.
- **Required fix:** Validate the complete installed `distribution.files`/`RECORD`, package tree, dist-info metadata, and allowed pip-generated files before import. Reject extra executable or import-affecting files and add hostile regressions.

### WR-02: Noise-scale oracle requirements encode floating-point noise as mandatory evidence

- **Files:** `scripts/artifact_oracle.json`, `scripts/smoke_wheel.py`, `tests/test_smoke_wheel.py`
- **Evidence:** The validator requires nine near-zero refinement terms to be strictly positive. A legitimate platform or improved implementation may round a true zero-scale discrepancy to exactly zero. After the affine diagnostic fix merged in PR #7, normalized alpha mean must equal h exactly and beta mean must equal p exactly; treating all four as unrelated empirical noise is mathematically wrong.
- **Required fix:** Validate exact `alpha == h` and `beta == p` mean-refinement identities. Restrict the machine-noise band to the remaining seven terms and accept `0 <= value <= 5e-14`. Retain anti-faking protection through all 19 substantive reference-bound components, reflection, independent oracle generation, and hostile mutations.

### WR-03: Tests omit the demonstrated security boundaries

- **Files:** `.github/workflows/ci.yml`, `tests/test_installed_package.py`, `tests/test_smoke_wheel.py`
- **Evidence:** Existing tests cover simple snapshot mutation and command order but not hostile pip configuration, outer startup hooks, mutation at pip consumption, pre-verification import execution, extra installed distribution files, or the sdist-built wheel. CI does not run the documented independent oracle regeneration check.
- **Required fix:** Add each hostile regression, include the independent oracle `--check` in one ordinary protected job, and retain focused coverage of the verifier's critical branches.

## Required re-review

Re-run the exact hostile probes, complete ordinary suite, independent oracle regeneration, lint, formatting, strict typing, all 11 protected matrix jobs, and clean wheel/sdist installation. The branch may merge only after independent packaging-security, numerical-science, and maintainability reviewers all return clean.
