# Contributing to stableboundary

`main` is the releasable branch. Development reaches it through a reviewed pull
request; do not push feature, fix, documentation, or release-preparation work
directly to `main`.

## Branches

- Start each branch from the current `main`.
- Keep one reviewable concern on each short-lived branch.
- Use a descriptive prefix such as `feat/`, `fix/`, `test/`, `docs/`, or `ci/`.
- Rebase or merge `main` into a branch only when needed to resolve genuine drift.
- Never rewrite shared `main` history. Delete a topic branch after it is merged.

## Commits

- Make each commit coherent and independently understandable.
- Commit tests with the behavior they verify.
- Use an imperative subject in the form `type(scope): outcome`, for example
  `fix(posterior): bind counts to their experiment`.
- Do not combine generated artifacts, unrelated formatting, and scientific logic
  in one commit.
- Do not commit secrets, local environments, build output, or exploratory data.

## Pull requests

Open a draft pull request early enough for the branch and CI result to be visible.
Before marking it ready:

1. Explain the behavior or scientific claim that changes and what does not.
2. Link every issue, manuscript claim, or review finding addressed.
3. Add focused regression tests and run the complete quality suite.
4. Confirm installation and smoke tests use the built wheel and source archive,
   not the repository checkout.
5. Obtain a skeptical review of numerical correctness, provenance, and packaging.
6. Resolve every required check and review conversation.

Pull requests are squash-merged. The pull-request title becomes the commit subject
on `main`, and GitHub deletes the merged branch automatically.

## Local quality gates

Use the commands defined by `pyproject.toml` and CI. At minimum, changes must pass
formatting, linting, strict type checking, the full test suite with the configured
coverage floor, package build validation, and installed-artifact smoke tests.

## Releases

Create releases from a clean `main` commit after CI succeeds. Use an annotated,
immutable version tag and publish exactly the artifacts built from that commit.
