# Changelog

## 0.1.0a1

First research alpha for Bayesian shape inference in univariate stable models
near the Gaussian boundary, with independently known location and scale.

- Exact finite three-cell posterior and explicit limiting approximation.
- Full stable-likelihood reference using every observation.
- Observed-data total variation and Hellinger posterior comparisons.
- Continuous posterior summaries, reduced-model predictive calculations,
  immutable numerical evidence, and JSON-compatible audit records.
- Numerical refinement checks that refuse unresolved calculations.
- Python 3.12 through 3.14, NumPy 2.5.2, and SciPy 1.18.1 support scope.

All results remain `research_uncertified`. This release does not implement
automatic reduction selection, automatic fallback, joint location/scale
inference, or mathematical error certificates. Full-likelihood evaluation is
slow and intended primarily as a reference. API compatibility is not promised
until a stable release. See `docs/validation.md` for evidence and limitations.
