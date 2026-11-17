# Changelog

## 0.1.0a1 (2026-11-17)

First research alpha for Bayesian shape inference in univariate stable models
near the Gaussian boundary, with independently known location and scale.

- Exact finite three-cell posterior and explicit limiting approximation.
- Full stable-likelihood reference using every observation.
- Observed-data total variation and Hellinger posterior comparisons.
- Continuous posterior summaries, reduced-model predictive calculations,
  immutable numerical evidence, and JSON-compatible audit records.
- Numerical refinement checks that refuse unresolved calculations.
- Fits and direct cell probabilities with the SciPy backend refuse
  `alpha < 1.10`, where SciPy's stable routines are unreliable near `alpha = 1`.
- The full fit refuses standardized observations beyond `1e4` scale units, where
  the SciPy density loses accuracy. Both limits are empirical cutoffs.
- Independent characteristic-function spot checks of the SciPy density and
  tails run in the test suite.
- Python 3.12 through 3.14, NumPy 2.5.2, and SciPy 1.18.1 support scope.

All results remain `research_uncertified`. This release does not implement
automatic reduction selection, automatic fallback, joint location/scale
inference, or mathematical error certificates. Full-likelihood evaluation is
slow and intended primarily as a reference. API compatibility is not promised
until a stable release. See `docs/validation.md` for evidence and limitations.
