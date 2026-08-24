# stableboundary

`stableboundary` provides Bayesian inference for univariate alpha-stable laws
near the Gaussian boundary. Its first workflow fits the exact finite three-cell
posterior when location and scale are known independently of the observations.
All public parameters use Nolan's continuous `S0` parameterization.

## Scientific scope

The current result status is always `research_uncertified`. Phase 1 does not
establish a mathematical certificate, make an automatic safe/fallback decision,
or estimate all four conventional stable parameters. It estimates the local
shape coordinates while treating the supplied `loc` and `scale` as externally
known. The finite-cell likelihood is exact for the declared three-cell model;
that description does not mean exact arithmetic or a finite-sample proof of the
reduction.

At the Gaussian boundary, `scale=1` in `S0` corresponds to a normal distribution
with standard deviation `sqrt(2)`, not standard deviation one. For every
supported posterior point `alpha < 2`, variance is infinite. Skewness can also
be weakly or not identified when signed-tail events are scarce; inspect the
structured identification diagnostics and warnings rather than reading a beta
interval in isolation.

## Installation

The project requires Python 3.12 or newer. From a built source tree, install the
package with:

```console
python -m pip install .
```

Released artifacts can instead be installed by passing the wheel or source
archive path to `python -m pip install`.

## Fixed-seed known-nuisance fit

The executable example first fixes a design for `n=5000`, derives the simulated
truth from that design, and only then simulates observations. It uses only the
top-level `stableboundary` API:

```console
python examples/known_nuisance_fit.py
```

The output records `S0`, the fixed `loc` and `scale` with their provenance,
signed cell counts, and posterior summaries for `h`, `p`, `alpha`, `beta`,
`tau_plus`, and `tau_minus`. It also includes normalization and quadrature
refinement evidence, identification diagnostics, and warnings. The signed gaps
`tau_plus` and `tau_minus` are local shape coordinates, not threshold-free tail
probabilities.

The example uses a deterministic seed to make the workflow auditable. Exact
floating-point values can still change across supported NumPy and SciPy
versions, so interpret the finite summaries and status rather than treating the
printed digits as a cross-version proof certificate.

## Maintainer checks

Run the checks in this order from the repository root:

```console
uv sync --extra dev --locked
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src
uv run --frozen coverage run --branch -m pytest -q -m "not installed"
uv run --frozen coverage report --fail-under=80
uv run --frozen python -m build
uv run --frozen twine check dist/*
uv run --frozen check-wheel-contents dist/*.whl
uv run --frozen pytest -q tests/test_installed_package.py -m installed
```

The final test installs both freshly built archives into separate clean virtual
environments and runs the same example from outside the checkout.

## Current limitations

- Input observations must be one-dimensional, finite, and independently drawn
  under the declared stable sampling model.
- Location and scale must be independently known; same-sample plug-in
  standardization is unsupported.
- The result is a reduced, exact finite-cell posterior only. Independent full
  stable-likelihood comparison and automatic fallback arrive in later phases.
- `research_uncertified` is deliberately retained even when ordinary
  floating-point refinement passes.
