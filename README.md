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

CI tests source archives in two deliberately distinct ways. The locked Linux
compatibility check creates a clean virtual environment, lets ordinary pip
resolve runtime dependencies and use its normal isolated PEP 517 build, runs
`pip check`, and imports `stableboundary` from outside the checkout. Separately,
the authenticated artifact smoke validates archive identity and integrity
before accepting its sdist-to-wheel execution path. The compatibility check is
evidence for the ordinary user install command; it is not a substitute for the
authenticated artifact proof.

## Audited known-nuisance fit

The executable artifact example separates inferential evidence from simulator
evidence. Its posterior is fitted to a deterministic 5000-observation witness
with signed-cell counts `(1, 4996, 3)`. Because the three-cell likelihood is
count-sufficient, the witness makes the inferential input transparent and keeps
it independent of changes to SciPy's random sampler. The example then runs a
separate fixed-seed `S0` simulation and reports the canonical little-endian raw
sample hash, five integer-quantized full-sample hashes, counts, extrema, summary
diagnostics, algorithms, and runtime versions. The source distribution includes
the example as `examples/known_nuisance_fit.py`. From a source checkout or an
unpacked source distribution, run:

```console
python examples/known_nuisance_fit.py
```

Wheel users do not need repository-only modules. After installing a wheel, save
the following as `quickstart.py` and run `python quickstart.py` from any working
directory:

```python
import json

import stableboundary as sb

n = 5_000
seed = 20_260_824
design = sb.LocalDesign.from_sample_size(n)
truth = sb.StableParams(
    alpha=2.0 - design.r * 1.5,
    beta=0.35,
    loc=0.0,
    scale=1.0,
)
observations = sb.simulate(truth, size=n, random_state=seed)
fit = sb.fit_known_nuisance(
    observations,
    loc=0.0,
    scale=1.0,
    design=design,
    prior=sb.LocalPrior.default(design),
    provenance="fixed by the simulation design",
    quadrature=sb.QuadratureConfig(
        base_nodes=20,
        refined_nodes=32,
        refinement_tolerance=0.002,
        common_grid_points=65,
    ),
)
print(json.dumps(fit.summary(), indent=2, sort_keys=True))
```

The output records `S0`, the fixed `loc` and `scale` with their provenance,
signed cell counts, and posterior summaries for `h`, `p`, `alpha`, `beta`,
`tau_plus`, and `tau_minus`. It also includes normalization and quadrature
refinement evidence, identification diagnostics, and warnings. The signed gaps
`tau_plus` and `tau_minus` are local shape coordinates, not threshold-free tail
probabilities.

The artifact smoke test accepts simulated evidence only for explicitly measured
operating-system, machine-architecture, NumPy, and SciPy combinations; an
unknown combination fails closed. The raw floating-point hash is diagnostic,
not normative: hosted CPUs and libm implementations can change insignificant
low bits. Acceptance instead uses the finest integer-quantization grid shown to
converge across independent hosts for that approved environment, together with
counts, extrema, and tolerance-bound summary diagnostics. Ordinary CI performs
two fresh isolated-interpreter simulations to detect process-state pollution.
A rejection prints the complete observed fingerprint so a maintainer can
review rather than blindly loosen the check. Posterior summaries are bound both
to retained regression values and to a higher-order reference generated
without importing `stableboundary`. These checks make the artifact reproducible
and sensitive to stale or hard-coded output, but they do not turn ordinary
floating-point calculations into a proof certificate.

## Maintainer checks

Run the checks in this order from the repository root:

```console
uv sync --extra dev --locked
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src
uv run --frozen python scripts/generate_artifact_oracle.py --check scripts/artifact_oracle.json
uv run --frozen coverage run --branch -m pytest -q -m "not installed"
uv run --frozen coverage report --fail-under=80
uv run --frozen python -m build --no-isolation
uv run --frozen twine check dist/*
uv run --frozen check-wheel-contents dist/*.whl
uv run --frozen pytest -q tests/test_installed_package.py -m installed
```

The final test installs both freshly built archives into separate clean virtual
environments and runs the same example from outside the checkout.

The oracle regeneration command independently evaluates the exact `S0`
three-cell likelihood with 48- and 64-node tensor Gauss--Legendre rules and
cross-checks selected cell probabilities by direct Gil--Pelaez Fourier
inversion. The generator imports NumPy and SciPy but is forbidden by test from
importing `stableboundary`. Review and explicitly approve any new
platform/architecture/NumPy/SciPy simulation key and its normative quantized
hash only after independent hosts agree and its counts, extrema, and summary
diagnostics have been checked. Raw hashes remain audit observations. The smoke
verifier intentionally rejects unmeasured combinations.

## Current limitations

- Input observations must be one-dimensional, finite, and independently drawn
  under the declared stable sampling model.
- Location and scale must be independently known; same-sample plug-in
  standardization is unsupported.
- The result is a reduced, exact finite-cell posterior only. Independent full
  stable-likelihood comparison and automatic fallback arrive in later phases.
- `research_uncertified` is deliberately retained even when ordinary
  floating-point refinement passes.
