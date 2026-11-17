# Figures

The three figures in the main README, with the data and scripts that made them.

| File | What it shows |
|---|---|
| `plot1_three_counts.png` | One simulated sample of 64 observations, the design threshold, and the three cell counts. |
| `plot2_posterior.png` | The joint posterior for tail index and skewness from those counts. |
| `plot3_comparison.png` | The same sample fitted from the counts and from every observation, with the total variation distance between the two posteriors. |

The sample: `n = 64`, seed `20261117`, true `alpha = 1.699` and `beta = 0.30`,
default prior, `stableboundary` 0.1.0a1. It is one illustrative example, not
evidence about the method in general.

## Reproduce

From the repository root, with the package installed:

```console
python figures/make_data.py reduced
python figures/make_data.py full
python figures/make_figures.py
```

`make_data.py full` runs the full-likelihood fit and takes several minutes. It
tries the default quadrature grid first and moves to a finer grid only if the
refinement check refuses. The saved `demo_*.npz` and `demo_*.json` files let
`make_figures.py` redraw the figures without refitting. `make_figures.py` needs
`matplotlib`, which is not a package dependency.

This folder is not part of the wheel or the source archive.
