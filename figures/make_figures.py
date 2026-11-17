"""Render the README figures from the saved package outputs.

Run with any Python that has numpy and matplotlib (matplotlib is not a
dependency of the package):
  python figures/make_figures.py
Inputs: demo_reduced.npz, demo_meta.json, and (for figure 3) demo_full.npz,
demo_full_meta.json, all written by make_data.py. No number is typed in by
hand here: every value on a figure is read from those files.

Style: scientific plots on a white background. Square, 2160 x 2160 px, with
type sized to stay readable on a phone. The fonts are Windows fonts; other
systems fall back to the matplotlib default and the layout may shift slightly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent

WHITE = "#ffffff"
INK = "#111827"
INK_2 = "#374151"
MUTED = "#6b7280"
AXIS = "#9ca3af"
GRID = "#e5e7eb"
BLUE = "#2a78d6"  # validated pair on a light surface
ORANGE = "#eb6834"
GREY_LINE = "#6b7280"
SANS = "Segoe UI"
SEMI = "Segoe UI Semibold"

plt.rcParams.update(
    {
        "font.family": SANS,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": AXIS,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.facecolor": WHITE,
    }
)

SIZE, DPI = 10.8, 200  # 2160 px square
LEFT, RIGHT = 0.075, 0.945
TICK, LABEL, NOTE = 17.5, 19, 18


def canvas(title: str, subtitle: str, meta):
    fig = plt.figure(figsize=(SIZE, SIZE), dpi=DPI)
    fig.text(LEFT, 0.955, title, fontsize=31, color=INK, family=SEMI, va="top")
    fig.text(LEFT, 0.895, subtitle, fontsize=19, color=INK_2, va="top", linespacing=1.4)
    fig.text(
        LEFT,
        0.032,
        f"Simulated data, n = {meta['n']}, seed {meta['seed']}",
        fontsize=13.5,
        color=MUTED,
        va="center",
    )
    fig.text(
        RIGHT,
        0.032,
        f"stableboundary {meta['versions']['stableboundary']}  |  "
        "github.com/moeketsims/stableboundary",
        fontsize=13.5,
        color=MUTED,
        va="center",
        ha="right",
    )
    return fig


def style_axes(ax, grid: str | None = "y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.3)
    ax.tick_params(length=6, width=1.2, color=AXIS, labelsize=TICK, pad=7)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1.1, zorder=0)
        ax.set_axisbelow(True)


def minus(text: str) -> str:
    return text.replace("-", "−")


def load():
    data = np.load(HERE / "demo_reduced.npz")
    meta = json.loads((HERE / "demo_meta.json").read_text())
    return data, meta


def figure_cells():
    data, meta = load()
    t = meta["threshold"]
    truth = meta["truth"]
    n_minus, n_zero, n_plus = meta["counts"]
    x = data["x"]
    stable = np.exp(data["logpdf_truth"])
    gauss = np.exp(data["logpdf_gauss"])
    obs = data["observations"]
    floor, ceiling = 2.5e-5, 40.0

    fig = canvas(
        f"{meta['n']} observations, summarised by three counts",
        "A threshold set by the sample size splits the data into three cells.\n"
        "The fast Bayesian fit uses only the counts in those cells.",
        meta,
    )
    ax = fig.add_axes([0.125, 0.165, 0.82, 0.595])
    style_axes(ax, grid=None)
    ax.set_yscale("log")
    ax.set_xlim(-9, 9)
    ax.set_ylim(floor, ceiling)

    for lo, hi in ((-9, -t), (t, 9)):
        ax.axvspan(lo, hi, color=ORANGE, alpha=0.11, lw=0, zorder=1)
    for edge in (-t, t):
        ax.axvline(edge, color=INK_2, lw=1.6, zorder=3)

    ax.plot(x, gauss, color=GREY_LINE, lw=3.0, zorder=4, solid_capstyle="round")
    ax.plot(x, stable, color=BLUE, lw=4.2, zorder=6, solid_capstyle="round")

    strip = ax.get_xaxis_transform()
    inside = np.abs(obs) <= t
    ax.scatter(
        obs[inside],
        np.full(inside.sum(), 0.05),
        transform=strip,
        s=120,
        color=BLUE,
        alpha=0.45,
        edgecolor=WHITE,
        linewidth=1.3,
        zorder=8,
    )
    ax.scatter(
        obs[~inside],
        np.full((~inside).sum(), 0.05),
        transform=strip,
        s=300,
        color=ORANGE,
        edgecolor=WHITE,
        linewidth=2.2,
        zorder=9,
    )

    for centre, count, label in (
        (-(t + 9) / 2, n_minus, "below"),
        (0.0, n_zero, "between"),
        ((t + 9) / 2, n_plus, "above"),
    ):
        ax.text(
            centre,
            0.975,
            f"{count}",
            transform=strip,
            ha="center",
            va="top",
            fontsize=62,
            family=SEMI,
            color=INK,
            zorder=10,
        )
        ax.text(
            centre,
            0.845,
            label,
            transform=strip,
            ha="center",
            va="top",
            fontsize=19,
            color=INK_2,
            zorder=10,
        )
    for edge, text in ((-t, minus(f"-{t:.1f}")), (t, f"+{t:.1f}")):
        ax.text(
            edge,
            1.012,
            text,
            transform=strip,
            ha="center",
            va="bottom",
            fontsize=TICK,
            color=INK_2,
        )

    ax.annotate(
        f"stable law, α = {truth['alpha']:.2f}",
        xy=(6.8, np.interp(6.8, x, stable)),
        xytext=(4.0, 0.06),
        fontsize=NOTE,
        color=INK,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.6, shrinkA=5, shrinkB=7),
        zorder=11,
    )
    ax.annotate(
        "Gaussian\nα = 2",
        xy=(-5.55, np.interp(-5.55, x, gauss)),
        xytext=(-8.75, 1.25e-4),
        fontsize=NOTE,
        color=INK,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=GREY_LINE, lw=1.6, shrinkA=5, shrinkB=7),
        zorder=11,
    )

    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0.0001", "0.001", "0.01", "0.1"])
    ax.minorticks_off()
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xticklabels([minus(str(v)) for v in (-8, -4, 0, 4, 8)])
    ax.set_xlabel("standardised observation", fontsize=LABEL, labelpad=12)
    ax.set_ylabel("density (log scale)", fontsize=LABEL, labelpad=10)
    fig.savefig(HERE / "plot1_three_counts.png", dpi=DPI)
    plt.close(fig)


def joint(meta, h_axis, p_axis, density):
    """Return alpha axis, beta axis and the density over (alpha, beta)."""
    r = meta["r"]
    alpha = 2.0 - r * h_axis[::-1]
    beta = 2.0 * p_axis - 1.0
    return alpha, beta, density[::-1, :] / (2.0 * r)


def trapezoid_weights(axis):
    w = np.zeros_like(axis)
    step = np.diff(axis)
    w[:-1] += step / 2
    w[1:] += step / 2
    return w


def hdr_levels(density, measure, masses):
    order = np.argsort(density.ravel())[::-1]
    cumulative = np.cumsum((density * measure).ravel()[order])
    cumulative /= cumulative[-1]
    return [density.ravel()[order][np.searchsorted(cumulative, m)] for m in masses]


def figure_posterior():
    data, meta = load()
    alpha, beta, density = joint(meta, data["h_axis"], data["p_axis"], data["density"])
    measure = np.multiply.outer(trapezoid_weights(alpha), trapezoid_weights(beta))
    truth = meta["truth"]
    summary = meta["reduced"]["summaries"]
    seconds = meta["reduced"]["attempts"][-1]["seconds"]
    counts = tuple(meta["counts"])

    fig = canvas(
        "Posterior for tail weight and skewness",
        f"Computed from the counts {counts} in {seconds:.1f} seconds.\n"
        "Darker blue means more probable; lines enclose 50% and 90% of the posterior.",
        meta,
    )
    ax = fig.add_axes([0.125, 0.165, 0.585, 0.595])
    style_axes(ax, grid=None)
    cmap = LinearSegmentedColormap.from_list(
        "white_blue",
        [
            (0.0, WHITE),
            (0.25, "#cde2fb"),
            (0.55, "#6da7ec"),
            (0.8, BLUE),
            (1.0, "#104281"),
        ],
    )
    ax.imshow(
        density.T,
        extent=[alpha[0], alpha[-1], beta[0], beta[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bicubic",
        zorder=1,
        vmin=0.0,
        vmax=float(density.max()),
    )
    level90, level50 = hdr_levels(density, measure, [0.90, 0.50])
    contours = ax.contour(
        alpha,
        beta,
        density.T,
        levels=[level90, level50],
        colors=[INK],
        linewidths=[1.5, 2.0],
        zorder=3,
    )
    ax.clabel(
        contours,
        fmt={level90: "90%", level50: "50%"},
        fontsize=15.5,
        inline=True,
        inline_spacing=10,
        colors=INK,
        manual=[(1.30, 0.25), (1.50, 0.62)],
    )

    ax.scatter(
        [truth["alpha"]],
        [truth["beta"]],
        s=330,
        marker="D",
        color=ORANGE,
        edgecolor=WHITE,
        linewidth=2.6,
        zorder=6,
    )
    mean_a, mean_b = summary["alpha"]["mean"], summary["beta"]["mean"]
    ax.scatter(
        [mean_a], [mean_b], s=260, color=INK, edgecolor=WHITE, linewidth=2.6, zorder=6
    )
    ax.set_xlim(alpha[0], alpha[-1])
    ax.set_ylim(beta[0], beta[-1])
    ax.set_xticks([1.2, 1.4, 1.6, 1.8])
    ax.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    ax.set_yticklabels([minus(f"{v:.1f}") for v in (-0.8, -0.4, 0.0, 0.4, 0.8)])
    ax.set_xlabel("tail index α  (2 = Gaussian)", fontsize=LABEL, labelpad=12)
    ax.set_ylabel("skewness β", fontsize=LABEL, labelpad=8)

    x0 = 0.755
    for row, (name, key) in enumerate(
        (("tail index α", "alpha"), ("skewness β", "beta"))
    ):
        y0 = 0.745 - row * 0.245
        s = summary[key]
        fig.text(x0, y0, name, fontsize=18, color=INK_2, va="top")
        fig.text(
            x0,
            y0 - 0.035,
            minus(f"{s['mean']:.2f}"),
            fontsize=54,
            family=SEMI,
            color=INK,
            va="top",
        )
        fig.text(
            x0,
            y0 - 0.128,
            minus(f"90% interval\n{s['lower']:.2f} to {s['upper']:.2f}"),
            fontsize=16.5,
            color=INK_2,
            va="top",
            linespacing=1.35,
        )
    key = fig.add_axes([x0, 0.165, 0.19, 0.09])
    key.set_axis_off()
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.scatter([0.06], [0.75], s=260, color=INK, edgecolor=WHITE, linewidth=2.6)
    key.text(0.17, 0.75, "posterior mean", fontsize=16.5, color=INK, va="center")
    key.scatter(
        [0.06], [0.22], s=300, marker="D", color=ORANGE, edgecolor=WHITE, linewidth=2.6
    )
    key.text(0.17, 0.22, "simulated truth", fontsize=16.5, color=INK, va="center")
    fig.savefig(HERE / "plot2_posterior.png", dpi=DPI)
    plt.close(fig)


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = seconds / 60
    return f"{minutes:.0f} min" if minutes >= 9.5 else f"{minutes:.1f} min"


def figure_comparison():
    data, meta = load()
    full = np.load(HERE / "demo_full.npz")
    full_meta = json.loads((HERE / "demo_full_meta.json").read_text())
    alpha, beta, reduced_density = joint(
        meta, data["h_axis"], data["p_axis"], data["density"]
    )
    _, _, full_density = joint(meta, full["h_axis"], full["p_axis"], full["density"])
    w_alpha, w_beta = trapezoid_weights(alpha), trapezoid_weights(beta)
    truth = meta["truth"]
    tv = full_meta["comparison"]["total_variation"]
    full_seconds = full_meta["full"]["attempts"][-1]["seconds"]
    reduced_seconds = meta["reduced"]["attempts"][-1]["seconds"]

    fig = canvas(
        "What does the three-count shortcut lose?",
        "The same sample fitted twice. The package measures the gap between\n"
        "the two posteriors instead of assuming it is small.",
        meta,
    )

    y = 0.752
    for row, (color, label, seconds) in enumerate(
        (
            (BLUE, "three counts", reduced_seconds),
            (ORANGE, "every observation", full_seconds),
        )
    ):
        yy = y - row * 0.04
        fig.add_artist(
            plt.Line2D(
                [0.10, 0.15], [yy, yy], color=color, lw=4.5, solid_capstyle="round"
            )
        )
        fig.text(
            0.165,
            yy,
            f"{label}  ({duration(seconds)})",
            fontsize=18.5,
            color=INK,
            va="center",
        )
    fig.text(
        RIGHT,
        y - 0.008,
        f"{tv:.2f}",
        fontsize=52,
        family=SEMI,
        color=INK,
        va="center",
        ha="right",
    )
    fig.text(
        RIGHT - 0.155,
        y - 0.008,
        "total variation distance\n0 = identical, 1 = no overlap",
        fontsize=15.5,
        color=INK_2,
        va="center",
        ha="right",
        linespacing=1.35,
    )

    panels = (
        (
            "tail index α  (2 = Gaussian)",
            alpha,
            reduced_density @ w_beta,
            full_density @ w_beta,
            truth["alpha"],
            [0.10, 0.165, 0.395, 0.47],
            [1.2, 1.4, 1.6, 1.8],
        ),
        (
            "skewness β",
            beta,
            w_alpha @ reduced_density,
            w_alpha @ full_density,
            truth["beta"],
            [0.55, 0.165, 0.395, 0.47],
            [-0.8, -0.4, 0.0, 0.4, 0.8],
        ),
    )
    for index, (
        name,
        axis,
        reduced_marginal,
        full_marginal,
        true_value,
        box,
        ticks,
    ) in enumerate(panels):
        ax = fig.add_axes(box)
        style_axes(ax, grid=None)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        top = 1.18 * max(reduced_marginal.max(), full_marginal.max())
        ax.set_xlim(axis[0], axis[-1])
        ax.set_ylim(0, top)
        ax.fill_between(
            axis, reduced_marginal, 0, color=BLUE, alpha=0.12, lw=0, zorder=2
        )
        ax.fill_between(
            axis, full_marginal, 0, color=ORANGE, alpha=0.12, lw=0, zorder=2
        )
        ax.plot(
            axis, reduced_marginal, color=BLUE, lw=4.0, zorder=6, solid_capstyle="round"
        )
        ax.plot(
            axis, full_marginal, color=ORANGE, lw=4.0, zorder=7, solid_capstyle="round"
        )
        ax.axvline(true_value, color=INK_2, lw=1.6, zorder=3)
        ax.text(
            true_value,
            top * 0.99,
            " truth",
            fontsize=16,
            color=INK_2,
            va="top",
            ha="left",
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([minus(f"{v:.1f}") for v in ticks])
        ax.set_xlabel(name, fontsize=LABEL, labelpad=12)
        if index == 0:
            ax.set_ylabel("posterior density", fontsize=LABEL, labelpad=10)
    fig.savefig(HERE / "plot3_comparison.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    figure_cells()
    figure_posterior()
    if (HERE / "demo_full.npz").exists():
        figure_comparison()
    print("done")
