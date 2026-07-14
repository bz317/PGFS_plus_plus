"""Shared MLMI-style violin panel styling for PGFS++ plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:  # pragma: no cover
    HAS_SEABORN = False

MLMI_HEADER_BLUE = "#256EBF"
MLMI_TEXT_DARK = "#2E608F"
MLMI_TEXT_MUTED = "#7A9BB8"
MLMI_GRID = "#C8DBEB"
MLMI_SPINE = "#A8C8E6"
MLMI_VIOLIN_EDGE = "#6FA3D6"
MLMI_VIOLIN_ALPHA = 0.78

VIOLIN_XLIM = 0.42
VIOLIN_SEABORN_WIDTH = 0.48
VIOLIN_MEDIAN_HALF = 0.055
VIOLIN_BW_ADJUST = 2.0


def _apply_mlmi_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.labelcolor": MLMI_TEXT_DARK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": MLMI_TEXT_MUTED,
            "ytick.color": MLMI_TEXT_MUTED,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": MLMI_SPINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _style_poster_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MLMI_SPINE)
    ax.spines["bottom"].set_color(MLMI_SPINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, color=MLMI_GRID, linestyle=(0, (4, 4)), linewidth=0.55, alpha=0.95)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MLMI_TEXT_MUTED, width=0.6, length=3)


def _plot_poster_cell_violin(
    ax: plt.Axes,
    values: list[float],
    *,
    color: str,
    y_min: float,
    y_max: float,
) -> None:
    """Violin with a white median marker."""
    if not values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(y_min, y_max)
        return

    if HAS_SEABORN:
        sns.violinplot(
            y=values,
            color=color,
            inner=None,
            cut=0,
            bw_adjust=VIOLIN_BW_ADJUST,
            linewidth=1.0,
            width=VIOLIN_SEABORN_WIDTH,
            ax=ax,
        )
        for violin in ax.collections:
            violin.set_alpha(MLMI_VIOLIN_ALPHA)
            violin.set_edgecolor(MLMI_VIOLIN_EDGE)
            violin.set_linewidth(0.65)
    else:
        parts = ax.violinplot(
            values,
            positions=[0],
            showmeans=False,
            showmedians=False,
            widths=VIOLIN_SEABORN_WIDTH,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(MLMI_VIOLIN_ALPHA)
            body.set_edgecolor(MLMI_VIOLIN_EDGE)
            body.set_linewidth(0.65)

    med = float(np.median(values))
    ax.plot(
        [-VIOLIN_MEDIAN_HALF, VIOLIN_MEDIAN_HALF],
        [med, med],
        color="white",
        linewidth=1.8,
        zorder=5,
        solid_capstyle="round",
    )
    ax.set_xlim(-VIOLIN_XLIM, VIOLIN_XLIM)
    ax.set_xticks([])
    ax.set_ylim(y_min, y_max)
