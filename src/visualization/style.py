"""
Global matplotlib style configuration for all paper figures.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Color palette for motif orders
ORDER_COLORS = {
    3: "#2E86AB",
    4: "#E84855",
    5: "#F4A261",
    6: "#2A9D8F",
    7: "#E9C46A",
    8: "#264653",
}

# Colors for team sides
SIDE_COLORS = {"home": "#1F77B4", "away": "#FF7F0E"}

# Colors for motif types
MOTIF_TYPE_COLORS = {
    "cooperative": "#2E86AB",
    "adversarial": "#E84855",
    "mixed": "#F4A261",
}

# Score state colors
SCORE_STATE_COLORS = {
    "leading": "#2A9D8F",
    "drawing": "#E9C46A",
    "trailing": "#E84855",
}

FIGURE_DPI = 300
FIGURE_WIDTH_SINGLE = 3.5   # inches (single column)
FIGURE_WIDTH_DOUBLE = 7.0   # inches (double column)
FIGURE_HEIGHT_DEFAULT = 4.5


def apply_paper_style():
    """Apply consistent matplotlib style for paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": FIGURE_DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "errorbar.capsize": 3,
        "savefig.bbox": "tight",
        "savefig.dpi": FIGURE_DPI,
    })


def get_figure(
    width: float = FIGURE_WIDTH_DOUBLE,
    height: float = FIGURE_HEIGHT_DEFAULT,
    n_cols: int = 1,
    n_rows: int = 1,
):
    """Create a figure with paper-consistent sizing."""
    apply_paper_style()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    return fig, axes


def color_for_zscore(z: float, vmin: float = -3.0, vmax: float = 3.0) -> str:
    """Return color string for a given z-score (diverging colormap)."""
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(z))
    return mpl.colors.to_hex(rgba)


def add_significance_stars(
    ax: plt.Axes,
    x: float,
    y: float,
    p_value: float,
    fontsize: int = 8,
) -> None:
    """Add significance stars to a plot at position (x, y)."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = ""
    if stars:
        ax.text(x, y, stars, ha="center", va="bottom", fontsize=fontsize)
