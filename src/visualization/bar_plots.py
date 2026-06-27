"""
Fig. 12: Heterogeneous motif errorbar plot.
Fig. 14: Context shift bar plots.
"""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import apply_paper_style, SIDE_COLORS, MOTIF_TYPE_COLORS, FIGURE_WIDTH_DOUBLE


def plot_hetero_errorbar(
    hetero_zscore_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 12: Errorbar plot of heterogeneous motif frequencies.
    Shows mean ± std for top N most frequent heterogeneous motifs.
    """
    apply_paper_style()

    # Select top N by mean count
    top = (
        hetero_zscore_df
        .sort_values("mu", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if top.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No heterogeneous motif data", ha="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE, 5.0))

    # Color by motif_type if available
    colors = [
        MOTIF_TYPE_COLORS.get(row.get("motif_type", "mixed"), "gray")
        for _, row in top.iterrows()
    ]

    y_pos = np.arange(len(top))

    ax.barh(
        y_pos,
        top["mu"],
        xerr=top["sigma"],
        color=colors,
        alpha=0.8,
        capsize=4,
        height=0.6,
    )

    # Mark significant bars
    if "significant" in top.columns:
        for i, (_, row) in enumerate(top.iterrows()):
            if row.get("significant", False):
                ax.text(
                    row["mu"] + row.get("sigma", 0) + 0.1,
                    i,
                    "*",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    # Y-tick labels
    if "pattern_key" in top.columns:
        labels = [str(k)[:30] for k in top["pattern_key"]]
    elif "motif_id" in top.columns:
        k_col = top.get("motif_order_k", pd.Series([3] * len(top)))
        labels = [f"M{int(mid)} k={int(k)}" for mid, k in zip(top["motif_id"], k_col)]
    else:
        labels = [f"Motif {i}" for i in range(len(top))]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean frequency (μ ± σ)")
    ax.set_title("Heterogeneous motif frequency distribution")
    ax.axvline(0, color="black", linewidth=0.5)

    # Legend for motif types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MOTIF_TYPE_COLORS[mt], label=mt.capitalize())
        for mt in MOTIF_TYPE_COLORS
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_motif_frequency_comparison(
    zscore_df: pd.DataFrame,
    top_n: int = 13,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Errorbar plot comparing home vs away motif frequencies.
    Analogous to original paper structure for homogeneous motifs.
    """
    apply_paper_style()

    if "team_side" not in zscore_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "team_side column required", ha="center", transform=ax.transAxes)
        return fig

    home_df = zscore_df[zscore_df["team_side"] == "home"].set_index("motif_id")
    away_df = zscore_df[zscore_df["team_side"] == "away"].set_index("motif_id")

    common_ids = sorted(set(home_df.index) & set(away_df.index))[:top_n]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE, 4.5))

    x = np.arange(len(common_ids))
    width = 0.35

    ax.bar(
        x - width / 2,
        [home_df.loc[mid, "mu"] if mid in home_df.index else 0 for mid in common_ids],
        width=width,
        yerr=[home_df.loc[mid, "sigma"] if mid in home_df.index else 0 for mid in common_ids],
        label="Home",
        color=SIDE_COLORS["home"],
        capsize=3,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        [away_df.loc[mid, "mu"] if mid in away_df.index else 0 for mid in common_ids],
        width=width,
        yerr=[away_df.loc[mid, "sigma"] if mid in away_df.index else 0 for mid in common_ids],
        label="Away",
        color=SIDE_COLORS["away"],
        capsize=3,
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"M{mid}" for mid in common_ids], rotation=45, ha="right")
    ax.set_ylabel("Mean frequency (μ ± σ)")
    ax.set_title("Motif frequency: home vs away (k=3)")
    ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
