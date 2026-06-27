"""
Fig. 11: z-score heatmap. Fig. 13: spatial heatmap on pitch.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .style import apply_paper_style, FIGURE_WIDTH_DOUBLE


def plot_zscore_heatmap(
    zscore_df: pd.DataFrame,
    motif_id_col: str = "motif_id",
    k_col: str = "motif_order_k",
    side_col: str = "team_side",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 11: z-score heatmap.
    Rows: (motif_id, k). Columns: home / away.
    Color: z-score (diverging red-blue).
    """
    apply_paper_style()

    # Only low-order motifs carry an interpretable, randomisation-tested z-score
    # (higher orders are not significance-tested and would produce thousands of
    # rows). Restrict the heatmap to the 13 canonical triadic motifs.
    df = zscore_df[zscore_df[k_col] == 3].copy()
    if df.empty:
        df = zscore_df[zscore_df["sigma_rnd"] > 0].copy() if "sigma_rnd" in zscore_df else zscore_df.copy()

    pivot = df.pivot_table(
        index=[motif_id_col, k_col],
        columns=side_col,
        values="z",
    )

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE * 0.7, max(4, len(pivot) * 0.3)))

    vmax = max(3.0, float(zscore_df["z"].abs().max()))
    im = ax.imshow(
        pivot.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"M{mid} (k={k})" for mid, k in pivot.index], fontsize=7)
    ax.set_xlabel("Team side")
    ax.set_title("Motif z-score heatmap")

    plt.colorbar(im, ax=ax, label="z-score")

    # Mark significance threshold
    sig_threshold = 1.96
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val) and abs(val) > sig_threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=1.5
                ))

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_spatial_heatmap(
    spatial_df: pd.DataFrame,
    selected_motifs: Optional[List] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 13: Spatial density heatmap on pitch background.
    """
    apply_paper_style()

    try:
        from mplsoccer import Pitch
        use_mplsoccer = True
    except ImportError:
        use_mplsoccer = False

    ZONE_CENTERS = {
        "defensive_left": (20, 13.3),
        "defensive_center": (20, 40.0),
        "defensive_right": (20, 66.7),
        "middle_left": (60, 13.3),
        "middle_center": (60, 40.0),
        "middle_right": (60, 66.7),
        "attacking_left": (100, 13.3),
        "attacking_center": (100, 40.0),
        "attacking_right": (100, 66.7),
    }

    if selected_motifs is not None:
        if "motif_id" in spatial_df.columns:
            plot_df = spatial_df[spatial_df["motif_id"].isin(selected_motifs)]
        else:
            plot_df = spatial_df
    else:
        plot_df = spatial_df

    # Aggregate density per zone
    zone_density = plot_df.groupby("zone")["motif_density"].mean().reset_index()

    if use_mplsoccer:
        pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")
        fig, ax = pitch.draw(figsize=(FIGURE_WIDTH_DOUBLE, 5.0))
    else:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE, 5.0))
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.set_facecolor("#2d9e2d")
        ax.set_aspect("equal")

    # Draw density as bubbles
    max_density = zone_density["motif_density"].max()
    if max_density > 0:
        for _, row in zone_density.iterrows():
            zone = row["zone"]
            if zone not in ZONE_CENTERS:
                continue
            cx, cy = ZONE_CENTERS[zone]
            size = (row["motif_density"] / max_density) * 2000 + 100
            ax.scatter(cx, cy, s=size, alpha=0.6, color="#FF7F0E", edgecolors="white", linewidth=1)
            ax.text(cx, cy, f"{row['motif_density']:.3f}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    ax.set_title("Motif spatial density distribution")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
