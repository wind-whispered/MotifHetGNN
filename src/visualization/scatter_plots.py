"""
Fig. 17: GNN attribution vs regression coefficient scatter plot.
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy import stats

from .style import apply_paper_style, ORDER_COLORS, SIDE_COLORS, FIGURE_WIDTH_DOUBLE


def plot_attribution_scatter(
    attribution_df: pd.DataFrame,
    regression_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 17: genuine, participation-based GNN motif attribution versus the
    structural significance |z| of the triadic motifs.

    Each point is one triadic motif class. The attribution is derived purely
    from the trained model by aggregating node-level integrated gradients onto
    the motifs through node participation, so a positive correlation with |z| is
    a non-circular confirmation that the model relies on the same motifs that the
    z-score flags as structurally significant.
    """
    apply_paper_style()
    df = attribution_df[attribution_df.get("motif_order_k", 3) == 3].copy()

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_DOUBLE, 4.0))
    for ax, side in zip(axes, ["home", "away"]):
        sub = df[df["team_side"] == side].copy()
        sub = sub.dropna(subset=["mean_z", "mean_attribution"])
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue
        x = sub["mean_z"].abs().values
        y = sub["mean_attribution"].values
        ax.scatter(x, y, c=SIDE_COLORS[side], s=36, alpha=0.85,
                   edgecolors="white", linewidth=0.6, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(str(int(r["motif_id"])),
                        (abs(r["mean_z"]), r["mean_attribution"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        # least-squares trend line
        if len(x) > 2:
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b0 + b1 * xs, "--", color="gray", alpha=0.7, linewidth=1)
            rho, p_val = stats.spearmanr(x, y)
            ax.text(0.04, 0.96, f"Spearman $\\rho$={rho:.2f}\n($p$={p_val:.3f})",
                    transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlabel("Structural significance $|z|$")
        ax.set_ylabel("GNN motif attribution")
        ax.set_title(f"{side.capitalize()} team motifs")

    fig.suptitle("Participation-based GNN attribution vs motif significance", y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig
