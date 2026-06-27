"""
Fig. 15: Tactical fingerprint radar chart per team.
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .style import apply_paper_style, FIGURE_WIDTH_DOUBLE

TEAM_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F",
]


def _radar_axes(ax: plt.Axes, n_vars: int) -> np.ndarray:
    """Set up radar chart axes."""
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return np.array(angles)


def plot_tactical_fingerprint_radar(
    motif_df: pd.DataFrame,
    team_names: List[str],
    team_match_ids: Dict[str, List[int]],
    n_top_motifs: int = 10,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 15: Radar chart showing tactical fingerprint per team.

    For each team, compute mean motif frequency across their matches,
    then plot on a radar chart with one axis per motif.

    Args:
        team_names: list of team names to include
        team_match_ids: dict mapping team_name -> list of match_ids (home + away)
        n_top_motifs: number of top motifs (by variance) to use as axes
    """
    apply_paper_style()

    # Select top motifs by cross-team variance
    all_teams_mean = {}
    for team, match_ids in team_match_ids.items():
        team_motifs = motif_df[motif_df["match_id"].isin(match_ids)]
        if team_motifs.empty:
            continue
        mean_counts = (
            team_motifs.groupby(["motif_id", "motif_order_k"])["count"]
            .mean()
            .reset_index()
        )
        mean_counts["label"] = mean_counts["motif_id"].astype(str) + "_k" + mean_counts["motif_order_k"].astype(str)
        all_teams_mean[team] = mean_counts.set_index("label")["count"]

    if not all_teams_mean:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha="center", transform=ax.transAxes)
        return fig

    combined = pd.DataFrame(all_teams_mean).fillna(0)
    top_motif_labels = combined.var(axis=1).nlargest(n_top_motifs).index.tolist()
    combined_top = combined.loc[top_motif_labels].T  # teams x motifs

    # Normalize per motif (0-1 range)
    col_max = combined_top.max(axis=0).replace(0, 1)
    combined_norm = combined_top / col_max

    n_vars = len(top_motif_labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(FIGURE_WIDTH_DOUBLE, FIGURE_WIDTH_DOUBLE))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_motif_labels, size=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6)

    for i, team in enumerate(team_names):
        if team not in combined_norm.index:
            continue
        values = combined_norm.loc[team].tolist()
        values += values[:1]
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        ax.plot(angles, values, "o-", linewidth=1.5, color=color, label=team, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Tactical fingerprint radar", pad=20)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig
