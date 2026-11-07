"""
Fig. 16: Multi-order motif temporal evolution for a single match.
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .style import apply_paper_style, ORDER_COLORS, FIGURE_WIDTH_DOUBLE


def plot_temporal_evolution(
    match_id: int,
    motif_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    events_raw: Optional[List[dict]] = None,
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    selected_motif_ids: Optional[List[int]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 16: Temporal evolution of motif cumulative frequency in one match.

    X-axis: continuous minute
    Y-axis: cumulative motif count per order k
    Vertical lines: goals, substitutions
    """
    apply_paper_style()

    match_motifs = motif_df[motif_df["match_id"] == match_id]
    match_passes = pass_df[pass_df["match_id"] == match_id]

    if match_motifs.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No motif data for match {match_id}", ha="center", transform=ax.transAxes)
        return fig

    # Build possession-minute mapping from passes
    poss_minutes = (
        match_passes.groupby("possession")["minute"]
        .mean()
        .reset_index()
        .rename(columns={"minute": "mean_minute"})
    )

    # Merge motif counts with possession times
    motif_with_time = match_motifs.merge(poss_minutes, on="possession", how="left") \
        if "possession" in match_motifs.columns else match_motifs.assign(mean_minute=45)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE, 4.5))

    k_values = sorted(match_motifs["motif_order_k"].unique())

    for k in k_values:
        k_data = motif_with_time[motif_with_time["motif_order_k"] == k].copy()
        if selected_motif_ids is not None:
            k_data = k_data[k_data["motif_id"].isin(selected_motif_ids)]

        if k_data.empty:
            continue

        # Sort by time and compute cumulative sum
        k_data = k_data.sort_values("mean_minute")
        cumulative = k_data.groupby("mean_minute")["count"].sum().cumsum().reset_index()
        cumulative.columns = ["minute", "cumulative_count"]

        color = ORDER_COLORS.get(k, "gray")
        ax.step(
            cumulative["minute"],
            cumulative["cumulative_count"],
            where="post",
            color=color,
            linewidth=1.5,
            label=f"k={k}",
        )

    # Mark goal events
    if events_raw is not None:
        SHOT_ID = 16
        GOAL_OUTCOME_ID = 97
        for ev in events_raw:
            if ev.get("type", {}).get("id") == SHOT_ID:
                shot_data = ev.get("shot", {}) or {}
                if shot_data.get("outcome", {}).get("id") == GOAL_OUTCOME_ID:
                    minute = ev.get("minute", 0)
                    team_id = ev.get("team", {}).get("id")
                    is_home = (team_id == home_team_id)
                    color = "#1F77B4" if is_home else "#FF7F0E"
                    ax.axvline(minute, color=color, linestyle="-", linewidth=2, alpha=0.7)
                    ax.text(
                        minute + 0.5,
                        ax.get_ylim()[1] * 0.95,
                        "⚽",
                        fontsize=10,
                        color=color,
                        va="top",
                    )

    # Mark half-time
    ax.axvline(45, color="black", linestyle="--", linewidth=1, alpha=0.4, label="Half-time")
    ax.axvline(90, color="black", linestyle=":", linewidth=1, alpha=0.4, label="Full-time")

    ax.set_xlabel("Match minute")
    ax.set_ylabel("Cumulative motif count")
    ax.set_title(f"Multi-order motif temporal evolution (match {match_id})")
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(0, 95)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_score_state_shift(
    motif_score_df: pd.DataFrame,
    selected_motifs: Optional[List] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 14: Mean triadic-motif count as a function of the team's score state
    (trailing / level / leading). Grouped bars for a few interpretable motifs
    reveal that trailing teams play more direct forward chains while leading
    teams recycle the ball through reciprocal motifs.
    """
    apply_paper_style()
    from .style import ORDER_COLORS

    # interpretable motifs: one direct chain vs. reciprocal/closed motifs
    if selected_motifs is None:
        selected_motifs = [12, 78, 238, 110]
    labels = {12: "Direct chain (12)", 38: "Redirect (38)", 78: "Hub triangle (78)",
              238: "Closed triangle (238)", 110: "Reciprocal (110)", 14: "Reciprocal (14)"}
    palette = {12: "#E84855", 38: "#F4A261", 78: "#2A9D8F",
               238: "#264653", 110: "#2E86AB", 14: "#8E7DBE"}

    states = ["trailing", "drawing", "leading"]
    df = motif_score_df.copy()
    df = df[df.get("motif_order_k", 3) == 3]

    # aggregate over team side, weighting by the number of matches
    def agg_mean(sub):
        w = sub["n_matches"].to_numpy(dtype=float)
        v = sub["mean_count"].to_numpy(dtype=float)
        return float(np.average(v, weights=w)) if w.sum() > 0 else float(v.mean())

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE * 0.62, 4.0))
    x = np.arange(len(states))
    motifs = [m for m in selected_motifs if m in df["motif_id"].unique()]
    width = 0.8 / max(len(motifs), 1)

    for i, mid in enumerate(motifs):
        sub = df[df["motif_id"] == mid]
        means = []
        for st in states:
            s = sub[sub["score_state"] == st]
            means.append(agg_mean(s) if len(s) else 0.0)
        ax.bar(x + i * width, means, width=width,
               color=palette.get(mid, ORDER_COLORS.get(3)),
               label=labels.get(mid, f"Motif {mid}"), alpha=0.85)

    ax.set_xticks(x + width * (len(motifs) - 1) / 2)
    ax.set_xticklabels([s.capitalize() for s in states])
    ax.set_xlabel("Score state of the team")
    ax.set_ylabel("Mean motif count per network")
    ax.set_title("Triadic-motif mixture by score state")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig
