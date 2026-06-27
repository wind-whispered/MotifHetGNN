"""
Fig. 5 / Fig. 10: Motif count decay curves and empirical distribution.
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .style import apply_paper_style, SIDE_COLORS, MOTIF_TYPE_COLORS, FIGURE_WIDTH_DOUBLE


def plot_decay_curve(
    homo_decay_df: pd.DataFrame,
    hetero_decay_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    k_max: int = 14,
) -> plt.Figure:
    """
    Fig. 5: All-order census size N(k) and occupancy per class.
    Left panel: N(k) for home/away (primary axis) + occupancy/class (secondary axis, red).
    Right panel (optional): heterogeneous triadic composition by motif_type.
    k range displayed: k=2 to k_max.
    """
    apply_paper_style()
    n_cols = 2 if hetero_decay_df is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(FIGURE_WIDTH_DOUBLE, 4.0))
    if n_cols == 1:
        axes = [axes]

    # ── Left: N(k) curves + occupancy twin axis ──────────────────────────
    ax = axes[0]
    ax_occ = ax.twinx()

    for side in ["home", "away"]:
        sub = homo_decay_df[homo_decay_df["team_side"] == side].sort_values("k")
        sub = sub[sub["k"] <= k_max]
        if sub.empty:
            continue
        ax.errorbar(
            sub["k"], sub["mean_total_count"],
            yerr=sub["std_total_count"],
            label=side.capitalize(),
            color=SIDE_COLORS[side],
            marker="o", markersize=4, capsize=3, linewidth=1.5,
        )

    # Occupancy per class: mean over home/away, plotted in red on twin axis
    occ_parts = []
    for side in ["home", "away"]:
        sub = homo_decay_df[homo_decay_df["team_side"] == side].sort_values("k")
        sub = sub[sub["k"] <= k_max].copy()
        if "n_observed_types" in sub.columns:
            sub["occ"] = sub["mean_total_count"] / sub["n_observed_types"].clip(lower=1)
            occ_parts.append(sub[["k", "occ"]])
    if occ_parts:
        occ_df = pd.concat(occ_parts).groupby("k")["occ"].mean().reset_index()
        ax_occ.plot(occ_df["k"], occ_df["occ"],
                    color="#E84855", linestyle="--", marker="s",
                    markersize=3, linewidth=1.2, label="Occupancy/class", alpha=0.85)
        ax_occ.set_yscale("log")
        ax_occ.set_ylabel("Mean occupancy per class", color="#E84855", fontsize=8)
        ax_occ.tick_params(axis="y", labelcolor="#E84855")

    # Mark k* = order with maximum N(k), averaged over sides
    mean_by_k = homo_decay_df[homo_decay_df["k"] <= k_max].groupby("k")["mean_total_count"].mean()
    if not mean_by_k.empty:
        k_peak = int(mean_by_k.idxmax())
        ax.axvline(k_peak, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        # use xaxis_transform: data x, axes-fraction y → stable in log scale
        ax.text(k_peak + 0.15, 0.97, f"$k^*\\!=\\!{k_peak}$",
                transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=7, color="gray")

    ax.set_xlabel("Motif order $k$")
    ax.set_ylabel("Mean census size $N(k)$")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title("Census size and class occupancy ($k=2$--$14$)")

    # Combined legend (N(k) lines + occupancy)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_occ.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              loc="upper right", fontsize=7, frameon=False)

    # ── Right: hetero triadic composition ────────────────────────────────
    if hetero_decay_df is not None and n_cols > 1:
        ax2 = axes[1]
        order = ["cooperative", "mixed", "adversarial"]
        present = [m for m in order if m in set(hetero_decay_df["motif_type"])]
        present += [m for m in hetero_decay_df["motif_type"].unique() if m not in present]
        xpos = np.arange(len(present))
        means = [
            float(hetero_decay_df.loc[hetero_decay_df["motif_type"] == m,
                                      "mean_total_count"].mean())
            for m in present
        ]
        errs = [
            float(hetero_decay_df.loc[hetero_decay_df["motif_type"] == m,
                                      "std_total_count"].mean())
            for m in present
        ]
        ax2.bar(
            xpos, means, yerr=errs, capsize=3,
            color=[MOTIF_TYPE_COLORS.get(m, "gray") for m in present],
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        ax2.set_yscale("log")
        ax2.set_xticks(xpos)
        ax2.set_xticklabels([m.capitalize() for m in present])
        ax2.set_ylabel("Mean census per match")
        ax2.set_xlabel("Heterogeneous motif type ($k=3$)")
        ax2.set_title("Cooperative--adversarial composition")
        for x, m in zip(xpos, means):
            ax2.text(x, m * 1.25, f"{m:.0f}" if m >= 1 else f"{m:.2f}",
                     ha="center", va="bottom", fontsize=7)
        ax2.set_ylim(top=max(means) * 4)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_empirical_distribution(
    homo_decay_df: pd.DataFrame,
    competition_decay_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 10: Empirical distribution of motif counts across matches.
    Shows boxplots of total count per match at each order k.
    """
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_DOUBLE, 4.0))

    for ax_idx, side in enumerate(["home", "away"]):
        ax = axes[ax_idx]
        sub = homo_decay_df[homo_decay_df["team_side"] == side].sort_values("k")
        k_vals = sorted(sub["k"].unique())

        data_for_box = [
            [sub[sub["k"] == k]["mean_total_count"].values[0]] * 10
            if not sub[sub["k"] == k].empty else [0]
            for k in k_vals
        ]

        ax.bar(
            range(len(k_vals)),
            sub["mean_total_count"].values,
            yerr=sub["std_total_count"].values,
            color=SIDE_COLORS[side],
            alpha=0.7,
            capsize=3,
        )
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels([f"k={k}" for k in k_vals], rotation=45, ha="right")
        ax.set_ylabel("Mean total count")
        ax.set_title(f"{side.capitalize()} team motif distribution")
        ax.set_yscale("log")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_w0_robustness(
    network_stats_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 9: robustness of the network measures to the link-weight threshold w0.
    A 2x3 grid of error-bar panels (mean +/- s.d. across matches) for density D,
    transitivity T, passing diversity phi, mean out-degree, mean betweenness and
    mean eigenvector centrality, for home and away teams.
    """
    apply_paper_style()
    measures = [
        ("density", "Density $D$"),
        ("transitivity", "Transitivity $T$"),
        ("pass_diversity", r"Diversity $\langle\phi\rangle$"),
        ("mean_outdegree", r"Out-degree $\langle k_{\mathrm{out}}\rangle$"),
        ("mean_betweenness", r"Betweenness $\langle b\rangle$"),
        ("mean_eigenvector", r"Eigenvector $\langle e\rangle$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_DOUBLE, 4.6))
    axes = axes.ravel()
    grp = network_stats_df.groupby(["w0", "team_side"])
    agg = grp.agg(["mean", "std"])

    for ax, (col, label) in zip(axes, measures):
        for side in ["home", "away"]:
            w0s, mus, sds = [], [], []
            for w0 in sorted(network_stats_df["w0"].unique()):
                try:
                    row = agg.loc[(w0, side), col]
                    w0s.append(w0)
                    mus.append(row["mean"])
                    sds.append(row["std"])
                except KeyError:
                    continue
            ax.errorbar(w0s, mus, yerr=sds, marker="o", capsize=3,
                        color=SIDE_COLORS[side], label=side.capitalize())
        ax.set_xlabel("$w_0$")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(network_stats_df["w0"].unique()))
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_saturation_curve(
    incremental_r2_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 10 (right axis): R² increment and significant motif count vs k.
    Shows information saturation point.
    """
    apply_paper_style()
    fig, ax1 = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE / 2, 4.0))

    ax2 = ax1.twinx()

    ax1.bar(
        incremental_r2_df["k"],
        incremental_r2_df.get("r2_increment", incremental_r2_df.get("r_squared_cumulative", 0)),
        alpha=0.6,
        color="#2E86AB",
        label="R² increment",
    )
    ax2.plot(
        incremental_r2_df["k"],
        incremental_r2_df.get("n_significant_motifs_at_k", 0),
        color="#E84855",
        marker="o",
        label="Distinct motif classes",
    )

    ax1.set_xlabel("Motif order $k$")
    ax1.set_ylabel("Incremental $R^2$", color="#2E86AB")
    ax2.set_ylabel("# distinct motif classes", color="#E84855")
    ax2.set_yscale("log")
    ax1.set_title("Information saturation by motif order")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
