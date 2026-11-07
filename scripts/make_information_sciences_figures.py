"""Generate revision figures exclusively from saved result tables."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.paths import figure_directory

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "information_sciences"
FIGURES = figure_directory()


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finite_size_figure():
    observed = pd.read_csv(RESULTS / "finite_size_observed_summary.csv")
    controlled = pd.read_csv(RESULTS / "finite_size_controlled_summary.csv")
    directed_controlled = (pd.read_csv(
        RESULTS / "finite_size_controlled_directed_degree_diagnostic_summary.csv")
        if (RESULTS / "finite_size_controlled_directed_degree_diagnostic_summary.csv").exists()
        else None)
    growth = pd.read_csv(
        RESULTS / "finite_size_growth_decomposition_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    axes = axes.ravel()
    raw = observed[observed.k <= 14]
    occupancy = observed[observed.k <= 7]
    axes[0].plot(raw.k, raw.raw_count_mean, "o-", color="#3B6FB6", ms=3)
    peak = raw.loc[raw.raw_count_mean.idxmax()]
    axes[0].annotate(
        f"peak $k={int(peak.k)}$", (peak.k, peak.raw_count_mean),
        xytext=(8, -18), textcoords="offset points", fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "0.4", "lw": .6})
    axes[0].set_xticks(range(2, 15, 2))
    axes[0].set(xlabel="Order $k$", ylabel="Raw mean count", title="(a) Raw census")
    axes[1].plot(occupancy.k, occupancy.q_mean, "o-", color="#D55E00", ms=3)
    axes[1].fill_between(
        occupancy.k,
        np.clip(occupancy.q_mean - occupancy.q_std, 0, 1),
        np.clip(occupancy.q_mean + occupancy.q_std, 0, 1),
        color="#D55E00", alpha=.15)
    axes[1].set_xticks(range(2, 8))
    axes[1].set(xlabel="Order $k$", ylabel="Connected occupancy $Q_k$",
                title="(b) Capacity normalisation", ylim=(0.25, .9))
    axes[2].axhline(0, color="0.45", lw=.8)
    axes[2].plot(controlled.k, controlled.q_excess_mean, "o-",
                 color="#009E73", ms=3, label="weak-projection degree-control")
    axes[2].fill_between(controlled.k,
                         controlled.q_excess_ci_low,
                         controlled.q_excess_ci_high,
                         color="#009E73", alpha=.15)
    if directed_controlled is not None:
        axes[2].plot(directed_controlled.k, directed_controlled.q_excess_mean, "s--",
                     color="#D55E00", ms=3,
                     label="directed-degree sensitivity")
        axes[2].fill_between(directed_controlled.k,
                             directed_controlled.q_excess_ci_low,
                             directed_controlled.q_excess_ci_high,
                             color="#D55E00", alpha=.12)
    axes[2].legend(frameon=False, fontsize=7, loc="best")
    axes[2].set(xlabel="Order $k$", ylabel="$Q_k^{obs}-E[Q_k^{null}]$",
                title="(c) Degree-controlled residual")
    axes[3].plot(growth.k, growth.extension_probability_mean, "o-",
                 color="#6A51A3", ms=3, label="$E_k$")
    axes[3].plot(growth.k, growth.raw_growth_threshold_mean, "d:",
                 color="#3B6FB6", ms=3,
                 label="$\\mu_{k+1}/(n-k)$")
    axes[3].plot(growth.k, growth.occupancy_growth_threshold_mean, "s--",
                 color="#E69F00", ms=3,
                 label="$\\mu_{k+1}/(k+1)$")
    axes[3].set(xlabel="Order $k$",
                ylabel="Extension probability / threshold",
                title="(d) Exact growth thresholds", ylim=(0, 1.35))
    axes[3].legend(frameon=False, fontsize=7, ncol=3, loc="lower center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=.5)
    fig.tight_layout(w_pad=1.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "Figure_03_Finite_Size_Controls.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_03_Finite_Size_Controls.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def model_figure():
    path = RESULTS / "model_results.parquet"
    if not path.exists():
        return
    data = pd.read_parquet(path)
    # Keep only seed-loop model runs for the core comparison panel.
    data = data[data["seed"] >= 0]
    labels = {
        "TeamPool-MLP": "TeamPool (no edges)",
        "HeteroSAGE-sum": "HeteroSAGE",
        "R-GCN": "R-GCN", "HGT": "HGT",
        "EdgeAware-Fixed": "Edge-aware (fixed)",
        "MotifGate-ObservedOnly": "MotifGate (observed)",
        "SPARC-GNN": "SPARC-GNN",
    }
    colors = {
        "TeamPool-MLP": "#999999", "HeteroSAGE-sum": "#4E79A7",
        "R-GCN": "#A0CBE8", "HGT": "#F28E2B",
        "EdgeAware-Fixed": "#76B7B2",
        "MotifGate-ObservedOnly": "#59A14F",
        "SPARC-GNN": "#D55E00",
    }
    order = [model for model in (
        "TeamPool-MLP", "HGT", "MotifGate-ObservedOnly",
        "EdgeAware-Fixed", "SPARC-GNN", "R-GCN", "HeteroSAGE-sum"
    ) if model in set(data.model)]
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.0))

    def absolute_panel(ax, metric, title):
        for position, model in enumerate(order):
            values = data.loc[data.model == model, metric].to_numpy()
            ax.errorbar(values.mean(), position, xerr=values.std(ddof=1),
                        fmt="o", color=colors[model], capsize=2.5,
                        markersize=4.5, lw=1.0, zorder=3)
            offsets = np.linspace(-.13, .13, len(values))
            ax.scatter(values, position + offsets, s=12, facecolor="white",
                       edgecolor=colors[model], linewidth=.65, zorder=2)
        ax.set_yticks(range(len(order)), [labels[model] for model in order])
        ax.invert_yaxis()
        ax.set_xlabel(title.split(" ", 1)[1])
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color="0.88", lw=.55)
        ax.spines[["top", "right"]].set_visible(False)

    absolute_panel(axes[0, 0], "macro_f1", r"(a) Macro-$F_1$")
    absolute_panel(axes[0, 1], "r2", r"(b) Goal-difference $R^2$")

    bootstrap_path = RESULTS / "hierarchical_paired_bootstrap.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    baseline_order = [model for model in order if model != "SPARC-GNN"]

    def contrast_panel(ax, metric, title):
        subset = bootstrap[bootstrap.metric == metric].set_index("baseline")
        for position, baseline in enumerate(baseline_order):
            row = subset.loc[baseline]
            value = row.delta_target_minus_baseline
            interval = np.asarray([[value - row.ci_low],
                                   [row.ci_high - value]])
            status_color = ("#009E73" if row.ci_low > 0 else
                            "#B2182B" if row.ci_high < 0 else "#777777")
            ax.errorbar(value, position, xerr=interval, fmt="o",
                        color=status_color, ecolor=status_color,
                        markersize=4.3, capsize=2.5, lw=1.0)
        ax.axvline(0, color="#333333", lw=.8, ls="--")
        ax.set_yticks(range(len(baseline_order)),
                      [labels[model] for model in baseline_order])
        ax.invert_yaxis()
        ax.set_xlabel("SPARC-GNN minus baseline")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color="0.88", lw=.55)
        ax.spines[["top", "right"]].set_visible(False)

    contrast_panel(axes[1, 0], "macro_f1",
                   r"(c) Paired $\Delta$ macro-$F_1$ (95\% interval)")
    contrast_panel(axes[1, 1], "draw_f1",
                   r"(d) Paired $\Delta$ draw-$F_1$ (95\% interval)")
    fig.text(.5, .006,
             "Top: five seed estimates and mean $\\pm$ s.d.  Bottom: hierarchical seed-by-match bootstrap.",
             ha="center", fontsize=7.2)
    fig.tight_layout(w_pad=1.4, h_pad=1.5, rect=(0, .025, 1, 1))
    fig.savefig(FIGURES / "Figure_08_Model_Comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_08_Model_Comparison.png", dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def main():
    style()
    finite_size_figure()
    model_figure()
    print(f"Information Sciences figures -> {FIGURES}")


if __name__ == "__main__":
    main()
