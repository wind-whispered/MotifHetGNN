"""Publication figures for method decomposition and generalisation evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from src.paths import figure_directory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "information_sciences"
FIG = figure_directory()
FIG.mkdir(parents=True, exist_ok=True)

MODEL_ALIASES = {
    "MCAR-Decomp": "SPARC-GNN",
    "MCAR-Decomp+Score": "SPARC-GNN+Score",
}

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25,
                     "savefig.bbox": "tight", "savefig.dpi": 600})

DISPLAY = {"HeteroSAGE-sum": "HeteroSAGE",
           "TeamPool-MLP": "TeamPool (no edges)",
           "R-GCN": "R-GCN",
           "HGT": "HGT",
           "MotifGate": "MotifGate",
           "MotifGate-ObservedOnly": "MotifGate (observed only)",
           "EdgeAware-Fixed": "Edge-aware (fixed)",
           "SPARC-GNN": "SPARC-GNN",
           "MCAR-Decomp": "SPARC-GNN",
           "TeamPool-MLP+Score": "TeamPool+Score (no edges)",
           "HeteroSAGE-sum+Score": "HeteroSAGE+Score",
           "SPARC-GNN+Score": "SPARC-GNN+Score",
           "MCAR-Decomp+Score": "SPARC-GNN+Score",
           "CurrentScore": "Current score",
           "CurrentScore-Calibrated": "Calibrated score",
           "Score+Strength": "Score + pre-match strength",
           "CausalMatchState-HGB": "Causal match state",
           "TeamPool-MLP+Score+ValBlend":
               "TeamPool+Score (val. calibrated)",
           "HeteroSAGE-sum+Score+ValBlend":
               "HeteroSAGE+Score (val. calibrated)",
           "SPARC-GNN+Score+ValBlend":
               "SPARC+Score (val. calibrated)"}

COLORS = {
    "TeamPool-MLP": "#9C9C9C",
    "HeteroSAGE-sum": "#4E79A7",
    "R-GCN": "#A0CBE8",
    "HGT": "#F28E2B",
    "MotifGate": "#59A14F",
    "MotifGate-ObservedOnly": "#59A14F",
    "EdgeAware-Fixed": "#76B7B2",
    "SPARC-GNN": "#D55E00",
    "MCAR-Decomp": "#D55E00",
    "TeamPool-MLP+Score": "#8C6BB1",
    "HeteroSAGE-sum+Score": "#1F3A93",
    "SPARC-GNN+Score": "#009E73",
    "MCAR-Decomp+Score": "#009E73",
    "CurrentScore": "#888888",
    "CurrentScore-Calibrated": "#333333",
    "Score+Strength": "#0072B2",
    "CausalMatchState-HGB": "#CC79A7",
    "TeamPool-MLP+Score+ValBlend": "#8C6BB1",
    "HeteroSAGE-sum+Score+ValBlend": "#1F3A93",
    "SPARC-GNN+Score+ValBlend": "#C51B7D",
}

GRAPH_MODEL_ORDER = [
    "TeamPool-MLP", "HeteroSAGE-sum", "HGT", "EdgeAware-Fixed",
    "SPARC-GNN",
]
TEMPORAL_MODEL_ORDER = [
    "CurrentScore", "CurrentScore-Calibrated",
    "HeteroSAGE-sum", "SPARC-GNN",
    "TeamPool-MLP+Score", "HeteroSAGE-sum+Score", "SPARC-GNN+Score",
]
MARKERS = {"TeamPool-MLP": "o", "HeteroSAGE-sum": "v", "HGT": "s",
           "EdgeAware-Fixed": "^", "SPARC-GNN": "D",
           "TeamPool-MLP+Score": "h",
           "HeteroSAGE-sum+Score": "X", "SPARC-GNN+Score": "P",
           "CurrentScore": "x", "CurrentScore-Calibrated": "*"}
MARKERS.update({"Score+Strength": "s", "CausalMatchState-HGB": "D",
                "TeamPool-MLP+Score+ValBlend": "h",
                "HeteroSAGE-sum+Score+ValBlend": "X",
                "SPARC-GNN+Score+ValBlend": "P"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def _performance_matrix(ax, frame, row_key, column_key, value_key,
                        rows, columns, column_labels, title, lower_is_better=False,
                        show_sd=True):
    grouped = frame.groupby([row_key, column_key])[value_key].agg(["mean", "std"])
    values = np.full((len(rows), len(columns)), np.nan)
    errors = np.full_like(values, np.nan)
    for i, model in enumerate(rows):
        for j, column in enumerate(columns):
            if (model, column) in grouped.index:
                values[i, j] = grouped.loc[(model, column), "mean"]
                errors[i, j] = grouped.loc[(model, column), "std"]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.set_axis_off()
        return
    pad = max((finite.max() - finite.min()) * .08, .01)
    cmap = "viridis_r" if lower_is_better else "viridis"
    image = ax.imshow(values, aspect="equal", cmap=cmap,
                      vmin=finite.min() - pad, vmax=finite.max() + pad)
    midpoint = (finite.min() + finite.max()) / 2
    for i in range(len(rows)):
        for j in range(len(columns)):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            if show_sd and np.isfinite(errors[i, j]):
                label = f"{value:.3f}\n$\\pm${errors[i, j]:.3f}"
            else:
                label = f"{value:.3f}"
            # Viridis is dark at the low end; reversed viridis is dark at the
            # high end. Select text colour from the rendered scalar position.
            dark = value < midpoint if not lower_is_better else value > midpoint
            ax.text(j, i, label, ha="center", va="center", fontsize=6.2,
                    color="white" if dark else "#111111")
    ax.set_xticks(range(len(columns)), column_labels, rotation=18, ha="right")
    ax.set_yticks(range(len(rows)), [DISPLAY.get(row, row) for row in rows])
    ax.set_title(title, fontweight="bold", pad=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def generalization():
    path = DATA / "generalization_results.parquet"
    if not path.exists(): return
    d = pd.read_parquet(path)
    protocols = ["random", "chronological", "competition", "team_grouped"]
    labels = ["Random", "Temporal", "Competition", "Team-disjoint"]
    models = [model for model in GRAPH_MODEL_ORDER if model in set(d["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))
    offsets = np.linspace(-.18, .18, len(models))
    for ax, metric, title in zip(
            axes, ["macro_f1", "r2"],
            [r"(a) Macro-$F_1$", r"(b) Goal-difference $R^2$"]):
        for model_index, model in enumerate(models):
            subset = d[d.model == model]
            grouped = subset.groupby("protocol")[metric].agg(["mean", "std"])
            means = np.asarray([grouped.loc[p, "mean"] for p in protocols])
            errors = np.asarray([grouped.loc[p, "std"] for p in protocols])
            positions = np.arange(len(protocols)) + offsets[model_index]
            ax.errorbar(positions, means, yerr=errors,
                        fmt=MARKERS.get(model, "o"), markersize=4.2,
                        capsize=2.2, elinewidth=1.05, linestyle="none",
                        color=COLORS.get(model, "#555555"),
                        label=DISPLAY.get(model, model))
        ax.set_xticks(range(len(protocols)), labels, rotation=18, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Evaluation protocol")
        ax.set_xlim(-.45, len(protocols) - .55)
        ax.grid(axis="y", color="0.88", lw=.6)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(models),
               frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, .91))
    save(fig, "Figure_09_Generalization")


def temporal():
    path = DATA / "temporal_results.parquet"
    causal_path = DATA / "temporal_causal_baselines.parquet"
    blend_path = DATA / "temporal_validation_blends.parquet"
    missing = [item.name for item in (path, causal_path, blend_path)
               if not item.exists()]
    if missing:
        raise RuntimeError(
            f"temporal figure requires completed sources; missing={missing}")
    raw = pd.read_parquet(path)
    raw["model"] = raw["model"].replace(MODEL_ALIASES)
    causal = pd.read_parquet(causal_path)
    blend = pd.read_parquet(blend_path)
    d = pd.concat([raw, causal, blend], ignore_index=True, sort=False)
    cutoffs = [15, 30, 45, 60, 75]
    control_models = [
        "CurrentScore", "CurrentScore-Calibrated",
        "Score+Strength", "CausalMatchState-HGB",
    ]
    graph_models = [
        "TeamPool-MLP+Score", "HeteroSAGE-sum+Score", "SPARC-GNN+Score",
    ]
    calibrated_models = [
        "TeamPool-MLP+Score+ValBlend",
        "HeteroSAGE-sum+Score+ValBlend",
        "SPARC-GNN+Score+ValBlend",
    ]
    expected_seed_counts = {
        **{model: 1 for model in control_models},
        **{model: 5 for model in graph_models},
        **{model: 5 for model in calibrated_models},
    }
    incomplete = []
    for model, expected_count in expected_seed_counts.items():
        subset = d[d.model.eq(model)]
        for cutoff in cutoffs:
            count = int(subset[subset.cutoff.eq(cutoff)]["seed"].nunique())
            if count != expected_count:
                incomplete.append((model, cutoff, count, expected_count))
    if incomplete:
        raise RuntimeError(
            "temporal figure refuses partial results: "
            f"{incomplete[:8]}{' ...' if len(incomplete) > 8 else ''}")

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.65), sharex=True,
                             sharey="col")
    specs = [("macro_f1", r"Macro-$F_1$"),
             ("draw_f1", r"Draw-$F_1$"),
             ("r2", r"Goal-difference $R^2$")]

    def plot_family(axis_row, models):
        for column, (metric, title) in enumerate(specs):
            ax = axes[axis_row, column]
            for model in models:
                subset = d[d.model.eq(model)]
                grouped = subset.groupby("cutoff")[metric].agg(["mean", "std"])
                means = grouped.loc[cutoffs, "mean"].to_numpy(float)
                errors = grouped.loc[cutoffs, "std"].to_numpy(float)
                color = COLORS.get(model, "#555555")
                linestyle = "--" if model in {
                    "CurrentScore", "CurrentScore-Calibrated", "SPARC-GNN"
                } else "-"
                finite = np.isfinite(errors)
                ax.errorbar(
                    cutoffs, means,
                    yerr=np.where(finite, errors, 0.0),
                    marker=MARKERS.get(model, "o"), markersize=3.7,
                    lw=1.1, ls=linestyle, capsize=2.0, color=color,
                    label=DISPLAY.get(model, model),
                )
            panel = chr(97 + axis_row * 3 + column)
            ax.set_title(f"({panel}) {title}", fontweight="bold")
            ax.set_xticks(cutoffs)
            ax.axvline(45, color="0.72", lw=.65, ls=":", zorder=0)
            ax.grid(axis="y", color="0.88", lw=.55)

    plot_family(0, control_models)
    plot_family(1, graph_models)
    plot_family(2, calibrated_models)
    for ax in axes[2, :]:
        ax.set_xlabel("Event cutoff (min)")
    axes[0, 0].set_ylabel("Causal controls")
    axes[1, 0].set_ylabel("Equal-budget pathways")
    axes[2, 0].set_ylabel("Validation calibration")
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    middle_handles, middle_labels = axes[1, 0].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(top_handles, top_labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(.5, .995), fontsize=7.1)
    fig.legend(middle_handles, middle_labels, loc="center", ncol=4,
               frameon=False, bbox_to_anchor=(.5, .665), fontsize=6.9)
    fig.legend(bottom_handles, bottom_labels, loc="center", ncol=3,
               frameon=False, bbox_to_anchor=(.5, .345), fontsize=6.9)
    fig.subplots_adjust(left=.09, right=.985, bottom=.075, top=.91,
                        wspace=.28, hspace=.82)
    save(fig, "Figure_10_Temporal_Cutoff")


def decomposition():
    path = DATA / "decomposition_diagnostics.parquet"
    utility_path = DATA / "relation_subset_paired_effects.csv"
    if not path.exists() or not utility_path.exists(): return
    d = pd.read_parquet(path)
    utility = pd.read_csv(utility_path)
    relation_order = ["pass", "adversarial", "turnover"]
    palette = {"pass": "#0072B2", "adversarial": "#D55E00",
               "turnover": "#009E73", "shared": "#6A51A3",
               "multi_relation_nodes": "#C51B7D"}

    def estimate_points(ax, quantity, order, title, log=False,
                        reference=None):
        subset = d[d.quantity == quantity]
        grouped = subset.groupby("component")["mean"].agg(["mean", "std"])
        means = [grouped.loc[item, "mean"] for item in order]
        errors = [grouped.loc[item, "std"] for item in order]
        for index, component in enumerate(order):
            points = subset.loc[subset.component == component, "mean"].to_numpy()
            offsets = np.linspace(-.13, .13, len(points)) if len(points) > 1 else [0]
            ax.scatter(index + np.asarray(offsets), points, s=12,
                       facecolor="white", edgecolor=palette[component],
                       linewidth=.75, alpha=.9, zorder=2)
            ax.errorbar(index, means[index], yerr=errors[index], fmt="o",
                        ms=4.3, capsize=2.8, elinewidth=1.1,
                        color=palette[component], markeredgecolor="white",
                        markeredgewidth=.45, zorder=4)
        labels = [item.replace("adversarial", "adv.").replace(
            "turnover", "turn.").replace("multi_relation_nodes", "multi-rel.")
                  for item in order]
        ax.set_xticks(range(len(order)), labels)
        ax.set_title(title, fontweight="bold", fontsize=8)
        if log:
            ax.set_yscale("log")
        if reference is not None:
            ax.axhline(reference, color="#555555", ls="--", lw=.8)
        ax.grid(False)
        ax.grid(axis="y", color="0.88", lw=.5)

    fig = plt.figure(figsize=(7.45, 4.85))
    grid = fig.add_gridspec(2, 2, hspace=.58, wspace=.50)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0])]
    specs = [
        ("relation_availability", relation_order,
         "(a) Observed-node support", False, None),
        ("gate_mass_multi_relation_nodes", relation_order,
         r"(b) Gate mass: $\geq2$ observed relations", False, None),
    ]
    for ax, spec in zip(axes[:2], specs):
        estimate_points(ax, *spec)
    axes[0].set_ylabel("Fraction of all nodes")
    axes[1].set_ylabel("Conditional gate mass")
    axes[0].set_ylim(0, .8)
    axes[1].set_ylim(-.06, 1.05)
    axes[1].set_yticks(np.linspace(0, 1, 6))

    # A two-condition estimate plot replaces the former single bar.  The
    # contrast distinguishes near-deterministic fusion over all observed nodes
    # from genuine mixing where at least two relation messages are available.
    ax = axes[2]
    conditions = [
        ("effective_relation_count", "observed_relations",
         "All observed", "shared"),
        ("effective_relation_count_multi", "multi_relation_nodes",
         r"$\geq2$ relations", "multi_relation_nodes"),
    ]
    for index, (quantity, component, _, colour_key) in enumerate(conditions):
        points = d.loc[(d.quantity == quantity) &
                       (d.component == component), "mean"].to_numpy()
        offsets = np.linspace(-.13, .13, len(points)) if len(points) > 1 else [0]
        colour = palette[colour_key]
        ax.scatter(index + np.asarray(offsets), points, s=12,
                   facecolor="white", edgecolor=colour, linewidth=.75,
                   alpha=.9, zorder=2)
        ax.errorbar(index, points.mean(), yerr=points.std(ddof=1), fmt="o",
                    ms=4.3, capsize=2.8, elinewidth=1.1, color=colour,
                    markeredgecolor="white", markeredgewidth=.45, zorder=4)
    ax.set_xticks(range(len(conditions)), [item[2] for item in conditions])
    ax.set_title("(c) Effective relation count", fontweight="bold", fontsize=8)
    ax.set_ylabel(r"$\exp\{H(\alpha_v)\}$")
    axes[2].axhline(1.0, color="#777777", ls=":", lw=.75)
    axes[2].axhline(3.0, color="#555555", ls="--", lw=.75)
    axes[2].set_ylim(.75, 3.12)
    axes[2].grid(False)
    axes[2].grid(axis="y", color="0.88", lw=.5)

    # The lower panel links local allocation diagnostics to the stronger
    # system-level test: matched retraining after adding one relation.
    ax = fig.add_subplot(grid[1, 1])
    contrasts = [
        ("pass_adversarial_over_pass", r"$P\to PA$"),
        ("pass_turnover_over_pass", r"$P\to PT$"),
        ("all_over_pass_turnover", r"$PT\to PAT$"),
        ("all_over_adversarial_turnover", r"$AT\to PAT$"),
    ]
    metrics = ["accuracy", "macro_f1", "draw_f1", "mae", "r2"]
    metric_labels = ["Acc", r"Macro-$F_1$", r"Draw-$F_1$",
                     r"$-$MAE", r"$R^2$"]
    matrix = np.full((len(contrasts), len(metrics)), np.nan)
    for row, (contrast, _) in enumerate(contrasts):
        rows = utility[utility["contrast"] == contrast].set_index("metric")
        for column, metric in enumerate(metrics):
            if metric in rows.index:
                matrix[row, column] = rows.loc[metric, "mean_improvement"]
    limit = np.nanmax(np.abs(matrix))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit,
                      aspect="equal", interpolation="nearest")
    ax.set_xticks(range(len(metrics)), metric_labels)
    ax.set_yticks(range(len(contrasts)), [label for _, label in contrasts])
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.set_title("(d) Marginal utility after retraining",
                 fontweight="bold", fontsize=8)
    ax.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            colour = "white" if abs(value) > .60 * limit else "#202020"
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center",
                    fontsize=6.8, color=colour)
    fig.subplots_adjust(left=.09, right=.92, top=.93, bottom=.09)
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + .018, pos.y0, .012, pos.height])
    colourbar = fig.colorbar(image, cax=cax)
    colourbar.set_label("Improvement", fontsize=7)
    colourbar.ax.tick_params(labelsize=6.5)
    save(fig, "Figure_11_Relation_Information_Decomposition")


def incremental_information():
    """Preserve the all-order information curve as a descriptive audit.

    The original figure mixed an in-sample cumulative regression curve with a
    separate cross-validation classification curve.  This replacement keeps
    one estimand throughout: cumulative in-sample R2 and its order-wise
    increment under the recorded screening rule.
    """
    path = ROOT / "data" / "analysis" / "incremental_r2_k7.parquet"
    if not path.exists():
        return
    d = pd.read_parquet(path).sort_values("k")
    ks = d["k"].to_numpy(dtype=int)
    cumulative = d["r_squared_cumulative"].to_numpy(dtype=float)
    increments = d["r2_increment"].to_numpy(dtype=float)
    candidates = d["n_observed_motif_types_at_k"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.85))
    ax = axes[0]
    ax.plot(ks, cumulative, marker="o", ms=4.4, lw=1.45,
            color="#0072B2")
    for x, value in zip(ks, cumulative):
        ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_xticks(ks)
    ax.set_ylim(.16, .305)
    ax.set_xlabel("Maximum motif order $K$")
    ax.set_ylabel("Cumulative in-sample $R^2$")
    ax.set_title("(a) Descriptive cumulative association", fontweight="bold")
    ax.grid(axis="y", color="0.88", lw=.6)

    ax = axes[1]
    ax.plot(ks[1:], increments[1:], marker="D", ms=4.0, lw=1.35,
            color="#D55E00")
    ax.set_xticks(ks[1:])
    ax.set_ylim(0, max(increments[1:]) * 1.30)
    ax.set_xlabel("Added motif order $k$")
    ax.set_ylabel(r"Incremental $\Delta R^2$")
    ax.set_title("(b) Gain versus screening burden", fontweight="bold")
    ax.grid(axis="y", color="0.88", lw=.6)
    burden = ax.twinx()
    burden.plot(ks[1:], candidates[1:], marker="s", ms=3.6, lw=1.1,
                ls="--", color="#6A51A3")
    burden.set_yscale("log")
    burden.set_ylabel("Corpus-realised classes (log scale)")
    burden.grid(False)
    ax.text(.06, .88, r"incremental $\Delta R^2$", color="#D55E00",
            transform=ax.transAxes, fontsize=6.7)
    ax.text(.46, .72, "corpus-realised classes", color="#6A51A3",
            transform=ax.transAxes, fontsize=6.7)

    fig.text(.5, .005,
             "All values are descriptive in-sample associations; feature screening and evaluation use the same corpus.",
             ha="center", fontsize=6.8)
    fig.tight_layout(w_pad=1.5, rect=(0, .045, 1, 1))
    save(fig, "Figure_06_Incremental_Predictive_Information")


def method_overview():
    # Dedicated, original vector artwork; the old boxed diagram is archived.
    from scripts.draw_sparc_architecture import draw
    draw()

if __name__ == "__main__":
    method_overview(); generalization(); temporal(); decomposition()
    incremental_information()
