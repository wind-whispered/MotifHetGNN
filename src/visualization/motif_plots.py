"""
Fig. 3: Pitch zone diagram.
Fig. 6: Triadic motif illustrations (13 canonical types, k=3).
Fig. 7: Higher-order homogeneous motif illustrations.
Fig. 8: Heterogeneous motif illustrations.
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx

from .style import apply_paper_style, SIDE_COLORS, FIGURE_WIDTH_DOUBLE, FIGURE_WIDTH_SINGLE


def _draw_small_graph(
    ax: plt.Axes,
    edges: List[tuple],
    node_labels: Optional[Dict] = None,
    node_colors: Optional[Dict] = None,
    edge_colors: Optional[Dict] = None,
    title: str = "",
    pos: Optional[Dict] = None,
) -> None:
    """Helper to draw a small labeled graph on a given axes."""
    G = nx.DiGraph()
    nodes = set()
    for edge in edges:
        nodes.update(edge[:2])
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if pos is None:
        if len(nodes) == 3:
            n = sorted(nodes)
            pos = {n[0]: (0, 0), n[1]: (1, 0), n[2]: (0.5, 0.866)}
        else:
            pos = nx.spring_layout(G, seed=42)

    default_node_color = "#4A90D9"
    node_color_list = [
        (node_colors or {}).get(n, default_node_color) for n in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list,
                           node_size=300, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels=node_labels or {n: str(n) for n in G.nodes()},
                            font_size=7, font_color="white")

    # Draw edges with colors
    for i, (u, v, *_) in enumerate(edges):
        color = (edge_colors or {}).get((u, v), "gray")
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            edge_color=color, width=1.5, alpha=0.8,
            arrows=True, arrowsize=12,
            connectionstyle="arc3,rad=0.15",
        )

    if title:
        ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")


def plot_motif_vocabulary_overview(
    census_stats: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 1 (single-column): complete directed motif vocabulary and all-order
    census statistics.

    Layout (top to bottom):
      Row 0      : k=2 — all 2 dyadic classes drawn as digraphs
      Rows 1–3   : k=3 — all 13 triadic classes drawn as digraphs (5 per row)
      Bottom panel: k=2..7 census statistics — realised class count and mean
                    occupancy per class, conveying the scale of the multi-order
                    vocabulary without drawing the exponentially many higher-order
                    subgraphs individually.

    Parameters
    ----------
    census_stats : dict, optional
        {'size': {k: {'home': float, 'away': float}},
         'summary': [{'k': int, 'classes_obs': int, 'occupancy': float}]}
        If None the bottom statistics panel is omitted.
    """
    from matplotlib.gridspec import GridSpec

    apply_paper_style()

    # ── complete k=2 vocabulary (2 classes) ──────────────────────────────────
    DYADS = [
        ("one-way\n$(i\\!\\to\\!j)$",      [(0, 1)]),
        ("reciprocal\n$(i\\!\\leftrightarrow\\!j)$", [(0, 1), (1, 0)]),
    ]
    POS2 = {0: (-0.5, 0), 1: (0.5, 0)}

    # ── complete k=3 vocabulary (13 classes, sorted by Milo id) ─────────────
    TRIADS_ALL = [
        (6,   [(0, 1), (0, 2)]),
        (12,  [(0, 1), (1, 2)]),
        (14,  [(0, 1), (1, 2), (0, 2)]),
        (36,  [(0, 1), (2, 1)]),
        (38,  [(0, 1), (2, 0), (2, 1)]),
        (46,  [(0, 1), (1, 0), (0, 2)]),
        (74,  [(0, 1), (1, 0), (0, 2), (2, 0)]),
        (78,  [(0, 1), (1, 0), (1, 2), (2, 1)]),
        (98,  [(0, 1), (1, 2), (2, 0)]),
        (102, [(0, 1), (1, 2), (2, 0), (0, 2)]),
        (108, [(0, 1), (1, 0), (1, 2), (2, 0)]),
        (110, [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)]),
        (238, [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]),
    ]
    POS3 = {0: (0, 0), 1: (1, 0), 2: (0.5, 0.866)}

    N_COLS = 5
    has_stats = census_stats is not None
    n_graph_rows = 4   # 1 (k=2) + 3 (k=3 in 5-col grid)
    row_h = 1.3
    stats_h = 2.1 if has_stats else 0.0
    fig_h = n_graph_rows * row_h + stats_h + 0.3

    fig = plt.figure(figsize=(FIGURE_WIDTH_SINGLE, fig_h))

    if has_stats:
        gs = GridSpec(n_graph_rows + 1, N_COLS, figure=fig,
                      height_ratios=[row_h] * n_graph_rows + [stats_h],
                      hspace=0.55, wspace=0.12)
    else:
        gs = GridSpec(n_graph_rows, N_COLS, figure=fig,
                      height_ratios=[row_h] * n_graph_rows,
                      hspace=0.55, wspace=0.12)

    # ── Row 0: k=2 dyads (2 panels centred, rest blank) ─────────────────────
    for col, (title, edges) in enumerate(DYADS):
        ax = fig.add_subplot(gs[0, col + 1])
        _draw_small_graph(ax, edges, node_labels={0: "A", 1: "B"},
                          pos=POS2, title=title)
    for col in [0] + list(range(len(DYADS) + 1, N_COLS)):
        fig.add_subplot(gs[0, col]).axis("off")
    # Row label
    fig.text(0.005, 1 - row_h / (2 * fig_h), "$k=2$",
             va="center", ha="left", fontsize=8, fontweight="bold",
             transform=fig.transFigure)

    # ── Rows 1–3: k=3 triads (13 panels in 5×3 grid) ────────────────────────
    for idx, (mid, edges) in enumerate(TRIADS_ALL):
        row = 1 + idx // N_COLS
        col = idx % N_COLS
        ax = fig.add_subplot(gs[row, col])
        _draw_small_graph(ax, edges, node_labels={0: "A", 1: "B", 2: "C"},
                          pos=POS3, title=f"$\\mathit{{{mid}}}$")
    # blank unused cells in the k=3 block
    for idx in range(len(TRIADS_ALL), 3 * N_COLS):
        row = 1 + idx // N_COLS
        col = idx % N_COLS
        fig.add_subplot(gs[row, col]).axis("off")
    # Row label (centred vertically across the 3 k=3 rows)
    fig.text(0.005, 1 - (row_h + 1.5 * row_h) / fig_h, "$k=3$",
             va="center", ha="left", fontsize=8, fontweight="bold",
             transform=fig.transFigure)

    # ── Bottom panel: census statistics for k=2..7 ──────────────────────────
    if has_stats:
        ax_s = fig.add_subplot(gs[n_graph_rows, :])
        summ = census_stats.get("summary", [])
        size_d = census_stats.get("size", {})

        # Build arrays
        ks   = sorted(int(r["k"]) for r in summ)
        cls  = [next(r["classes_obs"] for r in summ if int(r["k"]) == k) for k in ks]
        occ  = [next(r["occupancy"]   for r in summ if int(r["k"]) == k) for k in ks]

        ax_r = ax_s.twinx()
        x = np.arange(len(ks))
        w = 0.35
        ax_s.bar(x - w / 2, cls, width=w, color="#5B9BD5", alpha=0.8,
                 label="Realised classes")
        # census size (mean home+away)
        n_k = [(size_d.get(k, {}).get("home", 0) +
                size_d.get(k, {}).get("away", 0)) / 2
               for k in ks]
        ax_s.bar(x + w / 2, n_k, width=w, color="#ED7D31", alpha=0.8,
                 label="Mean $N(k)$")
        ax_r.plot(x, occ, marker="D", ms=3.5, color="#C00000", lw=1.3,
                  label="Occupancy / class")
        ax_r.set_yscale("log")
        ax_r.set_ylabel("Occupancy / class (log)", fontsize=6.5, color="#C00000")
        ax_r.tick_params(axis="y", labelcolor="#C00000", labelsize=6)
        ax_s.set_yscale("log")
        ax_s.set_xticks(x)
        ax_s.set_xticklabels([f"$k={k}$" for k in ks], fontsize=7)
        ax_s.set_ylabel("Count (log scale)", fontsize=6.5)
        ax_s.tick_params(axis="y", labelsize=6)
        h1, l1 = ax_s.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax_s.legend(h1 + h2, l1 + l2, loc="upper right",
                    frameon=False, fontsize=6)

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_triadic_motifs(output_path: Optional[str] = None) -> plt.Figure:
    """
    Fig. 6: All 13 canonical 3-motifs (directed).
    Layout: 4 rows x 4 cols (last cell empty).
    """
    apply_paper_style()

    # Canonical 3-motif edge lists (using Milo labeling scheme)
    TRIADIC_MOTIFS = {
        6:   [(0, 1), (0, 2)],
        12:  [(0, 1), (1, 2)],
        14:  [(0, 1), (1, 2), (0, 2)],
        36:  [(0, 1), (2, 1)],
        38:  [(0, 1), (2, 0), (2, 1)],
        46:  [(0, 1), (1, 0), (0, 2)],
        74:  [(0, 1), (1, 0), (0, 2), (2, 0)],
        78:  [(0, 1), (1, 0), (1, 2), (2, 1)],
        98:  [(0, 1), (1, 2), (2, 0)],
        102: [(0, 1), (1, 2), (2, 0), (0, 2)],
        108: [(0, 1), (1, 0), (1, 2), (2, 0)],
        110: [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)],
        238: [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],
    }

    n_motifs = len(TRIADIC_MOTIFS)
    n_cols = 5
    n_rows = (n_motifs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_WIDTH_DOUBLE, n_rows * 1.5))
    axes_flat = axes.flatten()

    fixed_pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 0.866)}

    for ax_idx, (motif_id, edges) in enumerate(sorted(TRIADIC_MOTIFS.items())):
        ax = axes_flat[ax_idx]
        _draw_small_graph(
            ax, edges,
            node_labels={0: "A", 1: "B", 2: "C"},
            pos=fixed_pos,
            title=f"ID={motif_id}",
        )

    # Hide unused subplots
    for ax_idx in range(n_motifs, len(axes_flat)):
        axes_flat[ax_idx].axis("off")

    fig.suptitle("13 canonical 3-motifs (directed)", y=1.01, fontsize=10)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_higher_order_motifs(
    motif_df: pd.DataFrame,
    k_values: List[int] = [4, 5],
    top_n_per_k: int = 4,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 7: Example higher-order homogeneous motifs.
    Shows the top N most frequent motifs at each order k.
    """
    import pandas as pd
    apply_paper_style()

    n_cols = top_n_per_k
    n_rows = len(k_values)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGURE_WIDTH_DOUBLE, n_rows * 1.7)
    )
    if n_rows == 1:
        axes = [axes]

    for row_idx, k in enumerate(k_values):
        k_df = motif_df[motif_df["motif_order_k"] == k] if "motif_order_k" in motif_df.columns else motif_df
        top_motifs = k_df.groupby("motif_id")["count"].mean().nlargest(n_cols)

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]

            if col_idx >= len(top_motifs):
                ax.axis("off")
                continue

            motif_id = int(top_motifs.index[col_idx])
            mean_count = top_motifs.iloc[col_idx]

            # Decode the *actual* canonical adjacency matrix from the Milo id:
            # cell (i,j) is bit (k*k - 1 - (i*k + j)) of the id.
            nbits = k * k
            edges = [
                (i, j)
                for i in range(k) for j in range(k)
                if i != j and (motif_id >> (nbits - 1 - (i * k + j))) & 1
            ]
            nodes = list(range(k))
            pos = {
                i: (np.cos(2 * np.pi * i / k + np.pi / 2),
                    np.sin(2 * np.pi * i / k + np.pi / 2))
                for i in nodes
            }
            _draw_small_graph(
                ax, edges,
                node_labels={i: chr(65 + i) for i in nodes},
                pos=pos,
                title=f"$k$={k}, id={motif_id}\n" + r"$\bar n$=" + f"{mean_count:.1f}",
            )

        # Row label
        fig.text(
            0.01, 1 - (row_idx + 0.5) / n_rows,
            f"k={k}",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
        )

    fig.suptitle("Top higher-order homogeneous motifs", y=1.02, fontsize=10)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_heterogeneous_motif_examples(output_path: Optional[str] = None) -> plt.Figure:
    """
    Fig. 8: Key heterogeneous motif examples with tactical labels.
    """
    apply_paper_style()

    # Pre-defined heterogeneous motif examples
    examples = [
        {
            "title": "H-01: Counterpress\ntrigger (k=3)",
            "nodes": [0, 1, 2],
            "edges": [(1, 0), (2, 1)],   # A intercepts H pass -> A counter-pass
            "node_colors": {0: SIDE_COLORS["home"], 1: SIDE_COLORS["away"], 2: SIDE_COLORS["away"]},
            "edge_colors": {(1, 0): "#E84855", (2, 1): SIDE_COLORS["away"]},
            "node_labels": {0: "H1", 1: "A1", 2: "A2"},
        },
        {
            "title": "H-02: Dispossession\nreorg (k=3)",
            "nodes": [0, 1, 2],
            "edges": [(0, 1), (1, 2)],   # H->A turnover, then A-A
            "node_colors": {0: SIDE_COLORS["home"], 1: SIDE_COLORS["away"], 2: SIDE_COLORS["away"]},
            "edge_colors": {(0, 1): "#2A9D8F", (1, 2): SIDE_COLORS["away"]},
            "node_labels": {0: "H1", 1: "A1", 2: "A2"},
        },
        {
            "title": "H-03: Triangle\npass chain (k=3)",
            "nodes": [0, 1, 2],
            "edges": [(0, 1), (1, 2), (2, 0)],
            "node_colors": {0: SIDE_COLORS["home"], 1: SIDE_COLORS["home"], 2: SIDE_COLORS["home"]},
            "edge_colors": {(0, 1): SIDE_COLORS["home"], (1, 2): SIDE_COLORS["home"], (2, 0): SIDE_COLORS["home"]},
            "node_labels": {0: "H1", 1: "H2", 2: "H3"},
        },
        {
            "title": "H-04: Interception\nto counter (k=4)",
            "nodes": [0, 1, 2, 3],
            "edges": [(0, 1), (1, 2), (2, 0), (2, 3)],
            "node_colors": {
                0: SIDE_COLORS["home"], 1: SIDE_COLORS["home"],
                2: SIDE_COLORS["away"], 3: SIDE_COLORS["away"]
            },
            "edge_colors": {
                (0, 1): SIDE_COLORS["home"],
                (1, 2): "#E84855",
                (2, 0): "#2A9D8F",
                (2, 3): SIDE_COLORS["away"],
            },
            "node_labels": {0: "H1", 1: "H2", 2: "A1", 3: "A2"},
        },
        {
            "title": "H-05: High-press\nchain (k=4)",
            "nodes": [0, 1, 2, 3],
            "edges": [(3, 2), (2, 1), (1, 0), (0, 3)],
            "node_colors": {
                0: SIDE_COLORS["away"], 1: SIDE_COLORS["away"],
                2: SIDE_COLORS["home"], 3: SIDE_COLORS["home"]
            },
            "edge_colors": {
                (3, 2): "#E84855", (2, 1): "#E84855",
                (1, 0): SIDE_COLORS["away"], (0, 3): SIDE_COLORS["away"],
            },
            "node_labels": {0: "A1", 1: "A2", 2: "H1", 3: "H2"},
        },
        {
            "title": "H-06: Under-pressure\npass chain (k=3)",
            "nodes": [0, 1, 2],
            "edges": [(0, 1), (1, 2)],
            "node_colors": {0: SIDE_COLORS["home"], 1: SIDE_COLORS["home"], 2: SIDE_COLORS["home"]},
            "edge_colors": {(0, 1): "#F4A261", (1, 2): "#F4A261"},
            "node_labels": {0: "H1", 1: "H2", 2: "H3"},
            "edge_style": "under_pressure",
        },
    ]
    # Fix syntax: remove trailing ] in edge_colors
    examples[5]["edge_colors"] = {(0, 1): "#F4A261", (1, 2): "#F4A261"}

    n_examples = len(examples)
    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_WIDTH_DOUBLE, n_rows * 2.8))
    axes_flat = axes.flatten() if n_rows > 1 else list(axes)

    for idx, (ex, ax) in enumerate(zip(examples, axes_flat)):
        nodes = ex["nodes"]
        k = len(nodes)
        pos = {
            n: (np.cos(2 * np.pi * i / k), np.sin(2 * np.pi * i / k))
            for i, n in enumerate(nodes)
        }
        _draw_small_graph(
            ax,
            edges=ex["edges"],
            node_labels=ex.get("node_labels"),
            node_colors=ex.get("node_colors"),
            edge_colors=ex.get("edge_colors"),
            pos=pos,
            title=ex["title"],
        )

    for idx in range(n_examples, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color=SIDE_COLORS["home"], label="Home player"),
        mpatches.Patch(color=SIDE_COLORS["away"], label="Away player"),
        plt.Line2D([0], [0], color=SIDE_COLORS["home"], lw=2, label="Pass edge"),
        plt.Line2D([0], [0], color="#E84855", lw=2, label="Adversarial edge"),
        plt.Line2D([0], [0], color="#2A9D8F", lw=2, label="Turnover edge"),
        plt.Line2D([0], [0], color="#F4A261", lw=2, label="Under pressure"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Heterogeneous motif examples", y=1.01, fontsize=10)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pitch_zones(output_path: Optional[str] = None) -> plt.Figure:
    """
    Fig. 3: Football pitch with 9-zone spatial partitioning.
    """
    apply_paper_style()

    try:
        from mplsoccer import Pitch
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#2d9e2d", line_color="white")
        fig, ax = pitch.draw(figsize=(FIGURE_WIDTH_DOUBLE * 0.8, 5.5))
    except ImportError:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE * 0.8, 5.5))
        ax.set_facecolor("#2d9e2d")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        # Draw pitch lines
        for x in [0, 120]:
            ax.plot([x, x], [0, 80], "white", lw=2)
        ax.plot([0, 120], [0, 0], "white", lw=2)
        ax.plot([0, 120], [80, 80], "white", lw=2)
        ax.plot([60, 60], [0, 80], "white", lw=1, linestyle="--")

    # Draw zone boundaries
    for x in [40, 80]:
        ax.axvline(x, color="yellow", linewidth=2, linestyle="--", alpha=0.8)
    for y in [26.67, 53.33]:
        ax.axhline(y, color="yellow", linewidth=2, linestyle="--", alpha=0.8)

    # Zone labels
    zone_centers = {
        "Defensive\nLeft":   (20, 13.3),
        "Defensive\nCenter": (20, 40.0),
        "Defensive\nRight":  (20, 66.7),
        "Middle\nLeft":      (60, 13.3),
        "Middle\nCenter":    (60, 40.0),
        "Middle\nRight":     (60, 66.7),
        "Attacking\nLeft":   (100, 13.3),
        "Attacking\nCenter": (100, 40.0),
        "Attacking\nRight":  (100, 66.7),
    }
    for label, (cx, cy) in zone_centers.items():
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7, color="white", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))

    ax.set_title("Pitch spatial partitioning (9 zones)\nStatsBomb coordinates: x∈[0,120], y∈[0,80]",
                 fontsize=9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig

