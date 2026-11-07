"""
Fig. 1: Homogeneous passing network visualization.
Fig. 2: Heterogeneous network schema illustration.
Fig. 4: Event flow -> graph construction diagram.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import networkx as nx

from .style import apply_paper_style, SIDE_COLORS, FIGURE_WIDTH_DOUBLE


def plot_passing_networks(
    G_home: nx.DiGraph,
    G_away: nx.DiGraph,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig. 1: Side-by-side passing networks, analogous to original paper Fig. 1.
    Node size proportional to total passes. Edge width proportional to weight.
    """
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_DOUBLE, 5.0))

    for ax, G, team_name, color in zip(
        axes,
        [G_home, G_away],
        [home_team_name, away_team_name],
        [SIDE_COLORS["home"], SIDE_COLORS["away"]],
    ):
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, f"No data for {team_name}", ha="center", transform=ax.transAxes)
            ax.set_title(team_name)
            continue

        pos = nx.spring_layout(G, seed=42, k=2.0)

        # Node sizes: proportional to out-degree (total passes)
        node_sizes = [
            max(50, G.out_degree(n, weight="weight") * 15)
            for n in G.nodes()
        ]

        # Edge widths: proportional to weight (capped)
        max_weight = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
        edge_widths = [
            max(0.3, G[u][v].get("weight", 1) / max_weight * 4)
            for u, v in G.edges()
        ]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=color, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color="gray", alpha=0.6,
                               arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.1")
        ax.set_title(team_name, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_heterogeneous_schema(output_path: Optional[str] = None) -> plt.Figure:
    """
    Fig. 2: Schematic illustration of heterogeneous graph structure.
    Shows node types (home/away) and edge types (pass/adversarial/turnover).
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE * 0.8, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Home team nodes (left side, blue circles)
    home_positions = [(2, 6), (2, 4), (2, 2)]
    home_labels = ["H1", "H2", "H3"]
    for (x, y), label in zip(home_positions, home_labels):
        circle = plt.Circle((x, y), 0.4, color=SIDE_COLORS["home"], zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)

    # Away team nodes (right side, orange circles)
    away_positions = [(8, 6), (8, 4), (8, 2)]
    away_labels = ["A1", "A2", "A3"]
    for (x, y), label in zip(away_positions, away_labels):
        circle = plt.Circle((x, y), 0.4, color=SIDE_COLORS["away"], zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2, color, style="-", lw=1.5, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                linestyle=style,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=2,
        )

    # Pass edges (home -> home, solid blue)
    draw_arrow(ax, 2, 5.6, 2, 4.4, SIDE_COLORS["home"], lw=2)
    draw_arrow(ax, 2, 3.6, 2, 2.4, SIDE_COLORS["home"], lw=2)
    ax.text(1.3, 5.0, "pass", fontsize=7, color=SIDE_COLORS["home"], ha="center")

    # Pass edges (away -> away, solid orange)
    draw_arrow(ax, 8, 5.6, 8, 4.4, SIDE_COLORS["away"], lw=2)
    draw_arrow(ax, 8, 3.6, 8, 2.4, SIDE_COLORS["away"], lw=2)

    # Adversarial edge (away -> home, red dashed)
    draw_arrow(ax, 7.6, 4.1, 2.4, 3.9, "#E84855", style="dashed", lw=1.5)
    ax.text(5, 4.4, "adversarial\n(intercept/tackle)", fontsize=7,
            color="#E84855", ha="center")

    # Turnover edge (home -> away, green dotted)
    ax.annotate(
        "", xy=(7.6, 2.1), xytext=(2.4, 1.9),
        arrowprops=dict(
            arrowstyle="->",
            color="#2A9D8F",
            lw=1.5,
            linestyle="dotted",
            connectionstyle="arc3,rad=-0.2",
        ),
    )
    ax.text(5, 1.4, "turnover\n(miscontrol/dispossessed)", fontsize=7,
            color="#2A9D8F", ha="center")

    # Team labels
    ax.text(2, 7.2, "Home team", ha="center", fontsize=9,
            color=SIDE_COLORS["home"], fontweight="bold")
    ax.text(8, 7.2, "Away team", ha="center", fontsize=9,
            color=SIDE_COLORS["away"], fontweight="bold")

    # Legend
    legend_elements = [
        mlines.Line2D([0], [0], color=SIDE_COLORS["home"], lw=2, label="Pass (same team)"),
        mlines.Line2D([0], [0], color="#E84855", lw=1.5, linestyle="dashed",
                      label="Adversarial (cross-team)"),
        mlines.Line2D([0], [0], color="#2A9D8F", lw=1.5, linestyle="dotted",
                      label="Turnover"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=7,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_title("Heterogeneous graph structure", fontsize=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def plot_event_flow_diagram(output_path: Optional[str] = None) -> plt.Figure:
    """
    Fig. 4: Event flow -> graph construction diagram.
    Shows how StatsBomb event stream maps to graph edges.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Event sequence boxes (top row)
    events = [
        ("Pass\n(H1→H2)", SIDE_COLORS["home"]),
        ("Pass\n(H2→H3)", SIDE_COLORS["home"]),
        ("Miscontrol\n(H3)", "#F4A261"),
        ("BallRecovery\n(A1)", SIDE_COLORS["away"]),
        ("Pass\n(A1→A2)", SIDE_COLORS["away"]),
    ]

    box_y = 4.5
    box_w, box_h = 1.8, 0.8
    x_starts = [0.5 + i * 2.3 for i in range(len(events))]

    for (label, color), x in zip(events, x_starts):
        rect = mpatches.FancyBboxPatch(
            (x, box_y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05",
            linewidth=1, edgecolor=color, facecolor=color, alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, box_y, label, ha="center", va="center",
                fontsize=7, color="black")

        # Arrow to next event
        if x != x_starts[-1]:
            ax.annotate(
                "", xy=(x + box_w + 0.4, box_y), xytext=(x + box_w + 0.05, box_y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            )

    # possession boundary
    ax.annotate(
        "", xy=(x_starts[2] + box_w / 2, box_y - 0.5),
        xytext=(x_starts[2] + box_w / 2, box_y - 1.2),
        arrowprops=dict(arrowstyle="->", color="#F4A261", lw=1.5),
    )
    ax.text(x_starts[2] + box_w / 2, box_y - 1.5,
            "possession\nboundary", ha="center", fontsize=7, color="#F4A261")

    # Graph edge labels (bottom row)
    graph_edges = [
        ("E_pass\nH1→H2", x_starts[0], SIDE_COLORS["home"]),
        ("E_pass\nH2→H3", x_starts[1], SIDE_COLORS["home"]),
        ("E_turn\nH3→A1", x_starts[2], "#2A9D8F"),
        ("E_pass\nA1→A2", x_starts[4], SIDE_COLORS["away"]),
    ]

    edge_y = 1.5
    for label, x, color in graph_edges:
        rect = mpatches.FancyBboxPatch(
            (x, edge_y - 0.4), box_w, 0.8,
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=color, facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, edge_y, label, ha="center", va="center",
                fontsize=7, color=color)

    ax.text(0.2, edge_y, "Graph\nedges:", ha="left", va="center",
            fontsize=8, fontweight="bold")

    ax.text(6, 0.3, "possession_id used to segment attack units; "
            "counterpress field marks 5-sec windows",
            ha="center", fontsize=7, style="italic", color="gray")

    ax.set_title("StatsBomb event stream → heterogeneous graph construction",
                 fontsize=9, fontweight="bold")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    return fig
