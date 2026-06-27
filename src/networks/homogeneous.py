"""
Task 2: Homogeneous passing network construction and topological statistics.
"""
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def build_passing_network(
    match_id: int,
    team_id: int,
    pass_df: pd.DataFrame,
    w0: int = 0,
) -> nx.DiGraph:
    """
    Build a directed weighted passing network for one team in one match.

    Args:
        match_id: match identifier
        team_id: team whose passes to use
        pass_df: DataFrame of successful passes
        w0: minimum edge weight threshold (edges with weight <= w0 removed)

    Returns:
        Directed weighted graph; edge attribute 'weight' = pass count.
    """
    subset = pass_df[
        (pass_df["match_id"] == match_id) & (pass_df["team_id"] == team_id)
    ].copy()

    G = nx.DiGraph()

    if subset.empty:
        return G

    # Count passes per (passer, recipient) pair
    edge_counts = (
        subset.groupby(["player_id", "recipient_id"])
        .size()
        .reset_index(name="weight")
    )

    for _, row in edge_counts.iterrows():
        u = int(row["player_id"])
        v = int(row["recipient_id"])
        w = int(row["weight"])
        if w > w0:
            G.add_edge(u, v, weight=w)

    return G


def build_all_networks_for_match(
    match_id: int,
    home_team_id: int,
    away_team_id: int,
    pass_df: pd.DataFrame,
    w0: int = 2,
) -> Tuple[nx.DiGraph, nx.DiGraph]:
    """Build home and away passing networks for one match."""
    G_home = build_passing_network(match_id, home_team_id, pass_df, w0)
    G_away = build_passing_network(match_id, away_team_id, pass_df, w0)
    return G_home, G_away


# ---------------------------------------------------------------------------
# Network metrics (replicating original paper)
# ---------------------------------------------------------------------------
def network_density(G: nx.DiGraph) -> float:
    """D = L / (N*(N-1))"""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    return G.number_of_edges() / (n * (n - 1))


def network_transitivity(G: nx.DiGraph) -> float:
    """Global clustering coefficient (transitivity)."""
    return nx.transitivity(G)


def pass_diversity(G: nx.DiGraph) -> float:
    """
    Average node pass diversity phi_i = -sum(p_ij * log(p_ij)) / log(k_i).
    Measures how evenly a player distributes passes across receivers.
    """
    diversities = []
    for node in G.nodes():
        successors = list(G.successors(node))
        if len(successors) < 2:
            continue
        weights = np.array([G[node][s]["weight"] for s in successors], dtype=float)
        total = weights.sum()
        if total == 0:
            continue
        probs = weights / total
        # Shannon entropy normalized by log(k_i)
        ent = entropy(probs)  # base e
        max_ent = np.log(len(successors))
        if max_ent > 0:
            diversities.append(ent / max_ent)
    return float(np.mean(diversities)) if diversities else 0.0


def mean_outdegree(G: nx.DiGraph) -> float:
    """Mean out-degree centrality."""
    if G.number_of_nodes() == 0:
        return 0.0
    out_degrees = [d for _, d in G.out_degree()]
    return float(np.mean(out_degrees))


def mean_betweenness(G: nx.DiGraph) -> float:
    """Mean betweenness centrality."""
    if G.number_of_nodes() < 2:
        return 0.0
    bc = nx.betweenness_centrality(G, normalized=True)
    return float(np.mean(list(bc.values())))


def mean_eigenvector(G: nx.DiGraph) -> float:
    """Mean eigenvector centrality."""
    if G.number_of_nodes() < 2:
        return 0.0
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        return float(np.mean(list(ec.values())))
    except Exception:
        return 0.0


def compute_network_stats(
    match_id: int,
    team_side: str,
    G: nx.DiGraph,
    w0: int,
) -> dict:
    """Compute all network-level metrics for one graph."""
    return {
        "match_id": match_id,
        "team_side": team_side,
        "w0": w0,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": network_density(G),
        "transitivity": network_transitivity(G),
        "pass_diversity": pass_diversity(G),
        "mean_outdegree": mean_outdegree(G),
        "mean_betweenness": mean_betweenness(G),
        "mean_eigenvector": mean_eigenvector(G),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_network(G: nx.DiGraph, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_network(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def network_to_edge_list(G: nx.DiGraph, path: str) -> None:
    """
    Write edge list to file in format required by gtrieScanner:
    one edge per line: 'u v'
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
