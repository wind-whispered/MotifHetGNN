"""Tests for motif enumeration utilities."""
import pytest
import networkx as nx

from src.motifs.homogeneous_enum import (
    determine_k_star, build_order_summary, build_motif_records,
)
from src.motifs.randomization import degree_preserving_randomize
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Randomization tests
# ---------------------------------------------------------------------------

def _make_ring_graph(n: int) -> nx.DiGraph:
    """Create a directed ring graph: 0->1->2->...->n-1->0."""
    G = nx.DiGraph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=1)
    return G


def test_randomization_preserves_degree_sequence():
    G = _make_ring_graph(10)
    G_rand = degree_preserving_randomize(G, seed=42)

    # Check degree sequences match
    orig_in = sorted(dict(G.in_degree()).values())
    orig_out = sorted(dict(G.out_degree()).values())
    rand_in = sorted(dict(G_rand.in_degree()).values())
    rand_out = sorted(dict(G_rand.out_degree()).values())

    assert orig_in == rand_in
    assert orig_out == rand_out


def test_randomization_preserves_node_count():
    G = _make_ring_graph(8)
    G_rand = degree_preserving_randomize(G, seed=0)
    assert G.number_of_nodes() == G_rand.number_of_nodes()


def test_randomization_preserves_edge_count():
    G = _make_ring_graph(8)
    G_rand = degree_preserving_randomize(G, seed=0)
    assert G.number_of_edges() == G_rand.number_of_edges()


def test_randomization_no_self_loops():
    G = _make_ring_graph(10)
    G_rand = degree_preserving_randomize(G, seed=42)
    for u, v in G_rand.edges():
        assert u != v, f"Self-loop found: {u} -> {v}"


# ---------------------------------------------------------------------------
# k* determination tests
# ---------------------------------------------------------------------------

def test_determine_k_star_basic():
    """k*=3 when only k=3 motifs present."""
    records = [
        {"match_id": 1, "team_side": "home", "motif_order_k": 3, "motif_id": 12, "count": 5.0},
        {"match_id": 1, "team_side": "home", "motif_order_k": 4, "motif_id": 1, "count": 0.0},
        {"match_id": 1, "team_side": "away", "motif_order_k": 3, "motif_id": 12, "count": 4.0},
        {"match_id": 1, "team_side": "away", "motif_order_k": 4, "motif_id": 1, "count": 0.0},
    ]
    motif_df = pd.DataFrame(records)
    k_star = determine_k_star(motif_df, zero_threshold=0.01)
    assert k_star["home"] == 3
    assert k_star["away"] == 3


def test_determine_k_star_higher():
    """k*=4 when k=4 motifs also present."""
    records = [
        {"match_id": i, "team_side": "home", "motif_order_k": 3, "motif_id": 12, "count": 5.0}
        for i in range(10)
    ] + [
        {"match_id": i, "team_side": "home", "motif_order_k": 4, "motif_id": 1, "count": 2.0}
        for i in range(10)
    ] + [
        {"match_id": i, "team_side": "home", "motif_order_k": 5, "motif_id": 1, "count": 0.0}
        for i in range(10)
    ] + [
        {"match_id": i, "team_side": "away", "motif_order_k": 3, "motif_id": 12, "count": 4.0}
        for i in range(10)
    ]
    motif_df = pd.DataFrame(records)
    k_star = determine_k_star(motif_df, zero_threshold=0.01)
    assert k_star["home"] == 4
    assert k_star.get("away", 3) == 3


# ---------------------------------------------------------------------------
# Order summary tests
# ---------------------------------------------------------------------------

def test_build_order_summary_columns():
    records = [
        {"match_id": 1, "team_side": "home", "motif_order_k": 3, "motif_id": 12, "count": 5.0},
        {"match_id": 2, "team_side": "home", "motif_order_k": 3, "motif_id": 12, "count": 3.0},
        {"match_id": 1, "team_side": "away", "motif_order_k": 3, "motif_id": 14, "count": 2.0},
    ]
    df = pd.DataFrame(records)
    summary = build_order_summary(df)
    assert "motif_order_k" in summary.columns
    assert "team_side" in summary.columns
    assert "observed_types" in summary.columns
    assert "mean_total_count" in summary.columns


def test_build_order_summary_values():
    records = [
        {"match_id": 1, "team_side": "home", "motif_order_k": 3, "motif_id": 12, "count": 5.0},
        {"match_id": 1, "team_side": "home", "motif_order_k": 3, "motif_id": 14, "count": 3.0},
        {"match_id": 2, "team_side": "home", "motif_order_k": 3, "motif_id": 12, "count": 4.0},
    ]
    df = pd.DataFrame(records)
    summary = build_order_summary(df)
    home_k3 = summary[(summary["team_side"] == "home") & (summary["motif_order_k"] == 3)]
    # 2 distinct motif IDs observed
    assert home_k3["observed_types"].iloc[0] == 2
    # match 1: 5+3=8, match 2: 4 -> mean = (8+4)/2 = 6
    assert abs(home_k3["mean_total_count"].iloc[0] - 6.0) < 1e-6


# ---------------------------------------------------------------------------
# Motif record building
# ---------------------------------------------------------------------------

def test_build_motif_records_flat():
    order_results = {
        3: {12: 5.0, 14: 3.0},
        4: {1: 2.0},
    }
    records = build_motif_records(match_id=1001, team_side="home", order_results=order_results)
    assert len(records) == 3
    ks = [r["motif_order_k"] for r in records]
    assert 3 in ks
    assert 4 in ks
    counts = {r["motif_id"]: r["count"] for r in records if r["motif_order_k"] == 3}
    assert counts[12] == 5.0
    assert counts[14] == 3.0
