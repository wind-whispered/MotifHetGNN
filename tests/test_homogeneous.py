"""Tests for homogeneous network construction and metrics."""
import pytest
import pandas as pd
import numpy as np
import networkx as nx

from src.networks.homogeneous import (
    build_passing_network,
    network_density,
    network_transitivity,
    pass_diversity,
    mean_outdegree,
    mean_betweenness,
    compute_network_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pass_df():
    """Minimal pass DataFrame for match_id=1, team_id=10."""
    rows = [
        # player 1 -> player 2 (3 times)
        {"match_id": 1, "team_id": 10, "player_id": 1, "recipient_id": 2},
        {"match_id": 1, "team_id": 10, "player_id": 1, "recipient_id": 2},
        {"match_id": 1, "team_id": 10, "player_id": 1, "recipient_id": 2},
        # player 2 -> player 3 (2 times)
        {"match_id": 1, "team_id": 10, "player_id": 2, "recipient_id": 3},
        {"match_id": 1, "team_id": 10, "player_id": 2, "recipient_id": 3},
        # player 3 -> player 1 (1 time)
        {"match_id": 1, "team_id": 10, "player_id": 3, "recipient_id": 1},
        # away team passes (should be ignored)
        {"match_id": 1, "team_id": 20, "player_id": 4, "recipient_id": 5},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def triangle_graph():
    """Simple triangle: 1->2->3->1 with weights."""
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=3)
    G.add_edge(2, 3, weight=2)
    G.add_edge(3, 1, weight=1)
    return G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_passing_network_basic(sample_pass_df):
    G = build_passing_network(1, 10, sample_pass_df, w0=0)
    assert G.number_of_nodes() >= 3
    assert G.has_edge(1, 2)
    assert G.has_edge(2, 3)
    assert G.has_edge(3, 1)
    # Away team should not be in home network
    assert not G.has_node(4)


def test_build_passing_network_w0_filter(sample_pass_df):
    """w0=2 should remove edge 3->1 (weight=1)."""
    G = build_passing_network(1, 10, sample_pass_df, w0=2)
    assert G.has_edge(1, 2)  # weight=3 > 2
    assert G.has_edge(2, 3)  # weight=2 > 2? No: 2 is NOT > 2
    assert not G.has_edge(3, 1)  # weight=1 <= 2


def test_build_passing_network_w0_strict(sample_pass_df):
    """w0=1 should include edge 2->3 (weight=2 > 1) and 3->1 removed (1 <= 1)."""
    G = build_passing_network(1, 10, sample_pass_df, w0=1)
    assert G.has_edge(1, 2)  # weight=3 > 1
    assert G.has_edge(2, 3)  # weight=2 > 1
    assert not G.has_edge(3, 1)  # weight=1 not > 1


def test_network_density_triangle(triangle_graph):
    d = network_density(triangle_graph)
    # 3 nodes, 3 edges: D = 3 / (3*2) = 0.5
    assert abs(d - 0.5) < 1e-6


def test_network_density_empty():
    G = nx.DiGraph()
    assert network_density(G) == 0.0


def test_network_density_single_node():
    G = nx.DiGraph()
    G.add_node(1)
    assert network_density(G) == 0.0


def test_pass_diversity_uniform(triangle_graph):
    """Triangle: each node sends to exactly one receiver -> diversity should be 0
    (only one receiver, log(1)=0 -> undefined; function returns 0 for k_i < 2)."""
    div = pass_diversity(triangle_graph)
    assert div == 0.0  # all nodes have exactly one out-neighbor


def test_pass_diversity_split():
    """Node splits passes evenly between two receivers -> high diversity."""
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=5)
    G.add_edge(1, 3, weight=5)
    div = pass_diversity(G)
    # log(2)/log(2) = 1.0
    assert abs(div - 1.0) < 1e-6


def test_mean_outdegree(triangle_graph):
    """Each node has out-degree 1 -> mean = 1."""
    assert abs(mean_outdegree(triangle_graph) - 1.0) < 1e-6


def test_compute_network_stats_returns_all_keys(triangle_graph):
    stats = compute_network_stats(match_id=1, team_side="home", G=triangle_graph, w0=2)
    expected_keys = [
        "match_id", "team_side", "w0", "n_nodes", "n_edges",
        "density", "transitivity", "pass_diversity",
        "mean_outdegree", "mean_betweenness", "mean_eigenvector",
    ]
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"


def test_edge_weight_in_graph(sample_pass_df):
    G = build_passing_network(1, 10, sample_pass_df, w0=0)
    assert G[1][2]["weight"] == 3
    assert G[2][3]["weight"] == 2
    assert G[3][1]["weight"] == 1
