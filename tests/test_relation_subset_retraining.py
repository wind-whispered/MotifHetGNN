import torch
from torch_geometric.data import HeteroData

from scripts.run_relation_subset_retraining import _restrict_relations


def _three_relation_graph():
    graph = HeteroData()
    graph["home_player"].x = torch.zeros(3, 39)
    graph["away_player"].x = torch.zeros(3, 39)
    edge_types = [
        ("home_player", "pass", "home_player"),
        ("away_player", "pass", "away_player"),
        ("home_player", "adversarial", "away_player"),
        ("away_player", "adversarial", "home_player"),
        ("home_player", "turnover", "away_player"),
        ("away_player", "turnover", "home_player"),
    ]
    for edge_type in edge_types:
        graph[edge_type].edge_index = torch.tensor([[0], [1]])
        graph[edge_type].edge_attr = torch.ones(1, 2)
    return graph


def test_relation_subset_removes_only_excluded_edges_and_rebuilds_context():
    graph = _three_relation_graph()
    restricted = _restrict_relations(graph, frozenset({"pass"}))

    assert all(restricted[edge_type].edge_index.size(1) ==
               (1 if edge_type[1] == "pass" else 0)
               for edge_type in restricted.edge_types)
    assert all(graph[edge_type].edge_index.size(1) == 1
               for edge_type in graph.edge_types)
    assert restricted["home_player"].motif_x.shape == (3, 7)
    assert restricted["away_player"].motif_x.shape == (3, 7)
