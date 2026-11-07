import pytest
import torch
from torch_geometric.data import HeteroData

from src.gnn.model import HeteroFootballGNN


def _cutoff_graph(include_score=True):
    graph = HeteroData()
    generator = torch.Generator().manual_seed(11)
    graph["home_player"].x = torch.randn(3, 39, generator=generator)
    graph["away_player"].x = torch.randn(3, 39, generator=generator)
    relations = (
        ("home_player", "pass", "home_player"),
        ("away_player", "pass", "away_player"),
        ("home_player", "adversarial", "away_player"),
        ("away_player", "adversarial", "home_player"),
        ("home_player", "turnover", "away_player"),
        ("away_player", "turnover", "home_player"),
    )
    for relation in relations:
        graph[relation].edge_index = torch.tensor(
            [[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    if include_score:
        graph.current_home_score = torch.tensor([1.0])
        graph.current_away_score = torch.tensor([0.0])
        graph.current_goal_diff = torch.tensor([1.0])
    return graph


def test_score_aware_heterosage_has_nested_score_input():
    torch.manual_seed(5)
    model = HeteroFootballGNN(
        hidden_dim=8, num_layers=1, dropout=0.0,
        use_current_score=True,
    ).eval()
    graph = _cutoff_graph()

    logits_a, regression_a = model(graph)
    graph.current_home_score = torch.tensor([0.0])
    graph.current_away_score = torch.tensor([1.0])
    graph.current_goal_diff = torch.tensor([-1.0])
    logits_b, regression_b = model(graph)

    assert model.cls_head[0].in_features == 20
    assert logits_a.shape == logits_b.shape == (1, 3)
    assert regression_a.shape == regression_b.shape == (1, 1)
    assert not torch.allclose(logits_a, logits_b)
    assert not torch.allclose(regression_a, regression_b)


def test_score_aware_heterosage_rejects_full_graph_without_cutoff_score():
    model = HeteroFootballGNN(
        hidden_dim=8, num_layers=1, dropout=0.0,
        use_current_score=True,
    ).eval()
    with pytest.raises(ValueError, match="cutoff score attributes"):
        model(_cutoff_graph(include_score=False))
