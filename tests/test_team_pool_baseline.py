import torch
import pytest
from torch_geometric.data import HeteroData

from src.gnn.team_pool import TeamPoolMLP


def _graph():
    graph = HeteroData()
    generator = torch.Generator().manual_seed(7)
    graph["home_player"].x = torch.randn(5, 39, generator=generator)
    graph["away_player"].x = torch.randn(4, 39, generator=generator)
    graph["home_player", "pass", "home_player"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return graph


def test_team_pool_is_invariant_to_edges_and_node_order():
    torch.manual_seed(3)
    model = TeamPoolMLP(dropout=0.0).eval()
    graph = _graph()
    reference = model(graph)

    changed = graph.clone()
    changed["home_player", "pass", "home_player"].edge_index = torch.tensor(
        [[0, 4, 4, 2], [4, 0, 3, 1]], dtype=torch.long)
    changed["home_player"].x = changed["home_player"].x[
        torch.tensor([4, 2, 0, 3, 1])]
    changed["away_player"].x = changed["away_player"].x[
        torch.tensor([2, 0, 3, 1])]
    observed = model(changed)

    assert torch.allclose(reference[0], observed[0], atol=1e-6)
    assert torch.allclose(reference[1], observed[1], atol=1e-6)


def test_score_aware_team_pool_requires_and_uses_cutoff_score():
    torch.manual_seed(5)
    model = TeamPoolMLP(dropout=0.0, use_current_score=True).eval()
    graph = _graph()
    with pytest.raises(ValueError, match="cutoff score"):
        model(graph)

    graph.current_home_score = 0.0
    graph.current_away_score = 0.0
    graph.current_goal_diff = 0.0
    level = model(graph)

    graph.current_home_score = 2.0
    graph.current_away_score = 0.0
    graph.current_goal_diff = 2.0
    leading = model(graph)

    assert not torch.allclose(level[0], leading[0])
    assert not torch.allclose(level[1], leading[1])
