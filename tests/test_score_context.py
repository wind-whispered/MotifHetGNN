import pytest
import torch
from torch_geometric.data import HeteroData

from src.gnn.score_context import cutoff_score_features


def test_cutoff_score_features_scale_without_clipping():
    graph = HeteroData()
    graph.current_home_score = torch.tensor([13.0])
    graph.current_away_score = torch.tensor([9.0])
    graph.current_goal_diff = torch.tensor([4.0])
    observed = cutoff_score_features(graph, torch.device("cpu"), "test")
    expected = torch.tensor([[2.6, 1.8, 0.8, 0.0]])
    assert torch.allclose(observed, expected)


def test_cutoff_score_features_reject_inconsistent_difference():
    graph = HeteroData()
    graph.current_home_score = 2.0
    graph.current_away_score = 1.0
    graph.current_goal_diff = 0.0
    with pytest.raises(ValueError, match="disagrees"):
        cutoff_score_features(graph, torch.device("cpu"), "test")
