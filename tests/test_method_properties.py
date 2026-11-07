"""Numerical checks for Proposition 2 on synthetic typed graphs."""
import torch
from src.gnn.mcar_model import MCARFootballGNN, observed_relation_mean, masked_pairwise_conflict
from tests.test_round2_validation_search import toy_batch


def test_node_permutation_transports_context_edges_and_anchor():
    torch.manual_seed(123)
    graph = toy_batch().to_data_list()[0]
    model = MCARFootballGNN(hidden_dim=8, num_layers=2, dropout=0., use_info_decomposition=True).eval()
    permutation = {"home_player": torch.tensor([2, 0, 1]), "away_player": torch.tensor([1, 2, 0])}
    inverse = {nt: torch.argsort(p) for nt, p in permutation.items()}
    other = graph.clone()
    for nt, order in permutation.items():
        other[nt].x = graph[nt].x[order]
        other[nt].motif_x = graph[nt].motif_x[order]
    for src, relation, dst in graph.edge_types:
        ei = graph[src, relation, dst].edge_index
        other[src, relation, dst].edge_index = torch.stack([inverse[src][ei[0]], inverse[dst][ei[1]]])
    with torch.no_grad():
        first, second = model.encode(graph), model.encode(other)
        for nt, order in permutation.items():
            torch.testing.assert_close(second[nt], first[nt][order], atol=1e-6, rtol=1e-5)
        for left, right in zip(model(graph), model(other)):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_empty_fusion_has_layer_norm_not_identity():
    graph = toy_batch().to_data_list()[0]
    model = MCARFootballGNN(hidden_dim=8, num_layers=2, dropout=0., use_info_decomposition=True).eval()
    for et in graph.edge_types:
        graph[et].edge_index = graph[et].edge_index[:, :0]
        graph[et].edge_attr = graph[et].edge_attr[:0]
    with torch.no_grad():
        encoded = model.encode(graph)
        for nt in model.NODE_TYPES:
            expected = model.input_proj[nt](graph[nt].x)
            for norms in model.norms:
                expected = norms[nt](expected)
            torch.testing.assert_close(encoded[nt], expected)
            assert model.last_decomposition[nt]["private_norm"].count_nonzero() == 0


def test_single_relation_and_observed_zero_vector_convention():
    messages = torch.tensor([[[1., 2.], [99., 99.], [0., 0.]]])
    available = torch.tensor([[True, False, False]])
    masked, mean, exists = observed_relation_mean(messages, available)
    assert exists.item() and torch.equal(mean, messages[:, 0])
    residual = (masked-mean[:, None])*available[:, :, None]
    assert residual.count_nonzero() == 0
    values, pairs = masked_pairwise_conflict(torch.zeros(1, 3, 2), torch.tensor([[True, True, False]]))
    assert pairs.tolist() == [[True, False, False]]
    assert values.tolist() == [[1., 0., 0.]]
