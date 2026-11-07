"""Synthetic contract checks for the development search, not a dataset rerun."""
import ast
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_round2_validation_search import toy_batch
from scripts.run_round3_hpo import search_space, make_model, MODELS
from src.gnn.mcar_model import MCARFootballGNN
from src.gnn.residual_candidate import RootPreservingSPARC


def test_grid_and_gradients():
    grid = search_space()
    assert len(grid) == 8 and grid == search_space()
    for cfg in grid:
        for name in MODELS:
            model = make_model(name, cfg)
            logits, reg = model(toy_batch())
            assert logits.shape == (3, 3) and reg.shape == (3, 1)
            (logits.square().mean()+reg.square().mean()).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_frozen_update_unchanged():
    model = MCARFootballGNN(hidden_dim=8, dropout=0.)
    x, update = torch.randn(5, 8), torch.randn(5, 8)
    norm = nn.LayerNorm(8)
    assert torch.equal(model._residual_update(x, update, norm), norm(x+F.relu(update)))


def test_isolated_root_and_assist_invariance():
    model = RootPreservingSPARC(hidden_dim=8, num_layers=2, dropout=0., use_info_decomposition=True)
    model.eval()
    data = toy_batch()
    reference = model(data)
    for et in data.edge_types:
        if et[1] == "pass":
            data[et].edge_attr[:, 10:12] = 10000.
    actual = model(data)
    assert all(torch.equal(a, b) for a, b in zip(reference, actual))
    for et in data.edge_types:
        data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[et].edge_attr = data[et].edge_attr[:0]
    encoded = model.encode(data)
    for nt in model.NODE_TYPES:
        assert torch.equal(encoded[nt], model.input_proj[nt](data[nt].x))
        assert model.last_gate_weights[nt].count_nonzero() == 0


def test_no_test_graph_loading():
    source = (ROOT / "scripts/run_round3_hpo.py").read_text(encoding="utf8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FootballHeteroDataset":
            ids = next(k.value for k in node.keywords if k.arg == "match_ids")
            assert ast.unparse(ids) == "train_ids + val_ids"
    assert "evaluate_on_test" not in source


if __name__ == "__main__":
    torch.set_num_threads(2)
    for check in [test_grid_and_gradients, test_frozen_update_unchanged,
                  test_isolated_root_and_assist_invariance, test_no_test_graph_loading]:
        check()
        print(f"PASS {check.__name__}", flush=True)
