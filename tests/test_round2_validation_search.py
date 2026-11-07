"""Validation-search contracts using synthetic graphs only."""
import ast
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch_geometric.data import HeteroData, Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_round2_validation_search import GRID, MODELS, SEEDS, make_model
from src.gnn.mcar_model import MCARFootballGNN


def toy_batch():
    graphs=[]
    for i in range(3):
        data=HeteroData()
        for nt in MCARFootballGNN.NODE_TYPES:
            data[nt].x=torch.rand(3,39)
            data[nt].motif_x=torch.rand(3,7)
        for et in MCARFootballGNN.EDGE_TYPES:
            data[et].edge_index=torch.tensor([[0,1,2],[1,2,0]])
            data[et].edge_attr=torch.rand(3,MCARFootballGNN.EDGE_DIMS[et[1]])
            if et[1]=="pass":
                data[et].edge_attr[:,10:12]=0.
        data.y=torch.tensor([i-1.])
        graphs.append(data)
    return Batch.from_data_list(graphs)


def test_all_configurations_have_finite_outputs_and_gradients():
    torch.set_num_threads(2)
    for name in MODELS:
        for cfg in GRID:
            model=make_model(name,cfg)
            logits,reg=model(toy_batch())
            assert logits.shape==(3,3) and reg.shape==(3,1)
            assert torch.isfinite(logits).all() and torch.isfinite(reg).all()
            (logits.square().mean()+reg.square().mean()).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_batchnorm_candidate_replaces_norm_not_gate_or_messages():
    model=make_model("SPARC-GNN", GRID[-1])
    assert all(isinstance(layer[nt],nn.BatchNorm1d) for layer in model.norms for nt in model.NODE_TYPES)
    assert model.use_info_decomposition and not model.use_assist_flags
    assert len(model.layers)==3 and len(model.gates)==3


def test_search_does_not_load_test_graphs_or_call_test_evaluator():
    source=(ROOT/"scripts/run_round2_validation_search.py").read_text(encoding="utf8")
    tree=ast.parse(source)
    assert "evaluate_on_test" not in source
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="FootballHeteroDataset":
            ids=next(k.value for k in node.keywords if k.arg=="match_ids")
            assert ast.unparse(ids)=="train_ids + val_ids"
    assert SEEDS == [7,13] and len(GRID) == 4
    protocol_path=ROOT/"data/information_sciences/round2_validation_search/protocol.json"
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        assert protocol["test_evaluated"] is False
        assert protocol["test_graphs_loaded"] is False
        assert protocol["seeds"]==SEEDS and len(protocol["grid"])==len(GRID)


def test_sage_lowdrop_and_batchnorm_are_equivalent_configurations():
    torch.manual_seed(7); left=make_model("HeteroSAGE-sum", GRID[2])
    torch.manual_seed(7); right=make_model("HeteroSAGE-sum", GRID[3])
    assert left.state_dict().keys()==right.state_dict().keys()
    assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())


if __name__ == "__main__":
    # The project interpreters do not ship pytest; these dependency-free
    # function checks can also be collected by pytest when it is available.
    for check in [test_all_configurations_have_finite_outputs_and_gradients,
                  test_batchnorm_candidate_replaces_norm_not_gate_or_messages,
                  test_search_does_not_load_test_graphs_or_call_test_evaluator,
                  test_sage_lowdrop_and_batchnorm_are_equivalent_configurations]:
        check()
        print(f"PASS {check.__name__}", flush=True)
