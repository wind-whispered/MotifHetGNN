"""Exact initialisation, edge sensitivity and missing-relation safeguards."""
from pathlib import Path
import sys
import torch
from torch_geometric.nn import GATv2Conv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gnn.edge_value_candidate import EdgeValueEncoder, add_edge_values
from src.gnn.mcar_model import MCARFootballGNN
from src.gnn.anchor_readout_candidate import AnchorReadoutCandidate
from tests.test_round2_validation_search import toy_batch


def test_initial_output_is_exact_base_encoder():
    torch.manual_seed(7)
    conv=GATv2Conv((8,8),8,heads=1,concat=False,edge_dim=4,add_self_loops=False,dropout=0)
    x=(torch.rand(3,8),torch.rand(3,8)); ei=torch.tensor([[0,1,2],[1,2,0]]); e=torch.rand(3,4)
    wrapped=EdgeValueEncoder(conv,4,8).eval()
    expected=conv(x,ei,edge_attr=e)
    assert torch.equal(wrapped(x,ei,e),expected)
    with torch.no_grad():
        wrapped.edge_value.weight.fill_(.1)
    assert not torch.allclose(wrapped(x,ei,e),expected)


def test_assist_columns_remain_inactive_in_candidate():
    model=add_edge_values(MCARFootballGNN(hidden_dim=8,num_layers=1,dropout=0,
                                        use_info_decomposition=True,use_assist_flags=False)).eval()
    for layer in model.layers:
        for conv in layer.values():
            torch.nn.init.normal_(conv.edge_value.weight,std=.1)
    data=toy_batch(); altered=data.clone()
    for et in model.EDGE_TYPES:
        if et[1]=="pass":
            altered[et].edge_attr[:,10:12]=999.
    with torch.no_grad():
        before=model(data); after=model(altered)
    assert all(torch.equal(a,b) for a,b in zip(before,after))


def test_absent_relation_is_masked_and_value_branch_can_learn():
    model=add_edge_values(MCARFootballGNN(hidden_dim=8,num_layers=1,dropout=0,
                                        use_info_decomposition=True,use_assist_flags=False))
    data=toy_batch()
    for et in model.EDGE_TYPES:
        if et[1]=="turnover":
            data[et].edge_index=torch.empty((2,0),dtype=torch.long)
            data[et].edge_attr=torch.empty((0,model.EDGE_DIMS["turnover"]))
    logits,reg=model(data)
    for nt in model.NODE_TYPES:
        assert torch.count_nonzero(model.last_gate_weights[nt][:,2])==0
    (logits.square().mean()+reg.square().mean()).backward()
    grads=[conv.edge_value.weight.grad for layer in model.layers for conv in layer.values()]
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    assert any(g is not None and torch.count_nonzero(g)>0 for g in grads)


def test_readout_initialisation_and_missing_incidence():
    settings=dict(hidden_dim=8,num_layers=1,dropout=0,use_info_decomposition=True,use_assist_flags=False)
    torch.manual_seed(21);base=MCARFootballGNN(**settings).eval()
    torch.manual_seed(21);candidate=AnchorReadoutCandidate(**settings).eval()
    data=toy_batch()
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(base(data),candidate(data)))
    for et in candidate.EDGE_TYPES:
        if et[1]!="pass":
            data[et].edge_index=torch.empty((2,0),dtype=torch.long)
            data[et].edge_attr=torch.empty((0,candidate.EDGE_DIMS[et[1]]))
    logits,reg=candidate(data)
    assert torch.isfinite(logits).all() and torch.isfinite(reg).all()
    (logits.square().mean()+reg.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


if __name__=="__main__":
    torch.set_num_threads(2)
    for check in [test_initial_output_is_exact_base_encoder,
                  test_assist_columns_remain_inactive_in_candidate,
                  test_absent_relation_is_masked_and_value_branch_can_learn,
                  test_readout_initialisation_and_missing_incidence]:
        check();print(f"PASS {check.__name__}",flush=True)
