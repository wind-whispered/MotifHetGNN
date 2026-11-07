"""Relation controls for the selected fixed-loss MCAR-Decomp model.

Passing endpoints are loop-free degree-marginal permuted.  Adversarial and turnover
events use a team anchor by construction, making endpoint permutation an
identity; for those relations we instead shift the complete anchor to another
opposing player while retaining edge counts, attributes and concentration.
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score)
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset,
                             split_dataset_by_metadata,
                             typed_triad_context)
from src.gnn.evaluator import evaluate_on_test
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import MCARFootballGNN, SPARC_REVISION
from src.gnn.trainer import CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from scripts.checkpoint_provenance import validated_ablation_checkpoint

SEEDS = [int(x) for x in os.environ.get("IS_SEEDS", "42,7,13,21,37").split(",")]
RELATIONS = ("pass", "adversarial", "turnover")
RELATION_CONTROL_REVISION = "simple_directed_double_edge_swap_v3"


def degree_marginal_permutation(
        edge_index, generator, swaps_per_edge=10,
        max_attempts_per_swap=100):
    """Return a simple directed double-edge-swap destination permutation.

    Every accepted swap replaces (u,v),(x,y) by (u,y),(x,v), rejecting
    self-loops and duplicate pairs. Thus the simple directed in/out partner
    degrees are exact invariants. Edge attributes follow destination tokens,
    preserving their global multiset and each destination's incoming token
    multiset; per-source weighted pass strength is deliberately not claimed as
    an invariant.
    """
    n_edges = int(edge_index.size(1))
    identity = torch.arange(n_edges, dtype=torch.long)
    if n_edges <= 1:
        return identity, 0

    source, destination = edge_index
    if bool((source == destination).any()):
        raise ValueError("pass permutation expects a loop-free input graph")

    if len({(int(u), int(v)) for u, v in zip(source, destination)}) != n_edges:
        raise ValueError("pass permutation expects a simple directed graph")

    order = identity.clone()
    pairs = {(int(source[i]), int(destination[i])) for i in range(n_edges)}
    # The extra swap avoids deterministic return to the original realisation
    # for degree sequences with only two reachable states.
    target_swaps = max(int(swaps_per_edge * n_edges) + 1, 1)
    maximum_attempts = max(target_swaps * int(max_attempts_per_swap), 1)
    accepted = 0
    for _ in range(maximum_attempts):
        if accepted >= target_swaps:
            break
        i, j = [int(value) for value in torch.randint(
            n_edges, (2,), generator=generator).tolist()]
        if i == j:
            continue
        source_i, source_j = int(source[i]), int(source[j])
        destination_i = int(destination[order[i]])
        destination_j = int(destination[order[j]])
        old_i = (source_i, destination_i)
        old_j = (source_j, destination_j)
        new_i = (source_i, destination_j)
        new_j = (source_j, destination_i)
        if (source_i == source_j or destination_i == destination_j
                or source_i == destination_j or source_j == destination_i
                or new_i == new_j):
            continue
        occupied_elsewhere = pairs.difference({old_i, old_j})
        if new_i in occupied_elsewhere or new_j in occupied_elsewhere:
            continue
        pairs.remove(old_i)
        pairs.remove(old_j)
        pairs.add(new_i)
        pairs.add(new_j)
        temporary = order[i].clone()
        order[i] = order[j]
        order[j] = temporary
        accepted += 1

    return order, accepted


def perturb(data, relation=None, mode="all", generator=None):
    graph = data.clone()
    graph.control_accepted_swaps = 0
    graph.control_changed_stores = 0
    if mode == "all":
        return graph
    for et in graph.edge_types:
        if et[1] != relation:
            continue
        store = graph[et]
        if mode == "delete":
            store.edge_index = torch.empty((2, 0), dtype=torch.long)
            store.edge_attr = store.edge_attr[:0]
        elif mode == "permute" and store.edge_index.size(1) > 1:
            order, accepted = degree_marginal_permutation(
                store.edge_index, generator)
            store.edge_index = torch.stack(
                [store.edge_index[0], store.edge_index[1, order]], dim=0)
            store.edge_attr = store.edge_attr[order]
            graph.control_accepted_swaps += int(accepted)
            graph.control_changed_stores += int(accepted > 0)
        elif mode == "anchor_shift" and store.edge_index.size(1) > 0:
            n_dst = int(graph[et[2]].num_nodes)
            if n_dst > 1:
                store.edge_index = store.edge_index.clone()
                store.edge_index[1] = (store.edge_index[1] + 1) % n_dst
    for nt, value in typed_triad_context(graph).items():
        graph[nt].motif_x = value
    return graph


def main():
    out = ROOT / "data/information_sciences"
    dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    _, _, test_ids = split_dataset_by_metadata(
        dataset, str(ROOT / "data/processed/matches_meta.parquet"),
        strategy="chronological")
    mapping = {mid: dataset[i] for i, mid in enumerate(dataset.match_ids)}
    test = [mapping[mid] for mid in test_ids if mid in mapping]
    conditions = [("all", None)] + [("delete", r) for r in RELATIONS] + [
        ("permute", "pass"),
        ("anchor_shift", "adversarial"),
        ("anchor_shift", "turnover"),
    ]
    rows = []
    for seed in SEEDS:
        checkpoint = validated_ablation_checkpoint(
            out, "MCAR-Decomp-FixedLoss", seed, SPARC_REVISION,
            CHRONOLOGICAL_SPLIT_REVISION,
            PRIMARY_FULL_MATCH_FEATURE_REVISION,
            HETEROGENEOUS_GRAPH_REVISION,
            FIXED_LOSS_REVISION,
            CLASS_WEIGHT_REVISION)
        if checkpoint is None:
            continue
        model = MCARFootballGNN(hidden_dim=64, num_layers=3, dropout=.3,
                               use_edge_attr=True, use_motif_context=True,
                               adaptive_gate=True,
                               use_info_decomposition=True,
                               use_assist_flags=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                         weights_only=True))
        for mode, relation in conditions:
            generator = torch.Generator().manual_seed(seed)
            graphs = [perturb(g, relation, mode, generator) for g in test]
            result = evaluate_on_test(model, DataLoader(graphs, batch_size=64),
                                      "cuda" if torch.cuda.is_available() else "cpu")
            p = result["predictions_df"]
            rows.append({
                "seed": seed, "condition": mode,
                "relation": relation or "all",
                "accuracy": accuracy_score(p.true_class, p.pred_class),
                "macro_f1": f1_score(p.true_class, p.pred_class,
                                     average="macro", zero_division=0),
                "draw_f1": f1_score(p.true_class, p.pred_class, labels=[1],
                                    average="macro", zero_division=0),
                "mae": mean_absolute_error(p.true_goal_diff, p.pred_goal_diff),
                "r2": r2_score(p.true_goal_diff, p.pred_goal_diff),
                "model_revision": SPARC_REVISION,
                "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                "loss_revision": FIXED_LOSS_REVISION,
                "class_weight_revision": CLASS_WEIGHT_REVISION,
                "control_revision": RELATION_CONTROL_REVISION,
                "control_accepted_swaps": (
                    sum(int(getattr(graph, "control_accepted_swaps", 0))
                        for graph in graphs)
                    if mode == "permute" else np.nan),
                "control_changed_graphs": (
                    sum(int(getattr(graph, "control_changed_stores", 0) > 0)
                        for graph in graphs)
                    if mode == "permute" else np.nan),
                "control_total_graphs": (
                    len(graphs) if mode == "permute" else np.nan),
            })
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "relation_controls_decomp.parquet", index=False)
    metrics = ["accuracy", "macro_f1", "draw_f1", "mae", "r2"]
    frame.groupby(["condition", "relation"])[metrics].agg(
        ["mean", "std"]).to_csv(out / "relation_controls_decomp_summary.csv")


if __name__ == "__main__":
    main()
