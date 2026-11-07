"""Retrain SPARC-GNN on every non-empty relation subset.

Inference-time deletion is an out-of-distribution reliance diagnostic.  This
experiment instead retrains the same architecture on all three singleton,
three pairwise, and the complete relation sets, allowing matched tests of
whether an additional relation supplies predictive information after the
model has adapted to the available channels.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.checkpoint_provenance import validated_ablation_checkpoint
from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset,
                             split_dataset_by_metadata,
                             typed_triad_context)
from src.gnn.evaluator import evaluate_on_test
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import MCARFootballGNN, SPARC_REVISION
from src.gnn.trainer import (CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION,
                             GNNTrainer, training_class_weights)
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION


SEEDS = [int(value) for value in os.environ.get(
    "IS_SEEDS", "42,7,13,21,37").split(",")]
ALL_RELATIONS = ("pass", "adversarial", "turnover")
SUBSETS = OrderedDict([
    ("pass", frozenset({"pass"})),
    ("adversarial", frozenset({"adversarial"})),
    ("turnover", frozenset({"turnover"})),
    ("pass+adversarial", frozenset({"pass", "adversarial"})),
    ("pass+turnover", frozenset({"pass", "turnover"})),
    ("adversarial+turnover", frozenset({"adversarial", "turnover"})),
    ("all", frozenset(ALL_RELATIONS)),
])
requested = [value.strip() for value in os.environ.get(
    "IS_RELATION_SUBSETS", "").split(",") if value.strip()]
if requested:
    unknown = sorted(set(requested) - set(SUBSETS))
    if unknown:
        raise ValueError(f"unknown relation subsets: {unknown}")
    SUBSETS = OrderedDict((name, SUBSETS[name]) for name in requested)

SUBSET_REVISION = "retrained_all_nonempty_relation_subsets_v1"


def _clean_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    required = {
        "model_revision": SPARC_REVISION,
        "subset_revision": SUBSET_REVISION,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
        "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
    }
    keep = np.ones(len(frame), dtype=bool)
    for column, value in required.items():
        keep &= frame.get(column, pd.Series(index=frame.index,
                                             dtype=object)).eq(value).to_numpy()
    frame = frame[keep].copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame = frame.dropna(subset=["seed"])
    frame["seed"] = frame["seed"].astype(int)
    frame = frame.drop_duplicates(["relation_subset", "seed"], keep="last")
    return frame.to_dict("records")


def _restrict_relations(graph, retained: frozenset[str]):
    if retained == frozenset(ALL_RELATIONS):
        return graph
    result = graph.clone()
    for edge_type in result.edge_types:
        if edge_type[1] in retained:
            continue
        store = result[edge_type]
        store.edge_index = torch.empty((2, 0), dtype=torch.long)
        if hasattr(store, "edge_attr"):
            store.edge_attr = store.edge_attr[:0]
    for node_type, context in typed_triad_context(result).items():
        result[node_type].motif_x = context
    return result


def _copy_full_rows(rows: list[dict], out: Path) -> list[dict]:
    completed = {(row["relation_subset"], int(row["seed"])) for row in rows}
    ablation = pd.read_parquet(out / "ablation_results.parquet")
    full = ablation[ablation["variant"] == "MCAR-Decomp-FixedLoss"]
    for record in full.to_dict("records"):
        seed = int(record["seed"])
        if ("all", seed) in completed or seed not in SEEDS:
            continue
        checkpoint = validated_ablation_checkpoint(
            out, "MCAR-Decomp-FixedLoss", seed, SPARC_REVISION,
            CHRONOLOGICAL_SPLIT_REVISION,
            PRIMARY_FULL_MATCH_FEATURE_REVISION,
            HETEROGENEOUS_GRAPH_REVISION, FIXED_LOSS_REVISION,
            CLASS_WEIGHT_REVISION)
        if checkpoint is None:
            raise RuntimeError(f"missing validated full checkpoint, seed={seed}")
        rows.append({
            "relation_subset": "all", "n_relations": 3, "seed": seed,
            "model_revision": SPARC_REVISION,
            "subset_revision": SUBSET_REVISION,
            "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
            "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
            "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
            "loss_revision": FIXED_LOSS_REVISION,
            "class_weight_revision": CLASS_WEIGHT_REVISION,
            "class_weight_home": float(record["class_weight_home"]),
            "class_weight_draw": float(record["class_weight_draw"]),
            "class_weight_away": float(record["class_weight_away"]),
            "n_train": int(record["n_train"]),
            "n_validation": int(record["n_validation"]),
            "n_test": int(record["n_test"]),
            "best_val_loss": float(record["best_val_loss"]),
            "best_val_task_loss": float(record["best_val_task_loss"]),
            "best_val_auxiliary_loss": float(
                record["best_val_auxiliary_loss"]),
            "best_epoch": int(record["best_epoch"]),
            "train_seconds": np.nan,
            "training_source": f"checkpoints_ablation/{checkpoint.name}",
            **{metric: float(record[metric]) for metric in
               ("accuracy", "macro_f1", "draw_f1", "mae", "r2")},
        })
        completed.add(("all", seed))
    return rows


def main() -> None:
    out = ROOT / "data" / "information_sciences"
    partial = out / "relation_subset_results_partial.parquet"
    final = out / "relation_subset_results.parquet"
    checkpoints = out / "checkpoints_relation_subsets"
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows = _copy_full_rows(_clean_existing(partial), out)
    pd.DataFrame(rows).to_parquet(partial, index=False)
    completed = {(row["relation_subset"], int(row["seed"])) for row in rows}

    dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    graph_by_id = {mid: dataset[index]
                   for index, mid in enumerate(dataset.match_ids)}
    train_ids, validation_ids, test_ids = split_dataset_by_metadata(
        dataset, str(ROOT / "data/processed/matches_meta.parquet"),
        strategy="chronological")
    split_ids = (train_ids, validation_ids, test_ids)
    class_weights = training_class_weights([
        float(graph_by_id[mid].y.item()) for mid in train_ids])

    for subset_name, retained in SUBSETS.items():
        missing_seeds = [seed for seed in SEEDS
                         if (subset_name, seed) not in completed]
        if not missing_seeds:
            print(f"skip completed subset: {subset_name}", flush=True)
            continue
        graphs = [[_restrict_relations(graph_by_id[mid], retained)
                   for mid in ids] for ids in split_ids]
        edge_counts = {
            relation: int(sum(
                graph[edge_type].edge_index.size(1)
                for graph in graphs[0] for edge_type in graph.edge_types
                if edge_type[1] == relation))
            for relation in ALL_RELATIONS}
        print(f"subset={subset_name}, train edge counts={edge_counts}",
              flush=True)
        for seed in missing_seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            loaders = [DataLoader(
                split, batch_size=32 if index == 0 else 64,
                shuffle=index == 0,
                generator=generator if index == 0 else None)
                for index, split in enumerate(graphs)]
            model = MCARFootballGNN(
                node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=.3,
                use_edge_attr=True, use_motif_context=True,
                adaptive_gate=True, use_info_decomposition=True,
                use_assist_flags=False)
            trainer = GNNTrainer(model, lr=1e-3, patience=20,
                                 class_weights=class_weights)
            checkpoint = checkpoints / f"{subset_name}_seed{seed}.pt"
            started = time.perf_counter()
            history = trainer.train(
                loaders[0], loaders[1], n_epochs=200,
                save_path=str(checkpoint))
            elapsed = time.perf_counter() - started
            prediction = evaluate_on_test(
                model, loaders[2], trainer.device)["predictions_df"]
            row = {
                "relation_subset": subset_name,
                "n_relations": len(retained), "seed": seed,
                "model_revision": SPARC_REVISION,
                "subset_revision": SUBSET_REVISION,
                "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                "loss_revision": FIXED_LOSS_REVISION,
                "class_weight_revision": CLASS_WEIGHT_REVISION,
                "class_weight_home": float(class_weights[0]),
                "class_weight_draw": float(class_weights[1]),
                "class_weight_away": float(class_weights[2]),
                "n_train": len(graphs[0]),
                "n_validation": len(graphs[1]),
                "n_test": len(graphs[2]),
                "best_val_loss": history["best_val_loss"],
                "best_val_task_loss": history["best_val_task_loss"],
                "best_val_auxiliary_loss": history[
                    "best_val_auxiliary_loss"],
                "best_epoch": history["best_epoch"],
                "train_seconds": elapsed,
                "training_source": "relation_subset_retraining",
                "accuracy": accuracy_score(
                    prediction.true_class, prediction.pred_class),
                "macro_f1": f1_score(
                    prediction.true_class, prediction.pred_class,
                    average="macro", zero_division=0),
                "draw_f1": f1_score(
                    prediction.true_class, prediction.pred_class, labels=[1],
                    average="macro", zero_division=0),
                "mae": mean_absolute_error(
                    prediction.true_goal_diff, prediction.pred_goal_diff),
                "r2": r2_score(
                    prediction.true_goal_diff, prediction.pred_goal_diff),
            }
            rows.append(row)
            completed.add((subset_name, seed))
            pd.DataFrame(rows).to_parquet(partial, index=False)
            print(row, flush=True)
        del graphs

    frame = pd.DataFrame(rows).sort_values(
        ["n_relations", "relation_subset", "seed"]).reset_index(drop=True)
    frame.to_parquet(final, index=False)
    frame.groupby("relation_subset")[[
        "accuracy", "macro_f1", "draw_f1", "mae", "r2"
    ]].agg(["mean", "std"]).to_csv(
        out / "relation_subset_results_summary.csv")
    print(f"wrote {len(frame)} rows -> {final}")


if __name__ == "__main__":
    main()
