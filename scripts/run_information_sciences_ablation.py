"""Component ablations for edge attributes, adaptive gates and motif context."""
from pathlib import Path
import logging
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset,
                             split_dataset_by_metadata)
from src.gnn.evaluator import evaluate_on_test
from src.gnn.feature_contracts import full_match_feature_revision
from src.gnn.mcar_model import MCARFootballGNN, SPARC_REVISION
from src.gnn.trainer import (
    CLASS_WEIGHT_REVISION, GNNTrainer, loss_revision_for,
    training_class_weights,
)
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")

SEEDS = [int(x) for x in os.environ.get("IS_SEEDS", "42,7,13,21,37").split(",")]
VARIANTS = {
    "NoEdgeAttr-Fixed": dict(use_edge_attr=False, use_motif_context=False,
                             adaptive_gate=False),
    "EdgeAware-Fixed": dict(use_edge_attr=True, use_motif_context=False,
                            adaptive_gate=False),
    "Adaptive-NoMotif": dict(use_edge_attr=True, use_motif_context=False,
                             adaptive_gate=True),
    "MotifGate": dict(use_edge_attr=True, use_motif_context=True,
                      adaptive_gate=True),
    # Matched semantic-absence control: retain the motif-conditioned gate but
    # remove shared/private/conflict decomposition while assigning zero mass
    # to relations without an incoming message.
    "MotifGate-ObservedOnly": dict(
        use_edge_attr=True, use_motif_context=True, adaptive_gate=True,
        mask_unavailable_relations=True),
    "MCAR-Decomp-FixedLoss": dict(use_edge_attr=True, use_motif_context=True,
                                   adaptive_gate=True,
                                   use_info_decomposition=True,
                                   loss_mode="fixed"),
    "MCAR-Decomp-NoConflict": dict(use_edge_attr=True, use_motif_context=True,
                                    adaptive_gate=True,
                                    use_info_decomposition=True,
                                    use_conflict_gate=False,
                                    loss_mode="fixed"),
    # Keep the structural mask in every valid decomposition model; this
    # ablation removes only the learned availability feature supplied to the
    # gate, rather than reintroducing messages for nonexistent relations.
    "MCAR-Decomp-NoAvailabilityFeature": dict(
        use_edge_attr=True, use_motif_context=True, adaptive_gate=True,
        use_info_decomposition=True, use_availability_gate=False,
        loss_mode="fixed"),
    "MCAR-Decomp-NoMotif": dict(use_edge_attr=True, use_motif_context=False,
                                 adaptive_gate=True,
                                 use_info_decomposition=True,
                                 loss_mode="fixed"),
    "MCAR-Decomp-NoOrthogonality": dict(
        use_edge_attr=True, use_motif_context=True, adaptive_gate=True,
        use_info_decomposition=True, auxiliary_weight=0.0,
        loss_mode="fixed"),
    # Deliberately invalid-as-primary upper bound. Every other variant masks
    # these retrospective annotations under the clean feature contract.
    "MCAR-Decomp-WithAssistProxy": dict(
        use_edge_attr=True, use_motif_context=True, adaptive_gate=True,
        use_info_decomposition=True, use_assist_flags=True,
        loss_mode="fixed"),
    "MCAR-Decomp-Uncertainty": dict(use_edge_attr=True,
                                     use_motif_context=True,
                                     adaptive_gate=True,
                                     use_info_decomposition=True,
                                     loss_mode="uncertainty"),
}
for _flags in VARIANTS.values():
    _flags.setdefault("use_assist_flags", False)
ALL_VARIANTS = {name: dict(flags) for name, flags in VARIANTS.items()}
REQUESTED_VARIANTS = [x for x in os.environ.get("IS_VARIANTS", "").split(",")
                      if x]
if REQUESTED_VARIANTS:
    unknown = sorted(set(REQUESTED_VARIANTS) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}")
    VARIANTS = {name: ALL_VARIANTS[name] for name in REQUESTED_VARIANTS}
RUN_TAG = os.environ.get("IS_RUN_TAG", "").strip()
if RUN_TAG and not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
    raise ValueError("IS_RUN_TAG may contain only letters, digits, '_' and '-'")


def _tagged_name(stem: str, suffix: str) -> str:
    tag = f"_{RUN_TAG}" if RUN_TAG else ""
    return f"{stem}{tag}{suffix}"


def main():
    out = ROOT / "data" / "information_sciences"
    checkpoint_dir = out / _tagged_name("checkpoints_ablation", "")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    train_ids, val_ids, test_ids = split_dataset_by_metadata(
        dataset, str(ROOT / "data/processed/matches_meta.parquet"),
        strategy="chronological")
    graph_by_id = {mid: dataset[i] for i, mid in enumerate(dataset.match_ids)}
    graphs = [[graph_by_id[mid] for mid in ids if mid in graph_by_id]
              for ids in (train_ids, val_ids, test_ids)]
    class_weights = training_class_weights(
        [float(graph.y.item()) for graph in graphs[0]])
    rows = []
    partial = out / _tagged_name("ablation_results_partial", ".parquet")
    if partial.exists():
        existing = pd.read_parquet(partial)
        if "model_revision" not in existing:
            existing["model_revision"] = pd.NA
        if "split_revision" not in existing:
            existing["split_revision"] = pd.NA
        if "feature_revision" not in existing:
            existing["feature_revision"] = pd.NA
        if "graph_revision" not in existing:
            existing["graph_revision"] = pd.NA
        if "loss_revision" not in existing:
            existing["loss_revision"] = pd.NA
        if "class_weight_revision" not in existing:
            existing["class_weight_revision"] = pd.NA
        if "loss_mode" not in existing:
            existing["loss_mode"] = np.where(
                existing["variant"].eq("MCAR-Decomp-Uncertainty"),
                "uncertainty", "fixed")
        decomposition = existing["variant"].astype(str).str.startswith(
            "MCAR-Decomp")
        if "auxiliary_weight" not in existing:
            existing["auxiliary_weight"] = np.where(decomposition, 0.01, 0.0)
        current = (~decomposition | (
            existing["model_revision"] == SPARC_REVISION)).fillna(False)
        current &= (existing["split_revision"] ==
                    CHRONOLOGICAL_SPLIT_REVISION).fillna(False)
        expected_feature = existing["variant"].map(
            lambda name: full_match_feature_revision(bool(
                ALL_VARIANTS.get(name, {}).get("use_assist_flags", False))))
        current &= (existing["feature_revision"] == expected_feature).fillna(False)
        current &= (existing["graph_revision"] ==
                    HETEROGENEOUS_GRAPH_REVISION).fillna(False)
        expected_mode = existing["variant"].map(
            lambda name: ALL_VARIANTS.get(name, {}).get(
                "loss_mode", "fixed"))
        current &= (existing["loss_mode"] == expected_mode).fillna(False)
        expected_loss = expected_mode.map(loss_revision_for)
        current &= (existing["loss_revision"] == expected_loss).fillna(False)
        current &= (existing["class_weight_revision"] ==
                    CLASS_WEIGHT_REVISION).fillna(False)
        existing = existing[current].copy()
        baseline = ~existing["variant"].astype(str).str.startswith(
            "MCAR-Decomp")
        existing.loc[baseline, "model_revision"] = (
            existing.loc[baseline, "model_revision"]
            .fillna("baseline_v1"))
        rows = existing.to_dict("records")
    completed = {(x["variant"], int(x["seed"])) for x in rows}
    for seed in SEEDS:
        for name, flags in VARIANTS.items():
            if (name, seed) in completed:
                continue
            print(f"training variant={name}, seed={seed}", flush=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            loaders = [DataLoader(g, batch_size=32 if i == 0 else 64,
                                  shuffle=(i == 0),
                                  generator=generator if i == 0 else None)
                       for i, g in enumerate(graphs)]
            flags = dict(flags)
            loss_mode = flags.pop("loss_mode", "fixed")
            auxiliary_weight = flags.pop("auxiliary_weight", 0.01)
            feature_revision = full_match_feature_revision(
                bool(flags.get("use_assist_flags", False)))
            model = MCARFootballGNN(hidden_dim=64, num_layers=3, dropout=.3,
                                    **flags)
            trainer = GNNTrainer(model, lr=1e-3, patience=20,
                                 loss_mode=loss_mode,
                                 auxiliary_weight=auxiliary_weight,
                                 class_weights=class_weights)
            history = trainer.train(
                loaders[0], loaders[1], n_epochs=200,
                save_path=str(checkpoint_dir / f"{name}_seed{seed}.pt"))
            result = evaluate_on_test(model, loaders[2], trainer.device)
            p = result["predictions_df"]
            rows.append({
                "variant": name, "seed": seed,
                "model_revision": (SPARC_REVISION
                                   if name.startswith("MCAR-Decomp")
                                   else "baseline_v1"),
                "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                "feature_revision": feature_revision,
                "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                "loss_revision": loss_revision_for(loss_mode),
                "class_weight_revision": CLASS_WEIGHT_REVISION,
                "class_weight_home": float(class_weights[0]),
                "class_weight_draw": float(class_weights[1]),
                "class_weight_away": float(class_weights[2]),
                "n_train": len(graphs[0]),
                "n_validation": len(graphs[1]),
                "n_test": len(graphs[2]),
                "loss_mode": loss_mode,
                "auxiliary_weight": auxiliary_weight,
                "best_val_loss": history["best_val_loss"],
                "best_val_task_loss": history["best_val_task_loss"],
                "best_val_auxiliary_loss": history[
                    "best_val_auxiliary_loss"],
                "best_epoch": history["best_epoch"],
                "task_precision_cls": history["task_precisions"][0],
                "task_precision_reg": history["task_precisions"][1],
                "accuracy": accuracy_score(p.true_class, p.pred_class),
                "macro_f1": f1_score(p.true_class, p.pred_class,
                                     average="macro", zero_division=0),
                "draw_f1": f1_score(p.true_class, p.pred_class, labels=[1],
                                    average="macro", zero_division=0),
                "mae": mean_absolute_error(p.true_goal_diff, p.pred_goal_diff),
                "r2": r2_score(p.true_goal_diff, p.pred_goal_diff),
            })
            pd.DataFrame(rows).to_parquet(partial, index=False)
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(
        out / _tagged_name("ablation_results", ".parquet"), index=False)
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame.groupby("variant")[numeric].agg(["mean", "std"]).to_csv(
        out / _tagged_name("ablation_results_summary", ".csv"))


if __name__ == "__main__":
    main()
