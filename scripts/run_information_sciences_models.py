"""Formal chronological-split comparison for the Information Sciences revision."""
from pathlib import Path
import json
import os
import collections
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset,
                             split_dataset_by_metadata)
from src.gnn.evaluator import evaluate_on_test
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import (HGTFootballGNN, MCARFootballGNN,
                                RGCNFootballGNN, SPARC_REVISION)
from src.gnn.model import HeteroFootballGNN
from src.gnn.team_pool import TeamPoolMLP
from src.gnn.trainer import (
    CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION, GNNTrainer,
    training_class_weights,
)
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from scripts.checkpoint_provenance import validated_ablation_checkpoint

SEEDS = [int(x) for x in os.environ.get("IS_SEEDS", "42,7,13,21,37").split(",")]
MODELS = collections.OrderedDict({
    "TeamPool-MLP": dict(factory=TeamPoolMLP, kwargs={}),
    "HeteroSAGE-sum": dict(factory=HeteroFootballGNN, kwargs={}),
    "R-GCN": dict(factory=RGCNFootballGNN, kwargs={}),
    "HGT": dict(factory=HGTFootballGNN, kwargs={}),
    # Motif-conditioned relation gating without the shared/private module.
    # This was called MCAR-HGNN in the first revision run; retaining it under a
    # descriptive name prevents it from being confused with the final method.
    "MotifGate-ObservedOnly": dict(
        factory=MCARFootballGNN,
        kwargs=dict(use_assist_flags=False,
                    mask_unavailable_relations=True),
        reuse_checkpoint="MotifGate-ObservedOnly"),
    "EdgeAware-Fixed": dict(
        factory=MCARFootballGNN,
        kwargs=dict(use_motif_context=False, adaptive_gate=False,
                    use_assist_flags=False),
        reuse_checkpoint="EdgeAware-Fixed",
    ),
    "SPARC-GNN": dict(
        factory=MCARFootballGNN,
        kwargs=dict(use_info_decomposition=True, use_assist_flags=False),
        reuse_checkpoint="MCAR-Decomp-FixedLoss",
    ),
})
LEGACY_NAMES = {"MCAR-HGNN": "MotifGate", "MCAR-Decomp": "SPARC-GNN"}
INVALID_FULL_MATCH_MODELS = {"CurrentScore", "CurrentScore-Calibrated"}


def _clean_rows(rows):
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["model"] = frame["model"].replace(LEGACY_NAMES)
    frame["model"] = frame["model"].where(frame["model"].notna(), other=pd.NA)
    frame["model"] = frame["model"].astype(str).replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    frame = frame.dropna(subset=["model"]).copy()
    frame = frame[~frame["model"].isin(INVALID_FULL_MATCH_MODELS)].copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["seed"])
    frame["seed"] = frame["seed"].astype(int)
    if "model_revision" not in frame:
        frame["model_revision"] = pd.NA
    if "split_revision" not in frame:
        frame["split_revision"] = pd.NA
    if "feature_revision" not in frame:
        frame["feature_revision"] = pd.NA
    if "graph_revision" not in frame:
        frame["graph_revision"] = pd.NA
    if "loss_revision" not in frame:
        frame["loss_revision"] = pd.NA
    if "class_weight_revision" not in frame:
        frame["class_weight_revision"] = pd.NA
    current = ((frame["model"] != "SPARC-GNN")
               | (frame["model_revision"] == SPARC_REVISION)).fillna(False)
    current &= (frame["split_revision"] ==
                CHRONOLOGICAL_SPLIT_REVISION).fillna(False)
    current &= (frame["feature_revision"] ==
                PRIMARY_FULL_MATCH_FEATURE_REVISION).fillna(False)
    current &= (frame["graph_revision"] ==
                HETEROGENEOUS_GRAPH_REVISION).fillna(False)
    current &= (frame["loss_revision"] == FIXED_LOSS_REVISION).fillna(False)
    current &= (frame["class_weight_revision"] ==
                CLASS_WEIGHT_REVISION).fillna(False)
    frame = frame[current].copy()
    frame.loc[frame["model"] != "SPARC-GNN", "model_revision"] = (
        frame.loc[frame["model"] != "SPARC-GNN", "model_revision"]
        .fillna("baseline_v1"))
    frame["training_source"] = frame.get("training_source", "main_comparison_training")
    return frame.to_dict("records")


def metrics_from_predictions(frame):
    yc, pc = frame.true_class.to_numpy(), frame.pred_class.to_numpy()
    yr, pr = frame.true_goal_diff.to_numpy(), frame.pred_goal_diff.to_numpy()
    return {
        "accuracy": accuracy_score(yc, pc),
        "macro_f1": f1_score(yc, pc, average="macro", zero_division=0),
        "weighted_f1": f1_score(yc, pc, average="weighted", zero_division=0),
        "draw_f1": f1_score(yc, pc, labels=[1], average="macro", zero_division=0),
        "kappa": cohen_kappa_score(yc, pc),
        "mae": mean_absolute_error(yr, pr),
        "rmse": float(np.sqrt(mean_squared_error(yr, pr))),
        "r2": r2_score(yr, pr),
        "pearson_r": float(pearsonr(yr, pr)[0]) if np.std(pr) > 0 else 0.0,
    }


def main():
    out = ROOT / "data" / "information_sciences"
    checkpoints = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    base = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    train_ids, val_ids, test_ids = split_dataset_by_metadata(
        base, str(ROOT / "data" / "processed" / "matches_meta.parquet"),
        strategy="chronological")
    split_payload = {"strategy": "chronological",
                     "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                     "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                     "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                     "loss_revision": FIXED_LOSS_REVISION,
                     "class_weight_revision": CLASS_WEIGHT_REVISION,
                     "train": train_ids,
                     "validation": val_ids, "test": test_ids}
    print(f"chronological split: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    # Match the established local pipeline: load each graph once, then reuse
    # the in-memory objects across epochs, models and seeds.
    motif_dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    graph_by_id = {}
    for idx, mid in enumerate(motif_dataset.match_ids, 1):
        graph_by_id[mid] = motif_dataset[idx - 1]
        if idx % 500 == 0:
            print(f"loaded {idx}/{len(motif_dataset)} graphs", flush=True)
    split_graphs = {
        "train": [graph_by_id[mid] for mid in train_ids if mid in graph_by_id],
        "validation": [graph_by_id[mid] for mid in val_ids if mid in graph_by_id],
        "test": [graph_by_id[mid] for mid in test_ids if mid in graph_by_id],
    }
    class_weights = training_class_weights(
        [float(graph.y.item()) for graph in split_graphs["train"]])
    split_payload["class_weights"] = {
        "home": float(class_weights[0]),
        "draw": float(class_weights[1]),
        "away": float(class_weights[2]),
    }
    (out / "chronological_split.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8")

    partial_results = out / "model_results_partial.parquet"
    partial_predictions = out / "predictions_partial.parquet"
    existing = (pd.read_parquet(partial_results)
                if partial_results.exists() else pd.DataFrame())
    rows = _clean_rows(existing.to_dict("records"))
    predictions = []
    if partial_predictions.exists():
        previous_predictions = pd.read_parquet(partial_predictions)
        previous_predictions["model"] = previous_predictions["model"].replace(
            LEGACY_NAMES)
        previous_predictions["model"] = previous_predictions["model"].where(
            previous_predictions["model"].notna(), other=pd.NA)
        previous_predictions["model"] = previous_predictions["model"].astype(str).replace(
            {"nan": pd.NA, "None": pd.NA, "": pd.NA})
        previous_predictions = previous_predictions.dropna(subset=["model"])
        previous_predictions = previous_predictions[
            ~previous_predictions["model"].isin(INVALID_FULL_MATCH_MODELS)].copy()
        previous_predictions["seed"] = pd.to_numeric(
            previous_predictions["seed"], errors="coerce")
        previous_predictions = previous_predictions.dropna(subset=["seed"])
        previous_predictions["seed"] = previous_predictions["seed"].astype(int)
        if "model_revision" not in previous_predictions:
            previous_predictions["model_revision"] = pd.NA
        if "split_revision" not in previous_predictions:
            previous_predictions["split_revision"] = pd.NA
        if "feature_revision" not in previous_predictions:
            previous_predictions["feature_revision"] = pd.NA
        if "graph_revision" not in previous_predictions:
            previous_predictions["graph_revision"] = pd.NA
        if "loss_revision" not in previous_predictions:
            previous_predictions["loss_revision"] = pd.NA
        if "class_weight_revision" not in previous_predictions:
            previous_predictions["class_weight_revision"] = pd.NA
        prediction_current = (
            (previous_predictions["model"] != "SPARC-GNN")
            | (previous_predictions["model_revision"] == SPARC_REVISION)
        ).fillna(False)
        prediction_current &= (
            previous_predictions["split_revision"]
            == CHRONOLOGICAL_SPLIT_REVISION).fillna(False)
        prediction_current &= (
            previous_predictions["feature_revision"]
            == PRIMARY_FULL_MATCH_FEATURE_REVISION).fillna(False)
        prediction_current &= (
            previous_predictions["graph_revision"]
            == HETEROGENEOUS_GRAPH_REVISION).fillna(False)
        prediction_current &= (
            previous_predictions["loss_revision"]
            == FIXED_LOSS_REVISION).fillna(False)
        prediction_current &= (
            previous_predictions["class_weight_revision"]
            == CLASS_WEIGHT_REVISION).fillna(False)
        previous_predictions = previous_predictions[prediction_current].copy()
        previous_predictions.loc[
            previous_predictions["model"] != "SPARC-GNN",
            "model_revision"] = previous_predictions.loc[
                previous_predictions["model"] != "SPARC-GNN",
                "model_revision"].fillna("baseline_v1")
        predictions.append(previous_predictions)
    completed = {(row["model"], row["seed"]) for row in rows}

    for seed in SEEDS:
        for name, spec in MODELS.items():
            if (name, seed) in completed:
                print(f"skip completed: {name}, seed={seed}")
                continue
            print(f"\n{name}, seed={seed}", flush=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(split_graphs["train"], batch_size=32, shuffle=True,
                                      generator=generator)
            val_loader = DataLoader(split_graphs["validation"], batch_size=64, shuffle=False)
            test_loader = DataLoader(split_graphs["test"], batch_size=64, shuffle=False)
            model = spec["factory"](
                node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3,
                **spec["kwargs"])
            ckpt = checkpoints / f"{name.replace('/', '_')}_seed{seed}.pt"
            reuse_name = spec.get("reuse_checkpoint")
            expected_revision = (SPARC_REVISION if name == "SPARC-GNN"
                                 else "baseline_v1")
            reuse_path = (validated_ablation_checkpoint(
                out, reuse_name, seed, expected_revision,
                CHRONOLOGICAL_SPLIT_REVISION,
                PRIMARY_FULL_MATCH_FEATURE_REVISION,
                HETEROGENEOUS_GRAPH_REVISION,
                FIXED_LOSS_REVISION,
                CLASS_WEIGHT_REVISION) if reuse_name else None)
            if reuse_path is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.load_state_dict(torch.load(
                    reuse_path, map_location=device, weights_only=True))
                model = model.to(device)
                best_val_loss = np.nan
                training_source = f"checkpoints_ablation/{reuse_path.name}"
            else:
                trainer = GNNTrainer(
                    model, lr=1e-3, patience=20,
                    class_weights=class_weights)
                history = trainer.train(train_loader, val_loader, n_epochs=200,
                                        save_path=str(ckpt))
                device = trainer.device
                best_val_loss = history["best_val_loss"]
                training_source = "main_comparison_training"
            result = evaluate_on_test(model, test_loader, device=device)
            pred = result.pop("predictions_df")
            pred["model"] = name; pred["seed"] = seed
            model_revision = expected_revision
            pred["model_revision"] = model_revision
            pred["split_revision"] = CHRONOLOGICAL_SPLIT_REVISION
            pred["feature_revision"] = PRIMARY_FULL_MATCH_FEATURE_REVISION
            pred["graph_revision"] = HETEROGENEOUS_GRAPH_REVISION
            pred["loss_revision"] = FIXED_LOSS_REVISION
            pred["class_weight_revision"] = CLASS_WEIGHT_REVISION
            predictions.append(pred)
            row = {"model": name, "seed": seed,
                   "model_revision": model_revision,
                   "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                   "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                   "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                   "loss_revision": FIXED_LOSS_REVISION,
                   "class_weight_revision": CLASS_WEIGHT_REVISION,
                   "class_weight_home": float(class_weights[0]),
                   "class_weight_draw": float(class_weights[1]),
                   "class_weight_away": float(class_weights[2]),
                   "n_train": len(split_graphs["train"]),
                   "n_validation": len(split_graphs["validation"]),
                   "n_test": len(split_graphs["test"]),
                   "best_val_loss": best_val_loss,
                   "training_source": training_source,
                   **metrics_from_predictions(pred)}
            rows.append(row)
            pd.DataFrame(rows).to_parquet(out / "model_results_partial.parquet", index=False)
            pd.concat(predictions, ignore_index=True).to_parquet(
                out / "predictions_partial.parquet", index=False)
            print(row, flush=True)

    order = list(MODELS)
    results = pd.DataFrame(rows)
    results["model"] = pd.Categorical(results["model"], order, ordered=True)
    results = results.sort_values(["model", "seed"]).reset_index(drop=True)
    results["model"] = results["model"].astype(str)
    results.to_parquet(out / "model_results.parquet", index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not prediction_frame.empty:
        prediction_frame["model"] = prediction_frame["model"].replace(LEGACY_NAMES)
        prediction_frame = prediction_frame.dropna(subset=["model"])
    prediction_frame["model"] = pd.Categorical(
        prediction_frame["model"], order, ordered=True)
    prediction_frame = prediction_frame.sort_values(
        ["model", "seed", "match_id"]).reset_index(drop=True)
    prediction_frame["model"] = prediction_frame["model"].astype(str)
    prediction_frame.to_parquet(out / "predictions.parquet", index=False)
    numeric = [c for c in results.columns
               if c not in {"model", "training_source", "model_revision",
                            "split_revision", "feature_revision",
                            "graph_revision", "loss_revision",
                            "class_weight_revision"}]
    summary = results.groupby("model", observed=True)[numeric].agg(["mean", "std"])
    summary.to_csv(out / "model_results_summary.csv")
    print(summary.to_string())

    pd.DataFrame(rows).to_parquet(out / "model_results_partial.parquet",
                                  index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(
            out / "predictions_partial.parquet", index=False)


if __name__ == "__main__":
    main()
