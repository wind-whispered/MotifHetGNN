"""Validation-only fusion of causal context and score-aware graph models.

The causal candidate families are exactly those used by
``evaluate_temporal_causal_baselines.py``.  A candidate and convex graph
weight are selected using only the chronological validation partition, then
frozen for one test evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_temporal_cutoff_graphs import TEMPORAL_GRAPH_REVISION
from scripts.checkpoint_provenance import validated_temporal_checkpoint
from scripts.evaluate_temporal_causal_baselines import (
    SCORE_FEATURES,
    add_score_features,
    linear_classifier_candidates,
    linear_regressor_candidates,
    state_classifier_candidates,
    state_regressor_candidates,
)
from scripts.evaluate_temporal_validation_blends import graph_predictions
from scripts.run_temporal_cutoff_experiment import MODELS, metric_row
from src.gnn.causal_context import (
    MATCH_STATE_FEATURE_NAMES,
    MATCH_STATE_REVISION,
    TEAM_STRENGTH_FEATURE_NAMES,
    TEAM_STRENGTH_REVISION,
    causal_team_strength_features,
)
from src.gnn.dataset import (
    CHRONOLOGICAL_SPLIT_REVISION,
    FootballHeteroDataset,
    split_dataset_by_metadata,
)
from src.gnn.mcar_model import SPARC_REVISION
from src.gnn.model import goal_diff_to_class
from src.gnn.score_context import SCORE_CONTEXT_REVISION
from src.gnn.trainer import CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.temporal import SCORE_TIMELINE_REVISION


CUTOFFS = tuple(int(value) for value in os.environ.get(
    "IS_CONTEXT_BLEND_CUTOFFS", "15,30,45,60,75").split(",") if value)
SEEDS = tuple(int(value) for value in os.environ.get(
    "IS_CONTEXT_BLEND_SEEDS", "42,7,13,21,37").split(",") if value)
MODEL_NAMES = tuple(value.strip() for value in os.environ.get(
    "IS_CONTEXT_BLEND_MODELS",
    "TeamPool-MLP+Score,HeteroSAGE-sum+Score,SPARC-GNN+Score",
).split(",") if value.strip())
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 11)
BLEND_REVISION = "validation_only_causal_context_graph_blend_v1"


def fitted_context_candidates(frame, split_ids, labels):
    """Fit the frozen candidate families on train and predict val/test."""
    frame = add_score_features(frame)
    feature_sets = (
        (
            "Score+Strength",
            list(SCORE_FEATURES) + list(TEAM_STRENGTH_FEATURE_NAMES),
            linear_classifier_candidates,
            linear_regressor_candidates,
        ),
        (
            "CausalMatchState-HGB",
            list(SCORE_FEATURES) + list(TEAM_STRENGTH_FEATURE_NAMES)
            + list(MATCH_STATE_FEATURE_NAMES),
            state_classifier_candidates,
            state_regressor_candidates,
        ),
    )
    y_train = labels.loc[split_ids["train"]].to_numpy(dtype=float)
    y_validation = labels.loc[
        split_ids["validation"]].to_numpy(dtype=float)
    y_test = labels.loc[split_ids["test"]].to_numpy(dtype=float)
    class_train = goal_diff_to_class(torch.tensor(y_train)).numpy()
    class_validation = goal_diff_to_class(
        torch.tensor(y_validation)).numpy()
    class_candidates = []
    reg_candidates = []

    for source, features, cls_factory, reg_factory in feature_sets:
        matrices = {
            name: frame.loc[ids, features].to_numpy(dtype=float)
            for name, ids in split_ids.items()
        }
        for name, estimator in cls_factory():
            estimator.fit(matrices["train"], class_train)
            class_candidates.append({
                "source": source,
                "candidate": name,
                "validation": estimator.predict_proba(
                    matrices["validation"]),
                "test": estimator.predict_proba(matrices["test"]),
            })
        for name, estimator in reg_factory():
            estimator.fit(matrices["train"], y_train)
            reg_candidates.append({
                "source": source,
                "candidate": name,
                "validation": estimator.predict(matrices["validation"]),
                "test": estimator.predict(matrices["test"]),
            })
    return (class_candidates, reg_candidates, y_validation, y_test,
            class_validation,
            goal_diff_to_class(torch.tensor(y_test)).numpy())


def choose_classification_context_blend(
    true_class, graph_probability, candidates,
):
    best = None
    for candidate in candidates:
        for graph_weight in BLEND_WEIGHTS:
            probability = (
                graph_weight * graph_probability
                + (1.0 - graph_weight) * candidate["validation"])
            probability = probability / probability.sum(axis=1, keepdims=True)
            prediction = probability.argmax(axis=1)
            objective = (
                f1_score(true_class, prediction, average="macro",
                         zero_division=0),
                f1_score(true_class, prediction, labels=[1], average="macro",
                         zero_division=0),
                accuracy_score(true_class, prediction),
                -log_loss(true_class, probability, labels=[0, 1, 2]),
            )
            if best is None or objective > best["objective"]:
                best = {
                    "objective": objective,
                    "candidate": candidate,
                    "graph_weight": float(graph_weight),
                }
    return best


def choose_regression_context_blend(
    true_goal_difference, graph_prediction, candidates,
):
    best = None
    for candidate in candidates:
        for graph_weight in BLEND_WEIGHTS:
            prediction = (
                graph_weight * graph_prediction
                + (1.0 - graph_weight) * candidate["validation"])
            objective = mean_absolute_error(true_goal_difference, prediction)
            if best is None or objective < best["objective"]:
                best = {
                    "objective": float(objective),
                    "candidate": candidate,
                    "graph_weight": float(graph_weight),
                }
    return best


def main():
    output_dir = ROOT / "data" / "information_sciences"
    metadata_path = ROOT / "data" / "processed" / "matches_meta.parquet"
    metadata = pd.read_parquet(metadata_path)
    labels = metadata.set_index("match_id")["goal_diff"].astype(float)
    state = pd.read_parquet(output_dir / "temporal_match_state.parquet")
    if set(state["match_state_revision"]) != {MATCH_STATE_REVISION}:
        raise RuntimeError("temporal match-state revision mismatch")
    if set(state["score_timeline_revision"]) != {SCORE_TIMELINE_REVISION}:
        raise RuntimeError("score timeline revision mismatch")
    strength = causal_team_strength_features(metadata)
    strength.index = strength.index.astype(int)

    base = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION,
    )
    train_ids, validation_ids, test_ids = split_dataset_by_metadata(
        base, str(metadata_path), strategy="chronological")
    split_ids = {
        "train": list(map(int, train_ids)),
        "validation": list(map(int, validation_ids)),
        "test": list(map(int, test_ids)),
    }

    partial = output_dir / "temporal_context_graph_blends_partial.parquet"
    rows = (pd.read_parquet(partial).to_dict("records")
            if partial.exists() else [])
    completed = {
        (int(row["cutoff"]), row["base_model"], int(row["seed"]))
        for row in rows
        if row.get("blend_revision") == BLEND_REVISION
        and pd.notna(row.get("checkpoint_provenance_verified"))
        and bool(row.get("checkpoint_provenance_verified"))
    }
    graph_root = output_dir / "temporal_graphs"

    for cutoff in CUTOFFS:
        context = state[state["cutoff"].eq(cutoff)].copy()
        context = context.set_index("match_id", verify_integrity=True)
        context = context.join(strength, how="left")
        candidates = fitted_context_candidates(context, split_ids, labels)
        (class_candidates, reg_candidates, validation_goal_difference,
         test_goal_difference, validation_class, test_class) = candidates

        graph_by_id = {}
        for path in (graph_root / str(cutoff)).glob("*.pt"):
            graph = torch.load(path, weights_only=False)
            if getattr(graph, "temporal_graph_revision", None) != TEMPORAL_GRAPH_REVISION:
                raise RuntimeError(f"temporal graph revision mismatch: {path}")
            if getattr(graph, "score_timeline_revision", None) != SCORE_TIMELINE_REVISION:
                raise RuntimeError(f"score timeline revision mismatch: {path}")
            graph_by_id[int(path.stem)] = graph
        graphs = {
            name: [graph_by_id[match_id] for match_id in ids]
            for name, ids in split_ids.items()
        }

        for model_name in MODEL_NAMES:
            if model_name not in MODELS:
                raise ValueError(f"unknown score-aware model: {model_name}")
            specification = MODELS[model_name]
            for seed in SEEDS:
                key = (cutoff, model_name, seed)
                if key in completed:
                    continue
                model_revision = (
                    SPARC_REVISION
                    if model_name == "SPARC-GNN+Score" else "baseline_v1")
                checkpoint = validated_temporal_checkpoint(
                    output_dir, cutoff, model_name, seed, model_revision,
                    CHRONOLOGICAL_SPLIT_REVISION, TEMPORAL_GRAPH_REVISION,
                    HETEROGENEOUS_GRAPH_REVISION, SCORE_TIMELINE_REVISION,
                    SCORE_CONTEXT_REVISION, FIXED_LOSS_REVISION,
                    CLASS_WEIGHT_REVISION)
                if checkpoint is None:
                    continue
                model = specification["factory"](
                    node_feature_dim=39,
                    hidden_dim=64,
                    num_layers=3,
                    dropout=0.3,
                    **specification.get("kwargs", {}),
                )
                model.load_state_dict(torch.load(
                    checkpoint, map_location="cpu", weights_only=True))
                (_, validation_class_check, validation_probability,
                 validation_regression) = graph_predictions(
                    model, graphs["validation"])
                (_, test_class_check, test_probability,
                 test_regression) = graph_predictions(model, graphs["test"])
                if not (np.array_equal(validation_class, validation_class_check)
                        and np.array_equal(test_class, test_class_check)):
                    raise RuntimeError("graph order/labels changed during inference")

                cls_choice = choose_classification_context_blend(
                    validation_class, validation_probability,
                    class_candidates)
                reg_choice = choose_regression_context_blend(
                    validation_goal_difference, validation_regression,
                    reg_candidates)
                probability = (
                    cls_choice["graph_weight"] * test_probability
                    + (1.0 - cls_choice["graph_weight"])
                    * cls_choice["candidate"]["test"])
                probability = probability / probability.sum(axis=1, keepdims=True)
                regression = (
                    reg_choice["graph_weight"] * test_regression
                    + (1.0 - reg_choice["graph_weight"])
                    * reg_choice["candidate"]["test"])
                row = {
                    "cutoff": cutoff,
                    "model": f"{model_name}+ContextBlend",
                    "base_model": model_name,
                    "seed": seed,
                    "blend_revision": BLEND_REVISION,
                    "checkpoint_provenance_verified": True,
                    "model_revision": model_revision,
                    "graph_revision": TEMPORAL_GRAPH_REVISION,
                    "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                    "match_state_revision": MATCH_STATE_REVISION,
                    "team_strength_revision": TEAM_STRENGTH_REVISION,
                    "score_timeline_revision": SCORE_TIMELINE_REVISION,
                    "score_context_revision": SCORE_CONTEXT_REVISION,
                    "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                    "n_train": len(train_ids),
                    "n_validation": len(validation_ids),
                    "n_test": len(test_ids),
                    "classification_context_source":
                        cls_choice["candidate"]["source"],
                    "classification_candidate":
                        cls_choice["candidate"]["candidate"],
                    "classification_graph_weight":
                        cls_choice["graph_weight"],
                    "validation_macro_f1": cls_choice["objective"][0],
                    "validation_draw_f1": cls_choice["objective"][1],
                    "regression_context_source":
                        reg_choice["candidate"]["source"],
                    "regression_candidate":
                        reg_choice["candidate"]["candidate"],
                    "regression_graph_weight": reg_choice["graph_weight"],
                    "validation_mae": reg_choice["objective"],
                    **metric_row(
                        test_class, probability.argmax(axis=1),
                        test_goal_difference, regression),
                }
                rows = [
                    old for old in rows
                    if (int(old["cutoff"]), old["base_model"],
                        int(old["seed"])) != key
                ]
                rows.append(row)
                pd.DataFrame(rows).to_parquet(partial, index=False)
                print(row, flush=True)

    result = pd.DataFrame(rows)
    if len(result):
        result = result[
            result["blend_revision"].eq(BLEND_REVISION)
        ].drop_duplicates(
            ["cutoff", "base_model", "seed"], keep="last")
        result.to_parquet(
            output_dir / "temporal_context_graph_blends.parquet",
            index=False)
        numeric = result.select_dtypes(include=[np.number]).columns
        result.groupby(["cutoff", "model"])[numeric].agg(
            ["mean", "std"]).to_csv(
                output_dir / "temporal_context_graph_blends_summary.csv")
        manifest = {
            "blend_revision": BLEND_REVISION,
            "match_state_revision": MATCH_STATE_REVISION,
            "team_strength_revision": TEAM_STRENGTH_REVISION,
            "score_timeline_revision": SCORE_TIMELINE_REVISION,
            "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
            "selection_partition": "validation",
            "test_used_for_selection": False,
            "checkpoint_provenance_required": True,
            "cutoffs": list(CUTOFFS),
            "base_models": list(MODEL_NAMES),
            "context_sources": ["Score+Strength", "CausalMatchState-HGB"],
            "blend_weights": BLEND_WEIGHTS.tolist(),
        }
        (output_dir / "temporal_context_graph_blends_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
