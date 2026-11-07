"""Validation-only calibration of score-aware temporal graph predictions.

This script never retrains a graph model and never selects a hyper-parameter on
the test set.  For each available score-aware checkpoint it:

1. fits score-only logistic/Ridge candidates on the chronological train split;
2. chooses the score-model candidate and convex blend weight on validation;
3. evaluates that frozen choice once on the chronological test split.

The result is an explicitly labelled calibrated ensemble, not a replacement
for the equal-budget primary model comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_temporal_cutoff_graphs import TEMPORAL_GRAPH_REVISION
from scripts.checkpoint_provenance import validated_temporal_checkpoint
from scripts.run_temporal_cutoff_experiment import MODELS, metric_row, score_matrix
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


CUTOFFS = [
    int(value) for value in os.environ.get(
        "IS_BLEND_CUTOFFS", "15,30,45,60,75").split(",") if value
]
SEEDS = [
    int(value) for value in os.environ.get(
        "IS_BLEND_SEEDS", "42,7,13,21,37").split(",") if value
]
MODEL_NAMES = [
    value.strip() for value in os.environ.get(
        "IS_BLEND_MODELS",
        "TeamPool-MLP+Score,HeteroSAGE-sum+Score,SPARC-GNN+Score",
    ).split(",") if value.strip()
]
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 11)
BLEND_REVISION = "validation_only_score_graph_convex_blend_v1"


def graph_predictions(model, graphs, device=None):
    """Return labels, class probabilities and regression outputs on CPU.

    Checkpoint evaluation is substantially faster on CUDA, but the returned
    arrays stay device-independent. ``IS_BLEND_DEVICE=cpu`` provides an
    explicit reproducibility/debug fallback.
    """
    if device is None:
        requested = os.environ.get("IS_BLEND_DEVICE", "auto").strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
    model = model.to(device)
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    model.eval()
    labels, probabilities, regression = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits, reg_out = model(batch)
            labels.extend(batch.y.reshape(-1).cpu().numpy().tolist())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            regression.extend(reg_out.reshape(-1).cpu().numpy().tolist())
    goal_diff = np.asarray(labels, dtype=float)
    true_class = goal_diff_to_class(torch.tensor(goal_diff)).numpy()
    return (goal_diff, true_class, np.concatenate(probabilities, axis=0),
            np.asarray(regression, dtype=float))


def score_candidates(train, validation, test):
    """Fit all score-only candidates on train and cache val/test outputs."""
    x_train, x_validation, x_test = map(
        score_matrix, (train, validation, test))
    y_train = np.asarray([float(graph.y.item()) for graph in train])
    y_validation = np.asarray(
        [float(graph.y.item()) for graph in validation])
    y_test = np.asarray([float(graph.y.item()) for graph in test])
    c_train = goal_diff_to_class(torch.tensor(y_train)).numpy()

    classifiers = []
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for class_weight in (None, "balanced"):
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=3000,
                    random_state=0,
                ),
            )
            estimator.fit(x_train, c_train)
            classifiers.append({
                "C": c_value,
                "balanced": class_weight == "balanced",
                "validation": estimator.predict_proba(x_validation),
                "test": estimator.predict_proba(x_test),
            })

    regressors = []
    for alpha in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        estimator.fit(x_train, y_train)
        regressors.append({
            "alpha": alpha,
            "validation": estimator.predict(x_validation),
            "test": estimator.predict(x_test),
        })
    return classifiers, regressors, y_validation, y_test


def choose_classification_blend(true_class, graph_probability, candidates):
    best = None
    for candidate in candidates:
        for graph_weight in BLEND_WEIGHTS:
            probability = (
                graph_weight * graph_probability
                + (1.0 - graph_weight) * candidate["validation"]
            )
            # Float32 graph probabilities can leave row sums a few ulps away
            # from one; normalise explicitly before log-loss evaluation.
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


def choose_regression_blend(true_goal_diff, graph_prediction, candidates):
    best = None
    for candidate in candidates:
        for graph_weight in BLEND_WEIGHTS:
            prediction = (
                graph_weight * graph_prediction
                + (1.0 - graph_weight) * candidate["validation"]
            )
            objective = mean_absolute_error(true_goal_diff, prediction)
            if best is None or objective < best["objective"]:
                best = {
                    "objective": float(objective),
                    "candidate": candidate,
                    "graph_weight": float(graph_weight),
                }
    return best


def main():
    out = ROOT / "data" / "information_sciences"
    graph_root = out / "temporal_graphs"
    partial = out / "temporal_validation_blends_partial.parquet"
    rows = (pd.read_parquet(partial).to_dict("records")
            if partial.exists() else [])
    completed = {
        (int(row["cutoff"]), row["base_model"], int(row["seed"]))
        for row in rows
        if row.get("blend_revision") == BLEND_REVISION
        and pd.notna(row.get("checkpoint_provenance_verified"))
        and bool(row.get("checkpoint_provenance_verified"))
    }

    base = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION,
    )
    train_ids, validation_ids, test_ids = split_dataset_by_metadata(
        base,
        str(ROOT / "data" / "processed" / "matches_meta.parquet"),
        strategy="chronological",
    )

    for cutoff in CUTOFFS:
        graph_by_id = {}
        for path in (graph_root / str(cutoff)).glob("*.pt"):
            graph = torch.load(path, weights_only=False)
            if getattr(graph, "temporal_graph_revision", None) != TEMPORAL_GRAPH_REVISION:
                raise RuntimeError(f"temporal graph revision mismatch: {path}")
            if getattr(graph, "score_timeline_revision", None) != SCORE_TIMELINE_REVISION:
                raise RuntimeError(f"score timeline revision mismatch: {path}")
            graph_by_id[int(path.stem)] = graph
        graphs = [
            [graph_by_id[match_id] for match_id in split_ids
             if match_id in graph_by_id]
            for split_ids in (train_ids, validation_ids, test_ids)
        ]
        if min(map(len, graphs)) == 0:
            continue

        (classification_candidates, regression_candidates,
         validation_goal_diff, test_goal_diff) = score_candidates(*graphs)
        validation_class = goal_diff_to_class(
            torch.tensor(validation_goal_diff)).numpy()
        test_class = goal_diff_to_class(torch.tensor(test_goal_diff)).numpy()

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
                    out, cutoff, model_name, seed, model_revision,
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
                 validation_regression) = graph_predictions(model, graphs[1])
                (_, test_class_check, test_probability,
                 test_regression) = graph_predictions(model, graphs[2])
                if not (np.array_equal(validation_class, validation_class_check)
                        and np.array_equal(test_class, test_class_check)):
                    raise RuntimeError("graph order/labels changed during inference")

                cls_choice = choose_classification_blend(
                    validation_class, validation_probability,
                    classification_candidates)
                reg_choice = choose_regression_blend(
                    validation_goal_diff, validation_regression,
                    regression_candidates)
                test_blended_probability = (
                    cls_choice["graph_weight"] * test_probability
                    + (1.0 - cls_choice["graph_weight"])
                    * cls_choice["candidate"]["test"]
                )
                test_blended_regression = (
                    reg_choice["graph_weight"] * test_regression
                    + (1.0 - reg_choice["graph_weight"])
                    * reg_choice["candidate"]["test"]
                )
                row = {
                    "cutoff": cutoff,
                    "model": f"{model_name}+ValBlend",
                    "base_model": model_name,
                    "seed": seed,
                    "blend_revision": BLEND_REVISION,
                    "checkpoint_provenance_verified": True,
                    "model_revision": model_revision,
                    "graph_revision": TEMPORAL_GRAPH_REVISION,
                    "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                    "score_timeline_revision": SCORE_TIMELINE_REVISION,
                    "score_context_revision": SCORE_CONTEXT_REVISION,
                    "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                    "n_train": len(graphs[0]),
                    "n_validation": len(graphs[1]),
                    "n_test": len(graphs[2]),
                    "classification_graph_weight":
                        cls_choice["graph_weight"],
                    "score_cls_C": cls_choice["candidate"]["C"],
                    "score_cls_balanced":
                        cls_choice["candidate"]["balanced"],
                    "validation_macro_f1": cls_choice["objective"][0],
                    "validation_draw_f1": cls_choice["objective"][1],
                    "regression_graph_weight": reg_choice["graph_weight"],
                    "score_reg_alpha": reg_choice["candidate"]["alpha"],
                    "validation_mae": reg_choice["objective"],
                    **metric_row(
                        test_class,
                        test_blended_probability.argmax(axis=1),
                        test_goal_diff,
                        test_blended_regression,
                    ),
                }
                rows = [
                    old for old in rows
                    if (int(old["cutoff"]), old["base_model"],
                        int(old["seed"])) != key
                ]
                rows.append(row)
                pd.DataFrame(rows).to_parquet(partial, index=False)
                print(row, flush=True)

    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame[
            frame["blend_revision"].eq(BLEND_REVISION)
        ].drop_duplicates(
            ["cutoff", "base_model", "seed"], keep="last")
        frame.to_parquet(
            out / "temporal_validation_blends.parquet", index=False)
        numeric = frame.select_dtypes(include=[np.number]).columns
        frame.groupby(["cutoff", "model"])[numeric].agg(
            ["mean", "std"]).to_csv(
                out / "temporal_validation_blends_summary.csv")
        manifest = {
            "blend_revision": BLEND_REVISION,
            "selection_partition": "validation",
            "test_used_for_selection": False,
            "checkpoint_provenance_required": True,
            "cutoffs": list(CUTOFFS),
            "base_models": list(MODEL_NAMES),
            "blend_weights": BLEND_WEIGHTS.tolist(),
        }
        (out / "temporal_validation_blends_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
