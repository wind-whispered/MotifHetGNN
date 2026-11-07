"""Evaluate strong score/strength and pre-cutoff match-state baselines.

All feature construction is causal.  Estimator and regularisation choices are
made on the chronological validation partition; the test partition is touched
once after those choices have been frozen.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_temporal_cutoff_experiment import metric_row
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
from src.gnn.model import goal_diff_to_class
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.temporal import SCORE_TIMELINE_REVISION


BASELINE_REVISION = "validation_selected_causal_context_v1"
CUTOFFS = tuple(int(value) for value in os.environ.get(
    "IS_CAUSAL_CUTOFFS", "15,30,45,60,75").split(",") if value)

SCORE_FEATURES = (
    "current_home_score",
    "current_away_score",
    "current_goal_difference",
    "current_goal_total",
    "current_home_leading",
    "current_drawing",
    "current_away_leading",
)


def add_score_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["current_goal_total"] = (
        result["current_home_score"] + result["current_away_score"])
    result["current_home_leading"] = (
        result["current_goal_difference"] > 0).astype(float)
    result["current_drawing"] = (
        result["current_goal_difference"] == 0).astype(float)
    result["current_away_leading"] = (
        result["current_goal_difference"] < 0).astype(float)
    return result


def linear_classifier_candidates():
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for class_weight in (None, "balanced"):
            name = f"logit_C{c_value:g}_{class_weight or 'unweighted'}"
            yield name, make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=4000,
                    random_state=0,
                ),
            )


def linear_regressor_candidates():
    for alpha in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        yield f"ridge_alpha{alpha:g}", make_pipeline(
            StandardScaler(), Ridge(alpha=alpha))


def state_classifier_candidates():
    for leaves, l2 in ((7, 1.0), (15, 3.0), (31, 5.0)):
        for class_weight in (None, "balanced"):
            name = f"hgb_leaf{leaves}_l2{l2:g}_{class_weight or 'unweighted'}"
            yield name, HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=leaves,
                min_samples_leaf=20,
                l2_regularization=l2,
                early_stopping=False,
                class_weight=class_weight,
                random_state=0,
            )


def state_regressor_candidates():
    for leaves, l2 in ((7, 1.0), (15, 3.0), (31, 5.0)):
        yield f"hgb_leaf{leaves}_l2{l2:g}", HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=leaves,
            min_samples_leaf=20,
            l2_regularization=l2,
            early_stopping=False,
            random_state=0,
        )


def select_classifier(candidates, x_train, y_train, x_validation,
                      y_validation):
    """Fit candidates on train and select lexicographically on validation."""
    best = None
    diagnostics = []
    for name, estimator in candidates:
        estimator.fit(x_train, y_train)
        probability = estimator.predict_proba(x_validation)
        prediction = estimator.classes_[probability.argmax(axis=1)]
        objective = (
            f1_score(y_validation, prediction, average="macro",
                     zero_division=0),
            f1_score(y_validation, prediction, labels=[1], average="macro",
                     zero_division=0),
            accuracy_score(y_validation, prediction),
            -log_loss(y_validation, probability, labels=[0, 1, 2]),
        )
        diagnostics.append({
            "candidate": name,
            "validation_macro_f1": objective[0],
            "validation_draw_f1": objective[1],
            "validation_accuracy": objective[2],
            "validation_log_loss": -objective[3],
        })
        if best is None or objective > best["objective"]:
            best = {"name": name, "estimator": estimator,
                    "objective": objective}
    return best, diagnostics


def select_regressor(candidates, x_train, y_train, x_validation,
                     y_validation):
    best = None
    diagnostics = []
    for name, estimator in candidates:
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_validation)
        mae = mean_absolute_error(y_validation, prediction)
        rmse = mean_squared_error(y_validation, prediction) ** 0.5
        objective = (mae, rmse)
        diagnostics.append({
            "candidate": name,
            "validation_mae": mae,
            "validation_rmse": rmse,
        })
        if best is None or objective < best["objective"]:
            best = {"name": name, "estimator": estimator,
                    "objective": objective}
    return best, diagnostics


def main():
    output_dir = ROOT / "data" / "information_sciences"
    metadata_path = ROOT / "data" / "processed" / "matches_meta.parquet"
    metadata = pd.read_parquet(metadata_path)
    state = pd.read_parquet(output_dir / "temporal_match_state.parquet")
    if set(state["match_state_revision"]) != {MATCH_STATE_REVISION}:
        raise RuntimeError("temporal match-state revision mismatch")
    if set(state["score_timeline_revision"]) != {SCORE_TIMELINE_REVISION}:
        raise RuntimeError("score timeline revision mismatch")
    strength = causal_team_strength_features(metadata)
    strength.index = strength.index.astype(int)

    dataset = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION,
    )
    train_ids, validation_ids, test_ids = split_dataset_by_metadata(
        dataset, str(metadata_path), strategy="chronological")
    split_ids = {
        "train": list(map(int, train_ids)),
        "validation": list(map(int, validation_ids)),
        "test": list(map(int, test_ids)),
    }
    labels = metadata.set_index("match_id")["goal_diff"].astype(float)

    result_rows = []
    prediction_rows = []
    validation_rows = []
    for cutoff in CUTOFFS:
        cutoff_frame = state[state["cutoff"].eq(cutoff)].copy()
        cutoff_frame = cutoff_frame.set_index("match_id", verify_integrity=True)
        cutoff_frame = cutoff_frame.join(strength, how="left")
        if cutoff_frame[list(TEAM_STRENGTH_FEATURE_NAMES)].isna().any().any():
            raise RuntimeError("team-strength join produced missing values")
        cutoff_frame = add_score_features(cutoff_frame)
        y_goal = {name: labels.loc[ids].to_numpy(dtype=float)
                  for name, ids in split_ids.items()}
        y_class = {
            name: goal_diff_to_class(
                torch.tensor(values)).numpy()
            for name, values in y_goal.items()
        }

        model_specs = (
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
        for model_name, features, cls_factory, reg_factory in model_specs:
            matrices = {
                name: cutoff_frame.loc[ids, features].to_numpy(dtype=float)
                for name, ids in split_ids.items()
            }
            classifier, cls_diagnostics = select_classifier(
                cls_factory(), matrices["train"], y_class["train"],
                matrices["validation"], y_class["validation"])
            regressor, reg_diagnostics = select_regressor(
                reg_factory(), matrices["train"], y_goal["train"],
                matrices["validation"], y_goal["validation"])

            test_probability = classifier["estimator"].predict_proba(
                matrices["test"])
            test_class = classifier["estimator"].classes_[
                test_probability.argmax(axis=1)]
            test_regression = regressor["estimator"].predict(
                matrices["test"])
            row = {
                "cutoff": cutoff,
                "model": model_name,
                "seed": -1,
                "model_revision": BASELINE_REVISION,
                "match_state_revision": MATCH_STATE_REVISION,
                "team_strength_revision": TEAM_STRENGTH_REVISION,
                "score_timeline_revision": SCORE_TIMELINE_REVISION,
                "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                "n_train": len(train_ids),
                "n_validation": len(validation_ids),
                "n_test": len(test_ids),
                "feature_count": len(features),
                "classification_candidate": classifier["name"],
                "regression_candidate": regressor["name"],
                "validation_macro_f1": classifier["objective"][0],
                "validation_draw_f1": classifier["objective"][1],
                "validation_accuracy": classifier["objective"][2],
                "validation_mae": regressor["objective"][0],
                **metric_row(
                    y_class["test"], test_class,
                    y_goal["test"], test_regression),
            }
            result_rows.append(row)
            for position, match_id in enumerate(test_ids):
                prediction_rows.append({
                    "cutoff": cutoff,
                    "model": model_name,
                    "match_id": int(match_id),
                    "true_class": int(y_class["test"][position]),
                    "predicted_class": int(test_class[position]),
                    "true_goal_difference": y_goal["test"][position],
                    "predicted_goal_difference": test_regression[position],
                    "prob_home": test_probability[position, 0],
                    "prob_draw": test_probability[position, 1],
                    "prob_away": test_probability[position, 2],
                    "model_revision": BASELINE_REVISION,
                })
            for diagnostic in cls_diagnostics:
                validation_rows.append({
                    "cutoff": cutoff, "model": model_name,
                    "task": "classification", **diagnostic,
                })
            for diagnostic in reg_diagnostics:
                validation_rows.append({
                    "cutoff": cutoff, "model": model_name,
                    "task": "regression", **diagnostic,
                })
            print(row, flush=True)

    pd.DataFrame(result_rows).to_parquet(
        output_dir / "temporal_causal_baselines.parquet", index=False)
    pd.DataFrame(prediction_rows).to_parquet(
        output_dir / "temporal_causal_baseline_predictions.parquet",
        index=False)
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "temporal_causal_baseline_validation.csv", index=False)
    manifest = {
        "model_revision": BASELINE_REVISION,
        "match_state_revision": MATCH_STATE_REVISION,
        "team_strength_revision": TEAM_STRENGTH_REVISION,
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
        "selection_partition": "validation",
        "test_used_for_selection": False,
        "cutoffs": list(CUTOFFS),
        "models": ["Score+Strength", "CausalMatchState-HGB"],
    }
    (output_dir / "temporal_causal_baselines_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
