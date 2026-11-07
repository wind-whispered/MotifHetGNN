"""Measure how much final-result information is carried by assist flags alone."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import (FootballHeteroDataset,
                             split_dataset_by_metadata)
from src.gnn.feature_contracts import ASSIST_PROXY_FEATURE_REVISION
from src.gnn.model import goal_diff_to_class
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION


def assist_features(graph):
    values = []
    for node_type in ("home_player", "away_player"):
        edge_type = (node_type, "pass", node_type)
        if edge_type in graph.edge_types:
            attributes = graph[edge_type].edge_attr
            values.extend([
                float(attributes[:, 10].sum()),
                float(attributes[:, 11].sum()),
            ])
        else:
            values.extend([0.0, 0.0])
    return values


def metrics(true_goal_diff, pred_class, pred_goal_diff):
    true_class = goal_diff_to_class(torch.tensor(true_goal_diff)).numpy()
    return {
        "accuracy": float(accuracy_score(true_class, pred_class)),
        "macro_f1": float(f1_score(
            true_class, pred_class, average="macro", zero_division=0)),
        "draw_f1": float(f1_score(
            true_class, pred_class, labels=[1], average="macro",
            zero_division=0)),
        "mae": float(mean_absolute_error(true_goal_diff, pred_goal_diff)),
        "r2": float(r2_score(true_goal_diff, pred_goal_diff)),
    }


def audit():
    dataset = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    train_ids, validation_ids, test_ids = split_dataset_by_metadata(
        dataset, str(ROOT / "data" / "processed" / "matches_meta.parquet"),
        strategy="chronological")
    ids = train_ids + validation_ids + test_ids
    id_set = set(ids)
    graphs = {mid: dataset[index]
              for index, mid in enumerate(dataset.match_ids) if mid in id_set}

    matrices = []
    labels = []
    for partition_ids in (train_ids, validation_ids, test_ids):
        selected = [graphs[mid] for mid in partition_ids if mid in graphs]
        matrices.append(np.asarray([assist_features(graph) for graph in selected]))
        labels.append(np.asarray([float(graph.y.item()) for graph in selected]))
    x_train, x_validation, x_test = matrices
    y_train, y_validation, y_test = labels
    c_train, c_validation, c_test = [
        goal_diff_to_class(torch.tensor(values)).numpy() for values in labels]

    best_classifier = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            estimator = make_pipeline(
                StandardScaler(), LogisticRegression(
                    C=c_value, class_weight=class_weight, max_iter=2000,
                    random_state=0))
            estimator.fit(x_train, c_train)
            prediction = estimator.predict(x_validation)
            score = (f1_score(c_validation, prediction, average="macro",
                              zero_division=0),
                     accuracy_score(c_validation, prediction))
            if best_classifier is None or score > best_classifier[0]:
                best_classifier = (score, estimator, c_value, class_weight)

    best_regressor = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        estimator.fit(x_train, y_train)
        score = mean_absolute_error(y_validation,
                                    estimator.predict(x_validation))
        if best_regressor is None or score < best_regressor[0]:
            best_regressor = (score, estimator, alpha)

    pred_class = best_classifier[1].predict(x_test)
    pred_goal_diff = best_regressor[1].predict(x_test)
    direct_goal_assist_diff = x_test[:, 0] - x_test[:, 2]
    direct_class = goal_diff_to_class(
        torch.tensor(direct_goal_assist_diff)).numpy()
    payload = {
        "feature_revision": ASSIST_PROXY_FEATURE_REVISION,
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "feature_order": [
            "home_goal_assist_pairs", "home_shot_assist_pairs",
            "away_goal_assist_pairs", "away_shot_assist_pairs",
        ],
        "split_sizes": {
            "train": len(x_train), "validation": len(x_validation),
            "test": len(x_test),
        },
        "validation_selection": {
            "classifier_C": best_classifier[2],
            "classifier_class_weight": best_classifier[3],
            "ridge_alpha": best_regressor[2],
        },
        "assist_only_calibrated": metrics(
            y_test, pred_class, pred_goal_diff),
        "direct_goal_assist_difference": metrics(
            y_test, direct_class, direct_goal_assist_diff),
        "test_correlations": {
            "goal_assist_pair_difference": float(np.corrcoef(
                direct_goal_assist_diff, y_test)[0, 1]),
            "shot_assist_pair_difference": float(np.corrcoef(
                x_test[:, 1] - x_test[:, 3], y_test)[0, 1]),
        },
        "interpretation": (
            "These four features are downstream outcome proxies in a "
            "complete-match graph; primary model comparisons should mask them "
            "if their stand-alone predictive power is material."
        ),
    }
    output = ROOT / "data" / "information_sciences" / (
        "full_match_assist_proxy_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output


def main():
    payload, output = audit()
    print(json.dumps(payload, indent=2))
    print(output)


if __name__ == "__main__":
    main()
