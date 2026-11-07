import numpy as np

from scripts.evaluate_temporal_validation_blends import (
    choose_classification_blend,
    choose_regression_blend,
)


def test_classification_blend_uses_validation_predictions_only():
    true_class = np.array([0, 1, 2, 1])
    graph_probability = np.full((4, 3), 1 / 3)
    score_probability = np.eye(3)[true_class] * .98 + .02 / 3
    candidate = {
        "C": 1.0,
        "balanced": True,
        "validation": score_probability,
        # Deliberately unrelated; selection must not inspect this field.
        "test": score_probability[::-1],
    }
    choice = choose_classification_blend(
        true_class, graph_probability, [candidate])
    assert choice["graph_weight"] == 0.0
    assert choice["objective"][0] == 1.0


def test_regression_blend_uses_validation_mae():
    true_goal_diff = np.array([-1.0, 0.0, 2.0])
    graph_prediction = np.array([3.0, 3.0, 3.0])
    candidate = {
        "alpha": 1.0,
        "validation": true_goal_diff.copy(),
        "test": np.array([99.0, 99.0, 99.0]),
    }
    choice = choose_regression_blend(
        true_goal_diff, graph_prediction, [candidate])
    assert choice["graph_weight"] == 0.0
    assert choice["objective"] == 0.0
