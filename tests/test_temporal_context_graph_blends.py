import numpy as np

from scripts.evaluate_temporal_context_graph_blends import (
    choose_classification_context_blend,
    choose_regression_context_blend,
)


def test_context_classification_blend_selection_ignores_test_predictions():
    true_class = np.array([0, 1, 2, 1])
    graph_probability = np.full((4, 3), 1 / 3)
    context_probability = np.eye(3)[true_class] * .98 + .02 / 3
    candidate = {
        "source": "unit-test",
        "candidate": "perfect-validation",
        "validation": context_probability,
        "test": context_probability[::-1],
    }
    choice = choose_classification_context_blend(
        true_class, graph_probability, [candidate])
    assert choice["graph_weight"] == 0.0
    assert choice["objective"][0] == 1.0


def test_context_regression_blend_selection_ignores_test_predictions():
    truth = np.array([-1.0, 0.0, 2.0])
    candidate = {
        "source": "unit-test",
        "candidate": "perfect-validation",
        "validation": truth.copy(),
        "test": np.array([99.0, 99.0, 99.0]),
    }
    choice = choose_regression_context_blend(
        truth, np.array([3.0, 3.0, 3.0]), [candidate])
    assert choice["graph_weight"] == 0.0
    assert choice["objective"] == 0.0
