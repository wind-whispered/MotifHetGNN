import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score)

from scripts.analyze_information_sciences_results import (
    _all_metric_arrays,
)


def test_fast_metric_bundle_matches_sklearn():
    true_class = np.asarray([0, 0, 1, 1, 2, 2, 2])
    pred_class = np.asarray([0, 1, 1, 2, 2, 0, 2])
    true_goal_diff = np.asarray([2, 1, 0, 0, -1, -2, -3], dtype=float)
    pred_goal_diff = np.asarray([1.5, 0.8, 0.2, -0.1, -0.5, -1.5, -2.0])

    observed = _all_metric_arrays(
        true_class, pred_class, true_goal_diff, pred_goal_diff)
    expected = np.asarray([
        accuracy_score(true_class, pred_class),
        f1_score(true_class, pred_class, average="macro", zero_division=0),
        f1_score(true_class, pred_class, labels=[1], average="macro",
                 zero_division=0),
        mean_absolute_error(true_goal_diff, pred_goal_diff),
        r2_score(true_goal_diff, pred_goal_diff),
    ])
    assert np.allclose(observed, expected)
