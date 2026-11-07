import numpy as np
import pandas as pd

from scripts.evaluate_temporal_causal_baselines import (
    add_score_features,
    select_classifier,
    select_regressor,
)


class FixedClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def fit(self, x, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, x):
        return self.probabilities


class FixedRegressor:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def fit(self, x, y):
        return self

    def predict(self, x):
        return self.predictions


def test_causal_baseline_selection_uses_validation_predictions():
    y = np.array([0, 1, 2])
    good = np.eye(3) * .98 + .02 / 3
    bad = good[::-1]
    selected, diagnostics = select_classifier(
        [("bad", FixedClassifier(bad)),
         ("good", FixedClassifier(good))],
        np.zeros((3, 1)), y, np.zeros((3, 1)), y)
    assert selected["name"] == "good"
    assert len(diagnostics) == 2


def test_causal_regression_selection_uses_validation_mae():
    y = np.array([-1.0, 0.0, 2.0])
    selected, diagnostics = select_regressor(
        [("bad", FixedRegressor(np.array([3.0, 3.0, 3.0]))),
         ("good", FixedRegressor(y.copy()))],
        np.zeros((3, 1)), y, np.zeros((3, 1)), y)
    assert selected["name"] == "good"
    assert selected["objective"][0] == 0.0
    assert len(diagnostics) == 2


def test_score_features_are_derived_only_from_current_score():
    frame = pd.DataFrame({
        "current_home_score": [2, 0],
        "current_away_score": [1, 0],
        "current_goal_difference": [1, 0],
    })
    result = add_score_features(frame)
    assert result["current_goal_total"].tolist() == [3, 0]
    assert result["current_home_leading"].tolist() == [1.0, 0.0]
    assert result["current_drawing"].tolist() == [0.0, 1.0]
    assert result["current_away_leading"].tolist() == [0.0, 0.0]
