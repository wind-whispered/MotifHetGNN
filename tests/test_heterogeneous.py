"""Tests for heterogeneous graph construction and feature encoding."""
import pytest
import numpy as np
import pandas as pd

from src.networks.node_features import build_node_feature, NODE_FEATURE_DIM
from src.networks.edge_features import (
    build_pass_edge_feature, build_adversarial_edge_feature,
    build_turnover_edge_feature,
    PASS_EDGE_DIM, ADV_EDGE_DIM, TURN_EDGE_DIM,
)
from src.networks.spatial import zone_index, zone_onehot, normalize_coords
from src.networks.temporal import get_period_label, get_score_state_at_minute


# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------

def test_zone_index_defensive_left():
    # x=10 (< 40) -> defensive, y=5 (< 26.67) -> left -> idx 0
    assert zone_index(10, 5) == 0


def test_zone_index_middle_center():
    # x=60, y=40 -> middle_center -> idx 4
    assert zone_index(60, 40) == 4


def test_zone_index_attacking_right():
    # x=100, y=70 -> attacking_right -> idx 8
    assert zone_index(100, 70) == 8


def test_zone_onehot_shape():
    vec = zone_onehot(60, 40)
    assert vec.shape == (9,)
    assert vec.sum() == 1.0


def test_zone_onehot_none():
    vec = zone_onehot(None, None)
    assert vec.shape == (9,)
    assert vec.sum() == 0.0


def test_normalize_coords():
    x_n, y_n = normalize_coords(60, 40)
    assert abs(x_n - 0.5) < 1e-6
    assert abs(y_n - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Temporal utilities
# ---------------------------------------------------------------------------

def test_period_label():
    assert get_period_label(15) == "0-30"
    assert get_period_label(45) == "30-60"
    assert get_period_label(75) == "60-90"
    assert get_period_label(95) == "90+"


def test_score_state_leading():
    timeline = [(0, 0, 0), (25, 1, 0)]
    state = get_score_state_at_minute(timeline, minute=30, is_home_team=True)
    assert state == "leading"


def test_score_state_trailing():
    timeline = [(0, 0, 0), (25, 0, 1)]
    state = get_score_state_at_minute(timeline, minute=30, is_home_team=True)
    assert state == "trailing"


def test_score_state_drawing():
    timeline = [(0, 0, 0)]
    state = get_score_state_at_minute(timeline, minute=30, is_home_team=True)
    assert state == "drawing"


# ---------------------------------------------------------------------------
# Node feature construction
# ---------------------------------------------------------------------------

def test_node_feature_shape():
    feat = build_node_feature(
        player_id=1, is_home=True, position_id=1,
        pass_count=10, receive_count=8,
        mean_x=60.0, mean_y=40.0,
    )
    assert feat.shape == (NODE_FEATURE_DIM,)


def test_node_feature_team_side():
    feat_home = build_node_feature(1, True, 1, 10, 8, 60, 40)
    feat_away = build_node_feature(1, False, 1, 10, 8, 60, 40)
    assert feat_home[25] == 1.0   # home flag
    assert feat_away[25] == 0.0   # away flag


def test_node_feature_position_onehot():
    feat = build_node_feature(1, True, 5, 10, 8, 60, 40)  # position_id=5 (LCB)
    assert feat[4] == 1.0   # 0-indexed: position 5 -> index 4
    # All other position bits should be 0
    assert sum(feat[:25]) == 1.0


def test_node_feature_no_position():
    feat = build_node_feature(1, True, None, 10, 8, 60, 40)
    assert sum(feat[:25]) == 0.0  # no position set


def test_node_feature_normalized_counts():
    feat = build_node_feature(1, True, 1, pass_count=100, receive_count=100,
                               mean_x=60, mean_y=40,
                               max_pass_count=100, max_receive_count=100)
    assert feat[26] == 1.0   # pass count maxed out
    assert feat[27] == 1.0   # receive count maxed out


# ---------------------------------------------------------------------------
# Edge feature construction
# ---------------------------------------------------------------------------

def test_pass_edge_feature_shape():
    row = pd.Series({
        "length": 20.0, "angle": 0.5, "height_id": 1,
        "under_pressure": True, "counterpress": False,
        "cross": False, "switch": False,
        "goal_assist": False, "shot_assist": False,
    })
    feat = build_pass_edge_feature(row)
    assert feat.shape == (PASS_EDGE_DIM,)


def test_pass_edge_feature_under_pressure():
    row = pd.Series({
        "length": 20.0, "angle": 0.5, "height_id": 1,
        "under_pressure": True, "counterpress": False,
        "cross": False, "switch": False,
        "goal_assist": False, "shot_assist": False,
    })
    feat = build_pass_edge_feature(row)
    assert feat[6] == 1.0  # under_pressure flag
    assert feat[7] == 0.0  # counterpress flag


def test_pass_edge_feature_height_onehot():
    for height_id in [1, 2, 3]:
        row = pd.Series({
            "length": 20.0, "angle": 0.5, "height_id": height_id,
            "under_pressure": False, "counterpress": False,
            "cross": False, "switch": False,
            "goal_assist": False, "shot_assist": False,
        })
        feat = build_pass_edge_feature(row)
        assert feat[3 + height_id - 1] == 1.0
        # Other height bits 0
        for h in [1, 2, 3]:
            if h != height_id:
                assert feat[3 + h - 1] == 0.0


def test_adversarial_edge_feature_shape():
    row = pd.Series({
        "event_type": "Interception",
        "outcome_id": 4,  # Won
        "counterpress": True,
        "location_x": 60.0,
        "location_y": 40.0,
    })
    feat = build_adversarial_edge_feature(row)
    assert feat.shape == (ADV_EDGE_DIM,)


def test_adversarial_edge_feature_type_onehot():
    for i, etype in enumerate(["Interception", "Tackle", "BallRecovery"]):
        row = pd.Series({
            "event_type": etype,
            "outcome_id": 4,
            "counterpress": False,
            "location_x": 60.0,
            "location_y": 40.0,
        })
        feat = build_adversarial_edge_feature(row)
        assert feat[i] == 1.0


def test_turnover_edge_feature_shape():
    row = pd.Series({
        "event_type": "Miscontrol",
        "location_x": 50.0,
        "location_y": 30.0,
    })
    feat = build_turnover_edge_feature(row)
    assert feat.shape == (TURN_EDGE_DIM,)


def test_turnover_edge_feature_type():
    row_m = pd.Series({"event_type": "Miscontrol", "location_x": 50.0, "location_y": 30.0})
    row_d = pd.Series({"event_type": "Dispossessed", "location_x": 50.0, "location_y": 30.0})
    feat_m = build_turnover_edge_feature(row_m)
    feat_d = build_turnover_edge_feature(row_d)
    assert feat_m[0] == 1.0  # Miscontrol at index 0
    assert feat_m[1] == 0.0
    assert feat_d[1] == 1.0  # Dispossessed at index 1
    assert feat_d[0] == 0.0
