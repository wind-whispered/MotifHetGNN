"""
Edge feature vector construction for heterogeneous graph.
"""
import numpy as np
import pandas as pd
from typing import Optional

# Pass edge feature layout:
# [0]    length_norm
# [1]    sin(angle)
# [2]    cos(angle)
# [3:6]  height one-hot (ground, low, high)
# [6]    under_pressure
# [7]    counterpress
# [8]    cross
# [9]    switch
# [10]   goal_assist
# [11]   shot_assist
# Total: 12 dimensions
PASS_EDGE_DIM = 12

# Adversarial edge feature layout:
# [0:3]  event_type one-hot (Interception, Tackle, BallRecovery)
# [3]    outcome_success (1=won/success, 0=lost/fail)
# [4]    counterpress
# [5:14] zone one-hot (9 zones)
# Total: 14 dimensions
ADV_EDGE_DIM = 14

# Turnover edge feature layout:
# [0:2]  event_type one-hot (Miscontrol, Dispossessed)
# [2:11] zone one-hot (9 zones)
# Total: 11 dimensions
TURN_EDGE_DIM = 11

MAX_PASS_LENGTH = 130.0

ADV_TYPE_ORDER = ["Interception", "Tackle", "BallRecovery"]
TURN_TYPE_ORDER = ["Miscontrol", "Dispossessed"]

SUCCESS_OUTCOME_IDS = {4, 15, 16, 17}  # Won, Success, Success In Play, Success Out


def build_pass_edge_feature(row: pd.Series) -> np.ndarray:
    """Build edge feature vector for a pass event row."""
    feat = np.zeros(PASS_EDGE_DIM, dtype=np.float32)

    length = row.get("length")
    feat[0] = min(float(length) / MAX_PASS_LENGTH, 1.0) if pd.notna(length) else 0.5

    angle = row.get("angle")
    if pd.notna(angle):
        feat[1] = float(np.sin(angle))
        feat[2] = float(np.cos(angle))

    height_id = row.get("height_id")
    if height_id in (1, 2, 3):
        feat[3 + int(height_id) - 1] = 1.0

    feat[6] = float(bool(row.get("under_pressure", False)))
    feat[7] = float(bool(row.get("counterpress", False)))
    feat[8] = float(bool(row.get("cross", False)))
    feat[9] = float(bool(row.get("switch", False)))
    feat[10] = float(bool(row.get("goal_assist", False)))
    feat[11] = float(bool(row.get("shot_assist", False)))

    return feat


def build_adversarial_edge_feature(row: pd.Series) -> np.ndarray:
    """Build edge feature vector for an adversarial event row."""
    from .spatial import zone_onehot
    feat = np.zeros(ADV_EDGE_DIM, dtype=np.float32)

    evt_type = row.get("event_type", "")
    if evt_type in ADV_TYPE_ORDER:
        feat[ADV_TYPE_ORDER.index(evt_type)] = 1.0

    outcome_id = row.get("outcome_id")
    feat[3] = 1.0 if (outcome_id in SUCCESS_OUTCOME_IDS) else 0.0
    feat[4] = float(bool(row.get("counterpress", False)))

    x = row.get("location_x")
    y = row.get("location_y")
    feat[5:14] = zone_onehot(x, y)

    return feat


def build_turnover_edge_feature(row: pd.Series) -> np.ndarray:
    """Build edge feature vector for a turnover event row."""
    from .spatial import zone_onehot
    feat = np.zeros(TURN_EDGE_DIM, dtype=np.float32)

    evt_type = row.get("event_type", "")
    if evt_type in TURN_TYPE_ORDER:
        feat[TURN_TYPE_ORDER.index(evt_type)] = 1.0

    x = row.get("location_x")
    y = row.get("location_y")
    feat[2:11] = zone_onehot(x, y)

    return feat
