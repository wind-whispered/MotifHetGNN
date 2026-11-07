"""
Spatial utilities: coordinate normalization, zone labeling, pitch region encoding.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
ZONE_X_BREAKS = [40.0, 80.0]
ZONE_Y_BREAKS = [26.67, 53.33]


def normalize_coords(x: float, y: float) -> Tuple[float, float]:
    """Normalize pitch coordinates to [0, 1]."""
    return x / PITCH_LENGTH, y / PITCH_WIDTH


def zone_index(x: float, y: float) -> int:
    """Return integer zone index 0-8 for 9-zone grid."""
    xi = 0 if x < ZONE_X_BREAKS[0] else (1 if x < ZONE_X_BREAKS[1] else 2)
    yi = 0 if y < ZONE_Y_BREAKS[0] else (1 if y < ZONE_Y_BREAKS[1] else 2)
    return xi * 3 + yi


def zone_onehot(x: Optional[float], y: Optional[float]) -> np.ndarray:
    """Return 9-dim one-hot zone vector. Returns zeros if coords are None."""
    vec = np.zeros(9, dtype=np.float32)
    if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
        idx = zone_index(float(x), float(y))
        vec[idx] = 1.0
    return vec


def pass_direction_features(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> np.ndarray:
    """
    Return a 4-dim feature vector for pass direction:
    [delta_x_norm, delta_y_norm, sin(angle), cos(angle)]
    """
    dx = (end_x - start_x) / PITCH_LENGTH
    dy = (end_y - start_y) / PITCH_WIDTH
    angle = np.arctan2(dy, dx)
    return np.array([dx, dy, np.sin(angle), np.cos(angle)], dtype=np.float32)


def compute_player_centroid(
    events_df: pd.DataFrame,
    player_id: int,
) -> Tuple[float, float]:
    """
    Compute the mean pitch location of a player across all their events.
    events_df should have columns: player_id, location_x, location_y.
    """
    subset = events_df[events_df["player_id"] == player_id]
    if subset.empty:
        return PITCH_LENGTH / 2, PITCH_WIDTH / 2
    x_mean = subset["location_x"].dropna().mean()
    y_mean = subset["location_y"].dropna().mean()
    return float(x_mean) if not np.isnan(x_mean) else PITCH_LENGTH / 2, \
           float(y_mean) if not np.isnan(y_mean) else PITCH_WIDTH / 2
