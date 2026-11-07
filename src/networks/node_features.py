"""
Node feature vector construction for heterogeneous graph.
Each player node gets a fixed-dimension feature vector.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .spatial import normalize_coords, zone_onehot, compute_player_centroid

# Feature vector layout:
# [0:25]   position one-hot (25 tactical positions)
# [25]     team side: 1=home, 0=away
# [26]     pass count (normalized)
# [27]     receive count (normalized)
# [28]     normalized mean x
# [29]     normalized mean y
# [30:39]  zone one-hot (9 zones) of mean position
# Total: 39 dimensions

NODE_FEATURE_DIM = 39
N_POSITIONS = 25
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def build_node_feature(
    player_id: int,
    is_home: bool,
    position_id: Optional[int],
    pass_count: int,
    receive_count: int,
    mean_x: float,
    mean_y: float,
    max_pass_count: float = 100.0,
    max_receive_count: float = 100.0,
) -> np.ndarray:
    """Construct node feature vector for one player."""
    feat = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

    # Position one-hot [0:25]
    if position_id is not None and 1 <= position_id <= N_POSITIONS:
        feat[position_id - 1] = 1.0

    # Team side [25]
    feat[25] = 1.0 if is_home else 0.0

    # Pass/receive counts [26, 27]
    feat[26] = min(pass_count / max(max_pass_count, 1), 1.0)
    feat[27] = min(receive_count / max(max_receive_count, 1), 1.0)

    # Normalized position [28, 29]
    feat[28] = mean_x / PITCH_LENGTH
    feat[29] = mean_y / PITCH_WIDTH

    # Zone one-hot [30:39]
    feat[30:39] = zone_onehot(mean_x, mean_y)

    return feat


def build_all_node_features(
    match_id: int,
    pass_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    player_side_map: Dict[int, str],
) -> Tuple[Dict[int, np.ndarray], List[int], List[int]]:
    """
    Build node feature vectors for all players in a match.

    Returns:
        features_dict: player_id -> feature vector
        home_players: list of home player IDs
        away_players: list of away player IDs
    """
    match_passes = pass_df[pass_df["match_id"] == match_id]
    match_lineups = lineups_df[lineups_df["match_id"] == match_id]
    match_structure = structure_df[structure_df["match_id"] == match_id]

    # Latest position assignment per player
    position_map: Dict[int, int] = {}
    if not match_structure.empty:
        # Use Starting XI (minute=0) first, then Tactical Shift overrides
        struct_sorted = match_structure.sort_values("minute")
        for _, row in struct_sorted.iterrows():
            if pd.notna(row.get("position_id")):
                position_map[int(row["player_id"])] = int(row["position_id"])

    # Pass/receive counts
    pass_counts = match_passes.groupby("player_id").size().to_dict()
    receive_counts = match_passes.groupby("recipient_id").size().to_dict()

    # Spatial centroids: combine pass start and adversarial locations
    all_locs = pd.concat([
        match_passes[["player_id", "location_x", "location_y"]],
        adv_df[adv_df["match_id"] == match_id][["player_id", "location_x", "location_y"]],
    ], ignore_index=True)

    centroid_df = all_locs.groupby("player_id")[["location_x", "location_y"]].mean()

    # All players in this match
    all_player_ids = set(match_lineups["player_id"].tolist())
    all_player_ids.update(match_passes["player_id"].tolist())
    all_player_ids.update(match_passes["recipient_id"].dropna().astype(int).tolist())

    max_pass = max(pass_counts.values(), default=1)
    max_recv = max(receive_counts.values(), default=1)

    features_dict: Dict[int, np.ndarray] = {}
    home_players: List[int] = []
    away_players: List[int] = []

    for pid in all_player_ids:
        pid = int(pid)
        side = player_side_map.get(pid, "home")
        is_home = (side == "home")

        if pid in centroid_df.index:
            mx = float(centroid_df.loc[pid, "location_x"])
            my = float(centroid_df.loc[pid, "location_y"])
        else:
            mx, my = PITCH_LENGTH / 2, PITCH_WIDTH / 2

        feat = build_node_feature(
            player_id=pid,
            is_home=is_home,
            position_id=position_map.get(pid),
            pass_count=pass_counts.get(pid, 0),
            receive_count=receive_counts.get(pid, 0),
            mean_x=mx,
            mean_y=my,
            max_pass_count=float(max_pass),
            max_receive_count=float(max_recv),
        )
        features_dict[pid] = feat

        if is_home:
            home_players.append(pid)
        else:
            away_players.append(pid)

    return features_dict, sorted(home_players), sorted(away_players)
