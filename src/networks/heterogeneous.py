"""
Task 3: Heterogeneous graph construction using PyTorch Geometric HeteroData.
"""
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import pandas as pd
# torch imported lazily inside functions to avoid hard dependency at module load

logger = logging.getLogger(__name__)


def _ids_to_index(player_ids: List[int]) -> Dict[int, int]:
    """Map player IDs to consecutive 0-based indices."""
    return {pid: idx for idx, pid in enumerate(player_ids)}


def build_heterogeneous_graph(
    match_id: int,
    home_team_id: int,
    away_team_id: int,
    pass_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    substitutions_df: pd.DataFrame,
    goal_diff: int,
    player_side_map: Dict[int, str],
    w0: int = 0,
):
    """
    Build a HeteroData graph for one match.
    Requires torch_geometric to be installed.

    Node types: 'home_player', 'away_player'
    Edge types:
        (home_player, pass, home_player)
        (away_player, pass, away_player)
        (home_player, adversarial, away_player)
        (away_player, adversarial, home_player)
        (home_player, turnover, away_player)
        (away_player, turnover, home_player)
    """
    import torch
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError("torch_geometric is required. Install with: pip install torch_geometric")

    from .node_features import build_all_node_features, NODE_FEATURE_DIM
    from .edge_features import (
        build_pass_edge_feature, build_adversarial_edge_feature,
        build_turnover_edge_feature, PASS_EDGE_DIM, ADV_EDGE_DIM, TURN_EDGE_DIM,
        TURN_TYPE_ORDER, ADV_TYPE_ORDER,
    )

    # Build node features
    feat_dict, home_players, away_players = build_all_node_features(
        match_id, pass_df, adv_df, lineups_df, structure_df, player_side_map
    )

    home_idx = _ids_to_index(home_players)
    away_idx = _ids_to_index(away_players)

    data = HeteroData()

    # Node feature matrices
    if home_players:
        home_feats = np.stack([feat_dict[pid] for pid in home_players])
        data["home_player"].x = torch.tensor(home_feats, dtype=torch.float)
        data["home_player"].player_ids = torch.tensor(home_players, dtype=torch.long)
    else:
        data["home_player"].x = torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float)

    if away_players:
        away_feats = np.stack([feat_dict[pid] for pid in away_players])
        data["away_player"].x = torch.tensor(away_feats, dtype=torch.float)
        data["away_player"].player_ids = torch.tensor(away_players, dtype=torch.long)
    else:
        data["away_player"].x = torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float)

    # Graph-level label
    data.y = torch.tensor([goal_diff], dtype=torch.float)
    data.match_id = match_id

    # ---------- Pass edges ----------
    match_passes = pass_df[pass_df["match_id"] == match_id]

    def _add_pass_edges(node_type_src: str, node_type_dst: str, team_id: int):
        idx_map = home_idx if node_type_src == "home_player" else away_idx
        team_passes = match_passes[match_passes["team_id"] == team_id]

        # Aggregate multiple passes between same pair
        edge_counts = (
            team_passes.groupby(["player_id", "recipient_id"])
            .agg(
                weight=("length", "count"),
                length_mean=("length", "mean"),
                angle_mean=("angle", "mean"),
                height_mode=("height_id", lambda x: x.mode()[0] if len(x) > 0 else 1),
                under_pressure_mean=("under_pressure", "mean"),
                counterpress_mean=("counterpress", "mean"),
                cross_mean=("cross", "mean"),
                switch_mean=("switch", "mean"),
                goal_assist_any=("goal_assist", "any"),
                shot_assist_any=("shot_assist", "any"),
            )
            .reset_index()
        )

        if w0 > 0:
            edge_counts = edge_counts[edge_counts["weight"] > w0]

        if edge_counts.empty:
            return

        src_list, dst_list, feat_list = [], [], []
        for _, row in edge_counts.iterrows():
            src_id = int(row["player_id"])
            dst_id = int(row["recipient_id"])
            if src_id not in idx_map or dst_id not in idx_map:
                continue
            src_list.append(idx_map[src_id])
            dst_list.append(idx_map[dst_id])
            # Build aggregated edge feature
            feat_row = {
                "length": row["length_mean"],
                "angle": row["angle_mean"],
                "height_id": row["height_mode"],
                "under_pressure": row["under_pressure_mean"] > 0.5,
                "counterpress": row["counterpress_mean"] > 0.5,
                "cross": row["cross_mean"] > 0.5,
                "switch": row["switch_mean"] > 0.5,
                "goal_assist": row["goal_assist_any"],
                "shot_assist": row["shot_assist_any"],
            }
            feat_list.append(build_pass_edge_feature(pd.Series(feat_row)))

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(np.array(feat_list), dtype=torch.float)
            key = (node_type_src, "pass", node_type_dst)
            data[key].edge_index = edge_index
            data[key].edge_attr = edge_attr

    _add_pass_edges("home_player", "home_player", home_team_id)
    _add_pass_edges("away_player", "away_player", away_team_id)

    # ---------- Adversarial and Turnover edges ----------
    match_adv = adv_df[adv_df["match_id"] == match_id]

    adv_types = ["Interception", "Tackle", "BallRecovery"]
    turn_types = ["Miscontrol", "Dispossessed"]

    def _add_adversarial_edges(attacker_type: str, defender_type: str, attacker_team_id: int):
        """Adversarial edge: from defending player to attacking player's team region."""
        # Events where the defender (opposite of attacker) acts
        defender_team_id = away_team_id if attacker_team_id == home_team_id else home_team_id
        def_idx = away_idx if attacker_type == "home_player" else home_idx
        att_idx = home_idx if attacker_type == "home_player" else away_idx

        subset = match_adv[
            (match_adv["event_type"].isin(adv_types)) &
            (match_adv["team_id"] == defender_team_id)
        ]
        if subset.empty:
            return

        src_list, dst_list, feat_list = [], [], []
        for _, row in subset.iterrows():
            src_id = int(row["player_id"])
            if src_id not in def_idx:
                continue
            # Connect to first available attacker as proxy (or skip if empty)
            if not att_idx:
                continue
            dst_id = list(att_idx.keys())[0]  # simplified: connect to first attacker
            src_list.append(def_idx[src_id])
            dst_list.append(att_idx[dst_id])
            feat_list.append(build_adversarial_edge_feature(row))

        if src_list:
            key = (defender_type, "adversarial", attacker_type)
            existing_ei = getattr(data[key], "edge_index", None)
            new_ei = torch.tensor([src_list, dst_list], dtype=torch.long)
            new_ea = torch.tensor(np.array(feat_list), dtype=torch.float)
            if existing_ei is not None:
                data[key].edge_index = torch.cat([existing_ei, new_ei], dim=1)
                data[key].edge_attr = torch.cat([data[key].edge_attr, new_ea], dim=0)
            else:
                data[key].edge_index = new_ei
                data[key].edge_attr = new_ea

    _add_adversarial_edges("home_player", "away_player", home_team_id)
    _add_adversarial_edges("away_player", "home_player", away_team_id)

    def _add_turnover_edges(losing_type: str, gaining_type: str, losing_team_id: int):
        losing_idx = home_idx if losing_type == "home_player" else away_idx
        gaining_idx = away_idx if losing_type == "home_player" else home_idx

        subset = match_adv[
            (match_adv["event_type"].isin(turn_types)) &
            (match_adv["team_id"] == losing_team_id)
        ]
        if subset.empty:
            return

        src_list, dst_list, feat_list = [], [], []
        for _, row in subset.iterrows():
            src_id = int(row["player_id"])
            if src_id not in losing_idx or not gaining_idx:
                continue
            dst_id = list(gaining_idx.keys())[0]
            src_list.append(losing_idx[src_id])
            dst_list.append(gaining_idx[dst_id])
            feat_list.append(build_turnover_edge_feature(row))

        if src_list:
            key = (losing_type, "turnover", gaining_type)
            data[key].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            data[key].edge_attr = torch.tensor(np.array(feat_list), dtype=torch.float)

    _add_turnover_edges("home_player", "away_player", home_team_id)
    _add_turnover_edges("away_player", "home_player", away_team_id)

    return data


def save_hetero_graph(data, path: str) -> None:
    import torch
    torch.save(data, path)


def load_hetero_graph(path: str):
    import torch
    return torch.load(path, weights_only=False)