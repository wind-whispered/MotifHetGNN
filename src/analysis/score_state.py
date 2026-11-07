"""
Dynamic score state computation and motif stratification by match situation.
"""
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SCORE_STATES = ["leading", "drawing", "trailing"]


def build_possession_score_state(
    match_id: int,
    events_raw: List[dict],
    pass_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    """
    For each possession in a match, determine the score state
    (leading/drawing/trailing) from the home team's perspective.

    Returns DataFrame with columns:
        match_id, possession, score_state_home, score_state_away,
        home_score_at_time, away_score_at_time, minute
    """
    from ..networks.temporal import build_score_timeline

    timeline = build_score_timeline(events_raw, home_team_id, away_team_id)

    # Get possessions and their representative minutes
    match_passes = pass_df[pass_df["match_id"] == match_id]
    if match_passes.empty:
        return pd.DataFrame()

    poss_minutes = (
        match_passes.groupby("possession")["minute"]
        .mean()
        .reset_index()
        .rename(columns={"minute": "mean_minute"})
    )

    records = []
    for _, row in poss_minutes.iterrows():
        poss = int(row["possession"])
        minute = float(row["mean_minute"])

        # Find score at this minute
        h_score, a_score = 0, 0
        for t_min, t_home, t_away in timeline:
            if t_min <= minute:
                h_score, a_score = t_home, t_away
            else:
                break

        diff = h_score - a_score
        home_state = "leading" if diff > 0 else ("trailing" if diff < 0 else "drawing")
        away_state = "leading" if diff < 0 else ("trailing" if diff > 0 else "drawing")

        records.append({
            "match_id": match_id,
            "possession": poss,
            "mean_minute": minute,
            "home_score_at_time": h_score,
            "away_score_at_time": a_score,
            "score_state_home": home_state,
            "score_state_away": away_state,
        })

    return pd.DataFrame(records)


def merge_motif_with_score_state(
    motif_df: pd.DataFrame,
    score_state_df: pd.DataFrame,
    team_side_col: str = "team_side",
) -> pd.DataFrame:
    """
    Merge motif counts with score state information.
    Since motif_df is per-match (not per-possession), we use the
    dominant score state across the match as approximation.
    """
    # Compute dominant score state per match per team
    dominant_state = (
        score_state_df.groupby("match_id")
        .apply(lambda g: pd.Series({
            "home_dominant_state": g["score_state_home"].value_counts().index[0],
            "away_dominant_state": g["score_state_away"].value_counts().index[0],
            "home_leading_pct": (g["score_state_home"] == "leading").mean(),
            "home_trailing_pct": (g["score_state_home"] == "trailing").mean(),
            "home_drawing_pct": (g["score_state_home"] == "drawing").mean(),
        }))
        .reset_index()
    )

    merged = motif_df.merge(dominant_state, on="match_id", how="left")

    def _get_state(row):
        if row.get(team_side_col) == "home":
            return row.get("home_dominant_state", "drawing")
        return row.get("away_dominant_state", "drawing")

    merged["dominant_score_state"] = merged.apply(_get_state, axis=1)
    return merged


def compute_motif_by_score_state(
    motif_df: pd.DataFrame,
    score_state_df: pd.DataFrame,
    groupby_motif: List[str] = ["motif_id", "motif_order_k", "team_side"],
) -> pd.DataFrame:
    """
    Compute mean motif frequency stratified by score state.
    Returns DataFrame with motif_cols + score_state + mean_count + std_count.
    """
    merged = merge_motif_with_score_state(motif_df, score_state_df)

    records = []
    for keys, group in merged.groupby(groupby_motif + ["dominant_score_state"]):
        key_dict = dict(zip(groupby_motif + ["score_state"], keys))
        per_match = group.groupby("match_id")["count"].sum()
        row = dict(key_dict)
        row["mean_count"] = per_match.mean()
        row["std_count"] = per_match.std()
        row["n_matches"] = len(per_match)
        records.append(row)

    return pd.DataFrame(records)
