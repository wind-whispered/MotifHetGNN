"""
Temporal utilities: possession unit segmentation, substitution handling,
match timeline reconstruction.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


def get_possession_units(pass_df: pd.DataFrame, match_id: int) -> List[Tuple[int, pd.DataFrame]]:
    """
    Split passes for one match into possession units.
    Returns list of (possession_id, sub_df) sorted by possession number.
    """
    match_passes = pass_df[pass_df["match_id"] == match_id].copy()
    match_passes = match_passes.sort_values(["possession", "minute", "second"])
    units = []
    for poss_id, group in match_passes.groupby("possession"):
        units.append((int(poss_id), group))
    return units


def build_score_timeline(
    events_raw: List[dict],
    home_team_id: int,
    away_team_id: int,
) -> List[Tuple[int, int, int]]:
    """
    Reconstruct score at each minute from raw events.
    Looks for Shot events with outcome=Goal.

    Returns list of (minute, home_score, away_score) sorted by minute.
    """
    SHOT_ID = 16
    GOAL_OUTCOME_ID = 97
    OWN_GOAL_FOR_ID = 25
    OWN_GOAL_AGAINST_ID = 20

    timeline = [(0, 0, 0)]  # (minute, home, away)
    home_score = 0
    away_score = 0

    for ev in sorted(events_raw, key=lambda e: (e.get("period", 1), e.get("minute", 0))):
        type_id = ev.get("type", {}).get("id")
        minute = ev.get("minute", 0)
        team_id = ev.get("team", {}).get("id")

        if type_id == SHOT_ID:
            shot_data = ev.get("shot", {}) or {}
            outcome_id = shot_data.get("outcome", {}).get("id")
            if outcome_id == GOAL_OUTCOME_ID:
                if team_id == home_team_id:
                    home_score += 1
                else:
                    away_score += 1
                timeline.append((minute, home_score, away_score))

        elif type_id == OWN_GOAL_FOR_ID:
            if team_id == home_team_id:
                home_score += 1
            else:
                away_score += 1
            timeline.append((minute, home_score, away_score))

        elif type_id == OWN_GOAL_AGAINST_ID:
            if team_id == home_team_id:
                away_score += 1
            else:
                home_score += 1
            timeline.append((minute, home_score, away_score))

    return timeline


def get_score_state_at_minute(
    timeline: List[Tuple[int, int, int]],
    minute: int,
    is_home_team: bool,
) -> str:
    """
    Return 'leading' | 'drawing' | 'trailing' at a given minute
    from the perspective of home or away team.
    """
    home_score, away_score = 0, 0
    for t_min, t_home, t_away in timeline:
        if t_min <= minute:
            home_score, away_score = t_home, t_away
        else:
            break

    diff = home_score - away_score
    if is_home_team:
        if diff > 0: return "leading"
        if diff < 0: return "trailing"
        return "drawing"
    else:
        if diff < 0: return "leading"
        if diff > 0: return "trailing"
        return "drawing"


def get_period_label(minute: int) -> str:
    """Return '0-30' | '30-60' | '60-90' | '90+' based on continuous minute."""
    if minute < 30: return "0-30"
    if minute < 60: return "30-60"
    if minute < 90: return "60-90"
    return "90+"


def build_player_team_map(
    lineups_df: pd.DataFrame,
    substitutions_df: pd.DataFrame,
    match_id: int,
    home_team_id: int,
) -> Dict[int, str]:
    """
    Build player_id -> 'home'|'away' mapping for a match,
    accounting for substitutions.
    """
    match_lineups = lineups_df[lineups_df["match_id"] == match_id]
    player_side = {}
    for _, row in match_lineups.iterrows():
        side = "home" if row["team_id"] == home_team_id else "away"
        player_side[int(row["player_id"])] = side

    # Substitutions: player coming on inherits the team side of player going off
    match_subs = substitutions_df[substitutions_df["match_id"] == match_id]
    for _, row in match_subs.iterrows():
        off_id = int(row["player_off_id"])
        on_id = int(row["player_on_id"])
        if off_id in player_side:
            player_side[on_id] = player_side[off_id]

    return player_side
