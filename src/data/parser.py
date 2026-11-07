"""
Task 1 - Part B: Field extraction from raw StatsBomb event/match/lineup dicts.
Each function takes raw JSON-derived dicts and returns flat record dicts.
"""
from typing import Dict, List, Optional, Tuple
import logging

from .schema import EventTypeID, TACKLE_TYPE_ID, ADVERSARIAL_EVENT_IDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Match meta parsing
# ---------------------------------------------------------------------------
def parse_match_meta(match_dict: dict) -> dict:
    """Extract flat match-level record from raw match dict."""
    home = match_dict.get("home_team", {})
    away = match_dict.get("away_team", {})
    comp = match_dict.get("competition", {})
    season = match_dict.get("season", {})
    stage = match_dict.get("competition_stage", {})
    meta = match_dict.get("metadata", {})

    home_score = int(match_dict.get("home_score", 0) or 0)
    away_score = int(match_dict.get("away_score", 0) or 0)

    return {
        "match_id": int(match_dict["match_id"]),
        "competition_id": int(comp.get("competition_id", -1)),
        "competition_name": comp.get("competition_name", ""),
        "season_id": int(season.get("season_id", -1)),
        "season_name": season.get("season_name", ""),
        "match_date": match_dict.get("match_date", ""),
        "home_team_id": int(home.get("home_team_id", -1)),
        "home_team_name": home.get("home_team_name", ""),
        "away_team_id": int(away.get("away_team_id", -1)),
        "away_team_name": away.get("away_team_name", ""),
        "home_score": home_score,
        "away_score": away_score,
        "goal_diff": home_score - away_score,
        "match_week": match_dict.get("match_week"),
        "competition_stage_id": stage.get("id"),
        "competition_stage_name": stage.get("name", ""),
        "data_version": meta.get("data_version", ""),
    }


# ---------------------------------------------------------------------------
# Lineup parsing
# ---------------------------------------------------------------------------
def parse_lineups(match_id: int, lineup_list: List[dict]) -> List[dict]:
    """
    Parse lineup JSON -> list of player records.
    lineup_list: list of two team dicts (home, away).
    """
    records = []
    for team_dict in lineup_list:
        team_id = int(team_dict.get("team_id", -1))
        team_name = team_dict.get("team_name", "")
        for player in team_dict.get("lineup", []):
            records.append({
                "match_id": match_id,
                "team_id": team_id,
                "team_name": team_name,
                "player_id": int(player.get("player_id", -1)),
                "player_name": player.get("player_name", ""),
                "jersey_number": player.get("jersey_number"),
                "country_id": player.get("country", {}).get("id"),
                "country_name": player.get("country", {}).get("name", ""),
            })
    return records


# ---------------------------------------------------------------------------
# Generic event field helpers
# ---------------------------------------------------------------------------
def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _get_common_fields(event: dict, match_id: int) -> dict:
    """Extract fields common to all event types."""
    loc = event.get("location") or [None, None]
    play_pat = event.get("play_pattern", {})
    team = event.get("team", {})
    player = event.get("player", {})
    return {
        "match_id": match_id,
        "event_id": event.get("id", ""),
        "index": event.get("index"),
        "period": event.get("period"),
        "timestamp": event.get("timestamp", ""),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "possession": event.get("possession"),
        "play_pattern_id": play_pat.get("id"),
        "play_pattern_name": play_pat.get("name", ""),
        "team_id": int(team.get("id", -1)) if team else -1,
        "team_name": team.get("name", "") if team else "",
        "player_id": int(player.get("id", -1)) if player else -1,
        "player_name": player.get("name", "") if player else "",
        "location_x": _safe_float(loc[0]) if len(loc) > 0 else None,
        "location_y": _safe_float(loc[1]) if len(loc) > 1 else None,
        "under_pressure": bool(event.get("under_pressure", False)),
        "counterpress": bool(event.get("counterpress", False)),
    }


# ---------------------------------------------------------------------------
# Pass event parsing
# ---------------------------------------------------------------------------
def parse_pass_event(event: dict, match_id: int) -> Optional[dict]:
    """
    Parse a single Pass event dict.
    Returns None if pass is not successful (outcome field present).
    """
    pass_data = event.get("pass", {})
    if not pass_data:
        return None

    # Successful pass: outcome key absent in pass sub-dict
    outcome = pass_data.get("outcome")
    if outcome is not None:
        return None  # incomplete / out / offside etc.

    recipient = pass_data.get("recipient", {}) or {}
    height = pass_data.get("height", {}) or {}
    body_part = pass_data.get("body_part", {}) or {}
    pass_type = pass_data.get("type", {}) or {}
    technique = pass_data.get("technique", {}) or {}
    end_loc = pass_data.get("end_location") or [None, None]

    common = _get_common_fields(event, match_id)
    common.update({
        "recipient_id": int(recipient.get("id", -1)) if recipient.get("id") else None,
        "recipient_name": recipient.get("name", ""),
        "end_x": _safe_float(end_loc[0]) if len(end_loc) > 0 else None,
        "end_y": _safe_float(end_loc[1]) if len(end_loc) > 1 else None,
        "length": _safe_float(pass_data.get("length")),
        "angle": _safe_float(pass_data.get("angle")),
        "height_id": height.get("id"),
        "height_name": height.get("name", ""),
        "body_part_id": body_part.get("id"),
        "body_part_name": body_part.get("name", ""),
        "pass_type_id": pass_type.get("id"),
        "pass_type_name": pass_type.get("name", ""),
        "technique_id": technique.get("id"),
        "technique_name": technique.get("name", ""),
        "cross": bool(pass_data.get("cross", False)),
        "switch": bool(pass_data.get("switch", False)),
        "goal_assist": bool(pass_data.get("goal_assist", False)),
        "shot_assist": bool(pass_data.get("shot_assist", False)),
        "backheel": bool(pass_data.get("backheel", False)),
        "deflected": bool(pass_data.get("deflected", False)),
        "miscommunication": bool(pass_data.get("miscommunication", False)),
        "outcome_id": None,   # successful -> no outcome
        "outcome_name": "",
    })
    return common


# ---------------------------------------------------------------------------
# Adversarial event parsing
# ---------------------------------------------------------------------------
def parse_adversarial_event(event: dict, match_id: int) -> Optional[dict]:
    """
    Parse Miscontrol / Dispossessed / Interception / Duel(Tackle) /
    BallRecovery events.
    Returns None if event type not in adversarial set.
    """
    type_id = event.get("type", {}).get("id")
    type_name = event.get("type", {}).get("name", "")

    if type_id not in ADVERSARIAL_EVENT_IDS:
        return None

    # For Duel: only keep Tackle sub-type
    if type_id == EventTypeID.DUEL:
        duel_data = event.get("duel", {}) or {}
        duel_type = duel_data.get("type", {}) or {}
        if duel_type.get("id") != TACKLE_TYPE_ID:
            return None
        duel_type_id = duel_type.get("id")
        duel_type_name = duel_type.get("name", "")
        outcome = duel_data.get("outcome", {}) or {}
        event_label = "Tackle"
    else:
        duel_type_id = None
        duel_type_name = ""
        # Get sub-dict outcome
        sub_key = {
            EventTypeID.MISCONTROL: None,
            EventTypeID.DISPOSSESSED: None,
            EventTypeID.INTERCEPTION: "interception",
            EventTypeID.BALL_RECOVERY: "ball_recovery",
        }.get(type_id)

        if sub_key:
            sub_data = event.get(sub_key, {}) or {}
            outcome = sub_data.get("outcome", {}) or {}
        else:
            outcome = {}

        label_map = {
            EventTypeID.MISCONTROL: "Miscontrol",
            EventTypeID.DISPOSSESSED: "Dispossessed",
            EventTypeID.INTERCEPTION: "Interception",
            EventTypeID.BALL_RECOVERY: "BallRecovery",
        }
        event_label = label_map.get(type_id, type_name)

    # offensive flag: relevant for BallRecovery
    br_data = event.get("ball_recovery", {}) or {}
    offensive = bool(br_data.get("offensive", False))

    common = _get_common_fields(event, match_id)
    common.update({
        "event_type": event_label,
        "outcome_id": outcome.get("id") if outcome else None,
        "outcome_name": outcome.get("name", "") if outcome else "",
        "offensive": offensive,
        "duel_type_id": duel_type_id,
        "duel_type_name": duel_type_name,
    })
    return common


# ---------------------------------------------------------------------------
# Structure event parsing (Starting XI / Tactical Shift)
# ---------------------------------------------------------------------------
def parse_structure_event(event: dict, match_id: int) -> List[dict]:
    """
    Parse Starting XI or Tactical Shift event.
    Returns list of per-player position records.
    """
    type_id = event.get("type", {}).get("id")
    type_name = event.get("type", {}).get("name", "Starting XI")
    if type_id not in (EventTypeID.STARTING_XI, EventTypeID.TACTICAL_SHIFT):
        return []

    team = event.get("team", {}) or {}
    tactics = event.get("tactics", {}) or {}
    formation = tactics.get("formation")
    lineup = tactics.get("lineup", []) or []
    minute = event.get("minute", 0)

    records = []
    for entry in lineup:
        player = entry.get("player", {}) or {}
        position = entry.get("position", {}) or {}
        records.append({
            "match_id": match_id,
            "event_id": event.get("id", ""),
            "event_type": type_name,
            "period": event.get("period"),
            "minute": minute,
            "team_id": int(team.get("id", -1)),
            "team_name": team.get("name", ""),
            "formation": formation,
            "player_id": int(player.get("id", -1)) if player.get("id") else -1,
            "player_name": player.get("name", ""),
            "position_id": position.get("id"),
            "position_name": position.get("name", ""),
            "valid_from_minute": minute,
        })
    return records


# ---------------------------------------------------------------------------
# Substitution parsing
# ---------------------------------------------------------------------------
def parse_substitution_event(event: dict, match_id: int) -> Optional[dict]:
    """Parse Substitution event."""
    if event.get("type", {}).get("id") != EventTypeID.SUBSTITUTION:
        return None
    sub_data = event.get("substitution", {}) or {}
    replacement = sub_data.get("replacement", {}) or {}
    player = event.get("player", {}) or {}
    team = event.get("team", {}) or {}
    return {
        "match_id": match_id,
        "minute": event.get("minute", 0),
        "team_id": int(team.get("id", -1)),
        "team_name": team.get("name", ""),
        "player_off_id": int(player.get("id", -1)) if player.get("id") else -1,
        "player_off_name": player.get("name", ""),
        "player_on_id": int(replacement.get("id", -1)) if replacement.get("id") else -1,
        "player_on_name": replacement.get("name", ""),
    }


# ---------------------------------------------------------------------------
# Batch parsing of all events for one match
# ---------------------------------------------------------------------------
def parse_all_events(
    match_id: int, events: List[dict]
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Parse all events for one match.
    Returns (pass_records, adversarial_records, structure_records, sub_records).
    """
    passes, adversarials, structures, subs = [], [], [], []

    for ev in events:
        type_id = ev.get("type", {}).get("id")

        if type_id == EventTypeID.PASS:
            rec = parse_pass_event(ev, match_id)
            if rec:
                passes.append(rec)

        elif type_id in ADVERSARIAL_EVENT_IDS:
            rec = parse_adversarial_event(ev, match_id)
            if rec:
                adversarials.append(rec)

        elif type_id in (EventTypeID.STARTING_XI, EventTypeID.TACTICAL_SHIFT):
            recs = parse_structure_event(ev, match_id)
            structures.extend(recs)

        elif type_id == EventTypeID.SUBSTITUTION:
            rec = parse_substitution_event(ev, match_id)
            if rec:
                subs.append(rec)

    return passes, adversarials, structures, subs
