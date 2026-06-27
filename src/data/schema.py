"""
Field definitions, data type constraints, and dataclasses
for StatsBomb event data.
"""
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Column lists for processed DataFrames
# ---------------------------------------------------------------------------
PASS_COLUMNS = [
    "match_id", "event_id", "period", "timestamp", "minute", "second",
    "possession", "play_pattern_id", "play_pattern_name",
    "team_id", "team_name", "player_id", "player_name",
    "recipient_id", "recipient_name",
    "location_x", "location_y", "end_x", "end_y",
    "length", "angle", "height_id", "height_name",
    "under_pressure", "counterpress",
    "cross", "switch", "goal_assist", "shot_assist",
    "body_part_id", "pass_type_id", "outcome_id",
]

ADVERSARIAL_COLUMNS = [
    "match_id", "event_id", "event_type",
    "period", "timestamp", "minute", "second",
    "possession", "play_pattern_id",
    "team_id", "team_name", "player_id", "player_name",
    "location_x", "location_y",
    "outcome_id", "outcome_name",
    "counterpress", "offensive",
    "duel_type_id", "duel_type_name",
]

MATCH_META_COLUMNS = [
    "match_id", "competition_id", "competition_name",
    "season_id", "season_name", "match_date",
    "home_team_id", "home_team_name",
    "away_team_id", "away_team_name",
    "home_score", "away_score", "goal_diff",
    "match_week", "competition_stage_id", "data_version",
]

# ---------------------------------------------------------------------------
# Event type IDs from StatsBomb spec
# ---------------------------------------------------------------------------
class EventTypeID:
    BALL_RECOVERY = 2
    DISPOSSESSED = 3
    DUEL = 4
    INTERCEPTION = 10
    SHOT = 16
    SUBSTITUTION = 19
    PASS = 30
    STARTING_XI = 35
    TACTICAL_SHIFT = 36
    MISCONTROL = 38
    CARRY = 43

TACKLE_TYPE_ID = 11

ADVERSARIAL_EVENT_IDS = [
    EventTypeID.MISCONTROL,
    EventTypeID.DISPOSSESSED,
    EventTypeID.INTERCEPTION,
    EventTypeID.DUEL,
    EventTypeID.BALL_RECOVERY,
]

# ---------------------------------------------------------------------------
# Position encoding (25 tactical positions)
# ---------------------------------------------------------------------------
POSITION_NAMES = {
    1: "Goalkeeper", 2: "Right Back", 3: "Right Center Back",
    4: "Center Back", 5: "Left Center Back", 6: "Left Back",
    7: "Right Wing Back", 8: "Left Wing Back",
    9: "Right Defensive Midfield", 10: "Center Defensive Midfield",
    11: "Left Defensive Midfield", 12: "Right Midfield",
    13: "Right Center Midfield", 14: "Center Midfield",
    15: "Left Center Midfield", 16: "Left Midfield",
    17: "Right Wing", 18: "Right Attacking Midfield",
    19: "Center Attacking Midfield", 20: "Left Attacking Midfield",
    21: "Left Wing", 22: "Right Center Forward",
    23: "Striker", 24: "Left Center Forward", 25: "Secondary Striker",
}

PLAY_PATTERN_MAP = {
    1: "Regular Play", 2: "From Corner", 3: "From Free Kick",
    4: "From Throw In", 5: "Other", 6: "From Counter",
    7: "From Goal Kick", 8: "From Keeper", 9: "From Kick Off",
}

# ---------------------------------------------------------------------------
# Spatial zones: x in [0,120], y in [0,80]
# ---------------------------------------------------------------------------
ZONE_X_BREAKS = [40.0, 80.0]
ZONE_Y_BREAKS = [26.67, 53.33]
ZONE_X_LABELS = ["defensive", "middle", "attacking"]
ZONE_Y_LABELS = ["left", "center", "right"]
ALL_ZONES = [f"{x}_{y}" for x in ZONE_X_LABELS for y in ZONE_Y_LABELS]


def get_zone(x: float, y: float) -> str:
    """Return 9-zone label for pitch coordinates."""
    xl = ZONE_X_LABELS[0] if x < ZONE_X_BREAKS[0] else (
        ZONE_X_LABELS[1] if x < ZONE_X_BREAKS[1] else ZONE_X_LABELS[2])
    yl = ZONE_Y_LABELS[0] if y < ZONE_Y_BREAKS[0] else (
        ZONE_Y_LABELS[1] if y < ZONE_Y_BREAKS[1] else ZONE_Y_LABELS[2])
    return f"{xl}_{yl}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MatchMeta:
    match_id: int
    competition_id: int
    competition_name: str
    season_id: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_score: int
    away_score: int
    goal_diff: int
    match_week: Optional[int] = None
    data_version: Optional[str] = None


@dataclass
class Substitution:
    match_id: int
    minute: int
    team_id: int
    player_off_id: int
    player_on_id: int
