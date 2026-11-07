"""Tests for event parsing utilities."""
import pytest
from src.data.parser import (
    parse_match_meta, parse_lineups, parse_pass_event,
    parse_adversarial_event, parse_structure_event,
    parse_substitution_event,
)
from src.data.schema import EventTypeID


# ---------------------------------------------------------------------------
# Sample raw dicts (minimal valid StatsBomb-style)
# ---------------------------------------------------------------------------

SAMPLE_MATCH = {
    "match_id": 1001,
    "competition": {"competition_id": 2, "competition_name": "Test League"},
    "season": {"season_id": 1, "season_name": "2020/21"},
    "match_date": "2020-09-01",
    "home_team": {"home_team_id": 10, "home_team_name": "Home FC"},
    "away_team": {"away_team_id": 20, "away_team_name": "Away FC"},
    "home_score": 2,
    "away_score": 1,
    "match_week": 1,
    "competition_stage": {"id": 1, "name": "Regular Season"},
    "metadata": {"data_version": "1.1.0"},
}

SAMPLE_PASS_EVENT = {
    "id": "aaa-111",
    "index": 5,
    "type": {"id": 30, "name": "Pass"},
    "period": 1,
    "timestamp": "00:05:10.000",
    "minute": 5,
    "second": 10,
    "possession": 3,
    "play_pattern": {"id": 1, "name": "Regular Play"},
    "team": {"id": 10, "name": "Home FC"},
    "player": {"id": 101, "name": "Player A"},
    "location": [30.0, 40.0],
    "under_pressure": True,
    "counterpress": False,
    "pass": {
        "recipient": {"id": 102, "name": "Player B"},
        "length": 15.0,
        "angle": -0.5,
        "height": {"id": 1, "name": "Ground Pass"},
        "end_location": [45.0, 38.0],
        "body_part": {"id": 40, "name": "Right Foot"},
        "type": None,
        "technique": None,
        # no "outcome" field -> successful pass
    },
}

SAMPLE_INCOMPLETE_PASS = {
    **SAMPLE_PASS_EVENT,
    "id": "bbb-222",
    "pass": {
        **SAMPLE_PASS_EVENT["pass"],
        "outcome": {"id": 9, "name": "Incomplete"},
    },
}

SAMPLE_INTERCEPTION_EVENT = {
    "id": "ccc-333",
    "index": 10,
    "type": {"id": EventTypeID.INTERCEPTION, "name": "Interception"},
    "period": 1,
    "timestamp": "00:10:00.000",
    "minute": 10,
    "second": 0,
    "possession": 5,
    "play_pattern": {"id": 1, "name": "Regular Play"},
    "team": {"id": 20, "name": "Away FC"},
    "player": {"id": 201, "name": "Player C"},
    "location": [55.0, 35.0],
    "counterpress": True,
    "interception": {
        "outcome": {"id": 4, "name": "Won"},
    },
}

SAMPLE_DUEL_TACKLE_EVENT = {
    "id": "ddd-444",
    "index": 12,
    "type": {"id": EventTypeID.DUEL, "name": "Duel"},
    "period": 1,
    "timestamp": "00:12:00.000",
    "minute": 12,
    "second": 0,
    "possession": 6,
    "play_pattern": {"id": 1, "name": "Regular Play"},
    "team": {"id": 20, "name": "Away FC"},
    "player": {"id": 202, "name": "Player D"},
    "location": [60.0, 40.0],
    "counterpress": False,
    "duel": {
        "type": {"id": 11, "name": "Tackle"},
        "outcome": {"id": 4, "name": "Won"},
    },
}

SAMPLE_STARTING_XI_EVENT = {
    "id": "eee-555",
    "index": 1,
    "type": {"id": EventTypeID.STARTING_XI, "name": "Starting XI"},
    "period": 1,
    "minute": 0,
    "team": {"id": 10, "name": "Home FC"},
    "tactics": {
        "formation": 433,
        "lineup": [
            {"player": {"id": 101, "name": "Player A"},
             "position": {"id": 1, "name": "Goalkeeper"}},
            {"player": {"id": 102, "name": "Player B"},
             "position": {"id": 2, "name": "Right Back"}},
        ],
    },
}

SAMPLE_SUBSTITUTION_EVENT = {
    "id": "fff-666",
    "index": 80,
    "type": {"id": EventTypeID.SUBSTITUTION, "name": "Substitution"},
    "period": 2,
    "minute": 70,
    "team": {"id": 10, "name": "Home FC"},
    "player": {"id": 103, "name": "Player C"},
    "substitution": {
        "replacement": {"id": 110, "name": "Player X"},
        "outcome": {"id": 103, "name": "Tactical"},
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_match_meta_basic():
    rec = parse_match_meta(SAMPLE_MATCH)
    assert rec["match_id"] == 1001
    assert rec["home_team_id"] == 10
    assert rec["away_team_id"] == 20
    assert rec["home_score"] == 2
    assert rec["away_score"] == 1
    assert rec["goal_diff"] == 1
    assert rec["competition_name"] == "Test League"


def test_parse_lineups():
    lineup_list = [
        {
            "team_id": 10,
            "team_name": "Home FC",
            "lineup": [
                {"player_id": 101, "player_name": "Player A",
                 "jersey_number": 9, "country": {"id": 1, "name": "England"}},
            ],
        },
    ]
    records = parse_lineups(1001, lineup_list)
    assert len(records) == 1
    assert records[0]["player_id"] == 101
    assert records[0]["team_id"] == 10


def test_parse_successful_pass():
    rec = parse_pass_event(SAMPLE_PASS_EVENT, match_id=1001)
    assert rec is not None
    assert rec["player_id"] == 101
    assert rec["recipient_id"] == 102
    assert rec["location_x"] == 30.0
    assert rec["location_y"] == 40.0
    assert rec["end_x"] == 45.0
    assert rec["under_pressure"] is True
    assert rec["outcome_id"] is None  # successful pass


def test_parse_incomplete_pass_returns_none():
    rec = parse_pass_event(SAMPLE_INCOMPLETE_PASS, match_id=1001)
    assert rec is None  # incomplete passes are filtered out


def test_parse_interception():
    rec = parse_adversarial_event(SAMPLE_INTERCEPTION_EVENT, match_id=1001)
    assert rec is not None
    assert rec["event_type"] == "Interception"
    assert rec["player_id"] == 201
    assert rec["counterpress"] is True
    assert rec["outcome_name"] == "Won"


def test_parse_duel_tackle():
    rec = parse_adversarial_event(SAMPLE_DUEL_TACKLE_EVENT, match_id=1001)
    assert rec is not None
    assert rec["event_type"] == "Tackle"
    assert rec["player_id"] == 202


def test_parse_starting_xi():
    records = parse_structure_event(SAMPLE_STARTING_XI_EVENT, match_id=1001)
    assert len(records) == 2
    assert records[0]["team_id"] == 10
    assert records[0]["position_id"] == 1
    assert records[1]["position_id"] == 2


def test_parse_substitution():
    rec = parse_substitution_event(SAMPLE_SUBSTITUTION_EVENT, match_id=1001)
    assert rec is not None
    assert rec["player_off_id"] == 103
    assert rec["player_on_id"] == 110
    assert rec["minute"] == 70
