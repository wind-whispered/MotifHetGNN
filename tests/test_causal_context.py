import numpy as np
import pandas as pd

from src.gnn.causal_context import (
    MATCH_STATE_FEATURE_NAMES,
    causal_team_strength_features,
    match_state_features,
)


def _state_dict(events, cutoff=30):
    values = match_state_features(events, 1, 2, cutoff)
    return dict(zip(MATCH_STATE_FEATURE_NAMES, values))


def test_match_state_is_half_open_and_does_not_read_assist_annotations():
    included_pass = {
        "period": 1, "minute": 29, "second": 59,
        "type": {"name": "Pass"}, "team": {"id": 1},
        "possession": 1, "possession_team": {"id": 1},
        "pass": {
            "end_location": [105, 40],
            "goal_assist": True,
            "shot_assist": True,
        },
    }
    excluded_shot = {
        "period": 1, "minute": 30, "second": 0,
        "type": {"name": "Shot"}, "team": {"id": 1},
        "shot": {"statsbomb_xg": 0.99,
                 "outcome": {"name": "Goal"},
                 "type": {"name": "Open Play"}},
    }
    shootout_shot = {
        "period": 5, "minute": 25, "second": 0,
        "type": {"name": "Shot"}, "team": {"id": 1},
        "shot": {"statsbomb_xg": 0.99,
                 "outcome": {"name": "Goal"},
                 "type": {"name": "Penalty"}},
    }
    with_assists = _state_dict(
        [included_pass, excluded_shot, shootout_shot])
    without_assists_event = {
        **included_pass,
        "pass": {**included_pass["pass"],
                 "goal_assist": False, "shot_assist": False},
    }
    without_assists = _state_dict(
        [without_assists_event, excluded_shot, shootout_shot])

    assert with_assists == without_assists
    assert with_assists["home_pass_count"] == 1
    assert with_assists["home_box_pass_count"] == 1
    assert with_assists["home_shot_count"] == 0
    assert with_assists["home_shot_xg"] == 0


def test_tackles_are_counted_from_the_duel_subtype():
    event = {
        "period": 1, "minute": 12, "second": 0,
        "type": {"name": "Duel"}, "team": {"id": 1},
        "duel": {
            "type": {"name": "Tackle"},
            "outcome": {"name": "Won"},
        },
    }
    state = _state_dict([event])
    assert state["home_duel_count"] == 1
    assert state["home_tackle_count"] == 1
    assert state["home_won_duel_count"] == 1


def test_same_date_outcomes_are_batch_updated_after_feature_recording():
    metadata = pd.DataFrame([
        {"match_id": 1, "match_date": "2020-01-01",
         "home_team_id": 10, "away_team_id": 20,
         "home_score": 2, "away_score": 0},
        {"match_id": 2, "match_date": "2020-01-01",
         "home_team_id": 10, "away_team_id": 30,
         "home_score": 0, "away_score": 1},
        {"match_id": 3, "match_date": "2020-01-02",
         "home_team_id": 10, "away_team_id": 40,
         "home_score": 1, "away_score": 1},
    ])
    features = causal_team_strength_features(metadata)

    assert features.loc[1, "pre_home_elo"] == 1500
    assert features.loc[2, "pre_home_elo"] == 1500
    assert features.loc[1, "pre_home_games"] == 0
    assert features.loc[2, "pre_home_games"] == 0
    assert features.loc[3, "pre_home_games"] == 2
    assert not np.isclose(features.loc[3, "pre_home_elo"], 1500)


def test_future_outcome_changes_only_later_strength_features():
    metadata = pd.DataFrame([
        {"match_id": 1, "match_date": "2020-01-01",
         "home_team_id": 10, "away_team_id": 20,
         "home_score": 2, "away_score": 0},
        {"match_id": 2, "match_date": "2020-01-02",
         "home_team_id": 10, "away_team_id": 30,
         "home_score": 1, "away_score": 1},
    ])
    changed = metadata.copy()
    changed.loc[changed.match_id.eq(1), ["home_score", "away_score"]] = [0, 3]
    original_features = causal_team_strength_features(metadata)
    changed_features = causal_team_strength_features(changed)

    np.testing.assert_allclose(
        original_features.loc[1], changed_features.loc[1])
    assert not np.allclose(
        original_features.loc[2], changed_features.loc[2])
