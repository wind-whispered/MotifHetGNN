"""Strictly pre-cutoff and pre-match features for strong causal baselines.

The helpers in this module deliberately keep labels outside the feature
extractors.  Match-state features use only events timestamped before a named
cutoff.  Team-strength features are recorded before a calendar-date batch is
updated, so two matches on the same date cannot leak outcomes to one another.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


MATCH_STATE_REVISION = "strict_prefix_score_xg_event_state_v3"
TEAM_STRENGTH_REVISION = "predate_batch_elo_form_v1"

TEAM_STATE_FEATURES = (
    "event_count",
    "possession_count",
    "pass_count",
    "completed_pass_count",
    "pass_completion_rate",
    "pressured_pass_count",
    "cross_count",
    "switch_count",
    "final_third_pass_count",
    "box_pass_count",
    "carry_count",
    "final_third_carry_count",
    "box_carry_count",
    "shot_count",
    "shot_xg",
    "shot_xg_per_shot",
    "max_shot_xg",
    "on_target_count",
    "open_play_shot_count",
    "set_piece_shot_count",
    "penalty_shot_count",
    "pressure_count",
    "interception_count",
    "tackle_count",
    "ball_recovery_count",
    "duel_count",
    "won_duel_count",
    "miscontrol_count",
    "dispossessed_count",
    "foul_won_count",
    "foul_committed_count",
    "corner_count",
)

MATCH_STATE_FEATURE_NAMES = tuple(
    f"{side}_{name}"
    for side in ("home", "away", "difference")
    for name in TEAM_STATE_FEATURES
) + ("home_possession_share",)

TEAM_STRENGTH_FEATURE_NAMES = (
    "pre_home_elo",
    "pre_away_elo",
    "pre_elo_difference",
    "pre_home_win_expectancy",
    "pre_home_games",
    "pre_away_games",
    "pre_home_unseen",
    "pre_away_unseen",
    "pre_home_form_result",
    "pre_away_form_result",
    "pre_form_result_difference",
    "pre_home_form_goal_difference",
    "pre_away_form_goal_difference",
    "pre_form_goal_difference_difference",
)


def _event_elapsed_minute(event: Mapping) -> float:
    minute = float(event.get("minute", 0) or 0)
    second = float(event.get("second", 0) or 0)
    return minute + second / 60.0


def _team_id(event: Mapping):
    team = event.get("team", {}) or {}
    value = team.get("id")
    return int(value) if value is not None else None


def _inside_box(location) -> bool:
    return bool(
        isinstance(location, (list, tuple))
        and len(location) >= 2
        and float(location[0]) >= 102.0
        and 18.0 <= float(location[1]) <= 62.0
    )


def _in_final_third(location) -> bool:
    return bool(
        isinstance(location, (list, tuple))
        and len(location) >= 1
        and float(location[0]) >= 80.0
    )


def match_state_features(
    events: Iterable[Mapping],
    home_team_id: int,
    away_team_id: int,
    cutoff: int,
) -> np.ndarray:
    """Summarise events in the strict half-open interval ``[0, cutoff)``.

    Shot outcome and StatsBomb xG are observed at the shot timestamp and are
    therefore valid match-state information.  Retrospective pass annotations
    such as ``goal_assist`` and ``shot_assist`` are never read.
    """
    teams = (int(home_team_id), int(away_team_id))
    counts = {team: defaultdict(float) for team in teams}
    possessions = {team: set() for team in teams}

    for event in events:
        if int(event.get("period", 1) or 1) == 5:
            continue
        if not (0.0 <= _event_elapsed_minute(event) < float(cutoff)):
            continue
        team = _team_id(event)
        if team not in counts:
            continue
        row = counts[team]
        row["event_count"] += 1.0
        possession = event.get("possession")
        possession_team = (event.get("possession_team", {}) or {}).get("id")
        if possession is not None and possession_team == team:
            possessions[team].add(int(possession))

        event_type = (event.get("type", {}) or {}).get("name", "")
        if event_type == "Pass":
            detail = event.get("pass", {}) or {}
            row["pass_count"] += 1.0
            if not detail.get("outcome"):
                row["completed_pass_count"] += 1.0
            row["pressured_pass_count"] += float(
                bool(event.get("under_pressure", False)))
            row["cross_count"] += float(bool(detail.get("cross", False)))
            row["switch_count"] += float(bool(detail.get("switch", False)))
            endpoint = detail.get("end_location")
            row["final_third_pass_count"] += float(_in_final_third(endpoint))
            row["box_pass_count"] += float(_inside_box(endpoint))
            play_pattern = (event.get("play_pattern", {}) or {}).get("name")
            row["corner_count"] += float(play_pattern == "From Corner")
        elif event_type == "Carry":
            endpoint = (event.get("carry", {}) or {}).get("end_location")
            row["carry_count"] += 1.0
            row["final_third_carry_count"] += float(_in_final_third(endpoint))
            row["box_carry_count"] += float(_inside_box(endpoint))
        elif event_type == "Shot":
            detail = event.get("shot", {}) or {}
            xg = float(detail.get("statsbomb_xg", 0.0) or 0.0)
            outcome = (detail.get("outcome", {}) or {}).get("name", "")
            shot_type = (detail.get("type", {}) or {}).get("name", "")
            row["shot_count"] += 1.0
            row["shot_xg"] += xg
            row["max_shot_xg"] = max(row["max_shot_xg"], xg)
            row["on_target_count"] += float(
                outcome == "Goal" or outcome.startswith("Saved"))
            row["open_play_shot_count"] += float(shot_type == "Open Play")
            is_penalty = shot_type == "Penalty"
            row["penalty_shot_count"] += float(is_penalty)
            row["set_piece_shot_count"] += float(
                shot_type not in {"", "Open Play", "Penalty"})
        elif event_type == "Pressure":
            row["pressure_count"] += 1.0
        elif event_type == "Interception":
            row["interception_count"] += 1.0
        elif event_type == "Tackle":
            row["tackle_count"] += 1.0
        elif event_type == "Ball Recovery":
            row["ball_recovery_count"] += 1.0
        elif event_type == "Duel":
            detail = event.get("duel", {}) or {}
            outcome = (detail.get("outcome", {}) or {}).get("name", "")
            duel_type = (detail.get("type", {}) or {}).get("name", "")
            row["duel_count"] += 1.0
            row["tackle_count"] += float(duel_type == "Tackle")
            row["won_duel_count"] += float(outcome in {
                "Won", "Success", "Success In Play", "Success Out"})
        elif event_type == "Miscontrol":
            row["miscontrol_count"] += 1.0
        elif event_type == "Dispossessed":
            row["dispossessed_count"] += 1.0
        elif event_type == "Foul Won":
            row["foul_won_count"] += 1.0
        elif event_type == "Foul Committed":
            row["foul_committed_count"] += 1.0

    vectors = []
    for team in teams:
        row = counts[team]
        row["possession_count"] = float(len(possessions[team]))
        row["pass_completion_rate"] = (
            row["completed_pass_count"] / row["pass_count"]
            if row["pass_count"] else 0.0)
        row["shot_xg_per_shot"] = (
            row["shot_xg"] / row["shot_count"]
            if row["shot_count"] else 0.0)
        vectors.append(np.asarray(
            [row[name] for name in TEAM_STATE_FEATURES], dtype=np.float64))

    home, away = vectors
    total_possessions = home[TEAM_STATE_FEATURES.index("possession_count")] + (
        away[TEAM_STATE_FEATURES.index("possession_count")])
    home_share = (
        home[TEAM_STATE_FEATURES.index("possession_count")] / total_possessions
        if total_possessions else 0.5)
    return np.concatenate([home, away, home - away, [home_share]])


def causal_team_strength_features(
    metadata: pd.DataFrame,
    elo_k: float = 20.0,
    home_advantage: float = 60.0,
    form_alpha: float = 0.25,
) -> pd.DataFrame:
    """Return team-strength features recorded before each date batch.

    Final outcomes are used only after every feature row for that calendar
    date has been recorded.  The returned frame is indexed by ``match_id``.
    """
    required = {
        "match_id", "match_date", "home_team_id", "away_team_id",
        "home_score", "away_score",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    frame = metadata.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    if bool(frame["match_date"].isna().any()):
        raise ValueError("causal strength features require every match date")
    frame = frame.sort_values(["match_date", "match_id"])

    ratings = defaultdict(lambda: 1500.0)
    games = defaultdict(int)
    form_result = defaultdict(float)
    form_goal_difference = defaultdict(float)
    records = []

    for _, date_group in frame.groupby(frame["match_date"].dt.normalize(),
                                       sort=True):
        rating_delta = defaultdict(float)
        daily_results = defaultdict(list)
        daily_goal_differences = defaultdict(list)

        for row in date_group.itertuples(index=False):
            home = int(row.home_team_id)
            away = int(row.away_team_id)
            home_rating = float(ratings[home])
            away_rating = float(ratings[away])
            expectancy = 1.0 / (
                1.0 + 10.0 ** (
                    -((home_rating + home_advantage) - away_rating) / 400.0))
            records.append({
                "match_id": int(row.match_id),
                "pre_home_elo": home_rating,
                "pre_away_elo": away_rating,
                "pre_elo_difference": home_rating - away_rating,
                "pre_home_win_expectancy": expectancy,
                "pre_home_games": float(games[home]),
                "pre_away_games": float(games[away]),
                "pre_home_unseen": float(games[home] == 0),
                "pre_away_unseen": float(games[away] == 0),
                "pre_home_form_result": form_result[home],
                "pre_away_form_result": form_result[away],
                "pre_form_result_difference": (
                    form_result[home] - form_result[away]),
                "pre_home_form_goal_difference": form_goal_difference[home],
                "pre_away_form_goal_difference": form_goal_difference[away],
                "pre_form_goal_difference_difference": (
                    form_goal_difference[home] - form_goal_difference[away]),
            })

            goal_difference = float(row.home_score - row.away_score)
            actual_home = 1.0 if goal_difference > 0 else (
                0.5 if goal_difference == 0 else 0.0)
            change = elo_k * (actual_home - expectancy)
            rating_delta[home] += change
            rating_delta[away] -= change
            signed_result = 1.0 if goal_difference > 0 else (
                0.0 if goal_difference == 0 else -1.0)
            daily_results[home].append(signed_result)
            daily_results[away].append(-signed_result)
            clipped = float(np.clip(goal_difference, -4.0, 4.0))
            daily_goal_differences[home].append(clipped)
            daily_goal_differences[away].append(-clipped)

        for team, delta in rating_delta.items():
            ratings[team] += delta
        for team, observations in daily_results.items():
            form_result[team] = (
                (1.0 - form_alpha) * form_result[team]
                + form_alpha * float(np.mean(observations)))
            form_goal_difference[team] = (
                (1.0 - form_alpha) * form_goal_difference[team]
                + form_alpha
                * float(np.mean(daily_goal_differences[team])))
            games[team] += len(observations)

    result = pd.DataFrame(records).set_index("match_id")
    return result.loc[:, TEAM_STRENGTH_FEATURE_NAMES]
