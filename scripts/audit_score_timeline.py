"""Audit cutoff-score reconstruction against official match scores.

The audit deliberately retains a local reproduction of the superseded score
logic.  This proves both that the corrected implementation matches the
official metadata and that the version change is material rather than a label
only: the old logic counted paired own-goal records twice and included period-5
shoot-out goals.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.temporal import (
    SCORE_TIMELINE_REVISION, build_score_timeline,
)


CUTOFFS = (15, 30, 45, 60, 75)
SHOT_ID = 16
GOAL_OUTCOME_ID = 97
OWN_GOAL_FOR_ID = 25
OWN_GOAL_AGAINST_ID = 20


def _score_at(timeline, cutoff=float("inf")):
    home = away = 0
    for elapsed, current_home, current_away in timeline:
        if elapsed < cutoff:
            home, away = int(current_home), int(current_away)
        else:
            break
    return home, away


def _legacy_timeline(events, home_team_id, away_team_id):
    """Exact reproduction of the invalid pre-audit implementation."""
    timeline = [(0.0, 0, 0)]
    home_score = away_score = 0
    for event in sorted(events, key=lambda item: (
            item.get("period", 1), item.get("minute", 0),
            item.get("second", 0), item.get("index", 0))):
        event_type = event.get("type", {}).get("id")
        elapsed = (float(event.get("minute", 0) or 0)
                   + float(event.get("second", 0) or 0) / 60.0)
        team_id = event.get("team", {}).get("id")
        if event_type == SHOT_ID:
            outcome = (event.get("shot", {}) or {}).get("outcome", {}).get("id")
            if outcome != GOAL_OUTCOME_ID:
                continue
            if team_id == home_team_id:
                home_score += 1
            else:
                away_score += 1
        elif event_type == OWN_GOAL_FOR_ID:
            if team_id == home_team_id:
                home_score += 1
            else:
                away_score += 1
        elif event_type == OWN_GOAL_AGAINST_ID:
            if team_id == home_team_id:
                away_score += 1
            else:
                home_score += 1
        else:
            continue
        timeline.append((elapsed, home_score, away_score))
    return timeline


def _own_goal_key(event):
    return (
        int(event.get("period", 1) or 1),
        int(event.get("minute", 0) or 0),
        float(event.get("second", 0) or 0),
    )


def _audit_match(record):
    event_path = ROOT / "StatsBomb" / "data" / "events" / (
        f"{int(record['match_id'])}.json")
    if not event_path.exists():
        return {"match_id": int(record["match_id"]), "missing": True}
    events = json.loads(event_path.read_text(encoding="utf-8"))
    corrected = build_score_timeline(
        events, int(record["home_team_id"]), int(record["away_team_id"]),
        include_seconds=True,
    )
    legacy = _legacy_timeline(
        events, int(record["home_team_id"]), int(record["away_team_id"]),
    )
    own_for = {_own_goal_key(event) for event in events
               if event.get("type", {}).get("id") == OWN_GOAL_FOR_ID}
    own_against = {_own_goal_key(event) for event in events
                   if event.get("type", {}).get("id") == OWN_GOAL_AGAINST_ID}
    corrected_final = _score_at(corrected)
    legacy_final = _score_at(legacy)
    official = (int(record["home_score"]), int(record["away_score"]))
    return {
        "match_id": int(record["match_id"]),
        "missing": False,
        "official_home_score": official[0],
        "official_away_score": official[1],
        "corrected_home_score": corrected_final[0],
        "corrected_away_score": corrected_final[1],
        "legacy_home_score": legacy_final[0],
        "legacy_away_score": legacy_final[1],
        "corrected_matches_official": corrected_final == official,
        "legacy_matches_official": legacy_final == official,
        "period5_present": any(
            int(event.get("period", 0) or 0) == 5 for event in events),
        "own_goal_for_records": sum(
            event.get("type", {}).get("id") == OWN_GOAL_FOR_ID
            for event in events),
        "own_goal_against_records": sum(
            event.get("type", {}).get("id") == OWN_GOAL_AGAINST_ID
            for event in events),
        "paired_own_goal_timestamps": len(own_for & own_against),
        "unpaired_own_goal_for_timestamps": len(own_for - own_against),
        "unpaired_own_goal_against_timestamps": len(own_against - own_for),
        "cutoff_changed": {
            str(cutoff): _score_at(corrected, cutoff) != _score_at(legacy, cutoff)
            for cutoff in CUTOFFS
        },
    }


def audit(workers=None):
    metadata = pd.read_parquet(
        ROOT / "data" / "processed" / "matches_meta.parquet")
    records = metadata.to_dict("records")
    workers = int(workers or os.environ.get("IS_SCORE_AUDIT_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        rows = list(pool.map(_audit_match, records))

    missing = [row for row in rows if row.get("missing")]
    available = [row for row in rows if not row.get("missing")]
    mismatches = [row for row in available
                  if not row["corrected_matches_official"]]
    legacy_mismatches = [row for row in available
                         if not row["legacy_matches_official"]]
    mismatch_columns = [
        "match_id", "official_home_score", "official_away_score",
        "corrected_home_score", "corrected_away_score",
        "legacy_home_score", "legacy_away_score", "period5_present",
        "own_goal_for_records", "own_goal_against_records",
        "paired_own_goal_timestamps", "unpaired_own_goal_for_timestamps",
        "unpaired_own_goal_against_timestamps",
    ]
    mismatch_path = (ROOT / "data" / "information_sciences" /
                     "score_timeline_mismatches.csv")
    pd.DataFrame(mismatches, columns=mismatch_columns).to_csv(
        mismatch_path, index=False)
    payload = {
        "status": "PASS" if not missing and not mismatches else "FAIL",
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "matches_expected": int(len(records)),
        "matches_audited": int(len(available)),
        "missing_event_files": int(len(missing)),
        "official_score_mismatches": int(len(mismatches)),
        "superseded_logic_score_mismatches": int(len(legacy_mismatches)),
        "period5_matches": int(sum(row["period5_present"] for row in available)),
        "own_goal_for_records": int(sum(
            row["own_goal_for_records"] for row in available)),
        "own_goal_against_records": int(sum(
            row["own_goal_against_records"] for row in available)),
        "paired_own_goal_timestamps": int(sum(
            row["paired_own_goal_timestamps"] for row in available)),
        "unpaired_own_goal_for_timestamps": int(sum(
            row["unpaired_own_goal_for_timestamps"] for row in available)),
        "unpaired_own_goal_against_timestamps": int(sum(
            row["unpaired_own_goal_against_timestamps"] for row in available)),
        "cutoff_changed_matches": {
            str(cutoff): int(sum(row["cutoff_changed"][str(cutoff)]
                                 for row in available))
            for cutoff in CUTOFFS
        },
        "mismatch_examples": [
            {key: row[key] for key in mismatch_columns}
            for row in mismatches[:25]
        ],
    }
    output = (ROOT / "data" / "information_sciences" /
              "score_timeline_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output, mismatch_path


def main():
    payload, output, mismatch_path = audit()
    print(json.dumps(payload, indent=2))
    print(output)
    print(mismatch_path)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
