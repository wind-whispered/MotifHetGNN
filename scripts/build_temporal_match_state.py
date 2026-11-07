"""Build resumable, strictly pre-cutoff tabular match-state features."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.causal_context import (
    MATCH_STATE_FEATURE_NAMES,
    MATCH_STATE_REVISION,
    match_state_features,
)
from src.networks.temporal import (
    SCORE_TIMELINE_REVISION,
    build_score_timeline,
)


CUTOFFS = tuple(int(value) for value in os.environ.get(
    "IS_STATE_CUTOFFS", "15,30,45,60,75").split(",") if value)
WORKERS = int(os.environ.get("IS_STATE_WORKERS", "4"))


def build_one(task):
    match_id, home_team_id, away_team_id, events_path = task
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    score_timeline = build_score_timeline(
        events, home_team_id, away_team_id, include_seconds=True)
    rows = []
    for cutoff in CUTOFFS:
        values = match_state_features(
            events, home_team_id, away_team_id, cutoff)
        home_score = away_score = 0
        for event_minute, event_home, event_away in score_timeline:
            if event_minute < cutoff:
                home_score, away_score = event_home, event_away
            else:
                break
        rows.append({
            "match_id": int(match_id),
            "cutoff": int(cutoff),
            "match_state_revision": MATCH_STATE_REVISION,
            "score_timeline_revision": SCORE_TIMELINE_REVISION,
            "current_home_score": int(home_score),
            "current_away_score": int(away_score),
            "current_goal_difference": int(home_score - away_score),
            **dict(zip(MATCH_STATE_FEATURE_NAMES, values.tolist())),
        })
    return rows


def main():
    output_dir = ROOT / "data" / "information_sciences"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "temporal_match_state_partial.parquet"
    final_path = output_dir / "temporal_match_state.parquet"
    metadata = pd.read_parquet(
        ROOT / "data" / "processed" / "matches_meta.parquet")
    events_dir = ROOT / "StatsBomb" / "data" / "events"

    rows = []
    if partial_path.exists():
        prior = pd.read_parquet(partial_path)
        prior = prior[
            prior["match_state_revision"].eq(MATCH_STATE_REVISION)
            & prior["cutoff"].isin(CUTOFFS)
        ].drop_duplicates(["match_id", "cutoff"], keep="last")
        rows = prior.to_dict("records")
    completed = {
        int(match_id)
        for match_id, group in pd.DataFrame(rows).groupby("match_id")
        if set(group["cutoff"].astype(int)) == set(CUTOFFS)
    } if rows else set()

    tasks = []
    for row in metadata.itertuples(index=False):
        match_id = int(row.match_id)
        if match_id in completed:
            continue
        path = events_dir / f"{match_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        tasks.append((match_id, int(row.home_team_id),
                      int(row.away_team_id), str(path)))

    print(f"building state features for {len(tasks)} matches at {CUTOFFS}",
          flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(build_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 200 == 0:
                frame = pd.DataFrame(rows).drop_duplicates(
                    ["match_id", "cutoff"], keep="last")
                frame.to_parquet(partial_path, index=False)
                print(f"{index}/{len(tasks)} matches", flush=True)

    frame = pd.DataFrame(rows).drop_duplicates(
        ["match_id", "cutoff"], keep="last")
    expected = len(metadata) * len(CUTOFFS)
    if len(frame) != expected:
        raise RuntimeError(
            f"state feature row count {len(frame)} != expected {expected}")
    if frame[list(MATCH_STATE_FEATURE_NAMES)].isna().any().any():
        raise RuntimeError("state feature matrix contains missing values")
    frame.sort_values(["cutoff", "match_id"]).to_parquet(
        final_path, index=False)
    frame.to_parquet(partial_path, index=False)
    manifest = {
        "match_state_revision": MATCH_STATE_REVISION,
        "cutoff_rule": "0 <= minute + second/60 < cutoff",
        "period_5_excluded": True,
        "retrospective_pass_annotations_used": False,
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "feature_count": len(MATCH_STATE_FEATURE_NAMES),
        "row_count": len(frame),
        "cutoffs": list(CUTOFFS),
    }
    (output_dir / "temporal_match_state_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
