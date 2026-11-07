"""Build leakage-controlled match graphs using events observed by each cutoff.

Full-time labels are retained as targets. Node/edge statistics, tactical
shifts, substitutions and motif context are recomputed from the event prefix.
The pre-match squad list is allowed; future event participation is not used.
The prefix is the half-open interval [0, cutoff): events timestamped exactly at
or after the declared minute are excluded.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, read_match_shard
from src.gnn.dataset import typed_triad_context
from src.networks.heterogeneous import (
    HETEROGENEOUS_GRAPH_REVISION, build_heterogeneous_graph,
)
from src.networks.edge_features import PASS_EDGE_DIM, ADV_EDGE_DIM, TURN_EDGE_DIM
from src.networks.temporal import (
    SCORE_TIMELINE_REVISION, build_player_team_map, build_score_timeline,
)

CUTOFFS = (15, 30, 45, 60, 75)
TEMPORAL_GRAPH_REVISION = "event_prefix_no_assist_pass_intensity_scorefix_v5"


def _filter_prefix(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Return events in [0, cutoff), using seconds whenever available."""
    if df.empty:
        return df
    if "minute" not in df.columns:
        return df
    minute = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype(int)
    if "second" in df.columns:
        second = pd.to_numeric(df["second"], errors="coerce").fillna(0)
        elapsed = minute.astype(float) + second / 60.0
    else:
        # Structure/substitution tables are minute-resolution. Strict '<'
        # remains conservative at the boundary instead of admitting a future
        # event from within the named cutoff minute.
        elapsed = minute.astype(float)
    return df[elapsed < float(cutoff)]


def _ensure_schema(
    frame: pd.DataFrame,
    columns: list,
) -> pd.DataFrame:
    """Add missing required columns so downstream key access remains safe on empty frames."""
    if frame is None:
        frame = pd.DataFrame()
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _mask_retrospective_pass_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove annotations whose truth depends on a later assisted shot.

    StatsBomb records ``goal_assist`` and ``shot_assist`` on the pass event.
    The referenced shot can occur after a temporal cutoff even when the pass is
    inside it, so these two channels are not valid event-prefix features.
    """
    result = df.copy()
    for column in ("goal_assist", "shot_assist"):
        if column in result:
            result[column] = False
    return result


def score_at(timeline, cutoff):
    """Score observed strictly before the cutoff."""
    home = away = 0
    for minute, h, a in timeline:
        if minute < cutoff:
            home, away = h, a
        else:
            break
    return home, away


def build_one(args):
    (row, cutoff, processed_dir, raw_dir, lineups, structure, substitutions,
     output_dir) = args
    match_id = int(row["match_id"])
    try:
        pass_cols = [
            "match_id", "minute", "second", "team_id", "player_id", "recipient_id", "length",
            "angle", "height_id", "under_pressure", "counterpress", "cross",
            "switch", "goal_assist", "shot_assist",
        ]
        adv_cols = [
            "match_id", "minute", "second", "team_id", "player_id", "event_type",
            "location_x", "location_y",
        ]
        lineups_cols = ["match_id", "team_id", "player_id"]
        structure_cols = ["match_id", "minute", "position_id", "player_id"]
        substitutions_cols = ["match_id", "minute", "player_on_id", "player_off_id"]

        passes = _filter_prefix(_ensure_schema(
            read_match_shard(processed_dir, match_id, "pass"), pass_cols),
            cutoff)
        passes = _mask_retrospective_pass_features(passes)
        adversarial = _filter_prefix(_ensure_schema(
            read_match_shard(processed_dir, match_id, "adv"), adv_cols),
            cutoff)
        structure = _filter_prefix(_ensure_schema(structure, structure_cols), cutoff)
        substitutions = _filter_prefix(_ensure_schema(substitutions,
                                                      substitutions_cols),
                                         cutoff)
        lineups = _ensure_schema(lineups, lineups_cols)
        side_map = build_player_team_map(
            lineups, substitutions, match_id, int(row["home_team_id"]))
        graph = build_heterogeneous_graph(
            match_id, int(row["home_team_id"]), int(row["away_team_id"]),
            passes, adversarial, lineups, structure, substitutions,
            int(row["goal_diff"]), side_map, w0=0)

        events_path = Path(raw_dir) / "events" / f"{match_id}.json"
        events = json.loads(events_path.read_text(encoding="utf-8"))
        timeline = build_score_timeline(
            events, int(row["home_team_id"]), int(row["away_team_id"]),
            include_seconds=True)
        home_score, away_score = score_at(timeline, cutoff)
        graph.current_home_score = int(home_score)
        graph.current_away_score = int(away_score)
        graph.current_goal_diff = int(home_score - away_score)
        graph.cutoff_minute = int(cutoff)
        graph.temporal_graph_revision = TEMPORAL_GRAPH_REVISION
        graph.score_timeline_revision = SCORE_TIMELINE_REVISION
        graph.assist_flags_masked = True
        context = typed_triad_context(graph)
        for node_type, value in context.items():
            graph[node_type].motif_x = value
        target = Path(output_dir) / str(cutoff)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(graph, target / f"{match_id}.pt")
        return cutoff, match_id, True, ""
    except Exception as exc:
        return cutoff, match_id, False, repr(exc)


def main():
    cfg = load_config("config.yaml")
    processed = ROOT / cfg["data"]["processed_dir"]
    raw = ROOT / cfg["data"]["raw_dir"]
    output = ROOT / "data" / "information_sciences" / "temporal_graphs"
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(processed / "matches_meta.parquet")
    lineups = pd.read_parquet(processed / "lineups.parquet")
    structure = pd.read_parquet(processed / "events_structure.parquet")
    substitutions = pd.read_parquet(processed / "substitutions.parquet")
    grouped = lambda frame: dict(tuple(frame.groupby("match_id")))
    lineups_by, structure_by, substitutions_by = map(
        grouped, (lineups, structure, substitutions))
    empty = pd.DataFrame()
    tasks = []
    for cutoff in CUTOFFS:
        for _, row in meta.iterrows():
            target = output / str(cutoff) / f"{int(row.match_id)}.pt"
            if target.exists():
                continue
            mid = int(row.match_id)
            tasks.append((row.to_dict(), cutoff, str(processed), str(raw),
                          lineups_by.get(mid, empty), structure_by.get(mid, empty),
                          substitutions_by.get(mid, empty), str(output)))
    print(f"building {len(tasks)} cutoff graphs", flush=True)
    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(build_one, task) for task in tasks]
        for future in as_completed(futures):
            cutoff, mid, ok, error = future.result()
            completed += 1
            if not ok:
                failures.append({"cutoff": cutoff, "match_id": mid,
                                 "error": error})
            if completed % 500 == 0:
                print(f"{completed}/{len(tasks)}, failures={len(failures)}",
                      flush=True)
    pd.DataFrame(failures).to_csv(
        output / "build_failures.csv", index=False)
    manifest = {str(c): len(list((output / str(c)).glob("*.pt")))
                for c in CUTOFFS}
    (output / "manifest.json").write_text(
        json.dumps({
            "counts": manifest,
            "failures": failures,
            "graph_revision": TEMPORAL_GRAPH_REVISION,
            "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
            "score_timeline_revision": SCORE_TIMELINE_REVISION,
            "cutoff_rule": "0 <= minute + second/60 < cutoff",
            "minute_resolution_fallback": "minute < cutoff",
            "assist_flags_masked": True,
            "masked_pass_edge_channels": ["goal_assist", "shot_assist"],
            "edge_dimensions": {
                "pass": PASS_EDGE_DIM,
                "adversarial": ADV_EDGE_DIM,
                "turnover": TURN_EDGE_DIM,
            },
        }, indent=2),
        encoding="utf-8")
    print(manifest, "failures", len(failures), flush=True)
    valid_full_ids = {int(p.stem) for p in
                      (ROOT / "data/networks/heterogeneous").glob("*.pt")}
    unexpected = [x for x in failures if int(x["match_id"]) in valid_full_ids]
    if unexpected:
        print("unexpected failures", unexpected[:10], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
