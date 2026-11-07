"""
Task 3: Build heterogeneous graphs for all matches.
Outputs: data/networks/heterogeneous/{match_id}.pt
"""
import logging
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, read_match_shard
from src.networks.temporal import build_player_team_map
from src.networks.heterogeneous import (
    HETEROGENEOUS_GRAPH_REVISION, build_heterogeneous_graph, save_hetero_graph,
)
from src.networks.edge_features import PASS_EDGE_DIM, ADV_EDGE_DIM, TURN_EDGE_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_one_match(args: Tuple) -> Tuple[int, bool, str]:
    """Worker: build and save one heterogeneous graph."""
    (match_id, home_team_id, away_team_id, goal_diff,
     processed_dir, lineups_df, struct_df, subs_df, out_dir) = args

    try:
        # Pass/adversarial come from this match's small shard; the small global
        # lineups/structure/substitution slices are passed in pre-filtered.
        pass_df = read_match_shard(processed_dir, match_id, "pass")
        adv_df = read_match_shard(processed_dir, match_id, "adv")

        player_side_map = build_player_team_map(
            lineups_df, subs_df, match_id, home_team_id
        )

        data = build_heterogeneous_graph(
            match_id=match_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            pass_df=pass_df,
            adv_df=adv_df,
            lineups_df=lineups_df,
            structure_df=struct_df,
            substitutions_df=subs_df,
            goal_diff=goal_diff,
            player_side_map=player_side_map,
            w0=0,
        )

        out_path = Path(out_dir) / f"{match_id}.pt"
        save_hetero_graph(data, str(out_path))
        return match_id, True, ""

    except Exception as e:
        logger.error(f"Error building graph for match {match_id}: {e}")
        return match_id, False, repr(e)


def main():
    cfg = load_config("config.yaml")
    processed_dir = Path(cfg["data"]["processed_dir"])
    net_dir = Path(cfg["data"]["networks_dir"]) / "heterogeneous"
    net_dir.mkdir(parents=True, exist_ok=True)

    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")
    logger.info(f"Building heterogeneous graphs for {len(match_meta_df)} matches...")

    # Check for torch_geometric
    try:
        import torch_geometric
    except ImportError:
        logger.error("torch_geometric not installed. Run: pip install torch_geometric")
        return

    # Small global tables loaded once and sliced per match (avoids re-reading
    # them in every worker).
    lineups_all = pd.read_parquet(processed_dir / "lineups.parquet")
    struct_all = pd.read_parquet(processed_dir / "events_structure.parquet") \
        if (processed_dir / "events_structure.parquet").exists() else pd.DataFrame()
    subs_all = pd.read_parquet(processed_dir / "substitutions.parquet") \
        if (processed_dir / "substitutions.parquet").exists() else pd.DataFrame()

    lineups_by = dict(tuple(lineups_all.groupby("match_id"))) if not lineups_all.empty else {}
    struct_by = dict(tuple(struct_all.groupby("match_id"))) if not struct_all.empty else {}
    subs_by = dict(tuple(subs_all.groupby("match_id"))) if not subs_all.empty else {}
    empty_lineups = pd.DataFrame(
        columns=["match_id", "team_id", "player_id"])
    empty_structure = pd.DataFrame(
        columns=["match_id", "minute", "position_id", "player_id"])
    empty_substitutions = pd.DataFrame(
        columns=["match_id", "minute", "player_on_id", "player_off_id"])

    args_list = [
        (
            int(row["match_id"]),
            int(row["home_team_id"]),
            int(row["away_team_id"]),
            int(row["goal_diff"]),
            str(processed_dir),
            lineups_by.get(int(row["match_id"]), empty_lineups),
            struct_by.get(int(row["match_id"]), empty_structure),
            subs_by.get(int(row["match_id"]), empty_substitutions),
            str(net_dir),
        )
        for _, row in match_meta_df.iterrows()
    ]

    manifest_path = net_dir / "manifest.json"
    previous_manifest = {}
    if manifest_path.exists():
        previous_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"))
    resumable = (
        previous_manifest.get("graph_revision") ==
        HETEROGENEOUS_GRAPH_REVISION
        and previous_manifest.get("edge_dimensions", {}).get("pass")
        == PASS_EDGE_DIM
        and os.environ.get("IS_FORCE_HETERO_REBUILD", "0") != "1"
    )
    if resumable:
        pending = [args for args in args_list
                   if not (net_dir / f"{args[0]}.pt").exists()]
        logger.info(
            "Resuming current graph revision: %d existing, %d pending",
            len(args_list) - len(pending), len(pending))
    else:
        pending = args_list

    success = len(args_list) - len(pending) if resumable else 0
    failed = 0
    failures = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(build_one_match, a): a[0] for a in pending}
        for fut in as_completed(futures):
            mid = futures[fut]
            _, ok, error = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
                failures.append({"match_id": int(mid), "error": error})
            processed = success + failed
            if processed % 100 == 0 or processed == len(args_list):
                logger.info(f"  Progress: {processed}/{len(args_list)} | "
                            f"OK={success} | Failed={failed}")

    logger.info(f"Heterogeneous graph building complete: {success} OK, {failed} failed")
    manifest = {
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "expected_matches": len(match_meta_df),
        "graph_files": len(list(net_dir.glob("*.pt"))),
        "successful_builds": success,
        "failed_builds": failed,
        "failures": sorted(failures, key=lambda item: item["match_id"]),
        "edge_dimensions": {
            "pass": PASS_EDGE_DIM,
            "adversarial": ADV_EDGE_DIM,
            "turnover": TURN_EDGE_DIM,
        },
    }
    (net_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
