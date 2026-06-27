"""
Task 3: Build heterogeneous graphs for all matches.
Outputs: data/networks/heterogeneous/{match_id}.pt
"""
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import pandas as pd

from src.data.loader import load_config, read_match_shard
from src.networks.temporal import build_player_team_map
from src.networks.heterogeneous import build_heterogeneous_graph, save_hetero_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_one_match(args: Tuple) -> Tuple[int, bool]:
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
        return match_id, True

    except Exception as e:
        logger.error(f"Error building graph for match {match_id}: {e}")
        return match_id, False


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
    empty = pd.DataFrame()

    args_list = [
        (
            int(row["match_id"]),
            int(row["home_team_id"]),
            int(row["away_team_id"]),
            int(row["goal_diff"]),
            str(processed_dir),
            lineups_by.get(int(row["match_id"]), empty),
            struct_by.get(int(row["match_id"]), empty),
            subs_by.get(int(row["match_id"]), empty),
            str(net_dir),
        )
        for _, row in match_meta_df.iterrows()
    ]

    success, failed = 0, 0
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(build_one_match, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            mid = futures[fut]
            _, ok = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
            if (success + failed) % 100 == 0:
                logger.info(f"  Progress: {success+failed}/{len(args_list)} | "
                            f"OK={success} | Failed={failed}")

    logger.info(f"Heterogeneous graph building complete: {success} OK, {failed} failed")


if __name__ == "__main__":
    main()
