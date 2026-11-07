"""
Task 5: Enumerate heterogeneous motifs for all matches.
Outputs: data/motifs/heterogeneous_motifs.parquet
"""
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List

import pandas as pd

from src.data.loader import load_config, read_match_shard
from src.networks.temporal import build_player_team_map
from src.motifs.heterogeneous_enum import (
    enumerate_hetero_motifs_for_match, hetero_motifs_to_records,
    build_hetero_order_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def enumerate_one_match(args: Tuple) -> List[dict]:
    """Worker: enumerate heterogeneous motifs for one match."""
    (match_id, home_team_id, processed_dir,
     lineups_df, subs_df, k_start, k_max) = args

    try:
        pass_df = read_match_shard(processed_dir, match_id, "pass")
        adv_df = read_match_shard(processed_dir, match_id, "adv")

        player_side_map = build_player_team_map(
            lineups_df, subs_df, match_id, home_team_id
        )

        order_results = enumerate_hetero_motifs_for_match(
            match_id=match_id,
            pass_df=pass_df,
            adv_df=adv_df,
            player_side_map=player_side_map,
            k_start=k_start,
            k_max=k_max,
        )

        return hetero_motifs_to_records(match_id, order_results)

    except Exception as e:
        logger.error(f"Hetero motif enum failed for match={match_id}: {e}")
        return []


def main():
    cfg = load_config("config.yaml")
    processed_dir = Path(cfg["data"]["processed_dir"])
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    motifs_dir.mkdir(parents=True, exist_ok=True)

    k_start = cfg["motif"]["k_start"]
    # Heterogeneous graphs include both full rosters (~40 nodes), so brute-force
    # labeled enumeration is O(N^k); cap at hetero_k_max (default 3) for tractability.
    k_max = min(cfg["motif"]["k_max_search"], cfg["motif"].get("hetero_k_max", 3))

    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")
    logger.info(f"Enumerating heterogeneous motifs for {len(match_meta_df)} matches...")

    lineups_all = pd.read_parquet(processed_dir / "lineups.parquet")
    subs_all = pd.read_parquet(processed_dir / "substitutions.parquet") \
        if (processed_dir / "substitutions.parquet").exists() else pd.DataFrame()
    lineups_by = dict(tuple(lineups_all.groupby("match_id"))) if not lineups_all.empty else {}
    subs_by = dict(tuple(subs_all.groupby("match_id"))) if not subs_all.empty else {}
    empty = pd.DataFrame()

    args_list = [
        (
            int(row["match_id"]),
            int(row["home_team_id"]),
            str(processed_dir),
            lineups_by.get(int(row["match_id"]), empty),
            subs_by.get(int(row["match_id"]), empty),
            k_start, k_max,
        )
        for _, row in match_meta_df.iterrows()
    ]

    all_records = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(enumerate_one_match, a): a[0] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            mid = futures[fut]
            try:
                records = fut.result()
                all_records.extend(records)
                done += 1
                if done % 200 == 0:
                    logger.info(f"  Progress: {done}/{len(args_list)}")
            except Exception as e:
                logger.error(f"Error for match={mid}: {e}")

    hetero_df = pd.DataFrame(all_records)
    if hetero_df.empty:
        logger.warning("No heterogeneous motif records generated!")
        return

    hetero_df.to_parquet(motifs_dir / "heterogeneous_motifs.parquet", index=False)
    logger.info(f"Saved heterogeneous_motifs: {len(hetero_df)} records")

    summary_df = build_hetero_order_summary(hetero_df)
    summary_df.to_parquet(motifs_dir / "hetero_order_summary.parquet", index=False)
    logger.info(f"Hetero order summary:\n{summary_df.to_string()}")


if __name__ == "__main__":
    main()
