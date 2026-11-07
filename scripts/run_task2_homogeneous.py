"""
Task 2: Build homogeneous passing networks and compute network statistics.
Outputs: data/networks/homogeneous/*.gpickle, network_stats.parquet
"""
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import pandas as pd
import yaml

from src.data.loader import load_config, read_match_shard
from src.networks.homogeneous import (
    build_all_networks_for_match, compute_network_stats,
    save_network, network_to_edge_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_one_match(
    args: Tuple,
) -> Tuple[int, list]:
    """Worker: build networks for one match and return stats records."""
    match_id, home_team_id, away_team_id, processed_dir, out_dir, w0_values = args

    # Read only this match's pass shard (small) instead of the full table.
    pass_df = read_match_shard(processed_dir, match_id, "pass")
    stats_records = []
    if pass_df.empty:
        return match_id, stats_records

    for w0 in w0_values:
        G_home, G_away = build_all_networks_for_match(
            match_id, home_team_id, away_team_id, pass_df, w0=w0
        )
        if w0 == 2:  # Save graphs only for default threshold
            save_network(G_home, str(Path(out_dir) / f"{match_id}_home.gpickle"))
            save_network(G_away, str(Path(out_dir) / f"{match_id}_away.gpickle"))
            # Also export edge lists for gtrieScanner
            network_to_edge_list(G_home, str(Path(out_dir) / f"{match_id}_home.edgelist"))
            network_to_edge_list(G_away, str(Path(out_dir) / f"{match_id}_away.edgelist"))

        stats_records.append(compute_network_stats(match_id, "home", G_home, w0))
        stats_records.append(compute_network_stats(match_id, "away", G_away, w0))

    return match_id, stats_records


def main():
    cfg = load_config("config.yaml")
    processed_dir = Path(cfg["data"]["processed_dir"])
    net_dir = Path(cfg["data"]["networks_dir"]) / "homogeneous"
    net_dir.mkdir(parents=True, exist_ok=True)

    w0_values = cfg["network"]["w0_values"]

    # Load data
    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")

    logger.info(f"Building homogeneous networks for {len(match_meta_df)} matches...")

    # Prepare args
    args_list = [
        (
            int(row["match_id"]),
            int(row["home_team_id"]),
            int(row["away_team_id"]),
            str(processed_dir),
            str(net_dir),
            w0_values,
        )
        for _, row in match_meta_df.iterrows()
    ]

    all_stats = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one_match, args): args[0] for args in args_list}
        done = 0
        for fut in as_completed(futures):
            match_id = futures[fut]
            try:
                _, stats_records = fut.result()
                all_stats.extend(stats_records)
                done += 1
                if done % 100 == 0:
                    logger.info(f"  Processed {done}/{len(args_list)} matches")
            except Exception as e:
                logger.error(f"Error for match {match_id}: {e}")

    stats_df = pd.DataFrame(all_stats)
    out_path = Path(cfg["data"]["processed_dir"]) / "network_stats.parquet"
    stats_df.to_parquet(out_path, index=False)
    logger.info(f"Saved network stats: {len(stats_df)} records -> {out_path}")

    # Summary: compare with original paper Table 3 for w0=2
    summary = stats_df[stats_df["w0"] == 2].groupby("team_side")[
        ["density", "transitivity", "pass_diversity", "mean_outdegree", "mean_betweenness"]
    ].agg(["mean", "std"])
    print("\n=== Network Stats Summary (w0=2) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
