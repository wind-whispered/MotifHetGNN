"""
Task 4: Enumerate homogeneous motifs for k=3..k* across all matches.
Outputs: data/motifs/homogeneous_motifs.parquet, motif_order_summary.parquet
"""
import logging
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List

import pandas as pd

from src.data.loader import load_config
from src.networks.homogeneous import load_network
from src.motifs.homogeneous_enum import (
    enumerate_all_orders, build_motif_records, build_order_summary,
)
from src.motifs.taxonomy import assign_motif_order_to_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def enumerate_one_network(args: Tuple) -> List[dict]:
    """Worker: enumerate motifs for one (match, side) network."""
    match_id, team_side, net_path, k_start, k_max = args

    try:
        G = load_network(net_path)
        if G.number_of_nodes() < k_start:
            return []

        order_results = enumerate_all_orders(
            match_id, team_side, G,
            k_start=k_start, k_max=k_max,
        )
        return build_motif_records(match_id, team_side, order_results)

    except Exception as e:
        logger.error(f"Motif enum failed for match={match_id} side={team_side}: {e}")
        return []


def main():
    cfg = load_config("config.yaml")
    net_dir = Path(cfg["data"]["networks_dir"]) / "homogeneous"
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    motifs_dir.mkdir(parents=True, exist_ok=True)

    k_start = cfg["motif"]["k_start"]
    k_max = cfg["motif"]["k_max_search"]

    # Collect all network files
    net_files = list(net_dir.glob("*.gpickle"))
    logger.info(f"Found {len(net_files)} network files")

    args_list = []
    for net_path in net_files:
        stem = net_path.stem  # e.g., "3788741_home"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        match_id = int(parts[0])
        team_side = parts[1]
        args_list.append((match_id, team_side, str(net_path), k_start, k_max))

    logger.info(f"Enumerating motifs for {len(args_list)} networks...")

    all_records = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(enumerate_one_network, a): a[:2] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            mid, side = futures[fut]
            try:
                records = fut.result()
                all_records.extend(records)
                done += 1
                if done % 200 == 0:
                    logger.info(f"  Progress: {done}/{len(args_list)}")
            except Exception as e:
                logger.error(f"Error for match={mid} side={side}: {e}")

    motif_df = pd.DataFrame(all_records)
    if motif_df.empty:
        logger.warning("No motif records generated!")
        return

    motif_df = assign_motif_order_to_df(motif_df)
    motif_df.to_parquet(motifs_dir / "homogeneous_motifs.parquet", index=False)
    logger.info(f"Saved homogeneous_motifs: {len(motif_df)} records")

    # Build order summary (Table 3)
    summary_df = build_order_summary(motif_df)
    summary_df.to_parquet(motifs_dir / "motif_order_summary.parquet", index=False)
    logger.info(f"Order summary:\n{summary_df.to_string()}")


if __name__ == "__main__":
    main()
