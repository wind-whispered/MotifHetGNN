"""
Task 6: z-score significance testing via randomization.
Outputs: data/motifs/homogeneous_zscore.parquet, heterogeneous_zscore.parquet
"""
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List

import pandas as pd

from src.data.loader import load_config
from src.networks.homogeneous import load_network
from src.motifs.zscore import run_randomization_for_match, compute_zscores_from_motif_df
from src.motifs.taxonomy import assign_motif_order_to_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def randomize_one_network(args: Tuple) -> List[dict]:
    """Worker: generate random networks and enumerate motifs."""
    match_id, team_side, net_path, k_values, n_random, base_seed = args

    try:
        G = load_network(net_path)
        if G.number_of_nodes() < 3:
            return []

        records = run_randomization_for_match(
            match_id, team_side, G, k_values,
            n_random=n_random, base_seed=base_seed,
        )
        return records
    except Exception as e:
        logger.error(f"Randomization failed for match={match_id} side={team_side}: {e}")
        return []


def main():
    cfg = load_config("config.yaml")
    net_dir = Path(cfg["data"]["networks_dir"]) / "homogeneous"
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    n_random = cfg["motif"]["n_random"]

    # Load observed motif data
    homo_motif_df = pd.read_parquet(motifs_dir / "homogeneous_motifs.parquet")
    # Significance testing is restricted to low-order motifs (k<=zscore_k_max):
    # higher orders explode combinatorially under 100x randomization and the
    # reference analysis itself only z-tests 3-motifs.
    zscore_k_max = cfg["motif"].get("zscore_k_max", 4)
    k_values = sorted(k for k in homo_motif_df["motif_order_k"].unique().tolist()
                      if k <= zscore_k_max)
    logger.info(f"k values to randomize: {k_values} (zscore_k_max={zscore_k_max})")

    # Collect network files
    net_files = list(net_dir.glob("*.gpickle"))
    args_list = []
    for net_path in net_files:
        parts = net_path.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        match_id, team_side = int(parts[0]), parts[1]
        args_list.append((match_id, team_side, str(net_path), k_values, n_random, 42))

    logger.info(f"Running randomization for {len(args_list)} networks "
                f"({n_random} random networks each)...")
    logger.info("WARNING: This is the most computationally intensive step.")

    all_random_records = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(randomize_one_network, a): a[:2] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            mid, side = futures[fut]
            try:
                records = fut.result()
                all_random_records.extend(records)
                done += 1
                if done % 50 == 0:
                    logger.info(f"  Progress: {done}/{len(args_list)}")
            except Exception as e:
                logger.error(f"Error for match={mid} side={side}: {e}")

    random_motif_df = pd.DataFrame(all_random_records)
    # Save intermediate (large file)
    random_motif_df.to_parquet(motifs_dir / "random_motifs.parquet", index=False)
    logger.info(f"Saved random motifs: {len(random_motif_df)} records")

    # Compute z-scores only for the motif orders that actually have a random
    # baseline (k in k_values); higher orders are observed but not z-tested.
    logger.info("Computing z-scores...")
    groupby_cols = ["motif_id", "motif_order_k", "team_side"]
    tested_obs = homo_motif_df[homo_motif_df["motif_order_k"].isin(k_values)]
    zscore_df = compute_zscores_from_motif_df(
        tested_obs, random_motif_df, groupby_cols=groupby_cols
    )
    zscore_df = assign_motif_order_to_df(zscore_df)
    zscore_df.to_parquet(motifs_dir / "homogeneous_zscore.parquet", index=False)
    logger.info(f"Saved homogeneous_zscore: {len(zscore_df)} records")

    # Summary
    sig = zscore_df[zscore_df["significant"] == True]
    logger.info(f"Significant motifs (|z|>1.96): {len(sig)}")
    print("\n=== z-score Summary ===")
    print(zscore_df.sort_values("z", ascending=False)[
        ["motif_id", "motif_order_k", "team_side", "mu", "mu_rnd", "z", "significant"]
    ].to_string())


if __name__ == "__main__":
    main()
