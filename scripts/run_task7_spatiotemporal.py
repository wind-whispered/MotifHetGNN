"""
Task 7: Spatiotemporal stratification of motif frequencies.
Outputs: motif_spatial.parquet, motif_temporal.parquet,
         motif_score_state.parquet, motif_play_pattern.parquet
"""
import json
import logging
from pathlib import Path

import pandas as pd

from src.data.loader import load_config
from src.analysis.spatiotemporal import (
    compute_motif_spatial_distribution,
    compute_motif_temporal_distribution,
    compute_motif_play_pattern_distribution,
)
from src.analysis.score_state import (
    build_possession_score_state, compute_motif_by_score_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config("config.yaml")
    processed_dir = Path(cfg["data"]["processed_dir"])
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    analysis_dir = Path(cfg["data"]["analysis_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cfg["data"]["raw_dir"]

    logger.info("Loading data...")
    pass_df = pd.read_parquet(processed_dir / "events_pass.parquet")
    homo_motif_df = pd.read_parquet(motifs_dir / "homogeneous_motifs.parquet")
    # Spatiotemporal stratification is interpretable only for the 13 canonical
    # triadic motifs; restricting to k=3 keeps the per-(motif x zone x period)
    # group counts tractable (higher orders have thousands of distinct ids).
    strat_k = int(cfg["motif"].get("spatiotemporal_k", 3))
    homo_motif_df = homo_motif_df[homo_motif_df["motif_order_k"] == strat_k].copy()
    logger.info(f"Spatiotemporal analysis restricted to k={strat_k} "
                f"({homo_motif_df['motif_id'].nunique()} motif types)")
    hetero_motif_df = pd.read_parquet(motifs_dir / "heterogeneous_motifs.parquet") \
        if (motifs_dir / "heterogeneous_motifs.parquet").exists() else pd.DataFrame()
    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")

    # ---- Spatial distribution ----
    logger.info("Computing spatial distributions...")
    spatial_df = compute_motif_spatial_distribution(homo_motif_df, pass_df)
    spatial_df.to_parquet(analysis_dir / "motif_spatial.parquet", index=False)
    logger.info(f"Saved motif_spatial: {len(spatial_df)} records")

    # ---- Temporal distribution ----
    logger.info("Computing temporal distributions...")
    temporal_df = compute_motif_temporal_distribution(homo_motif_df, pass_df)
    temporal_df.to_parquet(analysis_dir / "motif_temporal.parquet", index=False)
    logger.info(f"Saved motif_temporal: {len(temporal_df)} records")

    # ---- Play pattern distribution ----
    logger.info("Computing play pattern distributions...")
    pattern_df = compute_motif_play_pattern_distribution(homo_motif_df, pass_df)
    pattern_df.to_parquet(analysis_dir / "motif_play_pattern.parquet", index=False)
    logger.info(f"Saved motif_play_pattern: {len(pattern_df)} records")

    # ---- Score state: process sample of matches ----
    logger.info("Computing score state distributions (sample of matches)...")
    all_score_state_records = []

    sample_matches = match_meta_df.head(200)  # use sample for speed; adjust as needed
    for _, match_row in sample_matches.iterrows():
        match_id = int(match_row["match_id"])
        home_team_id = int(match_row["home_team_id"])
        away_team_id = int(match_row["away_team_id"])

        # Load raw events for score timeline
        event_path = Path(raw_dir) / "events" / f"{match_id}.json"
        if not event_path.exists():
            continue
        with open(event_path, encoding="utf-8") as f:
            events_raw = json.load(f)

        score_state_df = build_possession_score_state(
            match_id, events_raw, pass_df, home_team_id, away_team_id
        )
        all_score_state_records.append(score_state_df)

    if all_score_state_records:
        score_state_combined = pd.concat(all_score_state_records, ignore_index=True)

        motif_score_df = compute_motif_by_score_state(homo_motif_df, score_state_combined)
        motif_score_df.to_parquet(analysis_dir / "motif_score_state.parquet", index=False)
        logger.info(f"Saved motif_score_state: {len(motif_score_df)} records")
    else:
        logger.warning("No score state data computed")

    logger.info("Task 7 complete.")


if __name__ == "__main__":
    main()