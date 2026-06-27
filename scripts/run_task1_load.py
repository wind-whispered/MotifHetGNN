"""
Task 1: Load and parse all StatsBomb raw JSON files.

The full dataset contains millions of events, so raw event JSON is loaded and
parsed in **batches** of matches: each batch is loaded in parallel, parsed,
cleaned, appended to the growing parquet tables, and then discarded. This keeps
peak memory bounded (only one batch of raw events is alive at a time) and lets
the same code scale from a handful of matches to the entire dataset.

Outputs:
  processed/matches_meta.parquet, lineups.parquet,
  processed/events_pass.parquet, events_adversarial.parquet,
  events_structure.parquet, substitutions.parquet, quality_report.parquet
  processed/by_match/{match_id}/{pass,adv,struct,subs}.parquet   (per-match shards)
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.loader import (
    load_config, load_competitions,
    load_all_events_parallel, load_all_lineups_parallel,
)
from src.data.parser import parse_match_meta, parse_lineups, parse_all_events
from src.data.cleaner import (
    filter_pass_df, filter_adversarial_df,
    add_zone_column, add_continuous_minute, compute_data_quality_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 250  # matches per batch (bounds peak memory)


def _build_pass_df(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df, _ = filter_pass_df(df)
    df = add_zone_column(df)
    df = add_continuous_minute(df)
    return df


def _build_adv_df(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = filter_adversarial_df(df)
    df = add_zone_column(df)
    return df


def main():
    cfg = load_config("config.yaml")
    raw_dir = cfg["data"]["raw_dir"]
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "by_match"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: match metadata ----
    logger.info("Loading competitions...")
    competitions = load_competitions(raw_dir)

    match_meta_records = []
    for comp in competitions:
        cid, sid = comp["competition_id"], comp["season_id"]
        match_path = Path(raw_dir) / "matches" / str(cid) / f"{sid}.json"
        if not match_path.exists():
            continue
        with open(match_path, encoding="utf-8") as f:
            for m in json.load(f):
                match_meta_records.append(parse_match_meta(m))

    match_meta_df = pd.DataFrame(match_meta_records).drop_duplicates("match_id")

    max_matches = cfg["data"].get("max_matches")
    if max_matches:
        match_meta_df = (match_meta_df.sort_values("match_id")
                         .head(int(max_matches)).reset_index(drop=True))
        logger.info(f"Subsetting to first {len(match_meta_df)} matches (max_matches={max_matches})")
    match_meta_df.to_parquet(out_dir / "matches_meta.parquet", index=False)
    match_ids = match_meta_df["match_id"].tolist()
    logger.info(f"Total matches to process: {len(match_ids)}")

    # ---- Step 2: lineups (small) ----
    logger.info("Loading lineups...")
    lineups_map = load_all_lineups_parallel(raw_dir, match_ids, max_workers=8)
    lineup_records = []
    for mid, lineup_list in lineups_map.items():
        lineup_records.extend(parse_lineups(mid, lineup_list))
    lineups_df = pd.DataFrame(lineup_records)
    lineups_df.to_parquet(out_dir / "lineups.parquet", index=False)
    logger.info(f"Saved lineups: {len(lineups_df)} player-match records")
    del lineups_map, lineup_records

    # ---- Step 3: events, processed in batches ----
    pass_parts, adv_parts, struct_parts, sub_parts = [], [], [], []
    n_pass = n_adv = 0
    t0 = time.time()

    n_batches = (len(match_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch_ids = match_ids[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        events_map = load_all_events_parallel(raw_dir, batch_ids, max_workers=8)

        b_pass, b_adv, b_struct, b_sub = [], [], [], []
        for mid, events in events_map.items():
            passes, adversarials, structures, subs = parse_all_events(mid, events)
            b_pass.extend(passes)
            b_adv.extend(adversarials)
            b_struct.extend(structures)
            b_sub.extend(subs)
        del events_map  # free raw events for this batch immediately

        pass_df = _build_pass_df(b_pass)
        adv_df = _build_adv_df(b_adv)
        struct_df = pd.DataFrame(b_struct)
        sub_df = pd.DataFrame(b_sub)

        # Per-match shards for fast downstream reads (avoids re-reading the
        # monolithic table in every worker of Tasks 2/3/5).
        if not pass_df.empty:
            for mid, g in pass_df.groupby("match_id"):
                d = shard_dir / str(int(mid))
                d.mkdir(exist_ok=True)
                g.to_parquet(d / "pass.parquet", index=False)
        if not adv_df.empty:
            for mid, g in adv_df.groupby("match_id"):
                d = shard_dir / str(int(mid))
                d.mkdir(exist_ok=True)
                g.to_parquet(d / "adv.parquet", index=False)

        pass_parts.append(pass_df)
        adv_parts.append(adv_df)
        struct_parts.append(struct_df)
        sub_parts.append(sub_df)
        n_pass += len(pass_df)
        n_adv += len(adv_df)
        logger.info(f"  Batch {bi+1}/{n_batches}: {len(batch_ids)} matches, "
                    f"{len(pass_df)} passes (cum {n_pass}) | {time.time()-t0:.0f}s")

    logger.info(f"Event loading + parsing took {time.time() - t0:.1f}s")

    # ---- Step 4: concatenate and persist monolithic tables ----
    pass_df = pd.concat(pass_parts, ignore_index=True) if pass_parts else pd.DataFrame()
    adv_df = pd.concat(adv_parts, ignore_index=True) if adv_parts else pd.DataFrame()
    struct_df = pd.concat(struct_parts, ignore_index=True) if struct_parts else pd.DataFrame()
    sub_df = pd.concat(sub_parts, ignore_index=True) if sub_parts else pd.DataFrame()
    del pass_parts, adv_parts, struct_parts, sub_parts

    pass_df.to_parquet(out_dir / "events_pass.parquet", index=False)
    logger.info(f"Saved events_pass: {len(pass_df)} records")
    adv_df.to_parquet(out_dir / "events_adversarial.parquet", index=False)
    logger.info(f"Saved events_adversarial: {len(adv_df)} records")
    struct_df.to_parquet(out_dir / "events_structure.parquet", index=False)
    logger.info(f"Saved events_structure: {len(struct_df)} records")
    sub_df.to_parquet(out_dir / "substitutions.parquet", index=False)
    logger.info(f"Saved substitutions: {len(sub_df)} records")

    # ---- Step 5: quality report (Table 1) ----
    quality_df = compute_data_quality_report(pass_df, adv_df, len(match_ids))
    quality_df.to_parquet(out_dir / "quality_report.parquet", index=False)
    logger.info("Quality report saved.")
    print(quality_df.T.to_string())


if __name__ == "__main__":
    main()
