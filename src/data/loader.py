"""
Task 1 - Part A: JSON loading with parallel execution.
Reads raw StatsBomb files and returns dicts/lists for downstream parsing.
"""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_match_shard(processed_dir, match_id: int, kind: str):
    """
    Read a per-match event shard written by Task 1
    (processed/by_match/{match_id}/{kind}.parquet, kind in {pass, adv}).
    Returns an empty DataFrame if the shard does not exist.
    """
    import pandas as pd
    p = Path(processed_dir) / "by_match" / str(int(match_id)) / f"{kind}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_competitions(raw_dir: str) -> List[dict]:
    """Load competitions.json -> list of competition-season dicts."""
    path = Path(raw_dir) / "competitions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_matches_for_season(raw_dir: str, competition_id: int, season_id: int) -> List[dict]:
    """Load matches/{competition_id}/{season_id}.json."""
    path = Path(raw_dir) / "matches" / str(competition_id) / f"{season_id}.json"
    if not path.exists():
        logger.warning(f"Match file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_event_file(args: Tuple[str, int]) -> Tuple[int, Optional[List[dict]]]:
    """Worker function: load a single events/{match_id}.json."""
    raw_dir, match_id = args
    path = Path(raw_dir) / "events" / f"{match_id}.json"
    if not path.exists():
        return match_id, None
    with open(path, "r", encoding="utf-8") as f:
        return match_id, json.load(f)


def _load_lineup_file(args: Tuple[str, int]) -> Tuple[int, Optional[List[dict]]]:
    """Worker function: load a single lineups/{match_id}.json."""
    raw_dir, match_id = args
    path = Path(raw_dir) / "lineups" / f"{match_id}.json"
    if not path.exists():
        return match_id, None
    with open(path, "r", encoding="utf-8") as f:
        return match_id, json.load(f)


def load_all_match_ids(raw_dir: str) -> List[int]:
    """Collect all match IDs by scanning the matches/ directory."""
    matches_dir = Path(raw_dir) / "matches"
    match_ids = []
    for comp_dir in matches_dir.iterdir():
        if not comp_dir.is_dir():
            continue
        for season_file in comp_dir.iterdir():
            if season_file.suffix != ".json":
                continue
            with open(season_file, "r", encoding="utf-8") as f:
                for m in json.load(f):
                    match_ids.append(int(m["match_id"]))
    return sorted(set(match_ids))


def load_all_events_parallel(
    raw_dir: str,
    match_ids: List[int],
    max_workers: int = 8,
) -> Dict[int, List[dict]]:
    """
    Load event files for all match IDs in parallel.
    Returns dict: match_id -> list of event dicts.
    """
    events_map: Dict[int, List[dict]] = {}
    args_list = [(raw_dir, mid) for mid in match_ids]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_event_file, a): a[1] for a in args_list}
        for fut in as_completed(futures):
            mid = futures[fut]
            try:
                match_id, events = fut.result()
                if events is not None:
                    events_map[match_id] = events
                else:
                    logger.warning(f"No event file for match {mid}")
            except Exception as exc:
                logger.error(f"Error loading events for match {mid}: {exc}")

    logger.info(f"Loaded events for {len(events_map)}/{len(match_ids)} matches")
    return events_map


def load_all_lineups_parallel(
    raw_dir: str,
    match_ids: List[int],
    max_workers: int = 8,
) -> Dict[int, List[dict]]:
    """Load lineup files for all match IDs in parallel."""
    lineups_map: Dict[int, List[dict]] = {}
    args_list = [(raw_dir, mid) for mid in match_ids]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_lineup_file, a): a[1] for a in args_list}
        for fut in as_completed(futures):
            mid = futures[fut]
            try:
                match_id, lineups = fut.result()
                if lineups is not None:
                    lineups_map[match_id] = lineups
            except Exception as exc:
                logger.error(f"Error loading lineups for match {mid}: {exc}")

    logger.info(f"Loaded lineups for {len(lineups_map)}/{len(match_ids)} matches")
    return lineups_map


def load_360_frame(raw_dir: str, match_id: int) -> Optional[List[dict]]:
    """Load three-sixty/{match_id}.json if it exists."""
    path = Path(raw_dir) / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_available_360(raw_dir: str) -> List[int]:
    """Return list of match IDs that have 360 data."""
    three_sixty_dir = Path(raw_dir) / "three-sixty"
    if not three_sixty_dir.exists():
        return []
    return [int(p.stem) for p in three_sixty_dir.glob("*.json")]
