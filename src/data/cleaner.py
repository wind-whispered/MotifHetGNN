"""
Task 1 - Part C: Quality filtering, missing value handling, anomaly detection.
"""
from typing import Dict, Tuple
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Pitch coordinate bounds
PITCH_X_BOUNDS = (0.0, 120.0)
PITCH_Y_BOUNDS = (0.0, 80.0)
MAX_PASS_LENGTH_YARDS = 130.0


def filter_pass_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply quality filters to pass DataFrame.
    Returns filtered DataFrame and a dict of counts removed per filter.
    """
    original_len = len(df)
    removed = {}

    # 1. Must have valid player and recipient IDs
    mask = (df["player_id"] > 0) & (df["recipient_id"].notna()) & (df["recipient_id"] > 0)
    removed["missing_player_or_recipient"] = (~mask).sum()
    df = df[mask].copy()

    # 2. Self-passes (passer == recipient) are data artifacts
    mask = df["player_id"] != df["recipient_id"]
    removed["self_pass"] = (~mask).sum()
    df = df[mask].copy()

    # 3. Valid pitch coordinates
    mask = (
        df["location_x"].between(*PITCH_X_BOUNDS) &
        df["location_y"].between(*PITCH_Y_BOUNDS) &
        df["end_x"].between(*PITCH_X_BOUNDS) &
        df["end_y"].between(*PITCH_Y_BOUNDS)
    )
    removed["out_of_bounds_coords"] = (~mask).sum()
    df = df[mask].copy()

    # 4. Plausible pass length
    mask = df["length"].between(0.1, MAX_PASS_LENGTH_YARDS)
    removed["implausible_length"] = (~mask & df["length"].notna()).sum()
    df = df[mask | df["length"].isna()].copy()

    # 5. Valid minute
    mask = df["minute"].between(0, 130)
    removed["invalid_minute"] = (~mask).sum()
    df = df[mask].copy()

    total_removed = original_len - len(df)
    removed["total_removed"] = total_removed
    removed["total_kept"] = len(df)

    logger.info(f"Pass filter: kept {len(df)}/{original_len} | removed: {removed}")
    return df, removed


def filter_adversarial_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to adversarial event DataFrame."""
    original_len = len(df)

    # Must have valid player ID and location
    mask = (
        (df["player_id"] > 0) &
        df["location_x"].between(*PITCH_X_BOUNDS) &
        df["location_y"].between(*PITCH_Y_BOUNDS)
    )
    df = df[mask].copy()
    logger.info(f"Adversarial filter: kept {len(df)}/{original_len}")
    return df


def compute_data_quality_report(
    pass_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    match_ids_total: int,
) -> pd.DataFrame:
    """
    Generate Table 1 statistics.
    Returns a single-row DataFrame with all descriptive stats.
    """
    stats = {}
    stats["n_matches_total"] = match_ids_total
    stats["n_matches_with_events"] = pass_df["match_id"].nunique()

    # Pass stats
    stats["n_pass_events"] = len(pass_df)
    stats["n_unique_passers"] = pass_df["player_id"].nunique()
    stats["pct_under_pressure"] = (pass_df["under_pressure"].sum() / len(pass_df) * 100
                                   if len(pass_df) > 0 else 0.0)
    stats["pct_counterpress"] = (pass_df["counterpress"].sum() / len(pass_df) * 100
                                 if len(pass_df) > 0 else 0.0)
    stats["pct_cross"] = pass_df["cross"].sum() / len(pass_df) * 100 if len(pass_df) > 0 else 0.0
    stats["pct_switch"] = pass_df["switch"].sum() / len(pass_df) * 100 if len(pass_df) > 0 else 0.0
    stats["pct_goal_assist"] = pass_df["goal_assist"].sum() / len(pass_df) * 100 if len(pass_df) > 0 else 0.0

    # Play pattern distribution
    if "play_pattern_name" in pass_df.columns:
        pp_counts = pass_df["play_pattern_name"].value_counts(normalize=True) * 100
        for pp, pct in pp_counts.items():
            stats[f"pct_play_pattern_{pp.replace(' ', '_')}"] = round(pct, 2)

    # Adversarial stats
    for evt_type in ["Miscontrol", "Dispossessed", "Interception", "Tackle", "BallRecovery"]:
        stats[f"n_{evt_type}"] = (adv_df["event_type"] == evt_type).sum()
    stats["n_adversarial_total"] = len(adv_df)
    stats["pct_adversarial_counterpress"] = (
        adv_df["counterpress"].sum() / len(adv_df) * 100 if len(adv_df) > 0 else 0.0
    )

    return pd.DataFrame([stats])


def add_zone_column(df: pd.DataFrame, x_col: str = "location_x", y_col: str = "location_y") -> pd.DataFrame:
    """
    Add a 'zone' column based on pitch coordinates (fully vectorised; the
    previous row-wise ``apply`` was prohibitively slow on millions of events).
    """
    from .schema import ZONE_X_BREAKS, ZONE_Y_BREAKS, ZONE_X_LABELS, ZONE_Y_LABELS
    if df.empty:
        df = df.copy()
        df["zone"] = []
        return df
    df = df.copy()
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    # np.searchsorted-style bucketing into 3 bands each
    xi = (x >= ZONE_X_BREAKS[0]).astype("int8") + (x >= ZONE_X_BREAKS[1]).astype("int8")
    yi = (y >= ZONE_Y_BREAKS[0]).astype("int8") + (y >= ZONE_Y_BREAKS[1]).astype("int8")
    xlab = np.array(ZONE_X_LABELS, dtype=object)[xi.to_numpy()]
    ylab = np.array(ZONE_Y_LABELS, dtype=object)[yi.to_numpy()]
    zone = pd.Series([f"{a}_{b}" for a, b in zip(xlab, ylab)], index=df.index)
    zone[x.isna() | y.isna()] = None
    df["zone"] = zone
    return df


def add_continuous_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (period, minute) to a continuous game minute (vectorised).
    Extra-time periods (3, 4, 5) are offset from 90/105/120.
    """
    if df.empty:
        df = df.copy()
        df["continuous_minute"] = []
        return df
    df = df.copy()
    period_offsets = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    offset = df["period"].map(period_offsets).fillna(0)
    df["continuous_minute"] = offset + pd.to_numeric(df["minute"], errors="coerce").fillna(0)
    return df
