"""
Task 7: Spatiotemporal stratification of motif frequencies.
"""
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from ..data.schema import ALL_ZONES, get_zone, ZONE_X_LABELS, ZONE_Y_LABELS

logger = logging.getLogger(__name__)

PERIOD_LABELS = ["0-30", "30-60", "60-90", "90+"]


def assign_period_label(minute: float) -> str:
    if minute < 30:
        return "0-30"
    elif minute < 60:
        return "30-60"
    elif minute < 90:
        return "60-90"
    return "90+"


def compute_motif_spatial_distribution(
    motif_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    groupby_motif: List[str] = ["motif_id", "motif_order_k", "team_side"],
) -> pd.DataFrame:
    """
    Compute per-zone frequency for each motif type.
    Uses pass start location as proxy for motif location.

    Strategy:
    - For each match and motif, use the mean location of passes
      in possessions where that motif was observed.
    - Assign to a zone and aggregate.

    Returns DataFrame: motif_cols + zone + mean_density
    """
    # Ensure zone column on pass_df
    if "zone" not in pass_df.columns:
        pass_df = pass_df.copy()
        pass_df["zone"] = pass_df.apply(
            lambda r: get_zone(r["location_x"], r["location_y"])
            if pd.notna(r["location_x"]) else None,
            axis=1,
        )

    # Zone distribution of passes per match (used as spatial weight)
    zone_pass_counts = (
        pass_df.groupby(["match_id", "zone"])
        .size()
        .reset_index(name="n_passes_in_zone")
    )

    # Merge motif counts with zone pass distribution
    merged = motif_df.merge(
        zone_pass_counts, on="match_id", how="left"
    )

    # Weighted motif density per zone
    records = []
    for keys, group in merged.groupby(groupby_motif + ["zone"]):
        key_dict = dict(zip(groupby_motif + ["zone"], keys))
        total_motif = group["count"].sum()
        total_passes = group["n_passes_in_zone"].sum()
        density = total_motif / total_passes if total_passes > 0 else 0.0
        row = dict(key_dict)
        row["motif_density"] = density
        row["total_motif_count"] = total_motif
        records.append(row)

    return pd.DataFrame(records)


def compute_motif_temporal_distribution(
    motif_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    groupby_motif: List[str] = ["motif_id", "motif_order_k", "team_side"],
) -> pd.DataFrame:
    """
    Compute motif frequency distribution across time periods (0-30, 30-60, 60-90, 90+).

    Since motif_df is per-match (no per-possession timestamp),
    we approximate using mean motif count and total passes per period.
    """
    if "continuous_minute" not in pass_df.columns:
        pass_df = pass_df.copy()
        period_offsets = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
        pass_df["continuous_minute"] = (
            pass_df["period"].map(period_offsets).fillna(0) + pass_df["minute"].fillna(0)
        )

    pass_df = pass_df.copy()
    pass_df["period_label"] = pass_df["continuous_minute"].apply(assign_period_label)

    period_counts = (
        pass_df.groupby(["match_id", "period_label"])
        .size()
        .reset_index(name="n_passes")
    )

    records = []
    for keys, group in motif_df.groupby(groupby_motif):
        key_dict = dict(zip(groupby_motif, keys if isinstance(keys, tuple) else (keys,)))
        total_count = group["count"].sum()
        n_matches = group["match_id"].nunique()

        for period in PERIOD_LABELS:
            period_passes = period_counts[
                (period_counts["match_id"].isin(group["match_id"])) &
                (period_counts["period_label"] == period)
            ]["n_passes"].sum()

            row = dict(key_dict)
            row["period_label"] = period
            row["total_motif_count"] = total_count
            row["n_passes_in_period"] = period_passes
            row["motif_density"] = total_count / period_passes if period_passes > 0 else 0.0
            records.append(row)

    return pd.DataFrame(records)


def compute_motif_play_pattern_distribution(
    motif_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    groupby_motif: List[str] = ["motif_id", "motif_order_k", "team_side"],
) -> pd.DataFrame:
    """
    Compute motif frequency stratified by play_pattern_name.
    Uses possession-level play pattern from passes.
    """
    # Get dominant play pattern per possession per match
    poss_pattern = (
        pass_df.groupby(["match_id", "possession"])["play_pattern_name"]
        .agg(lambda x: x.mode()[0] if len(x) > 0 else "Regular Play")
        .reset_index()
    )

    pattern_counts = (
        poss_pattern.groupby(["match_id", "play_pattern_name"])
        .size()
        .reset_index(name="n_possessions")
    )

    records = []
    for keys, group in motif_df.groupby(groupby_motif):
        key_dict = dict(zip(groupby_motif, keys if isinstance(keys, tuple) else (keys,)))

        for pattern in pass_df["play_pattern_name"].unique():
            n_poss = pattern_counts[
                (pattern_counts["match_id"].isin(group["match_id"])) &
                (pattern_counts["play_pattern_name"] == pattern)
            ]["n_possessions"].sum()

            row = dict(key_dict)
            row["play_pattern"] = pattern
            row["total_motif_count"] = group["count"].sum()
            row["n_possessions_with_pattern"] = n_poss
            row["density"] = group["count"].sum() / n_poss if n_poss > 0 else 0.0
            records.append(row)

    return pd.DataFrame(records)
