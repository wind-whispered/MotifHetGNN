"""
Information saturation analysis: determine k* empirically and
plot the decay of observable motif counts with increasing order k.
"""
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ZERO_THRESHOLD = 0.01


def compute_decay_curve(
    motif_df: pd.DataFrame,
    zero_threshold: float = ZERO_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute the motif count decay curve: for each order k,
    compute mean and std of total motif count across matches.

    Returns DataFrame for Fig. 5 / Fig. 10.
    Columns: k, team_side, mean_total_count, std_total_count,
             n_observed_types, is_zero
    """
    records = []
    for (k, side), group in motif_df.groupby(["motif_order_k", "team_side"]):
        per_match_total = group.groupby("match_id")["count"].sum()
        mean_total = per_match_total.mean()
        std_total = per_match_total.std()
        n_types = group["motif_id"].nunique() if "motif_id" in group.columns else group.shape[0]

        records.append({
            "k": k,
            "team_side": side,
            "mean_total_count": mean_total,
            "std_total_count": std_total,
            "n_observed_types": n_types,
            "is_zero": mean_total < zero_threshold,
        })

    return pd.DataFrame(records).sort_values(["k", "team_side"])


def determine_k_star_from_decay(
    decay_df: pd.DataFrame,
    zero_threshold: float = ZERO_THRESHOLD,
) -> Dict[str, int]:
    """
    Determine empirical k* per team_side: last k where mean_total_count > threshold.
    """
    result = {}
    for side in ["home", "away"]:
        side_df = decay_df[decay_df["team_side"] == side].sort_values("k")
        k_star = 3  # default
        for _, row in side_df.iterrows():
            if row["mean_total_count"] > zero_threshold:
                k_star = int(row["k"])
            else:
                break
        result[side] = k_star
    return result


def compute_hetero_decay_curve(hetero_df: pd.DataFrame) -> pd.DataFrame:
    """Compute decay curve for heterogeneous motifs."""
    records = []
    for (k, mtype), group in hetero_df.groupby(["motif_order_k", "motif_type"]):
        per_match_total = group.groupby("match_id")["count"].sum()
        records.append({
            "k": k,
            "motif_type": mtype,
            "mean_total_count": per_match_total.mean(),
            "std_total_count": per_match_total.std(),
            "n_observed_patterns": group["pattern_key"].nunique()
            if "pattern_key" in group.columns else group.shape[0],
        })
    return pd.DataFrame(records).sort_values(["k", "motif_type"])


def build_table3_combined(
    homo_decay: pd.DataFrame,
    hetero_decay: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine homogeneous and heterogeneous decay stats into Table 3 format.
    Theoretical motif counts from directed graph theory.
    """
    THEORETICAL_HOMO = {3: 13, 4: 199, 5: 9364}

    homo_rows = []
    for _, row in homo_decay.iterrows():
        k = int(row["k"])
        homo_rows.append({
            "network_type": "homogeneous",
            "team_side": row["team_side"],
            "k": k,
            "theoretical_types": str(THEORETICAL_HOMO.get(k, "N/A")),
            "observed_types": int(row["n_observed_types"]),
            "mean_total_count": round(row["mean_total_count"], 3),
            "std_total_count": round(row["std_total_count"], 3),
        })

    hetero_rows = []
    for _, row in hetero_decay.iterrows():
        hetero_rows.append({
            "network_type": f"heterogeneous_{row['motif_type']}",
            "team_side": "both",
            "k": int(row["k"]),
            "theoretical_types": "N/A",
            "observed_types": int(row["n_observed_patterns"]),
            "mean_total_count": round(row["mean_total_count"], 3),
            "std_total_count": round(row["std_total_count"], 3),
        })

    return pd.DataFrame(homo_rows + hetero_rows).sort_values(["network_type", "k"])
