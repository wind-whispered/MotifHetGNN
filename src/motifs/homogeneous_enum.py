"""
Task 4: Homogeneous motif enumeration - k=3 to k* with k* determination.
"""
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from .enumerator import enumerate_motifs_for_graph

logger = logging.getLogger(__name__)

ZERO_THRESHOLD = 0.01  # mean frequency below this -> consider zero


def enumerate_all_orders(
    match_id: int,
    team_side: str,
    G: nx.DiGraph,
    k_start: int = 3,
    k_max: int = 15,
    zero_threshold: float = ZERO_THRESHOLD,
) -> Dict[int, Dict[int, float]]:
    """
    Enumerate motifs of all orders from k_start until frequency zeros out.
    Returns: {k: {motif_id: count}}
    """
    results = {}
    for k in range(k_start, k_max + 1):
        counts = enumerate_motifs_for_graph(G, k, directed=True)
        results[k] = counts

        # Check for zero: if no motifs found at this order, stop
        if not counts or sum(counts.values()) < zero_threshold:
            logger.debug(f"match={match_id} {team_side}: k* reached at k={k-1}")
            # Store the zero result then break
            break

    return results


def build_motif_records(
    match_id: int,
    team_side: str,
    order_results: Dict[int, Dict[int, float]],
) -> List[dict]:
    """Flatten order_results into list of records for DataFrame construction."""
    records = []
    for k, motif_counts in order_results.items():
        for motif_id, count in motif_counts.items():
            records.append({
                "match_id": match_id,
                "team_side": team_side,
                "motif_order_k": k,
                "motif_id": motif_id,
                "count": count,
            })
    return records


def determine_k_star(
    motif_df: pd.DataFrame,
    zero_threshold: float = ZERO_THRESHOLD,
) -> Dict[str, int]:
    """
    Determine k* (empirical upper limit) per team_side from the full motif DataFrame.
    k* is the highest k where mean count > zero_threshold across all matches.

    Returns: {"home": k*, "away": k*}
    """
    result = {}
    for side in ["home", "away"]:
        side_df = motif_df[motif_df["team_side"] == side]
        if side_df.empty:
            result[side] = 3
            continue

        k_vals = sorted(side_df["motif_order_k"].unique())
        k_star = k_vals[0]
        for k in k_vals:
            k_df = side_df[side_df["motif_order_k"] == k]
            mean_total = k_df.groupby("match_id")["count"].sum().mean()
            if mean_total > zero_threshold:
                k_star = k
            else:
                break

        result[side] = int(k_star)
    return result


def build_order_summary(motif_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Table 3 content: per-order statistics.
    Columns: k, team_side, n_observed_types, mean_total_count, std_total_count, n_significant
    (significance column filled after z-score computation in Task 6)
    """
    records = []
    # Theoretical motif counts per k for directed graphs
    # k=3: 13, k=4: 199, k=5: 9364 (known values)
    theoretical = {3: 13, 4: 199, 5: 9364}

    for (k, side), group in motif_df.groupby(["motif_order_k", "team_side"]):
        # Number of distinct motif IDs observed
        n_observed = group["motif_id"].nunique()
        # Total count per match, then average
        total_per_match = group.groupby("match_id")["count"].sum()
        records.append({
            "motif_order_k": k,
            "team_side": side,
            "theoretical_types": theoretical.get(k, None),
            "observed_types": n_observed,
            "mean_total_count": total_per_match.mean(),
            "std_total_count": total_per_match.std(),
        })

    return pd.DataFrame(records).sort_values(["motif_order_k", "team_side"])
