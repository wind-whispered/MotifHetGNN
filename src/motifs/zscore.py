"""
Task 6 - Part B: z-score significance testing for motif frequencies.
Computes mu, sigma, mu_rnd, sigma_rnd, z for each motif across all matches.
"""
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import networkx as nx

from .enumerator import enumerate_motifs_for_graph
from .randomization import generate_random_networks

logger = logging.getLogger(__name__)


def compute_zscore_for_motif(
    observed_counts: np.ndarray,
    random_counts_matrix: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Compute z-score statistics for one motif across all networks.

    Args:
        observed_counts: shape (n_networks,) - observed count per network
        random_counts_matrix: shape (n_networks, n_random) - random counts

    Returns:
        (mu, sigma, mu_rnd, sigma_rnd, z)
    """
    mu = float(np.mean(observed_counts))
    sigma = float(np.std(observed_counts))

    mu_rnd = float(np.mean(random_counts_matrix))
    sigma_rnd = float(np.std(random_counts_matrix.mean(axis=1)))

    if sigma_rnd == 0:
        z = 0.0
    else:
        z = (mu - mu_rnd) / sigma_rnd

    return mu, sigma, mu_rnd, sigma_rnd, z


def _mean_std_with_zeros(present_values: np.ndarray, n_total: int) -> Tuple[float, float]:
    """
    Mean and (population) std of a count distribution where ``present_values``
    are the non-zero observations and the remaining (n_total - len) entries are 0.
    """
    if n_total <= 0:
        return 0.0, 0.0
    s = float(present_values.sum())
    sq = float((present_values.astype(float) ** 2).sum())
    mu = s / n_total
    var = max(sq / n_total - mu * mu, 0.0)
    return mu, float(np.sqrt(var))


def compute_zscores_from_motif_df(
    motif_df: pd.DataFrame,
    random_motif_df: pd.DataFrame,
    groupby_cols: List[str] = ["motif_id", "motif_order_k", "team_side"],
) -> pd.DataFrame:
    """
    Compute z-scores by comparing observed motif_df with random_motif_df.

    Following Milo et al. and the reference football study, for every motif:

      * mu, sigma   -- mean and std, *across networks*, of the observed count
                       (networks where the motif never appears contribute 0);
      * mu_rnd      -- mean, across networks, of each network's mean count over
                       its degree-preserving randomisations;
      * sigma_rnd   -- std, across networks, of that per-network random mean
                       (the randomisation noise is averaged out first, so
                       sigma_rnd reflects genuine network-to-network spread, as
                       in the reference Table where e.g. motif 38 gives
                       (3.223-6.425)/2.239 = -1.43).

      z = (mu - mu_rnd) / sigma_rnd.

    Both DataFrames must have columns: match_id, motif_id, motif_order_k,
    team_side, count.  random_motif_df additionally has random_id.

    Returns DataFrame with columns:
        [groupby_cols] + [mu, sigma, mu_rnd, sigma_rnd, z, significant]
    """
    side_col = "team_side"
    # Number of networks per team side (denominator that turns "absent" motifs
    # into zeros when averaging across networks).
    n_networks_per_side = (
        motif_df.groupby(side_col)["match_id"].nunique().to_dict()
        if side_col in motif_df.columns else {}
    )
    n_random = (
        int(random_motif_df["random_id"].nunique())
        if (not random_motif_df.empty and "random_id" in random_motif_df.columns)
        else 1
    )

    # The full set of networks (match ids) per side, so a motif absent from a
    # network is correctly paired against a zero in both observed and random.
    side_matches = {}
    if side_col in motif_df.columns:
        for side, sub in motif_df.groupby(side_col):
            side_matches[side] = set(sub["match_id"].unique())

    # Pre-index, per grouping key, the per-network observed count and per-network
    # mean random count as match_id -> value mappings.
    rnd_keyed = {}
    if not random_motif_df.empty:
        for keys, grp in random_motif_df.groupby(groupby_cols):
            per_net_mean = grp.groupby("match_id")["count"].sum() / n_random
            rnd_keyed[keys] = per_net_mean.to_dict()

    records = []
    for keys, obs_group in motif_df.groupby(groupby_cols):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(groupby_cols, key_tuple))
        side = key_dict.get(side_col)

        n_net = n_networks_per_side.get(side, len(obs_group["match_id"].unique()))
        obs_map = obs_group.groupby("match_id")["count"].sum().to_dict()
        mu, sigma = _mean_std_with_zeros(np.array(list(obs_map.values())), n_net)

        rnd_map = rnd_keyed.get(key_tuple, {})
        if rnd_map:
            mu_rnd, sigma_rnd = _mean_std_with_zeros(
                np.array(list(rnd_map.values())), n_net
            )
        else:
            mu_rnd, sigma_rnd = 0.0, 0.0

        z = (mu - mu_rnd) / sigma_rnd if sigma_rnd > 0 else 0.0

        # Paired per-network difference (observed - random expectation), with a
        # one-sample t-test against 0 across all networks of this side. This is a
        # significance statement whose power grows with the number of matches.
        all_matches = side_matches.get(side, set(obs_map) | set(rnd_map))
        diffs = np.array([
            obs_map.get(m, 0.0) - rnd_map.get(m, 0.0) for m in all_matches
        ], dtype=float)
        if diffs.size > 1 and diffs.std(ddof=1) > 0:
            t_stat = float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(diffs.size)))
            from scipy import stats as _sps
            p_value = float(2 * _sps.t.sf(abs(t_stat), df=diffs.size - 1))
        else:
            t_stat, p_value = 0.0, 1.0

        row = dict(key_dict)
        row.update({
            "mu": mu,
            "sigma": sigma,
            "mu_rnd": mu_rnd,
            "sigma_rnd": sigma_rnd,
            "z": z,
            "delta": mu - mu_rnd,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
        })
        records.append(row)

    return pd.DataFrame(records)


def run_randomization_for_match(
    match_id: int,
    team_side: str,
    G: nx.DiGraph,
    k_values: List[int],
    n_random: int = 100,
    base_seed: int = 42,
) -> List[dict]:
    """
    Generate random networks for one match and enumerate motifs at all k.
    Returns list of records with random_id column.
    """
    from .randomization import generate_random_networks

    random_graphs = generate_random_networks(G, n_random=n_random, base_seed=base_seed)
    records = []

    for rand_id, G_rand in enumerate(random_graphs):
        for k in k_values:
            counts = enumerate_motifs_for_graph(G_rand, k, directed=True)
            for motif_id, count in counts.items():
                records.append({
                    "match_id": match_id,
                    "team_side": team_side,
                    "random_id": rand_id,
                    "motif_order_k": k,
                    "motif_id": motif_id,
                    "count": count,
                })
    return records


def filter_significant_motifs(
    zscore_df: pd.DataFrame,
    threshold: float = 1.96,
) -> pd.DataFrame:
    """Return only statistically significant motifs (|z| > threshold)."""
    return zscore_df[zscore_df["z"].abs() > threshold].copy()
