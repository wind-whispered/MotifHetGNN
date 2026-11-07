"""Empirically audit the exact finite-graph connected-subgraph growth law.

For each connected k-set S, P_k counts outside vertices whose addition keeps
the induced set connected.  Double counting the pairs (S, v) yields

    C_{k+1} = C_k (n-k) E_k / mu_{k+1},

where E_k is the extension probability and mu_{k+1} is the mean number of
non-cut vertices in a connected (k+1)-set.  The script uses exactly the graph
sample used by the degree/density-controlled finite-size experiment.
"""
from itertools import combinations
from math import comb
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.homogeneous import load_network


def load_weak_adjacency(path):
    graph = load_network(str(path))
    nodes = sorted(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = [0] * len(nodes)
    for raw_u, raw_v in graph.edges():
        if raw_u == raw_v:
            continue
        u, v = index[raw_u], index[raw_v]
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u
    return adjacency


def is_connected(combo, adjacency):
    target = sum(1 << node for node in combo)
    reached = frontier = 1 << combo[0]
    while frontier:
        neighbours = 0
        bits = frontier
        while bits:
            least = bits & -bits
            neighbours |= adjacency[least.bit_length() - 1]
            bits ^= least
        frontier = neighbours & target & ~reached
        reached |= frontier
    return reached == target, target


def graph_profile(path, k_max=7):
    adjacency = load_weak_adjacency(path)
    n = len(adjacency)
    full_mask = (1 << n) - 1
    counts, pair_counts = {}, {}
    for k in range(2, min(k_max + 1, n) + 1):
        connected = extensions = 0
        for combo in combinations(range(n), k):
            ok, target = is_connected(combo, adjacency)
            if not ok:
                continue
            connected += 1
            if k <= k_max:
                neighbours = 0
                for node in combo:
                    neighbours |= adjacency[node]
                extensions += ((neighbours & full_mask) & ~target).bit_count()
        counts[k] = connected
        if k <= k_max:
            pair_counts[k] = extensions

    rows = []
    for k in range(2, min(k_max, n - 1) + 1):
        c_k, c_next, pairs = counts[k], counts[k + 1], pair_counts[k]
        if c_k == 0 or c_next == 0:
            continue
        extension_probability = pairs / ((n - k) * c_k)
        removable_mean = pairs / c_next
        gamma = 2 * c_next / ((n - k) * c_k)
        raw_growth_ratio = c_next / c_k
        q_ratio = (c_next / comb(n, k + 1)) / (c_k / comb(n, k))
        raw_growth_threshold = removable_mean / (n - k)
        occupancy_growth_threshold = removable_mean / (k + 1)
        rows.append({
            "k": k, "n_nodes": n, "c_k": c_k, "c_next": c_next,
            "extension_pairs": pairs,
            "extension_probability": extension_probability,
            "removable_vertices_mean": removable_mean,
            "gamma_lower_bound": gamma,
            "raw_growth_ratio": raw_growth_ratio,
            "occupancy_ratio": q_ratio,
            "raw_growth_threshold": raw_growth_threshold,
            "occupancy_growth_threshold": occupancy_growth_threshold,
            "capacity_dominated_growth": bool(
                raw_growth_ratio > 1.0 and q_ratio < 1.0),
            "growth_identity_error": abs(
                c_next - c_k * (n - k) * extension_probability /
                removable_mean),
            "occupancy_identity_error": abs(
                q_ratio - (k + 1) * extension_probability /
                removable_mean),
        })
    return rows


def main():
    out = ROOT / "data" / "information_sciences"
    selected = pd.read_parquet(out / "finite_size_controlled.parquet")[
        ["match_id", "team_side"]].drop_duplicates()
    network_dir = ROOT / "data" / "networks" / "homogeneous"
    rows = []
    for item in selected.itertuples(index=False):
        path = network_dir / f"{int(item.match_id)}_{item.team_side}.gpickle"
        for row in graph_profile(path):
            rows.append({"match_id": int(item.match_id),
                         "team_side": item.team_side, **row})
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "finite_size_growth_decomposition.parquet",
                     index=False)
    summary = frame.groupby("k").agg(
        extension_probability_mean=("extension_probability", "mean"),
        extension_probability_std=("extension_probability", "std"),
        removable_vertices_mean=("removable_vertices_mean", "mean"),
        removable_vertices_std=("removable_vertices_mean", "std"),
        gamma_mean=("gamma_lower_bound", "mean"),
        raw_growth_ratio_mean=("raw_growth_ratio", "mean"),
        occupancy_ratio_mean=("occupancy_ratio", "mean"),
        raw_growth_threshold_mean=("raw_growth_threshold", "mean"),
        occupancy_growth_threshold_mean=("occupancy_growth_threshold", "mean"),
        capacity_dominated_growth_rate=("capacity_dominated_growth", "mean"),
        max_growth_identity_error=("growth_identity_error", "max"),
        max_occupancy_identity_error=("occupancy_identity_error", "max"),
        n_graphs=("match_id", "size"),
    ).reset_index()
    summary.to_csv(out / "finite_size_growth_decomposition_summary.csv",
                   index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
