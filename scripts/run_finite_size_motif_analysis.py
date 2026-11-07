"""Finite-size-normalised connected-subgraph analysis and null controls.

The primary null rewires the weak projection and exactly preserves weak
density and node degrees.  The sensitivity null rewires the directed graph and
exactly preserves every in/out degree; reciprocal dyads, and consequently weak
density, may change.  Keeping the two estimands explicit prevents the directed
diagnostic from being misreported as density matched.
"""
from argparse import ArgumentParser
from collections import Counter
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motifs.randomization import (directed_simple_edge_swap,
                                      undirected_simple_edge_swap)
from src.networks.homogeneous import load_network


NULL_MODELS = ("weak_projection", "directed_degree")
FINITE_SIZE_GRAPH_REVISION = "canonical_gpickle_v2"
OUTPUT_SUFFIX = {
    "weak_projection": "",
    "directed_degree": "_directed_degree_diagnostic",
}


def load_graph_edges(path):
    """Load exactly the graph used upstream, including any isolated nodes."""
    graph = load_network(str(path))
    nodes = sorted(graph.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    directed = {(index[u], index[v]) for u, v in graph.edges() if u != v}
    return len(nodes), directed


def connected_counts(n, directed_edges, k_max=None):
    adjacency = [0] * n
    for u, v in directed_edges:
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u
    k_max = min(k_max or n, n)
    out = {}
    for k in range(2, k_max + 1):
        total = 0
        for combo in combinations(range(n), k):
            target = sum(1 << x for x in combo)
            reached = 1 << combo[0]
            frontier = reached
            while frontier:
                neighbours = 0
                bits = frontier
                while bits:
                    lsb = bits & -bits
                    node = lsb.bit_length() - 1
                    neighbours |= adjacency[node]
                    bits ^= lsb
                new = neighbours & target & ~reached
                if not new:
                    break
                reached |= new
                frontier = new
            total += int(reached == target)
        out[k] = total
    return out


def weak_projection(edges):
    return {tuple(sorted((u, v))) for u, v in edges if u != v}


def _degree_signature(edges, directed):
    if directed:
        return (Counter(u for u, _ in edges), Counter(v for _, v in edges))
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return degree


def _replicate_seed(base_seed, match_id, team_side, replicate, null_model):
    """Stable per-realisation seed, independent of iteration order."""
    payload = (f"{base_seed}|{match_id}|{team_side}|{replicate}|{null_model}"
               .encode("utf-8"))
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(),
                          byteorder="big")


def select_networks(observed, n_networks, k_max, seed):
    keys = observed[["match_id", "team_side", "n_nodes"]].drop_duplicates()
    keys = keys[keys.n_nodes >= k_max]
    if len(keys) > n_networks:
        keys = keys.sample(n=n_networks, random_state=seed)
    return keys.sort_values(["match_id", "team_side"]).reset_index(drop=True)


def observed_profile():
    source = ROOT / "data" / "analysis" / "census_sizes_allk.parquet"
    frame = pd.read_parquet(source)
    network_dir = ROOT / "data" / "networks" / "homogeneous"
    node_rows = []
    for path in network_dir.glob("*.gpickle"):
        match_id, side = path.stem.rsplit("_", 1)
        n, _ = load_graph_edges(path)
        node_rows.append((int(match_id), side, n))
    nodes = pd.DataFrame(node_rows, columns=["match_id", "team_side", "n_nodes"])
    frame = frame.merge(nodes, on=["match_id", "team_side"], how="left")
    frame["capacity"] = [comb(int(n), int(k)) if k <= n else 0
                         for n, k in zip(frame.n_nodes, frame.k)]
    frame["connected_occupancy_q"] = frame.n_instances / frame.capacity
    nxt = frame[["match_id", "team_side", "k", "n_instances"]].copy()
    nxt["k"] -= 1
    nxt = nxt.rename(columns={"n_instances": "n_instances_next"})
    frame = frame.merge(nxt, on=["match_id", "team_side", "k"], how="left")
    denom = (frame.n_nodes - frame.k) * frame.n_instances
    frame["extension_efficiency_gamma"] = np.where(
        denom > 0, 2 * frame.n_instances_next / denom, np.nan)
    return frame


def null_analysis(observed, n_networks, n_random, k_max, seed,
                  null_model="weak_projection", swap_multiplier=10,
                  max_attempt_multiplier=30):
    if null_model not in NULL_MODELS:
        raise ValueError(f"Unknown null model: {null_model}")
    keys = select_networks(observed, n_networks, k_max, seed)
    rows = []
    diagnostic_rows = []
    network_dir = ROOT / "data" / "networks" / "homogeneous"
    for idx, item in enumerate(keys.itertuples(index=False), 1):
        path = network_dir / f"{item.match_id}_{item.team_side}.gpickle"
        n, directed_edges = load_graph_edges(path)
        if null_model == "weak_projection":
            base_edges = weak_projection(directed_edges)
            randomizer = undirected_simple_edge_swap
            directed_invariant = False
        else:
            base_edges = directed_edges
            randomizer = directed_simple_edge_swap
            directed_invariant = True
        target_swaps = max(1, swap_multiplier * len(base_edges))
        degree_signature = _degree_signature(base_edges, directed_invariant)
        original_weak_edges = len(weak_projection(directed_edges))
        for replicate in range(n_random):
            replicate_seed = _replicate_seed(
                seed, item.match_id, item.team_side, replicate, null_model)
            rng = random.Random(replicate_seed)
            random_edges, successful_swaps, attempts = randomizer(
                base_edges, target_swaps, rng,
                max_attempt_multiplier=max_attempt_multiplier)
            if _degree_signature(random_edges, directed_invariant) != degree_signature:
                raise RuntimeError(
                    f"{null_model} degree invariant failed for {path.name}, "
                    f"replicate {replicate}")
            random_weak_edges = len(weak_projection(random_edges))
            diagnostic_rows.append((
                item.match_id, item.team_side, replicate, null_model,
                replicate_seed, len(directed_edges), original_weak_edges,
                len(random_edges), random_weak_edges, target_swaps,
                successful_swaps, attempts,
                successful_swaps / target_swaps,
            ))
            counts = connected_counts(n, random_edges, k_max=k_max)
            for k, value in counts.items():
                rows.append((item.match_id, item.team_side, replicate, k, value,
                             value / comb(n, k)))
        if idx % 20 == 0:
            print(f"{null_model} controls {idx}/{len(keys)}", flush=True)
    null = pd.DataFrame(rows, columns=["match_id", "team_side", "replicate",
                                      "k", "null_count", "null_q"])
    diagnostics = pd.DataFrame(diagnostic_rows, columns=[
        "match_id", "team_side", "replicate", "null_model", "seed",
        "original_directed_edges", "original_weak_edges", "rewired_edges",
        "rewired_weak_edges", "target_swaps", "successful_swaps",
        "swap_attempts", "swap_completion",
    ])
    summary = null.groupby(["match_id", "team_side", "k"]).agg(
        null_q_mean=("null_q", "mean"), null_q_std=("null_q", "std"),
        null_count_mean=("null_count", "mean")).reset_index()
    selected = observed.merge(summary, on=["match_id", "team_side", "k"], how="inner")
    selected["q_excess"] = selected.connected_occupancy_q - selected.null_q_mean
    selected["q_z"] = selected.q_excess / selected.null_q_std.replace(0, np.nan)
    return null, selected, diagnostics, keys


def controlled_summary(controlled):
    rows = []
    for k, group in controlled.groupby("k"):
        values = group.q_excess.dropna().to_numpy(dtype=float)
        numerically_zero = bool(np.all(np.abs(values) < 1e-12))
        if numerically_zero:
            mean = median = ci_low = ci_high = std = 0.0
            effect = 0.0
        else:
            rng = np.random.default_rng(4200 + int(k))
            bootstrap = values[rng.integers(
                0, len(values), size=(10000, len(values)))].mean(axis=1)
            mean = float(values.mean())
            median = float(np.median(values))
            ci_low = float(np.quantile(bootstrap, 0.025))
            ci_high = float(np.quantile(bootstrap, 0.975))
            std = float(values.std(ddof=1))
            effect = float(mean / std) if std else np.nan
        rows.append({
            "k": int(k),
            "q_excess_mean": mean,
            "q_excess_std": std,
            "q_excess_median": median,
            "q_excess_ci_low": ci_low,
            "q_excess_ci_high": ci_high,
            "q_excess_cohen_dz": effect,
            "q_z_mean": float(group.q_z.mean()),
            "n_graphs": int(len(group)),
        })
    return pd.DataFrame(rows)


def observed_summary(observed):
    """Use all graph sides for raw Ck means and eligible graphs for Qk."""
    total_graphs = observed[["match_id", "team_side"]].drop_duplicates().shape[0]
    summary = observed.groupby("k").agg(
        raw_count_sum=("n_instances", "sum"),
        q_mean=("connected_occupancy_q", "mean"),
        q_std=("connected_occupancy_q", "std"),
        gamma_mean=("extension_efficiency_gamma", "mean"),
        gamma_std=("extension_efficiency_gamma", "std"),
        n_graphs=("match_id", "size"),
    ).reset_index()
    summary["raw_count_mean"] = summary.raw_count_sum / total_graphs
    summary["n_graphs_total"] = total_graphs
    return summary[[
        "k", "raw_count_mean", "q_mean", "q_std", "gamma_mean",
        "gamma_std", "n_graphs", "n_graphs_total",
    ]]


def _network_digest(keys):
    network_dir = ROOT / "data" / "networks" / "homogeneous"
    digest = hashlib.sha256()
    for item in keys.itertuples(index=False):
        path = network_dir / f"{item.match_id}_{item.team_side}.gpickle"
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_null_outputs(out, null_model, null, controlled, diagnostics, keys,
                      args):
    suffix = OUTPUT_SUFFIX[null_model]
    null.to_parquet(out / f"finite_size_null{suffix}.parquet", index=False)
    controlled.to_parquet(
        out / f"finite_size_controlled{suffix}.parquet", index=False)
    controlled_summary(controlled).to_csv(
        out / f"finite_size_controlled{suffix}_summary.csv", index=False)
    diagnostics.to_parquet(
        out / f"finite_size_rewiring_diagnostics{suffix}.parquet", index=False)
    manifest = {
        "null_model": null_model,
        "graph_source_revision": FINITE_SIZE_GRAPH_REVISION,
        "preserves": (
            ["node_count", "weak_edge_count", "weak_degree_sequence"]
            if null_model == "weak_projection" else
            ["node_count", "directed_edge_count", "in_degree_sequence",
             "out_degree_sequence"]
        ),
        "does_not_constrain": (
            [] if null_model == "weak_projection" else
            ["weak_edge_count", "weak_degree_sequence"]
        ),
        "seed": args.seed,
        "null_networks_requested": args.null_networks,
        "null_networks_used": int(len(keys)),
        "randomisations_per_network": args.n_random,
        "k_max": args.k_max,
        "target_accepted_swaps_per_edge": args.swap_multiplier,
        "max_attempts_per_target_swap": args.max_attempt_multiplier,
        "selected_network_sha256": _network_digest(keys),
    }
    (out / f"finite_size_null{suffix}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = ArgumentParser()
    parser.add_argument("--null-networks", type=int, default=200)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-model", choices=(*NULL_MODELS, "both"),
                        default="both")
    parser.add_argument("--swap-multiplier", type=int, default=10)
    parser.add_argument("--max-attempt-multiplier", type=int, default=30)
    parser.add_argument("--summaries-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "data" / "information_sciences"
    out.mkdir(parents=True, exist_ok=True)
    if args.summaries_only:
        observed = pd.read_parquet(out / "finite_size_observed.parquet")
        observed_summary(observed).to_csv(
            out / "finite_size_observed_summary.csv", index=False)
        for null_model in NULL_MODELS:
            suffix = OUTPUT_SUFFIX[null_model]
            path = out / f"finite_size_controlled{suffix}.parquet"
            controlled = pd.read_parquet(path)
            summary = controlled_summary(controlled)
            summary.to_csv(
                out / f"finite_size_controlled{suffix}_summary.csv",
                index=False,
            )
            print(f"\n{null_model} controlled summary")
            print(summary.to_string(index=False))
            manifest_path = out / f"finite_size_null{suffix}_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = (controlled[["match_id", "team_side"]].drop_duplicates()
                    .sort_values(["match_id", "team_side"])
                    .reset_index(drop=True))
            digest = _network_digest(keys)
            if manifest.get("selected_network_sha256") != digest:
                raise RuntimeError(
                    f"Selected-network digest mismatch for {null_model}")
            manifest["graph_source_revision"] = FINITE_SIZE_GRAPH_REVISION
            manifest["allorder_repair_revision"] = "skip_gtrie_header_v2"
            manifest_path.write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")
        return
    observed = observed_profile()
    observed.to_parquet(out / "finite_size_observed.parquet", index=False)
    summary = observed_summary(observed)
    summary.to_csv(out / "finite_size_observed_summary.csv", index=False)
    null_models = NULL_MODELS if args.null_model == "both" else (args.null_model,)
    for null_model in null_models:
        null, controlled, diagnostics, keys = null_analysis(
            observed, args.null_networks, args.n_random, args.k_max, args.seed,
            null_model=null_model, swap_multiplier=args.swap_multiplier,
            max_attempt_multiplier=args.max_attempt_multiplier)
        save_null_outputs(out, null_model, null, controlled, diagnostics, keys,
                          args)
        print(f"\n{null_model} controlled summary")
        print(controlled_summary(controlled).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
