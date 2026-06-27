"""
Task 5: Heterogeneous motif enumeration with node/edge type labels.
Implements labeled subgraph isomorphism for heterogeneous motifs.
"""
from typing import Dict, List, Optional, Tuple, Set
import logging
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)

# Heterogeneous node types
NODE_TYPE_HOME = 0
NODE_TYPE_AWAY = 1

# Heterogeneous edge types
EDGE_TYPE_PASS = 0
EDGE_TYPE_ADV = 1
EDGE_TYPE_TURN = 2

# Motif taxonomy labels
MOTIF_TYPE_COOPERATIVE = "cooperative"
MOTIF_TYPE_ADVERSARIAL = "adversarial"
MOTIF_TYPE_MIXED = "mixed"


def build_labeled_digraph(
    match_id: int,
    pass_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    player_side_map: Dict[int, str],
    w0: int = 2,
) -> nx.MultiDiGraph:
    """
    Build a labeled multi-directed graph where:
    - node attribute 'ntype': 0=home, 1=away
    - edge attribute 'etype': 0=pass, 1=adversarial, 2=turnover
    - edge attribute 'counterpress': bool
    - edge attribute 'under_pressure': bool
    """
    G = nx.MultiDiGraph()

    # Add nodes with type labels
    for pid, side in player_side_map.items():
        G.add_node(pid, ntype=NODE_TYPE_HOME if side == "home" else NODE_TYPE_AWAY)

    # Add pass edges
    match_passes = pass_df[pass_df["match_id"] == match_id]
    pass_counts = (
        match_passes.groupby(["player_id", "recipient_id"])
        .agg(
            weight=("length", "count"),
            under_pressure=("under_pressure", "any"),
            counterpress=("counterpress", "any"),
        )
        .reset_index()
    )

    for _, row in pass_counts.iterrows():
        u, v = int(row["player_id"]), int(row["recipient_id"])
        if row["weight"] <= w0:
            continue
        if u not in G.nodes:
            G.add_node(u, ntype=NODE_TYPE_HOME if player_side_map.get(u, "home") == "home" else NODE_TYPE_AWAY)
        if v not in G.nodes:
            G.add_node(v, ntype=NODE_TYPE_HOME if player_side_map.get(v, "home") == "home" else NODE_TYPE_AWAY)
        G.add_edge(u, v,
                   etype=EDGE_TYPE_PASS,
                   weight=int(row["weight"]),
                   under_pressure=bool(row["under_pressure"]),
                   counterpress=bool(row["counterpress"]))

    # Add adversarial and turnover edges
    TURN_TYPES = {"Miscontrol", "Dispossessed"}
    ADV_TYPES = {"Interception", "Tackle", "BallRecovery"}

    match_adv = adv_df[adv_df["match_id"] == match_id]
    for _, row in match_adv.iterrows():
        u = int(row["player_id"])
        evt_type = row.get("event_type", "")
        if u not in G.nodes:
            side = player_side_map.get(u, "home")
            G.add_node(u, ntype=NODE_TYPE_HOME if side == "home" else NODE_TYPE_AWAY)

        if evt_type in TURN_TYPES:
            # Turnover: self-loop marking loss of possession
            G.add_edge(u, u,
                       etype=EDGE_TYPE_TURN,
                       event_type=evt_type,
                       counterpress=bool(row.get("counterpress", False)))
        elif evt_type in ADV_TYPES:
            # Adversarial: needs target; use nearest player heuristic (simplified: mark on actor)
            G.add_edge(u, u,
                       etype=EDGE_TYPE_ADV,
                       event_type=evt_type,
                       counterpress=bool(row.get("counterpress", False)))

    return G


def classify_motif_type(
    subgraph: nx.MultiDiGraph,
    node_list: List[int],
) -> str:
    """Classify a subgraph motif into cooperative/adversarial/mixed."""
    node_types = set(subgraph.nodes[n].get("ntype", 0) for n in node_list)
    edge_types = set()
    for u, v, data in subgraph.edges(data=True):
        edge_types.add(data.get("etype", 0))

    has_cross_team = len(node_types) > 1
    has_pass = EDGE_TYPE_PASS in edge_types
    has_adv_or_turn = (EDGE_TYPE_ADV in edge_types) or (EDGE_TYPE_TURN in edge_types)

    if not has_cross_team and has_pass and not has_adv_or_turn:
        return MOTIF_TYPE_COOPERATIVE
    if has_cross_team and has_adv_or_turn and not has_pass:
        return MOTIF_TYPE_ADVERSARIAL
    return MOTIF_TYPE_MIXED


def enumerate_hetero_motifs_bruteforce(
    G: nx.MultiDiGraph,
    k: int,
) -> Dict[Tuple, int]:
    """
    Brute-force enumeration of k-node heterogeneous subgraph patterns.
    Returns dict: canonical_pattern_key -> count.

    canonical_pattern_key is a sorted tuple of:
    (sorted_node_types, sorted_edge_type_tuples)
    for isomorphism-class identification.

    NOTE: This is O(N^k) and suitable only for small k (3-5) and small graphs.
    """
    nodes = list(G.nodes())
    if len(nodes) < k:
        return {}

    pattern_counts: Dict[Tuple, int] = {}

    for node_subset in combinations(nodes, k):
        subG = G.subgraph(node_subset)
        if subG.number_of_edges() == 0:
            continue

        # Canonical key: node type sequence (sorted) + edge structure
        node_types_sorted = tuple(sorted(
            G.nodes[n].get("ntype", 0) for n in node_subset
        ))

        # Edge type multiset
        edge_tuples = []
        node_idx = {n: i for i, n in enumerate(sorted(node_subset))}
        for u, v, data in subG.edges(data=True):
            ui = node_idx[u]
            vi = node_idx[v]
            et = data.get("etype", 0)
            ut = G.nodes[u].get("ntype", 0)
            vt = G.nodes[v].get("ntype", 0)
            edge_tuples.append((ui, vi, et, ut, vt))

        edge_tuples_sorted = tuple(sorted(edge_tuples))
        pattern_key = (node_types_sorted, edge_tuples_sorted)
        pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1

    return pattern_counts


def enumerate_hetero_motifs_for_match(
    match_id: int,
    pass_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    player_side_map: Dict[int, str],
    k_start: int = 3,
    k_max: int = 8,
    w0: int = 2,
    zero_threshold: float = 0.01,
) -> Dict[int, Dict[Tuple, int]]:
    """
    Enumerate heterogeneous motifs for all orders k_start..k*.
    Returns: {k: {pattern_key: count}}
    """
    G = build_labeled_digraph(match_id, pass_df, adv_df, player_side_map, w0)

    if G.number_of_nodes() < k_start:
        return {}

    results = {}
    for k in range(k_start, k_max + 1):
        counts = enumerate_hetero_motifs_bruteforce(G, k)
        results[k] = counts

        total = sum(counts.values())
        if total < zero_threshold:
            logger.debug(f"Hetero k* reached at k={k-1} for match {match_id}")
            break

    return results


def hetero_motifs_to_records(
    match_id: int,
    order_results: Dict[int, Dict[Tuple, int]],
    G: Optional[nx.MultiDiGraph] = None,
) -> List[dict]:
    """Convert heterogeneous motif results to flat records."""
    records = []
    for k, pattern_counts in order_results.items():
        for pattern_key, count in pattern_counts.items():
            node_types, edge_tuples = pattern_key
            # Determine motif type
            has_cross = len(set(node_types)) > 1
            etypes_in_motif = set(et for _, _, et, _, _ in edge_tuples)
            has_pass = EDGE_TYPE_PASS in etypes_in_motif
            has_adv = (EDGE_TYPE_ADV in etypes_in_motif) or (EDGE_TYPE_TURN in etypes_in_motif)

            if not has_cross and has_pass and not has_adv:
                motif_type = MOTIF_TYPE_COOPERATIVE
            elif has_cross and has_adv and not has_pass:
                motif_type = MOTIF_TYPE_ADVERSARIAL
            else:
                motif_type = MOTIF_TYPE_MIXED

            records.append({
                "match_id": match_id,
                "motif_order_k": k,
                "pattern_key": str(pattern_key),
                "motif_type": motif_type,
                "count": count,
                "n_cross_team_nodes": int(has_cross),
                "has_pass_edge": int(has_pass),
                "has_adv_edge": int(has_adv),
            })
    return records


def build_hetero_order_summary(hetero_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize heterogeneous motif counts by order (Table 3 hetero sub-table)."""
    records = []
    for (k, mtype), group in hetero_df.groupby(["motif_order_k", "motif_type"]):
        total_per_match = group.groupby("match_id")["count"].sum()
        records.append({
            "motif_order_k": k,
            "motif_type": mtype,
            "observed_patterns": group["pattern_key"].nunique(),
            "mean_total_count": total_per_match.mean(),
            "std_total_count": total_per_match.std(),
        })
    return pd.DataFrame(records).sort_values(["motif_order_k", "motif_type"])
