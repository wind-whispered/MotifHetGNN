"""
Task 6 - Part A: Degree-preserving random network generation.
Used to compute baseline motif frequencies for z-score calculation.
"""
from typing import Dict, Iterable, List, Optional, Set, Tuple
import logging
import copy
import random

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


SimpleEdge = Tuple[int, int]


def undirected_simple_edge_swap(
    edges: Iterable[SimpleEdge],
    target_swaps: int,
    rng: random.Random,
    max_attempt_multiplier: int = 30,
) -> Tuple[Set[SimpleEdge], int, int]:
    """Randomise a simple undirected graph while preserving every degree.

    Parameters are expressed in *accepted* swaps.  The returned diagnostics
    make incomplete mixing visible instead of silently treating rejected swap
    proposals as successful randomisation.
    """
    rewired = {tuple(sorted((u, v))) for u, v in edges if u != v}
    edge_list = list(rewired)
    successful = attempts = 0
    max_attempts = max(100, target_swaps * max_attempt_multiplier)

    while (successful < target_swaps and attempts < max_attempts
           and len(edge_list) >= 2):
        attempts += 1
        i, j = rng.sample(range(len(edge_list)), 2)
        a, b = edge_list[i]
        c, d = edge_list[j]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            new_a, new_b = tuple(sorted((a, c))), tuple(sorted((b, d)))
        else:
            new_a, new_b = tuple(sorted((a, d))), tuple(sorted((b, c)))
        if (new_a in rewired or new_b in rewired
                or new_a[0] == new_a[1] or new_b[0] == new_b[1]):
            continue
        rewired.remove((a, b))
        rewired.remove((c, d))
        rewired.add(new_a)
        rewired.add(new_b)
        edge_list[i], edge_list[j] = new_a, new_b
        successful += 1

    return rewired, successful, attempts


def directed_simple_edge_swap(
    edges: Iterable[SimpleEdge],
    target_swaps: int,
    rng: random.Random,
    max_attempt_multiplier: int = 30,
) -> Tuple[Set[SimpleEdge], int, int]:
    """Randomise a simple digraph while preserving all in/out degrees.

    A proposal ``(a -> b, c -> d)`` becomes ``(a -> d, c -> b)``.  Self-loops
    and parallel directed edges are rejected.  Directed edge count and both
    degree sequences are exact invariants; the weak-projection edge count is
    deliberately *not* constrained because reciprocal dyads can be created or
    removed.  This distinction matters for weak-connectivity motif censuses.
    """
    rewired = {(u, v) for u, v in edges if u != v}
    edge_list = list(rewired)
    successful = attempts = 0
    max_attempts = max(100, target_swaps * max_attempt_multiplier)

    while (successful < target_swaps and attempts < max_attempts
           and len(edge_list) >= 2):
        attempts += 1
        i, j = rng.sample(range(len(edge_list)), 2)
        a, b = edge_list[i]
        c, d = edge_list[j]
        if a == c or b == d or a == d or c == b:
            continue
        new_a, new_b = (a, d), (c, b)
        if new_a in rewired or new_b in rewired:
            continue
        rewired.remove((a, b))
        rewired.remove((c, d))
        rewired.add(new_a)
        rewired.add(new_b)
        edge_list[i], edge_list[j] = new_a, new_b
        successful += 1

    return rewired, successful, attempts


def degree_preserving_randomize(
    G: nx.DiGraph,
    n_swaps_multiplier: int = 10,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Generate a random graph with the same degree sequence as G
    via edge-swapping (Markov chain Monte Carlo).

    n_swaps = n_swaps_multiplier * n_edges swap attempts.
    Self-loops and multi-edges are avoided.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G_rand = copy.deepcopy(G)
    edges = list(G_rand.edges())
    n_edges = len(edges)

    if n_edges < 2:
        return G_rand

    n_swaps = n_swaps_multiplier * n_edges
    successful_swaps = 0

    for _ in range(n_swaps):
        # Pick two random edges
        i, j = random.sample(range(n_edges), 2)
        u1, v1 = edges[i]
        u2, v2 = edges[j]

        # Attempt swap: (u1->v1, u2->v2) -> (u1->v2, u2->v1)
        if u1 == u2 or v1 == v2:
            continue
        if u1 == v2 or u2 == v1:
            continue  # would create self-loop
        if G_rand.has_edge(u1, v2) or G_rand.has_edge(u2, v1):
            continue  # would create multi-edge

        # Perform swap (read weights from the CURRENT graph, not the original G,
        # since edges[] has already been rewired by previous swaps).
        w1 = G_rand[u1][v1].get("weight", 1)
        w2 = G_rand[u2][v2].get("weight", 1)
        G_rand.remove_edge(u1, v1)
        G_rand.remove_edge(u2, v2)
        G_rand.add_edge(u1, v2, weight=w1)
        G_rand.add_edge(u2, v1, weight=w2)
        edges[i] = (u1, v2)
        edges[j] = (u2, v1)
        successful_swaps += 1

    logger.debug(f"Randomization: {successful_swaps}/{n_swaps} swaps successful")
    return G_rand


def generate_random_networks(
    G: nx.DiGraph,
    n_random: int = 100,
    n_swaps_multiplier: int = 10,
    base_seed: int = 42,
) -> List[nx.DiGraph]:
    """
    Generate n_random degree-preserving random networks from G.
    """
    randoms = []
    for i in range(n_random):
        G_rand = degree_preserving_randomize(G, n_swaps_multiplier, seed=base_seed + i)
        randoms.append(G_rand)
    return randoms


def generate_hetero_random_network(
    G_labeled: "nx.MultiDiGraph",
    n_swaps_multiplier: int = 10,
    seed: Optional[int] = None,
) -> "nx.MultiDiGraph":
    """
    Generate a random heterogeneous network preserving:
    - Node type distribution
    - Per-edge-type degree sequences

    Strategy: shuffle edges of each type independently.
    """
    import copy
    if seed is not None:
        random.seed(seed)

    G_rand = copy.deepcopy(G_labeled)

    # Separate edges by type and swap within each type
    for etype in [0, 1, 2]:  # pass, adversarial, turnover
        type_edges = [
            (u, v, k) for u, v, k, d in G_rand.edges(data=True, keys=True)
            if d.get("etype") == etype
        ]
        if len(type_edges) < 2:
            continue

        n_swaps = n_swaps_multiplier * len(type_edges)
        for _ in range(n_swaps):
            i, j = random.sample(range(len(type_edges)), 2)
            u1, v1, k1 = type_edges[i]
            u2, v2, k2 = type_edges[j]

            if u1 == u2 or v1 == v2:
                continue
            if u1 == v2 or u2 == v1:
                continue

            # Check node type compatibility: preserve cross-team structure
            u1_type = G_rand.nodes[u1].get("ntype", 0)
            u2_type = G_rand.nodes[u2].get("ntype", 0)
            v1_type = G_rand.nodes[v1].get("ntype", 0)
            v2_type = G_rand.nodes[v2].get("ntype", 0)

            # After swap: u1->v2 and u2->v1
            # For pass edges: must be same-team (same ntype)
            if etype == 0:  # pass
                if u1_type != v2_type or u2_type != v1_type:
                    continue

            data1 = dict(G_rand[u1][v1][k1])
            data2 = dict(G_rand[u2][v2][k2])

            G_rand.remove_edge(u1, v1, key=k1)
            G_rand.remove_edge(u2, v2, key=k2)
            new_k1 = G_rand.add_edge(u1, v2, **data1)
            new_k2 = G_rand.add_edge(u2, v1, **data2)
            type_edges[i] = (u1, v2, new_k1)
            type_edges[j] = (u2, v1, new_k2)

    return G_rand
