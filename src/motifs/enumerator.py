"""
Task 4/5 - Part A: motif enumeration.

Two back-ends are available:

1. A self-contained, pure-Python exact enumerator (the default). It enumerates
   every connected induced k-node subgraph and assigns each one the canonical
   motif id of Milo et al. (Science 2002) -- the minimum, over all node
   relabelings, of the row-major binary adjacency matrix read as an integer.
   This reproduces exactly the ids used by gtrieScanner / mfinder (e.g. the
   directed 3-motifs 6, 12, 14, 36, 38, 46, 74, 78, 98, 102, 108, 110, 238).

2. The original external ``gtrieScanner`` binary, used automatically when the
   environment variable ``GTRIE_SCANNER_BIN`` points to an existing executable
   (or ``USE_GTRIE_SCANNER=1`` is set). The pure-Python back-end is preferred
   because it needs no external compilation and is deterministic.
"""
import os
import subprocess
import tempfile
import re
from itertools import permutations, combinations
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import networkx as nx

logger = logging.getLogger(__name__)

# Path to gtrieScanner binary (optional; pure-Python back-end used otherwise)
GTRIE_SCANNER_BIN = os.environ.get("GTRIE_SCANNER_BIN", "gtrieScanner")


def _gtrie_available() -> bool:
    """True only if the user explicitly wired up a real gtrieScanner binary."""
    if os.environ.get("USE_GTRIE_SCANNER") == "1":
        return True
    bin_path = os.environ.get("GTRIE_SCANNER_BIN")
    return bool(bin_path) and Path(bin_path).exists()


# ---------------------------------------------------------------------------
# Pure-Python canonical motif enumeration (Milo et al. labeling)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _perm_bit_maps(k: int) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """
    Pre-compute, for each node permutation of size k, the list of
    (linear_adjacency_position, bit_weight) pairs used to turn an edge set
    into the row-major adjacency integer. Cached per motif order k.
    """
    maps = []
    nbits = k * k
    for perm in permutations(range(k)):
        # perm[m] = which matrix-row/col the original node m maps to
        # bit weight for matrix cell (a, b) is 2 ** (nbits - 1 - (a*k + b))
        cell = {}
        for orig in range(k):
            cell[orig] = perm[orig]
        maps.append(cell)
    return tuple(tuple(sorted(m.items())) for m in maps)


@lru_cache(maxsize=None)
def _permutations_cached(k: int) -> Tuple[Tuple[int, ...], ...]:
    return tuple(permutations(range(k)))


def canonical_motif_id(adj: Tuple[Tuple[int, int], ...], k: int) -> int:
    """
    Canonical motif id for a directed k-node subgraph.

    Args:
        adj: tuple of directed edges (a, b) with a, b in [0, k) (local indices).
        k:   number of nodes.

    Returns:
        The minimum, over all k! node relabelings, of the integer obtained by
        reading the row-major k*k adjacency matrix as a binary number
        (cell (i, j) is bit  k*k-1-(i*k+j) ).  This is the Milo/gtrieScanner id.
    """
    nbits = k * k
    edges = list(adj)
    best = None
    for perm in _permutations_cached(k):
        val = 0
        for a, b in edges:
            ra, rb = perm[a], perm[b]
            val |= 1 << (nbits - 1 - (ra * k + rb))
        if best is None or val < best:
            best = val
    return int(best if best is not None else 0)


def enumerate_motifs_python(G: nx.DiGraph, k: int) -> Dict[int, float]:
    """
    Exact enumeration of connected induced k-node subgraph motifs.

    A subgraph is counted when its k nodes are weakly connected (connected when
    edge directions are ignored), matching gtrieScanner's connected-subgraph
    semantics.  Returns {canonical_motif_id: count}.
    """
    n = G.number_of_nodes()
    if n < k or G.number_of_edges() == 0:
        return {}

    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    # adjacency as sets of local indices
    succ = {i: set() for i in range(n)}
    und = {i: set() for i in range(n)}  # undirected neighbours for connectivity
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        if iu == iv:
            continue
        succ[iu].add(iv)
        und[iu].add(iv)
        und[iv].add(iu)

    counts: Dict[int, int] = {}
    for combo in combinations(range(n), k):
        cset = set(combo)
        # weak-connectivity check via BFS on undirected neighbours
        start = combo[0]
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in und[x]:
                if y in cset and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != k:
            continue
        # local edge list
        local = {node: li for li, node in enumerate(combo)}
        edges = tuple(
            (local[a], local[b])
            for a in combo for b in succ[a] if b in cset
        )
        mid = canonical_motif_id(edges, k)
        counts[mid] = counts.get(mid, 0) + 1

    return {mid: float(c) for mid, c in counts.items()}


def write_graph_for_gtrie(G: nx.DiGraph, path: str) -> int:
    """
    Write graph in gtrieScanner edge-list format.
    Format: first line = N_nodes N_edges
             subsequent lines = src dst (0-indexed)
    Returns number of nodes written.
    """
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(nodes)} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")
    return len(nodes)


def run_gtrie_scanner(
    graph_path: str,
    motif_size: int,
    output_path: str,
    directed: bool = True,
    timeout: int = 300,
) -> bool:
    """
    Run gtrieScanner for a given motif size.
    Returns True if successful.
    """
    cmd = [
        GTRIE_SCANNER_BIN,
        "-s", str(motif_size),
        "-g", graph_path,
        "-o", output_path,
    ]
    if directed:
        cmd.append("-d")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.debug(f"gtrieScanner stderr: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"gtrieScanner timed out for motif size {motif_size}")
        return False
    except FileNotFoundError:
        raise RuntimeError(
            f"gtrieScanner binary not found at '{GTRIE_SCANNER_BIN}'. "
            "Set GTRIE_SCANNER_BIN environment variable or add to PATH."
        )


def parse_gtrie_output(output_path: str) -> Dict[int, float]:
    """
    Parse gtrieScanner output file.
    Returns dict: motif_id -> frequency (count).
    Format varies by version; handle common formats.
    """
    motif_counts: Dict[int, float] = {}
    if not Path(output_path).exists():
        return motif_counts

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Common format: "motif_id count" or "motif_id count zscore"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    motif_id = int(parts[0])
                    count = float(parts[1])
                    motif_counts[motif_id] = count
                except ValueError:
                    continue
    return motif_counts


def enumerate_motifs_for_graph(
    G: nx.DiGraph,
    k: int,
    directed: bool = True,
) -> Dict[int, float]:
    """
    Enumerate k-motifs for a single NetworkX graph.
    Uses temporary files for gtrieScanner I/O.
    Returns dict: motif_id -> count.
    """
    if G.number_of_nodes() < k or G.number_of_edges() == 0:
        return {}

    # Default: deterministic pure-Python canonical enumeration (no external deps)
    if not _gtrie_available():
        return enumerate_motifs_python(G, k)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.txt")
        output_file = os.path.join(tmpdir, "motifs.txt")

        write_graph_for_gtrie(G, graph_file)
        success = run_gtrie_scanner(graph_file, k, output_file, directed=directed)

        if not success:
            return enumerate_motifs_python(G, k)
        return parse_gtrie_output(output_file)


def enumerate_motifs_batch(
    graphs: List[Tuple[str, nx.DiGraph]],  # list of (label, graph)
    k: int,
    directed: bool = True,
) -> Dict[str, Dict[int, float]]:
    """
    Enumerate k-motifs for a list of graphs.
    Returns dict: label -> {motif_id -> count}.
    """
    results = {}
    for label, G in graphs:
        counts = enumerate_motifs_for_graph(G, k, directed=directed)
        results[label] = counts
    return results
