"""Compare gtrie edge-list headers with the canonical pickled graphs.

The first edge-list row stores ``n_nodes n_edges`` and must never be parsed as
an edge.  This audit quantifies the impact of doing so and verifies that the
header itself matches the graph used to create it.
"""

from __future__ import annotations

from collections import Counter
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = ROOT / "data" / "networks" / "homogeneous"
OUTPUT = ROOT / "data" / "analysis" / "edge_list_header_audit.json"


def main() -> None:
    counts = Counter()
    examples: list[dict] = []

    for graph_path in sorted(NETWORK_DIR.glob("*.gpickle")):
        with graph_path.open("rb") as handle:
            graph = pickle.load(handle)
        edge_path = graph_path.with_suffix(".edgelist")
        with edge_path.open("r", encoding="utf-8") as handle:
            header_fields = handle.readline().split()
        if len(header_fields) != 2:
            counts["malformed_header"] += 1
            continue

        header_n, header_m = map(int, header_fields)
        header_edge = (header_n, header_m)
        nodes = set(graph.nodes())
        present = sum(node in nodes for node in header_edge)

        counts["files"] += 1
        counts[f"header_endpoints_present_{present}"] += 1
        counts["header_overlaps_real_node"] += int(present > 0)
        counts["header_is_real_edge"] += int(graph.has_edge(*header_edge))
        counts["header_node_count_mismatch"] += int(
            header_n != graph.number_of_nodes())
        counts["header_edge_count_mismatch"] += int(
            header_m != graph.number_of_edges())
        counts["isolated_nodes"] += sum(
            graph.degree(node) == 0 for node in nodes)
        counts["empty_graphs"] += int(graph.number_of_edges() == 0)

        noteworthy = (
            present > 0
            or header_n != graph.number_of_nodes()
            or header_m != graph.number_of_edges()
        )
        if noteworthy and len(examples) < 20:
            examples.append({
                "file": graph_path.name,
                "header": [header_n, header_m],
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "header_endpoints_present": present,
                "header_is_real_edge": bool(graph.has_edge(*header_edge)),
            })

    payload = {"counts": dict(counts), "examples": examples}
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
