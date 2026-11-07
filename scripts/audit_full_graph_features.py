"""Audit every full-match graph against the current feature contract.

The audit reconstructs each aggregated pass pair from its processed event
shard.  It therefore checks not only that channel 12 exists, but that its
value exactly encodes the number of passes collapsed into that edge.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_config, read_match_shard
from src.networks.edge_features import (
    ADV_EDGE_DIM, PASS_EDGE_DIM, TURN_EDGE_DIM, build_pass_edge_feature,
)
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.node_features import NODE_FEATURE_DIM


def _expected_pass_intensity(pass_frame, team_id, player_ids):
    if pass_frame.empty:
        return {}
    required = {"team_id", "player_id", "recipient_id", "length"}
    if not required.issubset(pass_frame.columns):
        return {}
    selected = pass_frame[pass_frame["team_id"] == team_id]
    grouped = (selected.groupby(["player_id", "recipient_id"])
               .agg(weight=("length", "count")).reset_index())
    expected = {}
    for row in grouped.itertuples(index=False):
        source = int(row.player_id)
        target = int(row.recipient_id)
        if source not in player_ids or target not in player_ids:
            continue
        expected[(source, target)] = float(build_pass_edge_feature(
            pd.Series({"weight": row.weight}))[12])
    return expected


def audit():
    cfg = load_config(str(ROOT / "config.yaml"))
    processed = Path(cfg["data"]["processed_dir"])
    if not processed.is_absolute():
        processed = ROOT / processed
    metadata = pd.read_parquet(
        processed / "matches_meta.parquet").set_index("match_id")
    graph_dir = ROOT / "data" / "networks" / "heterogeneous"
    paths = sorted(graph_dir.glob("*.pt"), key=lambda path: int(path.stem))

    counts = Counter()
    maximum_error = 0.0
    invalid_examples = []

    def invalid(kind, path, detail=""):
        counts[kind] += 1
        if len(invalid_examples) < 25:
            invalid_examples.append({
                "kind": kind, "path": str(path), "detail": detail})

    dimensions = {
        "pass": PASS_EDGE_DIM,
        "adversarial": ADV_EDGE_DIM,
        "turnover": TURN_EDGE_DIM,
    }
    for index, path in enumerate(paths, 1):
        graph = torch.load(path, map_location="cpu", weights_only=False)
        counts["graphs"] += 1
        match_id = int(path.stem)
        if int(graph.match_id) != match_id:
            invalid("match_id_mismatches", path, str(graph.match_id))
        if getattr(graph, "graph_revision", None) != HETEROGENEOUS_GRAPH_REVISION:
            invalid("revision_mismatches", path,
                    repr(getattr(graph, "graph_revision", None)))
        if match_id not in metadata.index:
            invalid("metadata_mismatches", path, "missing metadata row")
            continue

        for node_type in graph.node_types:
            features = graph[node_type].x
            counts["node_rows"] += int(features.shape[0])
            if features.ndim != 2 or features.shape[1] != NODE_FEATURE_DIM:
                invalid("node_shape_mismatches", path, node_type)
            elif not bool(torch.isfinite(features).all()):
                invalid("nonfinite_node_features", path, node_type)

        pass_frame = read_match_shard(processed, match_id, "pass")
        meta = metadata.loc[match_id]
        for node_type, team_column in (
                ("home_player", "home_team_id"),
                ("away_player", "away_team_id")):
            edge_type = (node_type, "pass", node_type)
            player_tensor = getattr(graph[node_type], "player_ids", None)
            player_ids = ([] if player_tensor is None
                          else [int(value) for value in player_tensor.tolist()])
            player_set = set(player_ids)
            expected = _expected_pass_intensity(
                pass_frame, int(meta[team_column]), player_set)
            actual = {}
            if edge_type in graph.edge_types:
                store = graph[edge_type]
                attributes = store.edge_attr
                if attributes.ndim != 2 or attributes.shape[1] != PASS_EDGE_DIM:
                    invalid("pass_shape_mismatches", path, node_type)
                    continue
                nonfinite = int((~torch.isfinite(attributes)).sum().item())
                counts["nonfinite_pass_feature_values"] += nonfinite
                nonpositive = int((attributes[:, 12] <= 0).sum().item())
                counts["nonpositive_pass_intensity_values"] += nonpositive
                if nonfinite:
                    invalid("graphs_with_nonfinite_pass_features", path,
                            node_type)
                if nonpositive:
                    invalid("graphs_with_nonpositive_pass_intensity", path,
                            node_type)
                for column in range(int(store.edge_index.shape[1])):
                    source = player_ids[int(store.edge_index[0, column])]
                    target = player_ids[int(store.edge_index[1, column])]
                    pair = (source, target)
                    if pair in actual:
                        invalid("duplicate_pass_pairs", path, repr(pair))
                    actual[pair] = float(attributes[column, 12])

            if set(actual) != set(expected):
                invalid("pass_pair_set_mismatches", path, node_type)
            for pair in set(actual).intersection(expected):
                error = abs(actual[pair] - expected[pair])
                maximum_error = max(maximum_error, error)
                if error > 1e-6:
                    counts["pass_multiplicity_mismatches"] += 1
                    invalid("graphs_with_pass_multiplicity_mismatch", path,
                            f"{node_type}:{pair}:{error:.3e}")

        for edge_type in graph.edge_types:
            relation = edge_type[1]
            attributes = graph[edge_type].edge_attr
            counts[f"{relation}_edges"] += int(attributes.shape[0])
            expected_dimension = dimensions.get(relation)
            if (expected_dimension is None or attributes.ndim != 2
                    or attributes.shape[1] != expected_dimension):
                invalid("edge_shape_mismatches", path, repr(edge_type))
            elif not bool(torch.isfinite(attributes).all()):
                invalid("nonfinite_edge_features", path, repr(edge_type))

        if index % 500 == 0:
            print(f"audited {index}/{len(paths)} graphs", flush=True)

    violation_keys = {
        key for key in counts
        if key not in {"graphs", "node_rows", "pass_edges",
                       "adversarial_edges", "turnover_edges"}
        and counts[key] > 0
    }
    passed = bool(paths) and not violation_keys
    payload = {
        "status": "PASS" if passed else "FAIL",
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "pass_feature_dimension": PASS_EDGE_DIM,
        "maximum_pass_intensity_absolute_error": maximum_error,
        "counts": dict(counts),
        "invalid_examples": invalid_examples,
    }
    output = (ROOT / "data" / "information_sciences" /
              "full_graph_feature_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output


def main():
    payload, output = audit()
    print(json.dumps(payload, indent=2))
    print(output)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
