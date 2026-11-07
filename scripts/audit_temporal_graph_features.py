"""Verify every persisted temporal graph satisfies the current feature contract."""
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_temporal_cutoff_graphs import (
    CUTOFFS, TEMPORAL_GRAPH_REVISION,
)
from src.networks.node_features import NODE_FEATURE_DIM
from src.networks.edge_features import PASS_EDGE_DIM
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.temporal import SCORE_TIMELINE_REVISION

def audit():
    temporal_root = ROOT / "data" / "information_sciences" / "temporal_graphs"
    counts = Counter()
    by_cutoff = {}
    invalid_examples = []

    def invalid(kind, path):
        counts[kind] += 1
        if len(invalid_examples) < 25:
            invalid_examples.append({"kind": kind, "path": str(path)})

    for cutoff in CUTOFFS:
        paths = sorted((temporal_root / str(cutoff)).glob("*.pt"))
        by_cutoff[str(cutoff)] = len(paths)
        for path in paths:
            graph = torch.load(path, map_location="cpu", weights_only=False)
            counts["graphs"] += 1
            if getattr(graph, "temporal_graph_revision", None) != TEMPORAL_GRAPH_REVISION:
                invalid("invalid_revision", path)
            if getattr(graph, "graph_revision", None) != HETEROGENEOUS_GRAPH_REVISION:
                invalid("invalid_base_graph_revision", path)
            if (getattr(graph, "score_timeline_revision", None)
                    != SCORE_TIMELINE_REVISION):
                invalid("invalid_score_timeline_revision", path)
            if getattr(graph, "assist_flags_masked", None) is not True:
                invalid("missing_assist_mask_marker", path)
            if int(getattr(graph, "cutoff_minute", -1)) != cutoff:
                invalid("invalid_cutoff", path)
            if not all(hasattr(graph, name) for name in (
                    "current_home_score", "current_away_score",
                    "current_goal_diff", "y")):
                invalid("missing_score_or_label", path)
            elif int(graph.current_goal_diff) != (
                    int(graph.current_home_score) - int(graph.current_away_score)):
                invalid("inconsistent_current_score", path)

            for node_type in graph.node_types:
                features = graph[node_type].x
                counts["node_rows"] += int(features.shape[0])
                if features.ndim != 2 or features.shape[1] != NODE_FEATURE_DIM:
                    invalid("invalid_node_feature_shape", path)
                elif not bool(torch.isfinite(features).all()):
                    invalid("nonfinite_node_feature", path)

            for edge_type in graph.edge_types:
                if edge_type[1] != "pass":
                    continue
                attributes = graph[edge_type].edge_attr
                counts["pass_edges"] += int(attributes.shape[0])
                if attributes.ndim != 2 or attributes.shape[1] != PASS_EDGE_DIM:
                    invalid("invalid_pass_feature_shape", path)
                    continue
                nonfinite = int(torch.count_nonzero(
                    ~torch.isfinite(attributes)).item())
                counts["nonfinite_pass_feature_values"] += nonfinite
                if nonfinite:
                    invalid("nonfinite_pass_feature", path)
                nonzero = int(torch.count_nonzero(attributes[:, 10:12]).item())
                counts["nonzero_assist_values"] += nonzero
                if nonzero:
                    invalid("nonzero_assist_feature", path)
                zero_intensity = int(torch.count_nonzero(
                    attributes[:, 12] <= 0).item())
                counts["nonpositive_pass_intensity_values"] += zero_intensity
                if zero_intensity:
                    invalid("nonpositive_pass_intensity_feature", path)

    violation_keys = (
        "invalid_revision", "invalid_base_graph_revision",
        "invalid_score_timeline_revision", "inconsistent_current_score",
        "missing_assist_mask_marker", "invalid_cutoff",
        "missing_score_or_label", "invalid_node_feature_shape",
        "nonfinite_node_feature", "invalid_pass_feature_shape",
        "nonzero_assist_feature", "nonzero_assist_values",
        "nonfinite_pass_feature", "nonfinite_pass_feature_values",
        "nonpositive_pass_intensity_feature",
        "nonpositive_pass_intensity_values",
    )
    passed = counts["graphs"] == sum(by_cutoff.values()) \
        and counts["graphs"] > 0 \
        and not any(counts[key] for key in violation_keys)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "graph_revision": TEMPORAL_GRAPH_REVISION,
        "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "by_cutoff": by_cutoff,
        "counts": dict(counts),
        "node_feature_contract": {
            "dimension": NODE_FEATURE_DIM,
            "channels": [
                "tactical_position_one_hot", "home_away_side",
                "prefix_pass_count", "prefix_receive_count",
                "prefix_mean_event_coordinates", "pitch_zone_one_hot",
            ],
            "excluded": ["goals", "shots", "xG", "final_score"],
        },
        "pass_feature_mask": ["goal_assist", "shot_assist"],
        "pass_feature_dimension": PASS_EDGE_DIM,
        "invalid_examples": invalid_examples,
    }
    output = ROOT / "data" / "information_sciences" / (
        "temporal_graph_feature_audit.json")
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
