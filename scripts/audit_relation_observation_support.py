"""Quantify where node-level relation comparison is actually observable.

SPARC's pairwise conflict channels are defined only when both relation
messages have an incoming edge at the same target node.  This audit measures
that support directly from the graph topology, independently of a trained
model, and therefore bounds how widely the conflict gate can operate.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import FootballHeteroDataset, split_dataset_by_metadata
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION


RELATIONS = ("pass", "adversarial", "turnover")
NODE_TYPES = ("home_player", "away_player")
PAIRS = (("pass", "adversarial"), ("pass", "turnover"),
         ("adversarial", "turnover"))
OUT = ROOT / "data" / "information_sciences"


def main() -> None:
    dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=False,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    train, validation, test = split_dataset_by_metadata(
        dataset, str(ROOT / "data" / "processed" / "matches_meta.parquet"),
        strategy="chronological",
    )
    split_by_match = {
        int(match_id): split
        for split, ids in (("train", train), ("validation", validation),
                           ("test", test))
        for match_id in ids
    }

    rows = []
    for index in range(len(dataset)):
        data = dataset[index]
        match_id = int(data.match_id)
        for node_type in NODE_TYPES:
            n_nodes = int(data[node_type].num_nodes)
            available = {
                relation: torch.zeros(n_nodes, dtype=torch.bool)
                for relation in RELATIONS
            }
            edge_counts = {relation: 0 for relation in RELATIONS}
            for source, relation, target in data.edge_types:
                if target != node_type or relation not in available:
                    continue
                edge_index = data[(source, relation, target)].edge_index
                if edge_index is None or edge_index.numel() == 0:
                    continue
                available[relation][edge_index[1].unique()] = True
                edge_counts[relation] += int(edge_index.size(1))

            record = {
                "match_id": match_id,
                "split": split_by_match.get(match_id, "excluded"),
                "node_type": node_type,
                "n_nodes": n_nodes,
            }
            for relation in RELATIONS:
                record[f"edges_{relation}"] = edge_counts[relation]
                record[f"targets_{relation}"] = int(
                    available[relation].sum())
            stacked = torch.stack([available[r] for r in RELATIONS], dim=1)
            record["targets_at_least_one"] = int(stacked.any(dim=1).sum())
            record["targets_at_least_two"] = int(
                stacked.sum(dim=1).ge(2).sum())
            record["targets_all_three"] = int(stacked.all(dim=1).sum())
            for left, right in PAIRS:
                record[f"targets_{left}_{right}"] = int(
                    (available[left] & available[right]).sum())
            rows.append(record)

    frame = pd.DataFrame(rows).sort_values(
        ["match_id", "node_type"]).reset_index(drop=True)
    frame.to_parquet(OUT / "relation_observation_support.parquet", index=False)

    count_columns = [
        column for column in frame.columns
        if column.startswith("targets_") or column.startswith("edges_")
    ]
    pooled = {"graph_sides": int(len(frame)),
              "matches": int(frame.match_id.nunique()),
              "nodes": int(frame.n_nodes.sum())}
    for column in count_columns:
        pooled[column] = int(frame[column].sum())
        if column.startswith("targets_"):
            pooled[f"{column}_node_rate"] = float(
                frame[column].sum() / frame.n_nodes.sum())

    support = {
        "revision": "target_node_coobservation_v1",
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "pooled": pooled,
        "by_split": {},
        "target_concentration": {},
    }
    for split, group in frame.groupby("split"):
        support["by_split"][split] = {
            "graph_sides": int(len(group)),
            "nodes": int(group.n_nodes.sum()),
            **{
                f"{column}_node_rate": float(
                    group[column].sum() / group.n_nodes.sum())
                for column in count_columns if column.startswith("targets_")
            },
        }
    for relation in RELATIONS:
        column = f"targets_{relation}"
        support["target_concentration"][relation] = {
            "maximum_targets_per_graph_side": int(frame[column].max()),
            "graph_sides_with_at_most_one_target_rate": float(
                frame[column].le(1).mean()),
        }

    path = OUT / "relation_observation_support.json"
    path.write_text(json.dumps(support, indent=2), encoding="utf-8")
    print(json.dumps(support, indent=2))


if __name__ == "__main__":
    main()
