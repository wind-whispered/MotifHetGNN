"""Precompute outcome-independent typed-triad participation features."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import FootballHeteroDataset, typed_triad_context
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION


def main():
    dataset = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    out_dir = ROOT / "data" / "gnn" / "motif_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for i, match_id in enumerate(dataset.match_ids, 1):
        output = out_dir / f"{match_id}.pt"
        if not output.exists():
            data = torch.load(dataset.graph_dir / f"{match_id}.pt", weights_only=False)
            torch.save(typed_triad_context(data), output)
        completed += 1
        if i % 250 == 0:
            print(f"{i}/{len(dataset)} cached", flush=True)
    print(f"motif context ready: {completed} graphs -> {out_dir}")


if __name__ == "__main__":
    main()
