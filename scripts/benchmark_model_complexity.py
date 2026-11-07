"""Parameter and inference-cost audit on a fixed real-graph batch."""
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import FootballHeteroDataset
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import (HGTFootballGNN, MCARFootballGNN,
                                RGCNFootballGNN, SPARC_REVISION)
from src.gnn.model import HeteroFootballGNN
from src.gnn.team_pool import TeamPoolMLP
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION

BENCHMARK_REVISION = "real_graph_batch_128_v2"

MODELS = {
    "TeamPool-MLP": lambda: TeamPoolMLP(node_feature_dim=39),
    "HeteroSAGE-sum": lambda: HeteroFootballGNN(node_feature_dim=39),
    "R-GCN": lambda: RGCNFootballGNN(node_feature_dim=39),
    "HGT": lambda: HGTFootballGNN(node_feature_dim=39),
    "EdgeAware-Fixed": lambda: MCARFootballGNN(use_motif_context=False,
                                                adaptive_gate=False),
    "MotifGate-ObservedOnly": lambda: MCARFootballGNN(
        mask_unavailable_relations=True),
    "SPARC-GNN": lambda: MCARFootballGNN(use_info_decomposition=True),
}


def _score_baseline_row(device: str) -> dict:
    """Feature-only baseline: arithmetic on [home, away, diff, draw_flag]."""
    x = torch.randn((128, 4), device=device)
    # Warm-up
    for _ in range(10):
        _ = (x[:, 2] + (x[:, 3] > 0).float()) * 1.0
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5000):
        score_diff = x[:, 0] - x[:, 1]
        _ = torch.stack([(score_diff > 0).long(),
                        (score_diff == 0).long(),
                        (score_diff < 0).long()], dim=1)
        _ = score_diff
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 5000
    return {
        "model": "CurrentScore",
        "model_revision": "score_v1",
        "benchmark_revision": BENCHMARK_REVISION,
        "feature_revision": "cutoff_score_only_v1",
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "parameters": 0,
        "trainable_parameters": 0,
        "batch_size": 128,
        "device": device,
        "inference_ms_per_graph": elapsed * 1000 / 128,
    }


def main():
    dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    batch = next(iter(DataLoader([dataset[i] for i in range(128)], batch_size=128)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = batch.to(device)
    rows = []
    for name, factory in MODELS.items():
        model = factory().to(device).eval()
        with torch.no_grad():
            for _ in range(3): model(batch)
            if device == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(20): model(batch)
            if device == "cuda": torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / 20
        rows.append({"model": name,
                     "model_revision": (SPARC_REVISION if name == "SPARC-GNN"
                                        else "baseline_v1"),
                     "benchmark_revision": BENCHMARK_REVISION,
                     "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                     "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                 if p.requires_grad),
                     "batch_size": 128, "device": device,
                     "inference_ms_per_graph": elapsed * 1000 / 128})
    rows.append(_score_baseline_row(device))
    frame = pd.DataFrame(rows)
    path = ROOT / "data/information_sciences/model_complexity.parquet"
    frame.to_parquet(path, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
