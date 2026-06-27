"""
Task 9b: recompute *genuine* participation-based motif attribution from the
already-trained GNN (no retraining).

The original attribution stored ``IG * |z|`` with a single per-side IG scalar,
which makes the attribution trivially proportional to the z-score. This script
instead maps the node-level integrated gradients of the trained model onto the
triadic motifs through explicit node participation, so the resulting per-motif
attribution is an independent, model-derived quantity. It then reports the
honest correlation between attribution, structural significance (z) and the OLS
coefficient (beta), and overwrites data/gnn/motif_attribution.parquet.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_config
from src.gnn.dataset import FootballHeteroDataset, split_dataset, create_data_loaders
from src.gnn.model import HeteroFootballGNN
from src.gnn.attribution import compute_population_attribution
from src.networks.node_features import NODE_FEATURE_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config("config.yaml")
    gnn_cfg = cfg["gnn"]
    gnn_dir = Path(cfg["data"]["gnn_dir"])
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    analysis_dir = Path(cfg["data"]["analysis_dir"])
    homo_net_dir = str(Path(cfg["data"]["networks_dir"]) / "homogeneous")

    dataset = FootballHeteroDataset(root=".")
    train_ids, val_ids, test_ids = split_dataset(
        dataset,
        train_ratio=gnn_cfg["train_ratio"],
        val_ratio=gnn_cfg["val_ratio"],
        test_ratio=gnn_cfg["test_ratio"],
        seed=gnn_cfg["random_seed"],
    )
    logger.info(f"Test matches: {len(test_ids)}")
    _, _, test_loader = create_data_loaders(
        ".", train_ids, val_ids, test_ids, batch_size=gnn_cfg["batch_size"]
    )

    model = HeteroFootballGNN(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    )
    state = torch.load(gnn_dir / "gnn_model.pt", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded trained GNN weights.")

    zscore_df = pd.read_parquet(motifs_dir / "homogeneous_zscore.parquet")
    regression_df = pd.read_parquet(analysis_dir / "regression_results.parquet")

    attribution_df = compute_population_attribution(
        model, test_loader, homo_net_dir, zscore_df, regression_df, device="cpu",
    )
    attribution_df.to_parquet(gnn_dir / "motif_attribution.parquet", index=False)
    logger.info(f"Saved genuine attribution: {len(attribution_df)} rows")

    # ---- Honest correlation report (triadic, pooled over home & away) ----
    a = attribution_df.dropna(subset=["mean_z"]).copy()
    from scipy import stats
    def report(x, y, name):
        m = (~x.isna()) & (~y.isna())
        if m.sum() < 3:
            print(f"  {name}: n<3"); return
        r, p = stats.pearsonr(x[m], y[m])
        rho, pr = stats.spearmanr(x[m], y[m])
        print(f"  {name}: Pearson r={r:.3f} (p={p:.3f}), Spearman rho={rho:.3f} (p={pr:.3f}), n={m.sum()}")

    print("\n=== Genuine GNN attribution vs structure (triadic, pooled) ===")
    report(a["mean_attribution"], a["mean_z"].abs(), "attr vs |z|")
    if "beta" in a.columns:
        report(a["mean_attribution"], a["beta"].abs(), "attr vs |beta|")
    print("\nTop-attributed motifs:")
    print(a.sort_values("mean_attribution", ascending=False)
          [["motif_id", "team_side", "mean_attribution", "n_instances", "mean_z"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
