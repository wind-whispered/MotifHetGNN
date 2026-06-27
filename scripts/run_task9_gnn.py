"""
Task 9: GNN training, evaluation, and gradient attribution.
Outputs: data/gnn/gnn_model.pt, gnn_metrics.json, motif_attribution.parquet
"""
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from src.data.loader import load_config
from src.gnn.dataset import FootballHeteroDataset, split_dataset, create_data_loaders
from src.gnn.model import HeteroFootballGNN
from src.gnn.trainer import GNNTrainer
from src.gnn.evaluator import evaluate_on_test, compute_baseline_accuracy
from src.gnn.attribution import compute_population_attribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config("config.yaml")
    gnn_cfg = cfg["gnn"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    gnn_dir = Path(cfg["data"]["gnn_dir"])
    gnn_dir.mkdir(parents=True, exist_ok=True)

    root_dir = "."  # project root
    model_path = str(gnn_dir / "gnn_model.pt")
    metrics_path = str(gnn_dir / "gnn_metrics.json")

    # ---- Check dependencies ----
    try:
        import torch_geometric
    except ImportError:
        logger.error("torch_geometric not installed. Skipping Task 9.")
        return

    # ---- Dataset split ----
    logger.info("Loading dataset...")
    dataset = FootballHeteroDataset(root=root_dir)
    if len(dataset) == 0:
        logger.error("No heterogeneous graphs found. Run Task 3 first.")
        return

    train_ids, val_ids, test_ids = split_dataset(
        dataset,
        train_ratio=gnn_cfg["train_ratio"],
        val_ratio=gnn_cfg["val_ratio"],
        test_ratio=gnn_cfg["test_ratio"],
        seed=gnn_cfg["random_seed"],
    )
    logger.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir, train_ids, val_ids, test_ids,
        batch_size=gnn_cfg["batch_size"],
    )

    # ---- Model ----
    from src.networks.node_features import NODE_FEATURE_DIM
    model = HeteroFootballGNN(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ---- Baseline accuracy ----
    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")
    baseline_acc = compute_baseline_accuracy(match_meta_df)
    logger.info(f"Naive baseline accuracy: {baseline_acc:.3f}")

    # ---- Training ----
    trainer = GNNTrainer(
        model,
        lr=gnn_cfg["lr"],
        weight_cls=cfg.get("gnn", {}).get("loss_weight_cls", 0.5),
        weight_reg=cfg.get("gnn", {}).get("loss_weight_reg", 0.5),
        patience=gnn_cfg["patience"],
    )

    logger.info("Starting GNN training...")
    train_metrics = trainer.train(
        train_loader, val_loader,
        n_epochs=gnn_cfg["epochs"],
        save_path=model_path,
    )

    # ---- Evaluation ----
    logger.info("Evaluating on test set...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_results = evaluate_on_test(model, test_loader, device=device)

    final_metrics = {
        "baseline_accuracy": baseline_acc,
        "test_accuracy": test_results["accuracy"],
        "test_mae": test_results["mae"],
        "test_rmse": test_results["rmse"],
        "best_val_loss": train_metrics["best_val_loss"],
        "classification_report": test_results["classification_report"],
        "n_params": n_params,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Test accuracy: {test_results['accuracy']:.3f} "
                f"(baseline: {baseline_acc:.3f})")
    logger.info(f"Test MAE: {test_results['mae']:.3f}, RMSE: {test_results['rmse']:.3f}")

    # ---- Attribution ----
    logger.info("Computing motif attribution (Integrated Gradients)...")
    zscore_df = pd.read_parquet(motifs_dir / "homogeneous_zscore.parquet") \
        if (motifs_dir / "homogeneous_zscore.parquet").exists() else None
    analysis_dir = Path(cfg["data"]["analysis_dir"])
    regression_df = pd.read_parquet(analysis_dir / "regression_results.parquet") \
        if (analysis_dir / "regression_results.parquet").exists() else None
    homo_net_dir = str(Path(cfg["data"]["networks_dir"]) / "homogeneous")

    attribution_df = compute_population_attribution(
        model, test_loader, homo_net_dir, zscore_df, regression_df,
        device=device,
    )
    if not attribution_df.empty:
        attribution_df.to_parquet(gnn_dir / "motif_attribution.parquet", index=False)
        logger.info(f"Saved motif_attribution: {len(attribution_df)} records")
        print("\n=== Top 10 attributed motifs ===")
        print(attribution_df.sort_values("mean_attribution", ascending=False).head(10).to_string())

    logger.info("Task 9 complete.")


if __name__ == "__main__":
    main()
