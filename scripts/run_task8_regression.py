"""
Task 8: OLS regression - motif counts vs goal difference.
Outputs: data/analysis/regression_results.parquet
"""
import logging
from pathlib import Path

import pandas as pd

from src.data.loader import load_config
from src.analysis.regression import run_all_panels, compute_incremental_r2, compute_vif
from src.analysis.saturation import compute_decay_curve, build_table3_combined

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config("config.yaml")
    processed_dir = Path(cfg["data"]["processed_dir"])
    motifs_dir = Path(cfg["data"]["motifs_dir"])
    analysis_dir = Path(cfg["data"]["analysis_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    match_meta_df = pd.read_parquet(processed_dir / "matches_meta.parquet")
    homo_motif_df = pd.read_parquet(motifs_dir / "homogeneous_motifs.parquet")
    zscore_df = pd.read_parquet(motifs_dir / "homogeneous_zscore.parquet") \
        if (motifs_dir / "homogeneous_zscore.parquet").exists() else None

    hetero_motif_path = motifs_dir / "heterogeneous_motifs.parquet"
    hetero_motif_df = pd.read_parquet(hetero_motif_path) \
        if hetero_motif_path.exists() else pd.DataFrame()

    hetero_zscore_path = motifs_dir / "heterogeneous_zscore.parquet"
    hetero_zscore_df = pd.read_parquet(hetero_zscore_path) \
        if hetero_zscore_path.exists() else None

    # ---- Run all regression panels ----
    logger.info("Running OLS regression (three panels)...")
    regression_df = run_all_panels(
        homo_motif_df=homo_motif_df,
        hetero_motif_df=hetero_motif_df,
        match_meta_df=match_meta_df,
        homo_zscore_df=zscore_df,
        hetero_zscore_df=hetero_zscore_df,
    )
    regression_df.to_parquet(analysis_dir / "regression_results.parquet", index=False)
    logger.info(f"Saved regression_results: {len(regression_df)} records")

    # Print Panel A summary (should match original paper: F≈29.09)
    panel_a = regression_df[regression_df["panel"] == "A_k3_homo"]
    if not panel_a.empty:
        f_stat = panel_a["f_stat"].iloc[0]
        r2 = panel_a["r_squared"].iloc[0]
        n_obs = panel_a["n_obs"].iloc[0]
        logger.info(f"Panel A (k=3 homo): F={f_stat:.2f}, R²={r2:.4f}, N={n_obs}")
        print("\n=== Panel A Replication (original paper: F=29.09) ===")
        sig_rows = panel_a[panel_a["significant"] == True]
        print(sig_rows[["variable", "beta", "std_error", "t_stat", "p_value"]].to_string())

    # ---- Incremental R² (Fig. 10 saturation curve) ----
    logger.info("Computing incremental R² curve...")
    k_max = int(homo_motif_df["motif_order_k"].max())
    incr_r2_df = compute_incremental_r2(homo_motif_df, match_meta_df, k_max=k_max)
    incr_r2_df.to_parquet(analysis_dir / "incremental_r2.parquet", index=False)
    logger.info(f"Incremental R² curve:\n{incr_r2_df.to_string()}")

    # ---- VIF check for multicollinearity ----
    logger.info("Checking VIF for Panel A...")
    from src.analysis.regression import build_design_matrix
    X_a, _ = build_design_matrix(
        homo_motif_df, match_meta_df, k_filter=3, significant_only=False
    )
    if X_a.shape[1] > 1:
        vif_df = compute_vif(X_a)
        logger.info(f"VIF (top 5):\n{vif_df.head().to_string()}")
        high_vif = vif_df[vif_df["VIF"] > 10]
        if not high_vif.empty:
            logger.warning(f"High VIF features (>10):\n{high_vif.to_string()}")

    # ---- Table 3: Combined decay summary ----
    logger.info("Building Table 3 combined summary...")
    homo_decay = compute_decay_curve(homo_motif_df)

    if not hetero_motif_df.empty:
        from src.analysis.saturation import compute_hetero_decay_curve
        hetero_decay = compute_hetero_decay_curve(hetero_motif_df)
    else:
        hetero_decay = pd.DataFrame()

    table3_df = build_table3_combined(homo_decay, hetero_decay)
    table3_df.to_parquet(analysis_dir / "table3_combined.parquet", index=False)
    logger.info(f"Table 3:\n{table3_df.to_string()}")

    logger.info("Task 8 complete.")


if __name__ == "__main__":
    main()
