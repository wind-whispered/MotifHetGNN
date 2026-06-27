"""
Task 8: OLS regression analysis - motif counts vs goal difference.
Replicates and extends original paper Table 5 across three panels.
"""
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def build_design_matrix(
    motif_df: pd.DataFrame,
    match_meta_df: pd.DataFrame,
    motif_id_col: str = "motif_id",
    order_col: str = "motif_order_k",
    side_col: str = "team_side",
    count_col: str = "count",
    k_filter: Optional[int] = None,
    significant_only: bool = True,
    zscore_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build OLS design matrix X and response vector y.

    Each column: {motif_id}_{team_side} (e.g., "12_home", "12_away")
    y: goal_diff (home_score - away_score)

    Args:
        k_filter: if not None, only include this order k
        significant_only: if True, only include motifs with |z| > 1.96
        zscore_df: required if significant_only=True
    """
    if k_filter is not None:
        motif_df = motif_df[motif_df[order_col] == k_filter].copy()

    if significant_only and zscore_df is not None:
        sig_ids = set(
            zip(
                zscore_df[zscore_df["significant"]][motif_id_col].tolist(),
                zscore_df[zscore_df["significant"]][order_col].tolist(),
            )
        )
        motif_df = motif_df[
            motif_df.apply(lambda r: (r[motif_id_col], r[order_col]) in sig_ids, axis=1)
        ].copy()

    # Pivot: match_id -> one column per (motif_id, team_side)
    pivot = motif_df.groupby(["match_id", motif_id_col, side_col])[count_col].sum().reset_index()
    pivot["col_name"] = pivot[motif_id_col].astype(str) + "_" + pivot[side_col]
    wide = pivot.pivot_table(
        index="match_id", columns="col_name", values=count_col, fill_value=0
    ).reset_index()

    # Merge with goal_diff
    y_df = match_meta_df[["match_id", "goal_diff"]].copy()
    merged = wide.merge(y_df, on="match_id", how="inner")

    feature_cols = [c for c in merged.columns if c not in ("match_id", "goal_diff")]
    X = merged[feature_cols].copy()
    y = merged["goal_diff"].copy()

    return X, y


def run_ols_panel(
    X: pd.DataFrame,
    y: pd.Series,
    panel_name: str = "A",
) -> pd.DataFrame:
    """
    Run OLS regression and return results DataFrame.
    Replicates original paper Table 5 format.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    result = model.fit()

    records = []
    for var_name, beta, stderr, t_stat, p_val in zip(
        result.params.index,
        result.params.values,
        result.bse.values,
        result.tvalues.values,
        result.pvalues.values,
    ):
        # Parse column name: e.g., "12_home" -> motif_id=12, team_side="home"
        if "_" in var_name and var_name != "const":
            parts = var_name.rsplit("_", 1)
            try:
                motif_id = int(parts[0])
                team_side = parts[1]
            except (ValueError, IndexError):
                motif_id = None
                team_side = var_name
        else:
            motif_id = None
            team_side = "intercept"

        records.append({
            "panel": panel_name,
            "variable": var_name,
            "motif_id": motif_id,
            "team_side": team_side,
            "beta": float(beta),
            "std_error": float(stderr),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": float(p_val) < 0.05,
        })

    df_results = pd.DataFrame(records)
    df_results["f_stat"] = float(result.fvalue)
    df_results["f_pvalue"] = float(result.f_pvalue)
    df_results["r_squared"] = float(result.rsquared)
    df_results["r_squared_adj"] = float(result.rsquared_adj)
    df_results["n_obs"] = int(result.nobs)

    logger.info(
        f"Panel {panel_name}: F={result.fvalue:.2f} (p={result.f_pvalue:.4f}), "
        f"R²={result.rsquared:.4f}, N={result.nobs}"
    )
    return df_results


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for multicollinearity check."""
    X_clean = X.dropna(axis=1)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_clean.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_clean.values, i)
        for i in range(X_clean.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)


def run_all_panels(
    homo_motif_df: pd.DataFrame,
    hetero_motif_df: pd.DataFrame,
    match_meta_df: pd.DataFrame,
    homo_zscore_df: Optional[pd.DataFrame] = None,
    hetero_zscore_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run all three regression panels and concatenate results.

    Panel A: k=3 homogeneous (replication)
    Panel B: k>=4 homogeneous (extension)
    Panel C: heterogeneous motifs
    """
    all_results = []

    # Panel A: k=3 homo
    X_a, y_a = build_design_matrix(
        homo_motif_df, match_meta_df,
        k_filter=3, significant_only=False,
    )
    if X_a.shape[1] > 0:
        res_a = run_ols_panel(X_a, y_a, panel_name="A_k3_homo")
        all_results.append(res_a)

    # Panel B: k>=4 homo
    high_order_df = homo_motif_df[homo_motif_df["motif_order_k"] >= 4].copy()
    if not high_order_df.empty:
        X_b, y_b = build_design_matrix(
            high_order_df, match_meta_df,
            significant_only=True,
            zscore_df=homo_zscore_df,
        )
        if X_b.shape[1] > 0:
            res_b = run_ols_panel(X_b, y_b, panel_name="B_k4plus_homo")
            all_results.append(res_b)

    # Panel C: heterogeneous.
    # Individual labeled pattern keys number in the tens of thousands, which is
    # neither interpretable nor estimable with a few hundred matches. We instead
    # aggregate the per-match heterogeneous motif counts into a compact set of
    # tactically meaningful features: the total count of each motif *type*
    # (cooperative / adversarial / mixed) at each order k.
    if not hetero_motif_df.empty:
        agg = (
            hetero_motif_df
            .groupby(["match_id", "motif_type", "motif_order_k"])["count"].sum()
            .reset_index()
        )
        agg["col_name"] = agg["motif_type"] + "_k" + agg["motif_order_k"].astype(str)
        wide = agg.pivot_table(
            index="match_id", columns="col_name", values="count", fill_value=0
        ).reset_index()
        merged = wide.merge(match_meta_df[["match_id", "goal_diff"]], on="match_id", how="inner")
        feat_cols = [c for c in merged.columns if c not in ("match_id", "goal_diff")]
        # Standardise to keep coefficients comparable across types/orders
        X_c = merged[feat_cols].astype(float)
        y_c = merged["goal_diff"].astype(float)
        if X_c.shape[1] > 0 and len(X_c) > X_c.shape[1] + 1:
            res_c = run_ols_panel(X_c, y_c, panel_name="C_hetero")
            all_results.append(res_c)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def compute_incremental_r2(
    homo_motif_df: pd.DataFrame,
    match_meta_df: pd.DataFrame,
    k_max: int = 8,
) -> pd.DataFrame:
    """
    Compute R² increment as each order k is added to the regression.
    Used for Fig. 10 (information saturation curve).
    """
    # Cap the predictors contributed by each order to the TOP_K most frequent
    # motifs. Without this, order k=5 alone contributes thousands of columns
    # (p > n), the OLS saturates to R^2 = 1 by overfitting, and the genuine
    # "information saturation" signal is lost. Keeping n >> p yields an honest
    # marginal-R^2 curve.
    TOP_K = 25

    # Build, for each order, a per-match wide table of the TOP_K most frequent
    # motif counts, indexed by match_id so that orders can be merged cumulatively
    # without positional misalignment. The response (goal difference) is aligned
    # on the same match_id index.
    y_series = match_meta_df.set_index("match_id")["goal_diff"]

    records = []
    X_cumulative = pd.DataFrame(index=y_series.index)

    for k in range(3, k_max + 1):
        k_df = homo_motif_df[homo_motif_df["motif_order_k"] == k]
        if k_df.empty:
            break

        top_ids = (k_df.groupby("motif_id")["count"].sum()
                   .sort_values(ascending=False).head(TOP_K).index)
        k_df = k_df[k_df["motif_id"].isin(top_ids)]

        wide = (k_df.assign(col=k_df["motif_id"].astype(str) + "_"
                            + k_df["team_side"] + f"_k{k}")
                    .pivot_table(index="match_id", columns="col",
                                 values="count", aggfunc="sum", fill_value=0))
        X_cumulative = X_cumulative.join(wide, how="left").fillna(0.0)

        if X_cumulative.shape[1] == 0:
            continue

        y = y_series.reindex(X_cumulative.index)
        X_const = sm.add_constant(X_cumulative)
        try:
            model = sm.OLS(y.values, X_const.values).fit()
            r2 = float(model.rsquared)
        except Exception:
            r2 = 0.0

        n_types = int(k_df["motif_id"].nunique()) if False else \
            int(homo_motif_df[homo_motif_df["motif_order_k"] == k]["motif_id"].nunique())

        records.append({
            "k": k,
            "r_squared_cumulative": r2,
            "n_significant_motifs_at_k": n_types,
        })

    result = pd.DataFrame(records)
    if len(result) > 1:
        result["r2_increment"] = result["r_squared_cumulative"].diff().fillna(
            result["r_squared_cumulative"]
        )
    return result
