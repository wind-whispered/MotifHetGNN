"""Tests for regression analysis - Panel A replication check."""
import pytest
import numpy as np
import pandas as pd

from src.analysis.regression import (
    build_design_matrix, run_ols_panel, compute_vif,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_motif_df():
    """Minimal motif DataFrame: 100 matches, k=3, motif_ids 12 and 14."""
    np.random.seed(42)
    n = 100
    records = []
    for match_id in range(1, n + 1):
        for mid in [12, 14]:
            for side in ["home", "away"]:
                records.append({
                    "match_id": match_id,
                    "motif_id": mid,
                    "motif_order_k": 3,
                    "team_side": side,
                    "count": float(np.random.poisson(5)),
                })
    return pd.DataFrame(records)


@pytest.fixture
def simple_match_meta(simple_motif_df):
    """Match meta with synthetic goal_diff correlated with motif 12."""
    np.random.seed(42)
    match_ids = simple_motif_df["match_id"].unique()
    # goal_diff loosely correlated with home motif 12 count
    home_12 = (
        simple_motif_df[(simple_motif_df["motif_id"] == 12) &
                        (simple_motif_df["team_side"] == "home")]
        .set_index("match_id")["count"]
    )
    goal_diffs = {
        mid: int(home_12.get(mid, 0) * 0.2 + np.random.randn())
        for mid in match_ids
    }
    return pd.DataFrame([
        {"match_id": mid, "goal_diff": gd,
         "home_team_id": 1, "away_team_id": 2}
        for mid, gd in goal_diffs.items()
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_design_matrix_shape(simple_motif_df, simple_match_meta):
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    # 4 feature columns: 12_home, 12_away, 14_home, 14_away
    assert X.shape[1] == 4
    assert len(y) == simple_match_meta["match_id"].nunique()


def test_build_design_matrix_no_negative_counts(simple_motif_df, simple_match_meta):
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    assert (X.values >= 0).all()


def test_run_ols_panel_returns_dataframe(simple_motif_df, simple_match_meta):
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    results = run_ols_panel(X, y, panel_name="test")
    assert isinstance(results, pd.DataFrame)
    assert "beta" in results.columns
    assert "p_value" in results.columns
    assert "significant" in results.columns


def test_run_ols_panel_const_row(simple_motif_df, simple_match_meta):
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    results = run_ols_panel(X, y, panel_name="test")
    # Should have intercept row
    assert "const" in results["variable"].values


def test_run_ols_panel_f_stat_positive(simple_motif_df, simple_match_meta):
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    results = run_ols_panel(X, y, panel_name="test")
    f_stat = results["f_stat"].iloc[0]
    assert f_stat > 0


def test_run_ols_panel_n_obs(simple_motif_df, simple_match_meta):
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    results = run_ols_panel(X, y, panel_name="test")
    n_obs = results["n_obs"].iloc[0]
    assert n_obs == simple_match_meta["match_id"].nunique()


def test_compute_vif_output(simple_motif_df, simple_match_meta):
    X, _ = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    vif_df = compute_vif(X)
    assert "feature" in vif_df.columns
    assert "VIF" in vif_df.columns
    assert len(vif_df) == X.shape[1]
    assert (vif_df["VIF"] >= 0).all()


def test_goal_diff_column_in_meta(simple_match_meta):
    assert "goal_diff" in simple_match_meta.columns


def test_design_matrix_fillna(simple_motif_df, simple_match_meta):
    """Design matrix should have no NaN values."""
    X, y = build_design_matrix(
        simple_motif_df, simple_match_meta,
        k_filter=3, significant_only=False,
    )
    assert not X.isnull().any().any()
    assert not y.isnull().any()
