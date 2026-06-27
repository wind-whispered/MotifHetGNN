"""Tests for z-score computation."""
import pytest
import numpy as np
import pandas as pd

from src.motifs.zscore import compute_zscore_for_motif, filter_significant_motifs


def test_zscore_positive():
    """Observed mean > random mean -> positive z-score."""
    observed = np.array([10.0, 12.0, 11.0, 9.0, 10.5])
    # Random: mean ~5, std ~1
    random_matrix = np.random.default_rng(42).normal(5.0, 1.0, (5, 100))
    mu, sigma, mu_rnd, sigma_rnd, z = compute_zscore_for_motif(observed, random_matrix)
    assert z > 0
    assert mu > mu_rnd


def test_zscore_negative():
    """Observed mean < random mean -> negative z-score."""
    observed = np.array([2.0, 3.0, 2.5, 1.5, 2.0])
    random_matrix = np.random.default_rng(42).normal(10.0, 1.0, (5, 100))
    mu, sigma, mu_rnd, sigma_rnd, z = compute_zscore_for_motif(observed, random_matrix)
    assert z < 0


def test_zscore_zero_sigma_rnd():
    """When sigma_rnd == 0, z should be 0.0 (no division by zero)."""
    observed = np.array([5.0, 5.0, 5.0])
    # All random counts identical -> sigma_rnd = 0
    random_matrix = np.full((3, 10), 5.0)
    mu, sigma, mu_rnd, sigma_rnd, z = compute_zscore_for_motif(observed, random_matrix)
    assert z == 0.0


def test_zscore_returns_five_values():
    observed = np.array([5.0, 6.0])
    random_matrix = np.ones((2, 50)) * 4.0
    result = compute_zscore_for_motif(observed, random_matrix)
    assert len(result) == 5


def test_filter_significant_motifs():
    zscore_df = pd.DataFrame({
        "motif_id": [12, 14, 78, 38],
        "z": [2.5, -0.5, 3.1, -2.0],
        "significant": [True, False, True, False],
    })
    sig = filter_significant_motifs(zscore_df, threshold=1.96)
    assert len(sig) == 2
    assert set(sig["motif_id"].tolist()) == {12, 78}


def test_filter_significant_motifs_custom_threshold():
    zscore_df = pd.DataFrame({
        "motif_id": [1, 2, 3],
        "z": [1.0, 2.0, 3.0],
        "significant": [False, True, True],
    })
    sig = filter_significant_motifs(zscore_df, threshold=2.5)
    assert len(sig) == 1
    assert sig["motif_id"].iloc[0] == 3
