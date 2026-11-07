"""Seed-aware effect audit for strict event-prefix prediction.

Neural-versus-neural contrasts use matched initialisation seeds.  A score-only
baseline is deterministic, so its single result is broadcast across the five
neural seeds; the resulting interval reflects neural-seed variation only and
is labelled accordingly.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_ablation_paired_effects import (
    _exact_two_sided_sign_flip,
)


OUT = ROOT / "data" / "information_sciences"
INPUT = OUT / "temporal_results.parquet"
OUTPUT = OUT / "temporal_paired_effects.csv"
REPEATS = max(1000, int(os.environ.get(
    "IS_TEMPORAL_BOOTSTRAP_REPEATS", "20000")))
EXPECTED_SEEDS = {42, 7, 13, 21, 37}
CUTOFFS = (15, 30, 45, 60, 75)
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")
LOWER_IS_BETTER = {"mae"}
CONTRASTS = (
    ("sparc_over_edgeaware", "SPARC-GNN", "EdgeAware-Fixed"),
    ("sparc_over_team_pool", "SPARC-GNN", "TeamPool-MLP"),
    ("sparc_over_heterosage", "SPARC-GNN", "HeteroSAGE-sum"),
    ("score_augmented_over_graph_only", "SPARC-GNN+Score", "SPARC-GNN"),
    ("heterosage_score_augmented_over_graph_only", "HeteroSAGE-sum+Score",
     "HeteroSAGE-sum"),
    ("team_pool_score_augmented_over_graph_only", "TeamPool-MLP+Score",
     "TeamPool-MLP"),
    ("sparc_score_over_heterosage_score", "SPARC-GNN+Score",
     "HeteroSAGE-sum+Score"),
    ("sparc_score_over_team_pool_score", "SPARC-GNN+Score",
     "TeamPool-MLP+Score"),
    ("heterosage_score_over_team_pool_score", "HeteroSAGE-sum+Score",
     "TeamPool-MLP+Score"),
    ("score_augmented_over_calibrated_score", "SPARC-GNN+Score",
     "CurrentScore-Calibrated"),
    ("heterosage_score_over_calibrated_score", "HeteroSAGE-sum+Score",
     "CurrentScore-Calibrated"),
    ("team_pool_score_over_calibrated_score", "TeamPool-MLP+Score",
     "CurrentScore-Calibrated"),
    ("graph_only_over_calibrated_score", "SPARC-GNN",
     "CurrentScore-Calibrated"),
    ("heterosage_graph_only_over_calibrated_score", "HeteroSAGE-sum",
     "CurrentScore-Calibrated"),
)
FIXED_BASELINES = {"CurrentScore", "CurrentScore-Calibrated"}


def _rng_seed(cutoff: int, contrast: str, metric: str) -> int:
    digest = hashlib.sha256(
        f"{cutoff}:{contrast}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def paired_effects(frame: pd.DataFrame, repeats: int = REPEATS) -> pd.DataFrame:
    """Return matched-seed or neural-seed-versus-fixed-baseline effects."""
    frame = frame.copy()
    frame["model"] = frame["model"].replace({
        "MCAR-Decomp": "SPARC-GNN",
        "MCAR-Decomp+Score": "SPARC-GNN+Score",
    })
    key = ["cutoff", "model", "seed"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate cutoff-model-seed rows")
    indexed = frame.set_index(key).sort_index()
    rows: list[dict[str, object]] = []
    for cutoff in CUTOFFS:
        for contrast, target_name, reference_name in CONTRASTS:
            target_frame = indexed.loc[(cutoff, target_name)]
            target_seeds = set(target_frame.index.astype(int))
            if target_seeds != EXPECTED_SEEDS:
                raise ValueError(
                    f"incomplete target seeds for {cutoff}:{contrast}: "
                    f"{sorted(target_seeds)}")

            reference_frame = indexed.loc[(cutoff, reference_name)]
            reference_seeds = set(reference_frame.index.astype(int))
            if reference_name in FIXED_BASELINES:
                if reference_seeds != {-1}:
                    raise ValueError(
                        f"invalid fixed baseline for {cutoff}:{contrast}: "
                        f"{sorted(reference_seeds)}")
                pairing = "neural_seeds_vs_fixed_score_baseline"
            elif reference_seeds == EXPECTED_SEEDS:
                pairing = "matched_neural_seeds"
            else:
                raise ValueError(
                    f"incomplete reference seeds for {cutoff}:{contrast}: "
                    f"{sorted(reference_seeds)}")

            seeds = sorted(EXPECTED_SEEDS)
            for metric in METRICS:
                target = target_frame.loc[seeds, metric].to_numpy(float)
                if reference_name in FIXED_BASELINES:
                    reference = np.repeat(
                        float(reference_frame.loc[-1, metric]), len(seeds))
                else:
                    reference = reference_frame.loc[
                        seeds, metric].to_numpy(float)
                raw = target - reference
                improvement = -raw if metric in LOWER_IS_BETTER else raw
                rng = np.random.default_rng(
                    _rng_seed(cutoff, contrast, metric))
                sample = rng.integers(
                    0, len(seeds), size=(repeats, len(seeds)))
                bootstrap = improvement[sample].mean(axis=1)
                rows.append({
                    "cutoff": cutoff,
                    "contrast": contrast,
                    "target": target_name,
                    "reference": reference_name,
                    "pairing": pairing,
                    "metric": metric,
                    "direction": ("reference_minus_target"
                                  if metric in LOWER_IS_BETTER
                                  else "target_minus_reference"),
                    "raw_delta_target_minus_reference": float(raw.mean()),
                    "mean_improvement": float(improvement.mean()),
                    "ci_low": float(np.quantile(bootstrap, .025)),
                    "ci_high": float(np.quantile(bootstrap, .975)),
                    "target_better_seeds": int(np.sum(improvement > 0)),
                    "ties": int(np.sum(improvement == 0)),
                    "n_target_seeds": len(seeds),
                    "n_reference_seeds": (1 if reference_name in FIXED_BASELINES
                                          else len(seeds)),
                    "exact_two_sided_sign_flip_p":
                        _exact_two_sided_sign_flip(improvement),
                    "bootstrap_repeats": repeats,
                    "seedwise_improvements": ";".join(
                        f"{seed}:{value:.9g}"
                        for seed, value in zip(seeds, improvement)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    result = paired_effects(pd.read_parquet(INPUT))
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result)} effects -> {OUTPUT}")
    print(result[(result.contrast.isin({
        "sparc_over_edgeaware", "score_augmented_over_calibrated_score"}))
        & result.metric.isin(["macro_f1", "mae"])][[
            "cutoff", "contrast", "metric", "mean_improvement", "ci_low",
            "ci_high", "target_better_seeds", "pairing",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
