"""Paired effect audit for validation-calibrated temporal predictors.

This analysis keeps three questions separate:

1. Does score--graph calibration improve its equal-budget base model?
2. Does adding the stronger causal context improve the calibrated pathway?
3. Does a context--graph pathway outperform either causal context alone?

All learned pathways are paired by initialisation seed.  The two causal
context baselines are deterministic, so their test metric is broadcast over
the five neural seeds; the corresponding interval therefore measures neural
seed variation only.  No test value is used to select a candidate or weight.
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
BASE_INPUT = OUT / "temporal_results.parquet"
VAL_BLEND_INPUT = OUT / "temporal_validation_blends.parquet"
CONTEXT_BLEND_INPUT = OUT / "temporal_context_graph_blends.parquet"
CAUSAL_INPUT = OUT / "temporal_causal_baselines.parquet"
OUTPUT = OUT / "temporal_controlled_effects.csv"

REPEATS = max(1000, int(os.environ.get(
    "IS_TEMPORAL_BOOTSTRAP_REPEATS", "20000")))
EXPECTED_SEEDS = {42, 7, 13, 21, 37}
CUTOFFS = (15, 30, 45, 60, 75)
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")
LOWER_IS_BETTER = {"mae"}
FIXED_MODELS = {"Score+Strength", "CausalMatchState-HGB"}

FAMILIES = {
    "team_pool": (
        "TeamPool-MLP+Score",
        "TeamPool-MLP+Score+ValBlend",
        "TeamPool-MLP+Score+ContextBlend",
    ),
    "heterosage": (
        "HeteroSAGE-sum+Score",
        "HeteroSAGE-sum+Score+ValBlend",
        "HeteroSAGE-sum+Score+ContextBlend",
    ),
    "sparc": (
        "SPARC-GNN+Score",
        "SPARC-GNN+Score+ValBlend",
        "SPARC-GNN+Score+ContextBlend",
    ),
}


def _rng_seed(cutoff: int, contrast: str, metric: str) -> int:
    digest = hashlib.sha256(
        f"controlled:{cutoff}:{contrast}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_inputs() -> pd.DataFrame:
    frames = [
        pd.read_parquet(BASE_INPUT),
        pd.read_parquet(VAL_BLEND_INPUT),
        pd.read_parquet(CONTEXT_BLEND_INPUT),
        pd.read_parquet(CAUSAL_INPUT),
    ]
    required = {"cutoff", "model", "seed", *METRICS}
    for path, frame in zip(
            (BASE_INPUT, VAL_BLEND_INPUT, CONTEXT_BLEND_INPUT, CAUSAL_INPUT),
            frames):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} misses columns: {sorted(missing)}")
    combined = pd.concat(
        [frame.loc[:, sorted(required)] for frame in frames],
        ignore_index=True,
    )
    key = ["cutoff", "model", "seed"]
    if combined.duplicated(key).any():
        duplicate = combined.loc[combined.duplicated(key, keep=False), key]
        raise ValueError(
            "duplicate cutoff-model-seed rows:\n" + duplicate.to_string(index=False))
    if not np.isfinite(combined.loc[:, METRICS].to_numpy(float)).all():
        raise ValueError("non-finite controlled-effect input metric")
    return combined.set_index(key).sort_index()


def _contrasts() -> tuple[tuple[str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str]] = []
    for family, (base, val_blend, context_blend) in FAMILIES.items():
        rows.extend([
            (f"{family}_valblend_over_base", val_blend, base,
             "score_graph_calibration"),
            (f"{family}_contextblend_over_base", context_blend, base,
             "causal_context_calibration"),
            (f"{family}_contextblend_over_valblend", context_blend, val_blend,
             "stronger_context_increment"),
            (f"{family}_contextblend_over_score_strength", context_blend,
             "Score+Strength", "increment_over_causal_control"),
            (f"{family}_contextblend_over_match_state", context_blend,
             "CausalMatchState-HGB", "increment_over_causal_control"),
        ])
    rows.extend([
        ("sparc_context_over_heterosage_context", FAMILIES["sparc"][2],
         FAMILIES["heterosage"][2], "context_family_comparison"),
        ("sparc_context_over_team_pool_context", FAMILIES["sparc"][2],
         FAMILIES["team_pool"][2], "context_family_comparison"),
    ])
    return tuple(rows)


def controlled_effects(
        indexed: pd.DataFrame,
        repeats: int = REPEATS) -> pd.DataFrame:
    """Return seed-paired and neural-seed-versus-fixed-control effects."""
    rows: list[dict[str, object]] = []
    seeds = sorted(EXPECTED_SEEDS)
    for cutoff in CUTOFFS:
        for contrast, target_name, reference_name, claim_role in _contrasts():
            try:
                target_frame = indexed.loc[(cutoff, target_name)]
                reference_frame = indexed.loc[(cutoff, reference_name)]
            except KeyError as exc:
                raise ValueError(
                    f"missing controlled contrast input: {cutoff}:{contrast}") from exc

            target_seeds = set(target_frame.index.astype(int))
            if target_seeds != EXPECTED_SEEDS:
                raise ValueError(
                    f"incomplete target seeds for {cutoff}:{contrast}: "
                    f"{sorted(target_seeds)}")
            reference_seeds = set(reference_frame.index.astype(int))
            if reference_name in FIXED_MODELS:
                if reference_seeds != {-1}:
                    raise ValueError(
                        f"invalid fixed control for {cutoff}:{contrast}: "
                        f"{sorted(reference_seeds)}")
                pairing = "neural_seeds_vs_fixed_causal_control"
            elif reference_seeds == EXPECTED_SEEDS:
                pairing = "matched_neural_seeds"
            else:
                raise ValueError(
                    f"incomplete reference seeds for {cutoff}:{contrast}: "
                    f"{sorted(reference_seeds)}")

            for metric in METRICS:
                target = target_frame.loc[seeds, metric].to_numpy(float)
                if reference_name in FIXED_MODELS:
                    reference = np.repeat(
                        float(reference_frame.loc[-1, metric]), len(seeds))
                else:
                    reference = reference_frame.loc[seeds, metric].to_numpy(float)
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
                    "claim_role": claim_role,
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
                    "n_reference_seeds": (
                        1 if reference_name in FIXED_MODELS else len(seeds)),
                    "exact_two_sided_sign_flip_p":
                        _exact_two_sided_sign_flip(improvement),
                    "bootstrap_repeats": repeats,
                    "seedwise_improvements": ";".join(
                        f"{seed}:{value:.9g}"
                        for seed, value in zip(seeds, improvement)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    result = controlled_effects(_load_inputs())
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result)} controlled effects -> {OUTPUT}")
    focus = result[
        result["contrast"].str.startswith("sparc_")
        & result["metric"].isin(["macro_f1", "draw_f1", "mae"])
    ]
    print(focus[[
        "cutoff", "contrast", "metric", "mean_improvement", "ci_low",
        "ci_high", "target_better_seeds", "pairing",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
