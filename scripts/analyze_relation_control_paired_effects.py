"""Paired effect audit for post-training relation perturbation controls.

Every contrast compares the unmodified SPARC-GNN prediction with a matched
test-set perturbation from the same trained seed.  Positive ``mean_improvement``
always means that the unmodified graph performed better.  These contrasts are
diagnostic sensitivity tests, not retrained feature-ablation or causal effects.
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
INPUT = OUT / "relation_controls_decomp.parquet"
OUTPUT = OUT / "relation_control_paired_effects.csv"
REPEATS = max(1000, int(os.environ.get(
    "IS_RELATION_CONTROL_BOOTSTRAP_REPEATS", "20000")))
EXPECTED_SEEDS = {42, 7, 13, 21, 37}
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")
LOWER_IS_BETTER = {"mae"}
REFERENCE = ("all", "all")
CONTRASTS = (
    ("pass_availability", "delete", "pass"),
    ("adversarial_availability", "delete", "adversarial"),
    ("turnover_availability", "delete", "turnover"),
    ("pass_topology_degree_preserving", "permute", "pass"),
    ("adversarial_anchor_identity", "anchor_shift", "adversarial"),
    ("turnover_anchor_identity", "anchor_shift", "turnover"),
)


def _rng_seed(contrast: str, metric: str) -> int:
    digest = hashlib.sha256(f"{contrast}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def paired_effects(frame: pd.DataFrame, repeats: int = REPEATS) -> pd.DataFrame:
    """Return seed-paired effects for all predeclared controls."""
    key = ["condition", "relation", "seed"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate condition-relation-seed rows")
    indexed = frame.set_index(key).sort_index()
    reference_seeds = set(
        indexed.loc[REFERENCE].index.astype(int))
    if reference_seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"incomplete unmodified reference seeds: {sorted(reference_seeds)}")

    rows: list[dict[str, object]] = []
    for contrast, condition, relation in CONTRASTS:
        control_key = (condition, relation)
        control_seeds = set(indexed.loc[control_key].index.astype(int))
        common = sorted(reference_seeds & control_seeds)
        if set(common) != EXPECTED_SEEDS:
            raise ValueError(f"incomplete seeds for {contrast}: {common}")
        for metric in METRICS:
            reference = indexed.loc[(*REFERENCE, common), metric].to_numpy(float)
            control = indexed.loc[(*control_key, common), metric].to_numpy(float)
            raw = reference - control
            improvement = -raw if metric in LOWER_IS_BETTER else raw
            rng = np.random.default_rng(_rng_seed(contrast, metric))
            sample = rng.integers(0, len(common), size=(repeats, len(common)))
            bootstrap = improvement[sample].mean(axis=1)
            rows.append({
                "contrast": contrast,
                "reference_condition": REFERENCE[0],
                "reference_relation": REFERENCE[1],
                "control_condition": condition,
                "control_relation": relation,
                "metric": metric,
                "direction": ("control_minus_reference"
                              if metric in LOWER_IS_BETTER
                              else "reference_minus_control"),
                "raw_delta_reference_minus_control": float(raw.mean()),
                "mean_improvement": float(improvement.mean()),
                "ci_low": float(np.quantile(bootstrap, .025)),
                "ci_high": float(np.quantile(bootstrap, .975)),
                "reference_better_seeds": int(np.sum(improvement > 0)),
                "ties": int(np.sum(improvement == 0)),
                "n_seeds": len(common),
                "exact_two_sided_sign_flip_p":
                    _exact_two_sided_sign_flip(improvement),
                "bootstrap_repeats": repeats,
                "seedwise_improvements": ";".join(
                    f"{seed}:{value:.9g}"
                    for seed, value in zip(common, improvement)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    result = paired_effects(pd.read_parquet(INPUT))
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result)} effects -> {OUTPUT}")
    print(result[result.metric.isin(["macro_f1", "draw_f1", "mae"])][[
        "contrast", "metric", "mean_improvement", "ci_low", "ci_high",
        "reference_better_seeds", "exact_two_sided_sign_flip_p",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
