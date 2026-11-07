"""Paired effect audit for retrained relation-subset models."""
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
REPEATS = max(1000, int(os.environ.get(
    "IS_RELATION_SUBSET_BOOTSTRAP_REPEATS", "20000")))
EXPECTED_SEEDS = {42, 7, 13, 21, 37}
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")
LOWER_IS_BETTER = {"mae"}
CONTRASTS = (
    ("all_over_pass", "all", "pass"),
    ("all_over_adversarial", "all", "adversarial"),
    ("all_over_turnover", "all", "turnover"),
    ("all_over_pass_adversarial", "all", "pass+adversarial"),
    ("all_over_pass_turnover", "all", "pass+turnover"),
    ("all_over_adversarial_turnover", "all", "adversarial+turnover"),
    ("pass_adversarial_over_pass", "pass+adversarial", "pass"),
    ("pass_adversarial_over_adversarial", "pass+adversarial", "adversarial"),
    ("pass_turnover_over_pass", "pass+turnover", "pass"),
    ("pass_turnover_over_turnover", "pass+turnover", "turnover"),
    ("adversarial_turnover_over_adversarial",
     "adversarial+turnover", "adversarial"),
    ("adversarial_turnover_over_turnover",
     "adversarial+turnover", "turnover"),
)


def _rng_seed(contrast: str, metric: str) -> int:
    digest = hashlib.sha256(f"{contrast}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> None:
    source = OUT / "relation_subset_results.parquet"
    frame = pd.read_parquet(source)
    if frame.duplicated(["relation_subset", "seed"]).any():
        raise ValueError("duplicate relation-subset seed rows")
    indexed = frame.set_index(["relation_subset", "seed"]).sort_index()
    rows = []
    for contrast, target, baseline in CONTRASTS:
        common = sorted(
            set(indexed.loc[target].index.astype(int))
            & set(indexed.loc[baseline].index.astype(int)))
        if set(common) != EXPECTED_SEEDS:
            raise ValueError(f"incomplete seeds for {contrast}: {common}")
        for metric in METRICS:
            target_values = indexed.loc[(target, common), metric].to_numpy(float)
            baseline_values = indexed.loc[(baseline, common), metric].to_numpy(float)
            raw = target_values - baseline_values
            improvement = -raw if metric in LOWER_IS_BETTER else raw
            rng = np.random.default_rng(_rng_seed(contrast, metric))
            sample = rng.integers(0, len(common), size=(REPEATS, len(common)))
            bootstrap = improvement[sample].mean(axis=1)
            rows.append({
                "contrast": contrast, "target": target,
                "baseline": baseline, "metric": metric,
                "direction": ("baseline_minus_target"
                              if metric in LOWER_IS_BETTER
                              else "target_minus_baseline"),
                "raw_delta_target_minus_baseline": float(raw.mean()),
                "mean_improvement": float(improvement.mean()),
                "ci_low": float(np.quantile(bootstrap, .025)),
                "ci_high": float(np.quantile(bootstrap, .975)),
                "target_better_seeds": int(np.sum(improvement > 0)),
                "ties": int(np.sum(improvement == 0)),
                "n_seeds": len(common),
                "exact_two_sided_sign_flip_p":
                    _exact_two_sided_sign_flip(improvement),
                "bootstrap_repeats": REPEATS,
                "seedwise_improvements": ";".join(
                    f"{seed}:{value:.9g}"
                    for seed, value in zip(common, improvement)),
            })
    result = pd.DataFrame(rows)
    target = OUT / "relation_subset_paired_effects.csv"
    result.to_csv(target, index=False)
    print(f"wrote {len(result)} effects -> {target}")
    print(result[result.metric.isin(["macro_f1", "draw_f1", "mae"])][[
        "contrast", "metric", "mean_improvement", "ci_low", "ci_high",
        "target_better_seeds", "exact_two_sided_sign_flip_p",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
