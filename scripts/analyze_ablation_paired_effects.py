"""Audit paired multi-seed effects for the Information Sciences ablations.

The experiment has only five matched initialisation seeds.  We therefore
report the raw paired effect, a seed-bootstrap interval, directional
consistency, and an exact two-sided sign-flip probability.  With five seeds
the smallest attainable exact two-sided probability is 0.0625, so the output
is intended for effect-size reporting rather than binary significance claims.
"""
from __future__ import annotations

import hashlib
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "information_sciences"
INPUT = OUT / "ablation_results.parquet"
OUTPUT = OUT / "ablation_paired_effects.csv"
EXPECTED_SEEDS = {42, 7, 13, 21, 37}
REPEATS = max(1000, int(os.environ.get(
    "IS_ABLATION_BOOTSTRAP_REPEATS", "20000")))

# Each target differs from its baseline by the component named in `component`.
CONTRASTS = (
    ("edge_attributes", "EdgeAware-Fixed", "NoEdgeAttr-Fixed"),
    ("adaptive_relation_gate", "Adaptive-NoMotif", "EdgeAware-Fixed"),
    ("motif_conditioned_gate", "MotifGate", "Adaptive-NoMotif"),
    ("shared_private_conflict_decomposition", "MCAR-Decomp-FixedLoss",
     "MotifGate-ObservedOnly"),
    ("conflict_channel", "MCAR-Decomp-FixedLoss",
     "MCAR-Decomp-NoConflict"),
    ("availability_feature", "MCAR-Decomp-FixedLoss",
     "MCAR-Decomp-NoAvailabilityFeature"),
    ("motif_context_in_full_model", "MCAR-Decomp-FixedLoss",
     "MCAR-Decomp-NoMotif"),
    ("orthogonality_regularizer", "MCAR-Decomp-FixedLoss",
     "MCAR-Decomp-NoOrthogonality"),
    ("uncertainty_task_weighting", "MCAR-Decomp-Uncertainty",
     "MCAR-Decomp-FixedLoss"),
    ("retrospective_assist_proxy_upper_bound",
     "MCAR-Decomp-WithAssistProxy", "MCAR-Decomp-FixedLoss"),
)
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")
LOWER_IS_BETTER = {"mae"}


def _exact_two_sided_sign_flip(values: np.ndarray) -> float:
    """Return the exact randomisation tail probability for paired effects."""
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0),
                                               repeat=len(values))))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(null >= observed - 1e-15))


def _stable_seed(component: str, metric: str) -> int:
    digest = hashlib.sha256(f"{component}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> None:
    frame = pd.read_parquet(INPUT)
    duplicated = frame.duplicated(["variant", "seed"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate variant-seed rows in ablation results")

    indexed = frame.set_index(["variant", "seed"]).sort_index()
    rows: list[dict[str, object]] = []
    for component, target, baseline in CONTRASTS:
        target_seeds = set(indexed.loc[target].index.astype(int))
        baseline_seeds = set(indexed.loc[baseline].index.astype(int))
        common = sorted(target_seeds & baseline_seeds)
        if set(common) != EXPECTED_SEEDS:
            raise ValueError(
                f"Incomplete paired seeds for {component}: {common}")

        for metric in METRICS:
            target_values = indexed.loc[(target, common), metric].to_numpy(
                dtype=float)
            baseline_values = indexed.loc[(baseline, common), metric].to_numpy(
                dtype=float)
            raw_delta = target_values - baseline_values
            improvement = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
            rng = np.random.default_rng(_stable_seed(component, metric))
            sampled = rng.integers(0, len(common), size=(REPEATS, len(common)))
            bootstrap = improvement[sampled].mean(axis=1)
            rows.append({
                "component": component,
                "target": target,
                "baseline": baseline,
                "metric": metric,
                "direction": "baseline_minus_target" if metric in LOWER_IS_BETTER
                             else "target_minus_baseline",
                "raw_delta_target_minus_baseline": float(raw_delta.mean()),
                "mean_improvement": float(improvement.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
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
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result)} paired effects -> {OUTPUT}")
    print(result[
        (result.metric.isin(["macro_f1", "draw_f1", "mae"]))
    ][["component", "metric", "mean_improvement", "ci_low", "ci_high",
       "target_better_seeds", "exact_two_sided_sign_flip_p"]].to_string(
           index=False))


if __name__ == "__main__":
    main()
