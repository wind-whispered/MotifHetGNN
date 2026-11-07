"""Seed-paired model effects under the four generalisation protocols."""
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
INPUT = OUT / "generalization_results.parquet"
OUTPUT = OUT / "generalization_paired_effects.csv"
REPEATS = max(1000, int(os.environ.get(
    "IS_GENERALIZATION_BOOTSTRAP_REPEATS", "20000")))
EXPECTED_SEEDS = {42, 7, 13, 21, 37}
PROTOCOLS = ("random", "chronological", "competition", "team_grouped")
TARGET = "SPARC-GNN"
BASELINES = ("TeamPool-MLP", "HeteroSAGE-sum", "HGT", "EdgeAware-Fixed")
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")
LOWER_IS_BETTER = {"mae"}


def _rng_seed(protocol: str, baseline: str, metric: str) -> int:
    digest = hashlib.sha256(
        f"{protocol}:{baseline}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def paired_effects(frame: pd.DataFrame, repeats: int = REPEATS) -> pd.DataFrame:
    """Compare SPARC-GNN with each baseline within matched split/seed pairs."""
    frame = frame.copy()
    frame["model"] = frame["model"].replace({"MCAR-Decomp": TARGET})
    key = ["protocol", "model", "seed"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate protocol-model-seed rows")
    indexed = frame.set_index(key).sort_index()
    rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        target_seeds = set(
            indexed.loc[(protocol, TARGET)].index.astype(int))
        if target_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"incomplete SPARC-GNN seeds for {protocol}: "
                f"{sorted(target_seeds)}")
        for baseline in BASELINES:
            baseline_seeds = set(
                indexed.loc[(protocol, baseline)].index.astype(int))
            common = sorted(target_seeds & baseline_seeds)
            if set(common) != EXPECTED_SEEDS:
                raise ValueError(
                    f"incomplete {baseline} pairs for {protocol}: {common}")
            for metric in METRICS:
                target = indexed.loc[
                    (protocol, TARGET, common), metric].to_numpy(float)
                reference = indexed.loc[
                    (protocol, baseline, common), metric].to_numpy(float)
                raw = target - reference
                improvement = -raw if metric in LOWER_IS_BETTER else raw
                rng = np.random.default_rng(
                    _rng_seed(protocol, baseline, metric))
                sample = rng.integers(
                    0, len(common), size=(repeats, len(common)))
                bootstrap = improvement[sample].mean(axis=1)
                rows.append({
                    "protocol": protocol,
                    "target": TARGET,
                    "baseline": baseline,
                    "metric": metric,
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
    print(result[(result.baseline == "EdgeAware-Fixed")
                 & result.metric.isin(["macro_f1", "draw_f1", "mae"])][[
        "protocol", "metric", "mean_improvement", "ci_low", "ci_high",
        "target_better_seeds", "exact_two_sided_sign_flip_p",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
