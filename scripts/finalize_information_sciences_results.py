"""Canonicalise and validate result tables used by the manuscript.

The complete-match metadata contain the final score, so score-only rows in
the main or distribution-shift evaluations are label leakage.  They are
quarantined for auditability rather than silently discarded.  Cutoff-score
baselines remain valid in the event-prefix experiment.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "information_sciences"
INVALID_FULL_MATCH_MODELS = {"CurrentScore", "CurrentScore-Calibrated"}
MODEL_ALIASES = {
    "MCAR-HGNN": "MotifGate",
    "MCAR-Decomp": "SPARC-GNN",
    "MCAR-Decomp+Score": "SPARC-GNN+Score",
}


def _numeric_summary(frame, keys, path):
    numeric = [column for column in frame.select_dtypes(include=np.number)
               if column not in keys]
    frame.groupby(keys, observed=True)[numeric].agg(["mean", "std"]).to_csv(path)


def _canonicalise(frame, keys, allow_score):
    frame = frame.copy()
    frame["model"] = frame["model"].replace(MODEL_ALIASES)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    invalid = frame.iloc[0:0].copy()
    if not allow_score:
        invalid = frame[frame["model"].isin(INVALID_FULL_MATCH_MODELS)].copy()
        frame = frame[~frame["model"].isin(INVALID_FULL_MATCH_MODELS)].copy()
    frame = frame.drop_duplicates(subset=keys, keep="last")
    return frame, invalid


def _quarantine(frames):
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return
    quarantine = OUT / "quarantined_full_match_score_leakage.parquet"
    if quarantine.exists():
        frames.insert(0, pd.read_parquet(quarantine))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_parquet(quarantine, index=False)
    print(f"quarantined {len(combined)} invalid rows -> {quarantine}")


def finalize_main():
    quarantined = []
    for name in ("model_results_partial.parquet", "model_results.parquet"):
        path = OUT / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        clean, invalid = _canonicalise(frame, ["model", "seed"], allow_score=False)
        invalid["source_file"] = name
        quarantined.append(invalid)
        clean = clean.sort_values(["model", "seed"]).reset_index(drop=True)
        clean.to_parquet(path, index=False)

    for name in ("predictions_partial.parquet", "predictions.parquet"):
        path = OUT / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        clean, invalid = _canonicalise(
            frame, ["model", "seed", "match_id"], allow_score=False)
        invalid["source_file"] = name
        quarantined.append(invalid)
        clean = clean.sort_values(["model", "seed", "match_id"]).reset_index(drop=True)
        clean.to_parquet(path, index=False)

    _quarantine(quarantined)
    final = OUT / "model_results.parquet"
    if final.exists():
        frame = pd.read_parquet(final)
        _numeric_summary(frame, ["model"], OUT / "model_results_summary.csv")
        compact = frame.groupby("model", observed=True).agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            draw_f1_mean=("draw_f1", "mean"), draw_f1_std=("draw_f1", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        ).reset_index()
        compact.to_csv(OUT / "model_results_compact_summary.csv", index=False)
        print(f"main comparison: {len(frame)} valid rows")


def finalize_generalization():
    for name in ("generalization_results_partial.parquet",
                 "generalization_results.parquet"):
        path = OUT / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        clean, invalid = _canonicalise(
            frame, ["protocol", "model", "seed"], allow_score=False)
        if not invalid.empty:
            invalid["source_file"] = name
            _quarantine([invalid])
        clean = clean.sort_values(["protocol", "model", "seed"]).reset_index(drop=True)
        clean.to_parquet(path, index=False)
    final = OUT / "generalization_results.parquet"
    if final.exists():
        frame = pd.read_parquet(final)
        _numeric_summary(frame, ["protocol", "model"],
                         OUT / "generalization_results_summary.csv")
        print(f"generalization: {len(frame)} canonical rows")


def finalize_temporal():
    for name in ("temporal_results_partial.parquet", "temporal_results.parquet"):
        path = OUT / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        clean, _ = _canonicalise(
            frame, ["cutoff", "model", "seed"], allow_score=True)
        clean = clean.sort_values(["cutoff", "model", "seed"]).reset_index(drop=True)
        clean.to_parquet(path, index=False)
    final = OUT / "temporal_results.parquet"
    if final.exists():
        frame = pd.read_parquet(final)
        _numeric_summary(frame, ["cutoff", "model"],
                         OUT / "temporal_results_summary.csv")
        print(f"temporal cutoff: {len(frame)} canonical rows")


def main():
    parser = ArgumentParser()
    parser.add_argument("--scope", choices=("main", "generalization", "temporal", "all"),
                        default="all")
    args = parser.parse_args()
    if args.scope in {"main", "all"}:
        finalize_main()
    if args.scope in {"generalization", "all"}:
        finalize_generalization()
    if args.scope in {"temporal", "all"}:
        finalize_temporal()


if __name__ == "__main__":
    main()
