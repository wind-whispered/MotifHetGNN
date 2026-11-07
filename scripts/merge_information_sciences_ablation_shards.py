"""Merge independently written ablation shards into the canonical result set."""

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_information_sciences_ablation import ALL_VARIANTS, SEEDS
from src.gnn.dataset import CHRONOLOGICAL_SPLIT_REVISION
from src.gnn.feature_contracts import full_match_feature_revision
from src.gnn.mcar_model import SPARC_REVISION
from src.gnn.trainer import CLASS_WEIGHT_REVISION, loss_revision_for
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION


def _current_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant", "seed", "model_revision", "split_revision",
        "feature_revision", "graph_revision", "loss_revision",
        "class_weight_revision",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ablation shard is missing columns: {missing}")
    frame = frame.copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    unknown = sorted(set(frame["variant"]) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants in shards: {unknown}")
    expected_model = frame["variant"].map(
        lambda name: (SPARC_REVISION if name.startswith("MCAR-Decomp")
                      else "baseline_v1"))
    expected_feature = frame["variant"].map(
        lambda name: full_match_feature_revision(bool(
            ALL_VARIANTS[name].get("use_assist_flags", False))))
    expected_loss = frame["variant"].map(
        lambda name: loss_revision_for(
            ALL_VARIANTS[name].get("loss_mode", "fixed")))
    valid = (
        frame["model_revision"].eq(expected_model)
        & frame["split_revision"].eq(CHRONOLOGICAL_SPLIT_REVISION)
        & frame["feature_revision"].eq(expected_feature)
        & frame["graph_revision"].eq(HETEROGENEOUS_GRAPH_REVISION)
        & frame["loss_revision"].eq(expected_loss)
        & frame["class_weight_revision"].eq(CLASS_WEIGHT_REVISION)
    )
    if not valid.all():
        bad = frame.loc[~valid, ["variant", "seed"]].to_dict("records")
        raise ValueError(f"stale or mismatched shard rows: {bad}")
    return frame


def main():
    out = ROOT / "data" / "information_sciences"
    canonical = out / "ablation_results.parquet"
    shards = sorted(out.glob("ablation_results_shard_*.parquet"))
    sources = ([canonical] if canonical.exists() else []) + shards
    if not sources:
        raise FileNotFoundError("no canonical or shard ablation results found")

    frame = _current_rows(pd.concat(
        [pd.read_parquet(path) for path in sources], ignore_index=True))
    key_columns = ["variant", "seed"]
    duplicate_keys = frame.loc[
        frame.duplicated(key_columns, keep=False), key_columns]
    if not duplicate_keys.empty:
        raise ValueError(
            "duplicate variant/seed rows across shards: "
            f"{duplicate_keys.drop_duplicates().to_dict('records')}")

    expected = {(variant, seed) for variant in ALL_VARIANTS for seed in SEEDS}
    actual = set(frame[key_columns].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(
            f"incomplete ablation merge: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}")

    checkpoint_dir = out / "checkpoints_ablation"
    checkpoint_dir.mkdir(exist_ok=True)
    shard_checkpoint_dirs = sorted(out.glob("checkpoints_ablation_shard_*"))
    for variant, seed in sorted(expected):
        destination = checkpoint_dir / f"{variant}_seed{seed}.pt"
        candidates = ([destination] if destination.exists() else []) + [
            directory / destination.name for directory in shard_checkpoint_dirs
            if (directory / destination.name).exists()
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one checkpoint for {variant}, seed={seed}; "
                f"found={candidates}")
        if candidates[0] != destination:
            shutil.copy2(candidates[0], destination)

    order = list(ALL_VARIANTS)
    frame["variant"] = pd.Categorical(
        frame["variant"], categories=order, ordered=True)
    frame = frame.sort_values(["variant", "seed"]).reset_index(drop=True)
    frame["variant"] = frame["variant"].astype(str)
    frame.to_parquet(canonical, index=False)
    frame.to_parquet(out / "ablation_results_partial.parquet", index=False)
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame.groupby("variant")[numeric].agg(["mean", "std"]).to_csv(
        out / "ablation_results_summary.csv")
    print(f"merged {len(frame)} rows from {len(sources)} result files")
    print(f"canonical checkpoints: {len(list(checkpoint_dir.glob('*.pt')))}")


if __name__ == "__main__":
    main()
