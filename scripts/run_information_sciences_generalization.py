"""Four-level generalisation audit for the Information Sciences revision.

Protocols: random, chronological, competition-held-out, and team-disjoint.
Each protocol compares an edge-free TeamPool-MLP, HGT, an otherwise matched
edge-aware fixed-fusion control, and SPARC-GNN. Results are resumable at
protocol/model/seed level.
"""
from collections import OrderedDict
from pathlib import Path
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset, split_dataset,
                             split_dataset_by_metadata)
from src.gnn.evaluator import evaluate_on_test
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import SPARC_REVISION
from src.gnn.trainer import (
    CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION, GNNTrainer,
    training_class_weights,
)
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from scripts.run_information_sciences_models import MODELS as BASELINE_MODELS

SEEDS = [int(x) for x in os.environ.get("IS_SEEDS", "42,7,13,21,37").split(",")]
HOLDOUT = os.environ.get("IS_HOLDOUT_COMPETITION", "Premier League")
PROTOCOLS = [x for x in os.environ.get(
    "IS_PROTOCOLS", "random,chronological,competition,team_grouped").split(",") if x]
SPLIT_AUDIT_ONLY = os.environ.get("IS_SPLIT_AUDIT_ONLY", "0") == "1"

MODELS = OrderedDict()
for name, spec in BASELINE_MODELS.items():
    MODELS[name] = spec

_requested_models = [item.strip() for item in os.environ.get("IS_MODELS", "").split(",") if item.strip()]
MODEL_ALIASES = {"MCAR-Decomp": "SPARC-GNN"}
DEFAULT_GENERALIZATION_MODELS = [
    "TeamPool-MLP", "HeteroSAGE-sum", "HGT", "EdgeAware-Fixed",
    "SPARC-GNN",
]


def _normalise_requested_models(requested):
    if not requested:
        return []
    normalised = []
    unknown = []
    for name in requested:
        if name in MODELS:
            normalised.append(name)
        elif name in MODEL_ALIASES and MODEL_ALIASES[name] in MODELS:
            normalised.append(MODEL_ALIASES[name])
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unknown model(s) in IS_MODELS: {unknown}")
    return normalised


SELECTED_MODELS = (_normalise_requested_models(_requested_models)
                  if _requested_models else DEFAULT_GENERALIZATION_MODELS)

LEGACY_NAMES = {"MCAR-Decomp": "SPARC-GNN"}
INVALID_FULL_MATCH_MODELS = {"CurrentScore", "CurrentScore-Calibrated"}
NON_TEMPORAL_SPLIT_REVISION = "metadata_protocol_v1"


def split_revision_for(protocol):
    return (CHRONOLOGICAL_SPLIT_REVISION if protocol == "chronological"
            else NON_TEMPORAL_SPLIT_REVISION)


def load_existing_rows(path: Path):
    """Canonicalise resumable rows and reject full-time score leakage.

    The metadata scores are final scores, so they are labels rather than
    admissible inputs for a complete-match generalisation experiment.  Score
    baselines belong exclusively to the event-prefix experiment, where the
    score is observed at the declared cutoff.
    """
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    frame["model"] = frame["model"].replace(LEGACY_NAMES)
    frame = frame[~frame["model"].isin(INVALID_FULL_MATCH_MODELS)].copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if "model_revision" not in frame:
        frame["model_revision"] = pd.NA
    if "split_revision" not in frame:
        frame["split_revision"] = pd.NA
    if "feature_revision" not in frame:
        frame["feature_revision"] = pd.NA
    if "graph_revision" not in frame:
        frame["graph_revision"] = pd.NA
    if "loss_revision" not in frame:
        frame["loss_revision"] = pd.NA
    if "class_weight_revision" not in frame:
        frame["class_weight_revision"] = pd.NA
    current = ((frame["model"] != "SPARC-GNN")
               | (frame["model_revision"] == SPARC_REVISION)).fillna(False)
    expected_split = frame["protocol"].map(split_revision_for)
    current &= (frame["split_revision"] == expected_split).fillna(False)
    current &= (frame["feature_revision"] ==
                PRIMARY_FULL_MATCH_FEATURE_REVISION).fillna(False)
    current &= (frame["graph_revision"] ==
                HETEROGENEOUS_GRAPH_REVISION).fillna(False)
    current &= (frame["loss_revision"] == FIXED_LOSS_REVISION).fillna(False)
    current &= (frame["class_weight_revision"] ==
                CLASS_WEIGHT_REVISION).fillna(False)
    frame = frame[current].copy()
    baseline = frame["model"] != "SPARC-GNN"
    frame.loc[baseline, "model_revision"] = frame.loc[
        baseline, "model_revision"].fillna("baseline_v1")
    frame = frame.drop_duplicates(
        subset=["protocol", "model", "seed"], keep="last")
    return frame.to_dict("records")


def reuse_chronological_rows(rows, out: Path):
    """Merge the identical chronological main-comparison runs.

    The main comparison and this protocol use the same split, model registry,
    seeds and optimisation budget. Reusing those rows avoids treating a
    duplicate stochastic fit as a separate generalisation result.
    """
    source = out / "model_results.parquet"
    if not source.exists():
        return rows
    completed = {(row["protocol"], row["model"], int(row["seed"]))
                 for row in rows}
    main = pd.read_parquet(source)
    main["model"] = main["model"].replace(LEGACY_NAMES)
    for record in main.to_dict("records"):
        model = record["model"]
        seed = int(record["seed"])
        key = ("chronological", model, seed)
        if model not in SELECTED_MODELS or seed < 0 or key in completed:
            continue
        if (model == "SPARC-GNN"
                and record.get("model_revision") != SPARC_REVISION):
            continue
        if record.get("split_revision") != CHRONOLOGICAL_SPLIT_REVISION:
            continue
        if record.get("feature_revision") != PRIMARY_FULL_MATCH_FEATURE_REVISION:
            continue
        if record.get("graph_revision") != HETEROGENEOUS_GRAPH_REVISION:
            continue
        if record.get("loss_revision") != FIXED_LOSS_REVISION:
            continue
        if record.get("class_weight_revision") != CLASS_WEIGHT_REVISION:
            continue
        rows.append({
            "protocol": "chronological", "model": model, "seed": seed,
            "model_revision": record.get(
                "model_revision", SPARC_REVISION if model == "SPARC-GNN"
                else "baseline_v1"),
            "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
            "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
            "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
            "loss_revision": FIXED_LOSS_REVISION,
            "class_weight_revision": CLASS_WEIGHT_REVISION,
            "class_weight_home": float(record["class_weight_home"]),
            "class_weight_draw": float(record["class_weight_draw"]),
            "class_weight_away": float(record["class_weight_away"]),
            "n_train": int(record["n_train"]),
            "n_validation": int(record["n_validation"]),
            "n_test": int(record["n_test"]),
            "best_val_loss": record.get("best_val_loss", np.nan),
            "best_val_task_loss": record.get(
                "best_val_task_loss", record.get("best_val_loss", np.nan)),
            "best_val_auxiliary_loss": record.get(
                "best_val_auxiliary_loss", np.nan),
            "train_seconds": np.nan,
            "training_source": "model_results.parquet",
            **{metric: record[metric] for metric in
               ("accuracy", "macro_f1", "draw_f1", "mae", "r2")},
        })
        completed.add(key)
    return rows


def split_for(protocol, dataset, metadata_path, seed):
    if protocol == "random":
        return split_dataset(dataset, seed=seed)
    kwargs = {}
    if protocol == "competition":
        kwargs["holdout_competition"] = HOLDOUT
    return split_dataset_by_metadata(dataset, str(metadata_path),
                                     strategy=protocol, seed=seed, **kwargs)


def audit_split(protocol, split, metadata):
    """Validate and summarise the leakage constraints of one split."""
    names = ("train", "validation", "test")
    id_sets = [set(map(int, ids)) for ids in split]
    id_overlap = {
        f"{names[i]}_{names[j]}": len(id_sets[i].intersection(id_sets[j]))
        for i in range(3) for j in range(i + 1, 3)
    }
    if any(id_overlap.values()):
        raise ValueError(f"match leakage in {protocol}: {id_overlap}")

    teams = []
    date_ranges = {}
    for name, ids in zip(names, split):
        subset = metadata.loc[list(map(int, ids))]
        teams.append(set(subset["home_team_id"]).union(subset["away_team_id"]))
        dates = pd.to_datetime(subset["match_date"], errors="coerce").dropna()
        date_ranges[name] = {
            "min": dates.min().isoformat() if not dates.empty else None,
            "max": dates.max().isoformat() if not dates.empty else None,
        }
    team_overlap = {
        f"{names[i]}_{names[j]}": len(teams[i].intersection(teams[j]))
        for i in range(3) for j in range(i + 1, 3)
    }

    if protocol == "team_grouped" and any(team_overlap.values()):
        raise ValueError(f"team identity leakage: {team_overlap}")
    if protocol == "competition":
        train_val = metadata.loc[list(id_sets[0].union(id_sets[1]))]
        test = metadata.loc[list(id_sets[2])]
        if (train_val["competition_name"] == HOLDOUT).any():
            raise ValueError("holdout competition leaked into train/validation")
        if not (test["competition_name"] == HOLDOUT).all():
            raise ValueError("competition test set contains a non-holdout match")
    if protocol == "chronological":
        train_max = pd.Timestamp(date_ranges["train"]["max"])
        val_min = pd.Timestamp(date_ranges["validation"]["min"])
        val_max = pd.Timestamp(date_ranges["validation"]["max"])
        test_min = pd.Timestamp(date_ranges["test"]["min"])
        if train_max >= val_min or val_max >= test_min:
            raise ValueError(f"chronological order violated: {date_ranges}")
    if protocol == "competition":
        train_max = pd.Timestamp(date_ranges["train"]["max"])
        val_min = pd.Timestamp(date_ranges["validation"]["min"])
        if train_max >= val_min:
            raise ValueError(
                f"competition train/validation chronology violated: {date_ranges}")

    return {
        "match_overlap": id_overlap,
        "team_overlap": team_overlap,
        "unique_teams": dict(zip(names, map(len, teams))),
        "date_ranges": date_ranges,
    }


def main():
    out = ROOT / "data" / "information_sciences"
    ckpts = out / "checkpoints_generalization"
    ckpts.mkdir(parents=True, exist_ok=True)
    partial = out / "generalization_results_partial.parquet"
    rows = load_existing_rows(partial)
    rows = reuse_chronological_rows(rows, out)

    if rows:
        # Persist the canonical method name before resuming, so an interrupted
        # run cannot reintroduce duplicate MCAR-Decomp/SPARC-GNN rows.
        pd.DataFrame(rows).to_parquet(partial, index=False)

    completed = {(r["protocol"], r["model"], int(r["seed"])) for r in rows}

    dataset = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    graph_by_id = {mid: dataset[i] for i, mid in enumerate(dataset.match_ids)}
    metadata_path = ROOT / "data" / "processed" / "matches_meta.parquet"
    metadata = pd.read_parquet(metadata_path).set_index("match_id")
    split_records = {}

    for protocol in PROTOCOLS:
        for seed in SEEDS:
            split = split_for(protocol, dataset, metadata_path, seed)
            split_records[f"{protocol}:{seed}"] = {
                "train": split[0], "validation": split[1], "test": split[2],
                "split_revision": split_revision_for(protocol),
                "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                "loss_revision": FIXED_LOSS_REVISION,
                "class_weight_revision": CLASS_WEIGHT_REVISION,
                "holdout_competition": HOLDOUT if protocol == "competition" else None,
                "audit": audit_split(protocol, split, metadata),
            }
            graphs = [[graph_by_id[mid] for mid in ids if mid in graph_by_id]
                      for ids in split]
            if min(map(len, graphs)) == 0:
                raise RuntimeError(f"empty partition for {protocol}, seed={seed}")
            class_weights = training_class_weights(
                [float(graph.y.item()) for graph in graphs[0]])
            split_records[f"{protocol}:{seed}"]["class_weights"] = {
                "home": float(class_weights[0]),
                "draw": float(class_weights[1]),
                "away": float(class_weights[2]),
            }
            # Persist the audited split before any expensive fit.  This keeps
            # the protocol and class-balance evidence inspectable after an
            # interrupted resumable run.
            (out / "generalization_splits.json").write_text(
                json.dumps(split_records, indent=2), encoding="utf-8")

            if SPLIT_AUDIT_ONLY:
                print(f"audited {protocol}:{seed}: "
                      f"{len(split[0])}/{len(split[1])}/{len(split[2])}",
                      flush=True)
                continue

            for name in SELECTED_MODELS:
                cfg = MODELS[name]
                key = (protocol, name, seed)
                if key in completed:
                    print("skip", key, flush=True)
                    continue
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                loaders = [DataLoader(g, batch_size=32 if i == 0 else 64,
                                      shuffle=(i == 0),
                                      generator=generator if i == 0 else None)
                           for i, g in enumerate(graphs)]
                model = cfg["factory"](
                    node_feature_dim=39, hidden_dim=64, num_layers=3,
                    dropout=0.3, **cfg.get("kwargs", {}))
                trainer = GNNTrainer(model, lr=1e-3, patience=20,
                                     loss_mode=cfg.get("loss_mode", "fixed"),
                                     class_weights=class_weights)
                checkpoint = ckpts / f"{protocol}_{name}_seed{seed}.pt"
                started = time.perf_counter()
                history = trainer.train(loaders[0], loaders[1], n_epochs=200,
                                        save_path=str(checkpoint))
                train_seconds = time.perf_counter() - started
                pred = evaluate_on_test(model, loaders[2], trainer.device)["predictions_df"]
                row = {
                    "protocol": protocol, "model": name, "seed": seed,
                    "model_revision": (SPARC_REVISION
                                       if name == "SPARC-GNN"
                                       else "baseline_v1"),
                    "split_revision": split_revision_for(protocol),
                    "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
                    "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                    "loss_revision": FIXED_LOSS_REVISION,
                    "class_weight_revision": CLASS_WEIGHT_REVISION,
                    "class_weight_home": float(class_weights[0]),
                    "class_weight_draw": float(class_weights[1]),
                    "class_weight_away": float(class_weights[2]),
                    "n_train": len(graphs[0]), "n_validation": len(graphs[1]),
                    "n_test": len(graphs[2]),
                    "best_val_loss": history["best_val_loss"],
                    "best_val_task_loss": history["best_val_task_loss"],
                    "best_val_auxiliary_loss": history[
                        "best_val_auxiliary_loss"],
                    "train_seconds": train_seconds,
                    "accuracy": accuracy_score(pred.true_class, pred.pred_class),
                    "macro_f1": f1_score(pred.true_class, pred.pred_class,
                                         average="macro", zero_division=0),
                    "draw_f1": f1_score(pred.true_class, pred.pred_class,
                                        labels=[1], average="macro", zero_division=0),
                    "mae": mean_absolute_error(pred.true_goal_diff,
                                               pred.pred_goal_diff),
                    "r2": r2_score(pred.true_goal_diff, pred.pred_goal_diff),
                }
                rows.append(row)
                pd.DataFrame(rows).to_parquet(partial, index=False)
                print(row, flush=True)

    (out / "generalization_splits.json").write_text(
        json.dumps(split_records, indent=2), encoding="utf-8")
    if SPLIT_AUDIT_ONLY:
        print(f"split audit complete: {len(split_records)} protocol-seed records")
        return
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "generalization_results.parquet", index=False)
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame.groupby(["protocol", "model"])[numeric].agg(["mean", "std"]).to_csv(
        out / "generalization_results_summary.csv")


if __name__ == "__main__":
    main()
