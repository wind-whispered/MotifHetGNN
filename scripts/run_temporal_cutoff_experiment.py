"""True event-prefix prediction at 15/30/45/60/75 minutes.

Every graph is rebuilt from events available by the cutoff. The current-score
baseline predicts the final outcome and goal difference from the score observed
at that cutoff. Neural results are resumable by cutoff/model/seed.
"""
from collections import OrderedDict
from pathlib import Path
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset,
                             split_dataset_by_metadata)
from src.gnn.evaluator import evaluate_on_test
from src.gnn.model import goal_diff_to_class
from src.gnn.mcar_model import SPARC_REVISION
from src.gnn.score_context import SCORE_CONTEXT_REVISION
from src.gnn.trainer import (
    CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION, GNNTrainer,
    training_class_weights,
)
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.temporal import SCORE_TIMELINE_REVISION
from scripts.run_information_sciences_models import MODELS as BASELINE_MODELS
from scripts.build_temporal_cutoff_graphs import TEMPORAL_GRAPH_REVISION

CUTOFFS = [int(x) for x in os.environ.get("IS_CUTOFFS", "15,30,45,60,75").split(",")]
SEEDS = [int(x) for x in os.environ.get("IS_SEEDS", "42,7,13,21,37").split(",")]
FORCE_REBUILD = os.environ.get("IS_FORCE_TEMPO_REBUILD", "0") == "1"
MODELS = OrderedDict()
for name, spec in BASELINE_MODELS.items():
    MODELS[name] = spec
MODELS["SPARC-GNN+Score"] = dict(
    factory=BASELINE_MODELS["SPARC-GNN"]["factory"],
    kwargs=dict(
        use_edge_attr=True,
        use_motif_context=True,
        adaptive_gate=True,
        use_info_decomposition=True,
        use_current_score=True,
        use_assist_flags=False,
    ),
    loss_mode="fixed",
)
MODELS["HeteroSAGE-sum+Score"] = dict(
    factory=BASELINE_MODELS["HeteroSAGE-sum"]["factory"],
    kwargs=dict(use_current_score=True),
    loss_mode="fixed",
)
MODELS["TeamPool-MLP+Score"] = dict(
    factory=BASELINE_MODELS["TeamPool-MLP"]["factory"],
    kwargs=dict(use_current_score=True),
    loss_mode="fixed",
)
MODEL_ALIASES = {
    "MCAR-Decomp": "SPARC-GNN",
    "MCAR-Decomp+Score": "SPARC-GNN+Score",
}
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

_requested_models = [item.strip() for item in os.environ.get("IS_MODELS", "").split(",") if item.strip()]
DEFAULT_TEMPORAL_MODELS = [
    "TeamPool-MLP", "HeteroSAGE-sum", "HGT", "EdgeAware-Fixed",
    "SPARC-GNN", "TeamPool-MLP+Score", "HeteroSAGE-sum+Score",
    "SPARC-GNN+Score",
]
SELECTED_MODELS = (_normalise_requested_models(_requested_models)
                   if _requested_models else DEFAULT_TEMPORAL_MODELS)
SCORE_LOSS_REVISION = "not_applicable_score_v1"


def metric_row(true_cls, pred_cls, true_gd, pred_gd):
    return {
        "accuracy": accuracy_score(true_cls, pred_cls),
        "macro_f1": f1_score(true_cls, pred_cls, average="macro", zero_division=0),
        "draw_f1": f1_score(true_cls, pred_cls, labels=[1], average="macro",
                            zero_division=0),
        "mae": mean_absolute_error(true_gd, pred_gd),
        "r2": r2_score(true_gd, pred_gd),
    }


def load_existing_rows(path):
    """Canonicalise an interrupted run before deciding what to resume."""
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    frame["model"] = frame["model"].replace(MODEL_ALIASES)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if "graph_revision" not in frame:
        frame["graph_revision"] = pd.NA
    frame = frame[
        (frame["graph_revision"] == TEMPORAL_GRAPH_REVISION).fillna(False)
    ].copy()
    if "base_graph_revision" not in frame:
        frame["base_graph_revision"] = pd.NA
    frame = frame[(frame["base_graph_revision"] ==
                   HETEROGENEOUS_GRAPH_REVISION).fillna(False)].copy()
    if "score_timeline_revision" not in frame:
        frame["score_timeline_revision"] = pd.NA
    frame = frame[(frame["score_timeline_revision"] ==
                   SCORE_TIMELINE_REVISION).fillna(False)].copy()
    if "score_context_revision" not in frame:
        frame["score_context_revision"] = pd.NA
    frame = frame[(frame["score_context_revision"] ==
                   SCORE_CONTEXT_REVISION).fillna(False)].copy()
    if "split_revision" not in frame:
        frame["split_revision"] = pd.NA
    frame = frame[(frame["split_revision"] ==
                   CHRONOLOGICAL_SPLIT_REVISION).fillna(False)].copy()
    if "model_revision" not in frame:
        frame["model_revision"] = pd.NA
    sparc = frame["model"].isin({"SPARC-GNN", "SPARC-GNN+Score"})
    frame = frame[(~sparc | (
        frame["model_revision"] == SPARC_REVISION)).fillna(False)].copy()
    baseline = ~frame["model"].isin({"SPARC-GNN", "SPARC-GNN+Score"})
    frame.loc[baseline, "model_revision"] = frame.loc[
        baseline, "model_revision"].fillna("baseline_or_score_v1")
    if "loss_revision" not in frame:
        frame["loss_revision"] = pd.NA
    expected_loss = frame["model"].map(
        lambda name: (SCORE_LOSS_REVISION if name in {
            "CurrentScore", "CurrentScore-Calibrated"}
                      else FIXED_LOSS_REVISION))
    frame = frame[(frame["loss_revision"] == expected_loss).fillna(False)].copy()
    if "class_weight_revision" not in frame:
        frame["class_weight_revision"] = pd.NA
    expected_class_weight = frame["model"].map(
        lambda name: (SCORE_LOSS_REVISION if name in {
            "CurrentScore", "CurrentScore-Calibrated"}
                      else CLASS_WEIGHT_REVISION))
    frame = frame[(frame["class_weight_revision"] ==
                   expected_class_weight).fillna(False)].copy()
    frame = frame.drop_duplicates(
        subset=["cutoff", "model", "seed"], keep="last")
    return frame.to_dict("records")


def score_matrix(graphs):
    """Features available from the scoreboard at the cutoff only."""
    home = np.asarray([float(g.current_home_score) for g in graphs])
    away = np.asarray([float(g.current_away_score) for g in graphs])
    diff = home - away
    return np.column_stack([
        home, away, diff, home + away,
        diff > 0, diff == 0, diff < 0,
    ]).astype(float)


def calibrated_score_predictions(train, val, test):
    """Validation-selected logistic/Ridge current-score baseline."""
    x_train, x_val, x_test = map(score_matrix, (train, val, test))
    y_train = np.asarray([float(g.y.item()) for g in train])
    y_val = np.asarray([float(g.y.item()) for g in val])
    y_test = np.asarray([float(g.y.item()) for g in test])
    c_train = goal_diff_to_class(torch.tensor(y_train)).numpy()
    c_val = goal_diff_to_class(torch.tensor(y_val)).numpy()

    best_cls = None
    for c_value in (0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, class_weight=class_weight,
                                   max_iter=2000, random_state=0))
            estimator.fit(x_train, c_train)
            prediction = estimator.predict(x_val)
            score = (f1_score(c_val, prediction, average="macro",
                              zero_division=0),
                     accuracy_score(c_val, prediction))
            if best_cls is None or score > best_cls[0]:
                best_cls = (score, estimator, c_value, class_weight)

    best_reg = None
    for alpha in (0.1, 1.0, 10.0):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        estimator.fit(x_train, y_train)
        score = mean_absolute_error(y_val, estimator.predict(x_val))
        if best_reg is None or score < best_reg[0]:
            best_reg = (score, estimator, alpha)

    return (y_test, best_cls[1].predict(x_test),
            best_reg[1].predict(x_test), best_cls[2],
            best_cls[3] == "balanced", best_reg[2])


def main():
    out = ROOT / "data" / "information_sciences"
    temporal_root = out / "temporal_graphs"
    checkpoint_dir = out / "checkpoints_temporal"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partial = out / "temporal_results_partial.parquet"
    rows = []
    if partial.exists() and not FORCE_REBUILD:
        rows = load_existing_rows(partial)
        pd.DataFrame(rows).to_parquet(partial, index=False)
    elif partial.exists():
        # In full-rebuild mode we intentionally drop the checkpoint list and
        # regenerate all cutoff/model/seed rows from scratch.
        partial.unlink()
    completed = {(int(r["cutoff"]), r["model"], int(r["seed"])) for r in rows}
    base = FootballHeteroDataset(
        root=str(ROOT),
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    train_ids, val_ids, test_ids = split_dataset_by_metadata(
        base, str(ROOT / "data/processed/matches_meta.parquet"),
        strategy="chronological")

    for cutoff in CUTOFFS:
        graph_dir = temporal_root / str(cutoff)
        graph_by_id = {}
        for path in graph_dir.glob("*.pt"):
            graph = torch.load(path, weights_only=False)
            if (getattr(graph, "temporal_graph_revision", None)
                    != TEMPORAL_GRAPH_REVISION):
                raise RuntimeError(
                    f"temporal graph revision mismatch for {path}")
            if (getattr(graph, "graph_revision", None)
                    != HETEROGENEOUS_GRAPH_REVISION):
                raise RuntimeError(
                    f"base graph revision mismatch for {path}")
            if (getattr(graph, "score_timeline_revision", None)
                    != SCORE_TIMELINE_REVISION):
                raise RuntimeError(
                    f"score timeline revision mismatch for {path}")
            graph_by_id[int(path.stem)] = graph
        graphs = [[graph_by_id[mid] for mid in ids if mid in graph_by_id]
                  for ids in (train_ids, val_ids, test_ids)]
        if min(map(len, graphs)) == 0:
            raise RuntimeError(f"cutoff {cutoff} graphs missing; run builder first")
        class_weights = training_class_weights(
            [float(graph.y.item()) for graph in graphs[0]])

        baseline_key = (cutoff, "CurrentScore", -1)
        if baseline_key not in completed:
            test = graphs[2]
            true_gd = np.array([float(g.y.item()) for g in test])
            current_gd = np.array([float(g.current_goal_diff) for g in test])
            true_cls = goal_diff_to_class(torch.tensor(true_gd)).numpy()
            pred_cls = goal_diff_to_class(torch.tensor(current_gd)).numpy()
            rows.append({"cutoff": cutoff, "model": "CurrentScore", "seed": -1,
                         "model_revision": "score_v1",
                         "graph_revision": TEMPORAL_GRAPH_REVISION,
                         "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                         "score_timeline_revision": SCORE_TIMELINE_REVISION,
                         "score_context_revision": SCORE_CONTEXT_REVISION,
                         "loss_revision": SCORE_LOSS_REVISION,
                         "class_weight_revision": SCORE_LOSS_REVISION,
                         "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                         "n_train": len(graphs[0]),
                         "n_validation": len(graphs[1]),
                         "n_test": len(test),
                         **metric_row(true_cls, pred_cls, true_gd, current_gd)})
            pd.DataFrame(rows).to_parquet(partial, index=False)

        calibrated_key = (cutoff, "CurrentScore-Calibrated", -1)
        if calibrated_key not in completed:
            true_gd, pred_cls, pred_gd, cls_c, balanced, reg_alpha = (
                calibrated_score_predictions(*graphs))
            true_cls = goal_diff_to_class(torch.tensor(true_gd)).numpy()
            rows.append({
                "cutoff": cutoff, "model": "CurrentScore-Calibrated",
                "seed": -1, "n_test": len(graphs[2]),
                "model_revision": "score_v1",
                "graph_revision": TEMPORAL_GRAPH_REVISION,
                "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                "score_timeline_revision": SCORE_TIMELINE_REVISION,
                "score_context_revision": SCORE_CONTEXT_REVISION,
                "loss_revision": SCORE_LOSS_REVISION,
                "class_weight_revision": SCORE_LOSS_REVISION,
                "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                "n_train": len(graphs[0]),
                "n_validation": len(graphs[1]),
                "score_cls_C": cls_c, "score_cls_balanced": balanced,
                "score_reg_alpha": reg_alpha,
                **metric_row(true_cls, pred_cls, true_gd, pred_gd),
            })
            pd.DataFrame(rows).to_parquet(partial, index=False)

        for seed in SEEDS:
            for name in SELECTED_MODELS:
                cfg = MODELS[name]
                key = (cutoff, name, seed)
                if key in completed:
                    continue
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                generator = torch.Generator().manual_seed(seed)
                loaders = [DataLoader(g, batch_size=32 if i == 0 else 64,
                                      shuffle=(i == 0),
                                      generator=generator if i == 0 else None)
                           for i, g in enumerate(graphs)]
                factory = cfg["factory"]
                model_kwargs = cfg.get("kwargs", {})
                model = factory(
                    node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3,
                    **model_kwargs)
                trainer = GNNTrainer(model, lr=1e-3, patience=20,
                                     loss_mode=cfg.get("loss_mode", "fixed"),
                                     class_weights=class_weights)
                started = time.perf_counter()
                history = trainer.train(
                    loaders[0], loaders[1], n_epochs=200,
                    save_path=str(checkpoint_dir / f"t{cutoff}_{name}_seed{seed}.pt"))
                train_seconds = time.perf_counter() - started
                pred = evaluate_on_test(model, loaders[2], trainer.device)["predictions_df"]
                rows.append({
                    "cutoff": cutoff, "model": name, "seed": seed,
                    "model_revision": (SPARC_REVISION
                                       if name in {"SPARC-GNN", "SPARC-GNN+Score"}
                                       else "baseline_v1"),
                    "graph_revision": TEMPORAL_GRAPH_REVISION,
                    "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
                    "score_timeline_revision": SCORE_TIMELINE_REVISION,
                    "score_context_revision": SCORE_CONTEXT_REVISION,
                    "loss_revision": FIXED_LOSS_REVISION,
                    "class_weight_revision": CLASS_WEIGHT_REVISION,
                    "class_weight_home": float(class_weights[0]),
                    "class_weight_draw": float(class_weights[1]),
                    "class_weight_away": float(class_weights[2]),
                    "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
                    "n_train": len(graphs[0]),
                    "n_validation": len(graphs[1]),
                    "n_test": len(graphs[2]), "best_val_loss": history["best_val_loss"],
                    "best_val_task_loss": history["best_val_task_loss"],
                    "best_val_auxiliary_loss": history[
                        "best_val_auxiliary_loss"],
                    "train_seconds": train_seconds,
                    **metric_row(pred.true_class, pred.pred_class,
                                 pred.true_goal_diff, pred.pred_goal_diff),
                })
                pd.DataFrame(rows).to_parquet(partial, index=False)
                print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "temporal_results.parquet", index=False)
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame.groupby(["cutoff", "model"])[numeric].agg(["mean", "std"]).to_csv(
        out / "temporal_results_summary.csv")


if __name__ == "__main__":
    main()
