"""Bounded, validation-only development; never load or evaluate test graphs.

This is a development experiment, not a replacement for the frozen main table.
The search grid, seeds and primary selection score are written before fitting.
All three models receive exactly the same four settings and training budget.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnn.dataset import FootballHeteroDataset, CHRONOLOGICAL_SPLIT_REVISION
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import MCARFootballGNN, RGCNFootballGNN, SPARC_REVISION
from src.gnn.model import HeteroFootballGNN, goal_diff_to_class
from src.gnn.trainer import training_class_weights
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION

OUT = ROOT / "data/information_sciences/round2_validation_search"
SEEDS = [7, 13]
MODELS = ["SPARC-GNN", "HeteroSAGE-sum", "R-GCN"]
# Change one factor at a time around the scale-balanced setting. BatchNorm is
# an experimental architecture variant, not a silent change to published SPARC.
GRID = [
    dict(name="reference", layers=3, hidden=64, dropout=.3, lr=.001,
         norm="native", reg_scale="raw", weight_decay=0.),
    dict(name="balanced", layers=3, hidden=64, dropout=.3, lr=.001,
         norm="native", reg_scale="train_variance", weight_decay=0.),
    dict(name="balanced_lowdrop", layers=3, hidden=64, dropout=.1, lr=.001,
         norm="native", reg_scale="train_variance", weight_decay=0.),
    dict(name="balanced_batchnorm", layers=3, hidden=64, dropout=.1, lr=.001,
         norm="batch", reg_scale="train_variance", weight_decay=0.),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(name, cfg):
    kwargs = dict(node_feature_dim=39, hidden_dim=cfg["hidden"],
                  num_layers=cfg["layers"], dropout=cfg["dropout"])
    if name == "SPARC-GNN":
        model = MCARFootballGNN(**kwargs, use_info_decomposition=True,
                                use_assist_flags=False)
    elif name == "HeteroSAGE-sum":
        model = HeteroFootballGNN(**kwargs)
    elif name == "R-GCN":
        model = RGCNFootballGNN(**kwargs)
    else:
        raise ValueError(name)
    if cfg["norm"] == "batch" and hasattr(model, "norms"):
        for i, norm in enumerate(model.norms):
            if isinstance(norm, nn.ModuleDict):
                for key in list(norm):
                    norm[key] = nn.BatchNorm1d(cfg["hidden"])
            else:
                model.norms[i] = nn.BatchNorm1d(cfg["hidden"])
    return model


@torch.no_grad()
def validation(model, loader, device, weights, train_variance):
    model.eval()
    ys, logits_all, reg_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits, reg = model(batch)
        ys.append(batch.y.reshape(-1))
        logits_all.append(logits)
        reg_all.append(reg.reshape(-1))
    y = torch.cat(ys)
    logits = torch.cat(logits_all)
    pred = torch.cat(reg_all)
    cls = goal_diff_to_class(y)
    ce = nn.functional.cross_entropy(logits, cls, weight=weights).item()
    mse = nn.functional.mse_loss(pred, y).item()
    y_np, pred_np = y.cpu().numpy(), pred.cpu().numpy()
    cls_np, guessed = cls.cpu().numpy(), logits.argmax(-1).cpu().numpy()
    return dict(selection_score=.5 * ce + .5 * mse / train_variance,
                ce=ce, mse=mse, accuracy=accuracy_score(cls_np, guessed),
                macro_f1=f1_score(cls_np, guessed, average="macro", zero_division=0),
                draw_f1=f1_score(cls_np, guessed, labels=[1], average="macro", zero_division=0),
                mae=mean_absolute_error(y_np, pred_np), r2=r2_score(y_np, pred_np)), (
                    y_np, cls_np, logits.softmax(-1).cpu().numpy(), pred_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    split_file = ROOT / "data/information_sciences/chronological_split.json"
    split = json.loads(split_file.read_text(encoding="utf-8"))
    assert split["split_revision"] == CHRONOLOGICAL_SPLIT_REVISION
    train_ids, val_ids = split["train"], split["validation"]
    assert not (set(train_ids) & set(val_ids))
    assert not ((set(train_ids) | set(val_ids)) & set(split["test"]))
    protected = [ROOT / "data/information_sciences" / n for n in
                 ["model_results.parquet", "predictions.parquet", "ablation_results.parquet"]]
    protocol = dict(revision="round2_validation_only_v1", grid=GRID,
                    seeds=SEEDS, models=MODELS, epochs=args.epochs,
                    patience=args.patience, batch_size=args.batch_size,
                    split_revision=CHRONOLOGICAL_SPLIT_REVISION,
                    feature_revision=PRIMARY_FULL_MATCH_FEATURE_REVISION,
                    graph_revision=HETEROGENEOUS_GRAPH_REVISION,
                    sparc_revision=SPARC_REVISION, split_sha256=digest(split_file),
                    source_sha256=digest(Path(__file__)),
                    selection="mean across seeds of validation (0.5 weighted CE + 0.5 MSE / training target variance)",
                    checkpoint_rule="same normalized validation score for every candidate; auxiliary penalty excluded",
                    test_evaluated=False, test_graphs_loaded=False,
                    limitations="Two development seeds; already-inspected test set is not a fresh holdout. No SOTA claim.",
                    protected_hashes={p.name: digest(p) for p in protected})
    path = OUT / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError("Protocol changed: use a new output directory; do not mix runs")
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    graphs = {}
    dataset = FootballHeteroDataset(str(ROOT), match_ids=train_ids+val_ids,
            use_motif_context=True, required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    for i, mid in enumerate(train_ids+val_ids):
        graph = dataset[i]
        # Physically redact proxy columns for every model, not only SPARC.
        for et in graph.edge_types:
            if et[1] == "pass":
                graph[et].edge_attr[:, 10:12] = 0.
                assert graph[et].edge_attr[:, 10:12].count_nonzero() == 0
        graphs[mid] = graph
        if (i+1) % 500 == 0:
            print(f"loaded train/validation {i+1}/{len(dataset)}", flush=True)
    train_graphs = [graphs[mid] for mid in train_ids]
    val_graphs = [graphs[mid] for mid in val_ids]
    y_train = torch.cat([g.y.reshape(-1) for g in train_graphs]).float()
    variance = y_train.var(unbiased=False).item()
    weights = training_class_weights(y_train).to(device)
    print(f"Device={device}; target variance={variance:.4f}; NO TEST GRAPHS", flush=True)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False)
    rows = []
    # Model interleaving gives an early matched comparison rather than all
    # configurations of SPARC first and a late, unfinished baseline sweep.
    for cfg in GRID:
        for seed in SEEDS:
            for name in args.models:
                stem = f"{name}_{cfg['name']}_seed{seed}"
                record = OUT / f"{stem}.json"
                if record.exists():
                    rows.append(json.loads(record.read_text(encoding="utf-8")))
                    continue
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model = make_model(name, cfg).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                             weight_decay=cfg["weight_decay"])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=.5, patience=7)
                loader = DataLoader(train_graphs, batch_size=args.batch_size,
                                    shuffle=True, generator=torch.Generator().manual_seed(seed))
                best, patience, history = float("inf"), 0, []
                started = time.perf_counter()
                ckpt = OUT / f"{stem}.pt"
                print(f"START {stem}", flush=True)
                for epoch in range(1, args.epochs+1):
                    model.train()
                    loss_sum, n_train = 0., 0
                    for batch in loader:
                        batch = batch.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits, reg = model(batch)
                        y = batch.y.reshape(-1).float()
                        ce = nn.functional.cross_entropy(logits, goal_diff_to_class(y), weight=weights)
                        mse = nn.functional.mse_loss(reg.reshape(-1), y)
                        scale = variance if cfg["reg_scale"] == "train_variance" else 1.
                        loss = .5 * ce + .5 * mse / scale
                        if hasattr(model, "auxiliary_loss"):
                            loss = loss + .01 * model.auxiliary_loss()
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite loss in {stem}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        loss_sum += loss.item()*len(y); n_train += len(y)
                    metrics, _ = validation(model, val_loader, device, weights, variance)
                    history.append(dict(epoch=epoch, train_loss=loss_sum/n_train, **metrics))
                    scheduler.step(metrics["selection_score"])
                    if metrics["selection_score"] < best:
                        best, best_epoch, patience = metrics["selection_score"], epoch, 0
                        torch.save(model.state_dict(), ckpt)
                    else:
                        patience += 1
                    if epoch % 10 == 0 or patience >= args.patience:
                        print(f"{stem} epoch={epoch} score={metrics['selection_score']:.4f} valF1={metrics['macro_f1']:.4f} best={best:.4f}", flush=True)
                    if patience >= args.patience:
                        break
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                metrics, prediction = validation(model, val_loader, device, weights, variance)
                y, cls, prob, pred = prediction
                pd.DataFrame(dict(match_id=val_ids, true_goal_diff=y, true_class=cls,
                    prob_home=prob[:,0], prob_draw=prob[:,1], prob_away=prob[:,2],
                    pred_goal_diff=pred)).to_parquet(OUT/f"{stem}_validation.parquet", index=False)
                pd.DataFrame(history).to_csv(OUT/f"{stem}_history.csv", index=False)
                row = dict(model=name, config=cfg["name"], seed=seed,
                           best_epoch=best_epoch, epochs_run=epoch,
                           seconds=time.perf_counter()-started,
                           checkpoint_sha256=digest(ckpt), split="validation",
                           test_evaluated=False, **metrics)
                record.write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/"validation_results.csv", index=False)
                print(f"DONE {stem} {json.dumps(row)}", flush=True)
                del model, optimizer
                if device == "cuda":
                    torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "config"])[["selection_score", "macro_f1", "r2", "mae"]].agg(["mean", "std"])
    summary.to_csv(OUT/"validation_summary.csv")
    winners = frame.groupby(["model", "config"]).selection_score.mean().reset_index()
    winners = winners.sort_values(["model", "selection_score", "config"]).groupby("model").head(1)
    (OUT/"frozen_validation_selection.json").write_text(json.dumps(dict(
        selection_rule=protocol["selection"], selections=winners.to_dict("records"),
        test_evaluated=False, confirmation_required=True), indent=2), encoding="utf-8")
    assert all(digest(p) == protocol["protected_hashes"][p.name] for p in protected)
    print(summary.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
