"""Equal-budget, two-stage validation search; no test graphs are loaded.

Eight shared settings (two mechanistic controls plus six Sobol proposals) are
screened at seed 7. Each model's top two are repeated at seeds 13 and 21.
The mean of all three seeds selects a configuration, not a lucky seed.
This development protocol does not replace Table 3 or constitute SOTA evidence.
"""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch
from torch import nn
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_round2_validation_search import digest, validation
from src.gnn.dataset import FootballHeteroDataset, CHRONOLOGICAL_SPLIT_REVISION
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import MCARFootballGNN, RGCNFootballGNN, HGTFootballGNN, SPARC_REVISION
from src.gnn.model import HeteroFootballGNN, goal_diff_to_class
from src.gnn.team_pool import TeamPoolMLP
from src.gnn.residual_candidate import RootPreservingSPARC
from src.gnn.trainer import training_class_weights
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION

OUT = ROOT / "data/information_sciences/round3_hpo"
MODELS = ["SPARC-GNN", "HeteroSAGE-sum", "R-GCN", "HGT",
          "EdgeAware-Fixed", "MotifGate-ObservedOnly", "TeamPool-MLP"]
SEEDS = [7, 13, 21]


def search_space():
    control = dict(hidden=64, layers=3, dropout=.1, lr=.001,
                   weight_decay=0., reg_scale="train_variance")
    grid = [dict(name="post_control", update="post", **control),
            dict(name="root_control", update="root", **control)]
    points = qmc.Sobol(d=6, scramble=True, seed=20260906).random_base2(3)
    for i, point in enumerate(points[:6]):
        grid.append(dict(name=f"sobol_{i:02d}", update="root",
            hidden=[32, 48, 64][min(int(point[0]*3), 2)],
            layers=[1, 2, 3][min(int(point[1]*3), 2)],
            dropout=round(float(.05+.3*point[2]), 6),
            lr=float(10**(-3.7+1.2*point[3])),
            weight_decay=float(10**(-6+4*point[4])),
            reg_scale="raw" if point[5] < .5 else "train_variance"))
    return grid


def make_model(name, cfg):
    kw = dict(node_feature_dim=39, hidden_dim=cfg["hidden"],
              num_layers=cfg["layers"], dropout=cfg["dropout"])
    if name in {"SPARC-GNN", "EdgeAware-Fixed", "MotifGate-ObservedOnly"}:
        cls = RootPreservingSPARC if cfg["update"] == "root" else MCARFootballGNN
        model = cls(**kw, use_info_decomposition=name == "SPARC-GNN",
            adaptive_gate=name != "EdgeAware-Fixed",
            use_motif_context=name != "EdgeAware-Fixed",
            mask_unavailable_relations=name != "EdgeAware-Fixed",
            use_assist_flags=False)
        if cfg["update"] == "post":
            for norms in model.norms:
                for nt in model.NODE_TYPES:
                    norms[nt] = nn.BatchNorm1d(cfg["hidden"])
        return model
    constructors = {"HeteroSAGE-sum": HeteroFootballGNN, "R-GCN": RGCNFootballGNN,
                    "HGT": HGTFootballGNN, "TeamPool-MLP": TeamPoolMLP}
    return constructors[name](**kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # PyG GPU scatter reductions can remain nondeterministic; record rather
    # than claiming bitwise reproducibility that the backend cannot guarantee.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "data/information_sciences/chronological_split.json"
    split = json.loads(split_path.read_text(encoding="utf8"))
    assert split["split_revision"] == CHRONOLOGICAL_SPLIT_REVISION
    train_ids, val_ids = split["train"], split["validation"]
    assert not set(train_ids) & set(val_ids)
    assert not (set(train_ids) | set(val_ids)) & set(split["test"])
    protected = [ROOT / "data/information_sciences" / f for f in
                 ["model_results.parquet", "predictions.parquet", "ablation_results.parquet"]]
    source_paths = [Path(__file__), ROOT / "scripts/run_round2_validation_search.py",
                   *[ROOT / "src/gnn" / f for f in ["residual_candidate.py", "mcar_model.py",
                      "model.py", "team_pool.py", "dataset.py", "feature_contracts.py"]]]
    grid = search_space()
    protocol = dict(revision="round3_sobol_root_validation_v1", grid=grid,
        models=MODELS, seeds=SEEDS, screen_seed=7, finalists_per_model=2,
        epochs=args.epochs, patience=args.patience, batch_size=64,
        optimizer="AdamW", scheduler="ReduceLROnPlateau factor .5 patience 4",
        selection="mean normalized validation loss across seeds 7,13,21; top two screened at seed 7",
        validation_loss="0.5 weighted CE + 0.5 MSE / training target variance; no auxiliary penalty",
        test_evaluated=False, test_graphs_loaded=False,
        equal_budget="8 trials at seed 7, top 2 at two further seeds per model",
        baseline_controls="post/root controls are repeated native configurations for non-SPARC-family models",
        limitations="Development on previously inspected validation set; test set already known historically. No fresh-holdout or SOTA claim.",
        determinism="CUBLAS and cuDNN deterministic; torch warn_only; GPU scatter may remain nondeterministic",
        split_sha256=digest(split_path), graph_revision=HETEROGENEOUS_GRAPH_REVISION,
        feature_revision=PRIMARY_FULL_MATCH_FEATURE_REVISION, sparc_revision=SPARC_REVISION,
        sources={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        protected_hashes={p.name: digest(p) for p in protected})
    pp = OUT / "protocol.json"
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, "Use new output directory for a changed protocol"
    else:
        pp.write_text(json.dumps(protocol, indent=2), encoding="utf8")
    dataset = FootballHeteroDataset(str(ROOT), match_ids=train_ids+val_ids,
        use_motif_context=True, required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    graphs = []
    for index in range(len(dataset)):
        graph = dataset[index]
        for et in graph.edge_types:
            if et[1] == "pass":
                graph[et].edge_attr[:, 10:12] = 0.
                assert graph[et].edge_attr[:, 10:12].count_nonzero() == 0
        graphs.append(graph)
        if (index+1) % 500 == 0:
            print(f"Loaded {index+1}/{len(dataset)} development graphs", flush=True)
    train_graphs, val_graphs = graphs[:len(train_ids)], graphs[len(train_ids):]
    targets = torch.cat([g.y.reshape(-1) for g in train_graphs]).float()
    variance = targets.var(unbiased=False).item()
    weights = training_class_weights(targets).to(device)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False)
    rows = []

    def fit(name, cfg, seed, phase):
        stem = f"{name}_{cfg['name']}_seed{seed}"
        record = OUT / f"{stem}.json"
        if record.exists():
            row = json.loads(record.read_text())
            assert row["checkpoint_sha256"] == digest(OUT/f"{stem}.pt")
            rows.append(row)
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = make_model(name, cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4)
        loader = DataLoader(train_graphs, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        best, stale, history = float("inf"), 0, []
        started = time.perf_counter()
        ckpt = OUT / f"{stem}.pt"
        print(f"START {phase} {stem}", flush=True)
        for epoch in range(1, args.epochs+1):
            model.train()
            loss_sum, count = 0., 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, reg = model(batch)
                y = batch.y.reshape(-1).float()
                ce = nn.functional.cross_entropy(logits, goal_diff_to_class(y), weight=weights)
                mse = nn.functional.mse_loss(reg.reshape(-1), y)
                loss = .5*ce + .5*mse/(variance if cfg["reg_scale"] == "train_variance" else 1.)
                if hasattr(model, "auxiliary_loss"):
                    loss = loss + .01*model.auxiliary_loss()
                assert torch.isfinite(loss), stem
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_sum += loss.item()*len(y)
                count += len(y)
            metrics, _ = validation(model, val_loader, device, weights, variance)
            history.append(dict(epoch=epoch, train_loss=loss_sum/count,
                                lr=optimizer.param_groups[0]["lr"], **metrics))
            scheduler.step(metrics["selection_score"])
            if metrics["selection_score"] < best:
                best, best_epoch, stale = metrics["selection_score"], epoch, 0
                torch.save(model.state_dict(), ckpt)
            else:
                stale += 1
            if epoch % 10 == 0:
                print(f"{stem} e{epoch} best={best:.4f} F1={metrics['macro_f1']:.4f}", flush=True)
            if stale >= args.patience:
                break
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        metrics, (y, cls, prob, pred) = validation(model, val_loader, device, weights, variance)
        pd.DataFrame(dict(match_id=val_ids, true_goal_diff=y, true_class=cls,
            prob_home=prob[:, 0], prob_draw=prob[:, 1], prob_away=prob[:, 2],
            pred_goal_diff=pred)).to_parquet(OUT/f"{stem}_validation.parquet", index=False)
        pd.DataFrame(history).to_csv(OUT/f"{stem}_history.csv", index=False)
        row = dict(model=name, config=cfg["name"], seed=seed, phase=phase,
            best_epoch=best_epoch, epochs_run=epoch, parameters=sum(p.numel() for p in model.parameters()),
            seconds=time.perf_counter()-started, checkpoint_sha256=digest(ckpt),
            test_evaluated=False, split="validation", **metrics)
        record.write_text(json.dumps(row, indent=2), encoding="utf8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/"validation_results.csv", index=False)
        print(f"DONE {stem} {json.dumps(row)}", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

    for cfg in grid:
        for name in MODELS:
            fit(name, cfg, 7, "screen")
    screen = pd.DataFrame(rows)
    selected = screen.sort_values(["model", "selection_score", "config"]).groupby("model").head(2)
    frozen = dict(rule=protocol["selection"], test_evaluated=False,
                  finalists=selected[["model", "config"]].to_dict("records"))
    fp = OUT / "frozen_finalists.json"
    if fp.exists():
        assert json.loads(fp.read_text()) == frozen
    else:
        fp.write_text(json.dumps(frozen, indent=2), encoding="utf8")
    for seed in [13, 21]:
        for entry in frozen["finalists"]:
            cfg = next(c for c in grid if c["name"] == entry["config"])
            fit(entry["model"], cfg, seed, "replicate")
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "config"]).agg(n=("seed", "count"),
        score=("selection_score", "mean"), f1=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        r2=("r2", "mean"), r2_std=("r2", "std"), mae=("mae", "mean")).reset_index()
    summary.to_csv(OUT/"validation_summary.csv", index=False)
    winners = summary[summary.n == 3].sort_values(["model", "score", "config"]).groupby("model").head(1)
    (OUT/"frozen_validation_selection.json").write_text(json.dumps(dict(
        selections=winners.to_dict("records"), rule=protocol["selection"],
        test_evaluated=False, confirmation_required=True), indent=2), encoding="utf8")
    assert len(frame) == 84
    assert all(digest(p) == protocol["protected_hashes"][p.name] for p in protected)
    print(winners.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
