"""
Run the full ablation study (Table tab:ablation) with real experiments.

Models (identical 70/10/20 split, seed 42):
  * Majority / mean          -- trivial baseline
  * Linear, topology         -- LogisticRegression / LinearRegression on 16 topo feats
  * Linear, motifs k=3       -- on 26 triadic motif counts (home+away)
  * XGBoost, motifs k=3      -- GradientBoosting on the 26 triadic counts
  * AdvGNN                   -- HetGNN trained on adversarial+turnover edges only
  * HomGNN                   -- HetGNN trained on pass edges only
  * HetGNN (full)            -- all three edge types (stored model re-evaluated)

Full classification + regression metrics are computed on the held-out test set.
Output: data/analysis/ablation_full.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnn.dataset import FootballHeteroDataset, split_dataset
from src.gnn.model import HeteroFootballGNN, goal_diff_to_class
from src.gnn.trainer import GNNTrainer
from torch_geometric.loader import DataLoader

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             r2_score, mean_absolute_error, mean_squared_error)
from scipy.stats import pearsonr

DATA = ROOT / "data"
ADV_TURN = [("home_player", "adversarial", "away_player"),
            ("away_player", "adversarial", "home_player"),
            ("home_player", "turnover", "away_player"),
            ("away_player", "turnover", "home_player")]
PASS = [("home_player", "pass", "home_player"),
        ("away_player", "pass", "away_player")]


def full_metrics(y_cls, p_cls, y_reg, p_reg):
    return {
        "Acc": accuracy_score(y_cls, p_cls),
        "macro_F1": f1_score(y_cls, p_cls, average="macro"),
        "wtd_F1": f1_score(y_cls, p_cls, average="weighted"),
        "F1_H": f1_score(y_cls, p_cls, labels=[0], average="macro"),
        "F1_D": f1_score(y_cls, p_cls, labels=[1], average="macro"),
        "F1_A": f1_score(y_cls, p_cls, labels=[2], average="macro"),
        "kappa": cohen_kappa_score(y_cls, p_cls),
        "MAE": mean_absolute_error(y_reg, p_reg),
        "RMSE": float(np.sqrt(mean_squared_error(y_reg, p_reg))),
        "R2": r2_score(y_reg, p_reg),
        "r": float(pearsonr(y_reg, p_reg)[0]) if np.std(p_reg) > 0 else 0.0,
    }


def mask_graph(data, keep):
    """Return a shallow-modified copy keeping only `keep` edge types (others emptied)."""
    g = data.clone()
    for et in g.edge_types:
        if et not in keep:
            g[et].edge_index = torch.empty((2, 0), dtype=torch.long)
    return g


@torch.no_grad()
def gnn_predict(model, loader, device):
    model.eval()
    yc, pc, yr, pr = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        cls_logits, reg_out = model(batch)
        gd = batch.y.to(device).squeeze()
        yc.extend(goal_diff_to_class(gd).cpu().numpy().tolist())
        pc.extend(cls_logits.argmax(-1).cpu().numpy().tolist())
        yr.extend(gd.cpu().numpy().tolist())
        pr.extend(reg_out.squeeze().cpu().numpy().tolist())
    return np.array(yc), np.array(pc), np.array(yr), np.array(pr)


def preload(ids, graph_dir):
    out = []
    for mid in ids:
        p = graph_dir / f"{mid}.pt"
        if p.exists():
            out.append(torch.load(p, weights_only=False))
    return out


def main():
    device = "cpu"
    graph_dir = DATA / "networks" / "heterogeneous"
    ds = FootballHeteroDataset(root=str(ROOT))
    train_ids, val_ids, test_ids = split_dataset(ds, 0.7, 0.1, 0.2, 42)
    print(f"split: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    print("preloading graphs into memory ...")
    t0 = time.time()
    g_train = preload(train_ids, graph_dir)
    g_val = preload(val_ids, graph_dir)
    g_test = preload(test_ids, graph_dir)
    print(f"  loaded {len(g_train)+len(g_val)+len(g_test)} graphs in {time.time()-t0:.0f}s")

    results = {}

    def train_eval_gnn(name, keep_edges, epochs=150):
        print(f"\n=== {name} (keep {len(keep_edges)} edge types) ===")
        if keep_edges == "all":
            gtr, gva, gte = g_train, g_val, g_test
        else:
            gtr = [mask_graph(d, keep_edges) for d in g_train]
            gva = [mask_graph(d, keep_edges) for d in g_val]
            gte = [mask_graph(d, keep_edges) for d in g_test]
        trl = DataLoader(gtr, batch_size=32, shuffle=True)
        val = DataLoader(gva, batch_size=64, shuffle=False)
        tel = DataLoader(gte, batch_size=64, shuffle=False)
        torch.manual_seed(42); np.random.seed(42)
        model = HeteroFootballGNN(node_feature_dim=39, hidden_dim=64,
                                  num_layers=3, dropout=0.3)
        tr = GNNTrainer(model, lr=1e-3, patience=20, device=device)
        ckpt = str(DATA / "gnn" / f"hetgnn_{name}.pt")
        t1 = time.time()
        tr.train(trl, val, n_epochs=epochs, save_path=ckpt)
        yc, pc, yr, pr = gnn_predict(model, tel, device)
        m = full_metrics(yc, pc, yr, pr)
        print(f"  {name}: acc={m['Acc']:.3f} macroF1={m['macro_F1']:.3f} "
              f"kappa={m['kappa']:.3f} mae={m['MAE']:.3f}  ({time.time()-t1:.0f}s)")
        results[name] = m

    # ---- GNN variants ----
    # Full HetGNN: re-evaluate the stored trained model on the same test split
    print("\n=== HetGNN (full, stored model) ===")
    model = HeteroFootballGNN(node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3)
    state = torch.load(DATA / "gnn" / "gnn_model.pt", weights_only=True)
    model.load_state_dict(state)
    tel = DataLoader(g_test, batch_size=64, shuffle=False)
    yc, pc, yr, pr = gnn_predict(model, tel, device)
    results["HetGNN"] = full_metrics(yc, pc, yr, pr)
    print(f"  HetGNN: acc={results['HetGNN']['Acc']:.3f} "
          f"macroF1={results['HetGNN']['macro_F1']:.3f} kappa={results['HetGNN']['kappa']:.3f}")

    train_eval_gnn("HomGNN", PASS)
    train_eval_gnn("AdvGNN", ADV_TURN)

    # ---- tabular baselines on the same split ----
    print("\n=== tabular baselines ===")
    mm = pd.read_parquet(DATA / "processed" / "matches_meta.parquet").set_index("match_id")
    ns = pd.read_parquet(DATA / "processed" / "network_stats.parquet")
    hom = pd.read_parquet(DATA / "motifs" / "homogeneous_motifs.parquet")
    hom = hom[hom.motif_order_k == 3]
    RM = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]

    def motif_feats():
        out = {}
        for side in ("home", "away"):
            p = (hom[hom.team_side == side].pivot_table(index="match_id", columns="motif_id",
                 values="count", aggfunc="sum", fill_value=0).reindex(columns=RM, fill_value=0))
            p.columns = [f"{side}_{i}" for i in RM]
            out[side] = p
        return out["home"].join(out["away"])

    def topo_feats():
        n2 = ns[ns.w0 == 2]
        cols = ["density", "transitivity", "pass_diversity", "mean_outdegree",
                "mean_betweenness", "mean_eigenvector", "n_nodes", "n_edges"]
        out = {}
        for side in ("home", "away"):
            t = n2[n2.team_side == side].set_index("match_id")[cols]
            t.columns = [f"{side}_{c}" for c in cols]
            out[side] = t
        return out["home"].join(out["away"])

    mfeat = motif_feats()
    tfeat = topo_feats()
    common = [i for i in mm.index if i in mfeat.index and i in tfeat.index]
    y = mm.loc[common, "goal_diff"].astype(float)
    ycls = np.sign(y.values).astype(int)
    ycls = np.where(ycls == 1, 0, np.where(ycls == 0, 1, 2))  # 0=home,1=draw,2=away
    tr_mask = np.array([c in set(train_ids) for c in common])
    te_mask = np.array([c in set(test_ids) for c in common])

    def run_tab(X, clf, reg):
        Xs = StandardScaler().fit(X[tr_mask]).transform(X)
        clf.fit(Xs[tr_mask], ycls[tr_mask])
        reg.fit(Xs[tr_mask], y.values[tr_mask])
        pc = clf.predict(Xs[te_mask]); pr = reg.predict(Xs[te_mask])
        return full_metrics(ycls[te_mask], pc, y.values[te_mask], pr)

    results["Linear_topology"] = run_tab(tfeat.loc[common].values,
        LogisticRegression(max_iter=2000), LinearRegression())
    results["Linear_motifs"] = run_tab(mfeat.loc[common].values,
        LogisticRegression(max_iter=2000), LinearRegression())
    results["XGBoost_motifs"] = run_tab(mfeat.loc[common].values,
        GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42))

    # Majority / mean
    maj = np.bincount(ycls[tr_mask]).argmax()
    meanv = y.values[tr_mask].mean()
    pc = np.full(te_mask.sum(), maj); pr = np.full(te_mask.sum(), meanv)
    results["Majority"] = full_metrics(ycls[te_mask], pc, y.values[te_mask], pr)

    out_path = DATA / "analysis" / "ablation_full.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print("\nwrote", out_path)
    order = ["Majority", "Linear_topology", "Linear_motifs", "XGBoost_motifs",
             "AdvGNN", "HomGNN", "HetGNN"]
    print(f"\n{'model':16} {'Acc':>6} {'mF1':>6} {'wF1':>6} {'F1H':>6} {'F1D':>6} "
          f"{'F1A':>6} {'kap':>6} {'MAE':>6} {'RMSE':>6} {'R2':>6} {'r':>6}")
    for k in order:
        m = results[k]
        print(f"{k:16} " + " ".join(f"{m[c]:6.3f}" for c in
              ["Acc","macro_F1","wtd_F1","F1_H","F1_D","F1_A","kappa","MAE","RMSE","R2","r"]))


if __name__ == "__main__":
    main()
