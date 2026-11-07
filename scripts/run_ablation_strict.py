"""
Stricter ablation: remove not only the edges but also the cooperative
*node-level* information leaking into the adversarial-only model.

Node feature layout (39-d):
  [0:25]  tactical position one-hot (from lineup; not passing-derived)
  [25]    team side
  [26]    normalised pass count        }
  [27]    normalised reception count   }  passing-derived
  [28,29] mean (x,y) location          }  (cooperative)
  [30:39] zone one-hot                 }

Variants:
  HetGNN  : full node features + all edges                       (stored model)
  HomGNN  : full node features + pass edges only                 (cooperative)
  AdvGNN  : passing-derived features [26:39] zeroed
            + adversarial/turnover edges only                     (adversarial)

This denies the adversarial-only model the cooperative participation signal
(pass/reception counts, passing footprint) that otherwise lets it match the
full model on node features alone.
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
from src.gnn.model import HeteroFootballGNN
from src.gnn.trainer import GNNTrainer
from torch_geometric.loader import DataLoader
import run_ablation_table as A   # full_metrics, mask_graph, gnn_predict, preload, PASS, ADV_TURN

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

DATA = ROOT / "data"
COOP_FEAT_IDX = list(range(26, 39))   # passing-derived node features


def mask_features(data, zero_idx):
    g = data.clone()
    for nt in ["home_player", "away_player"]:
        if hasattr(g[nt], "x") and g[nt].x is not None:
            x = g[nt].x.clone()
            x[:, zero_idx] = 0.0
            g[nt].x = x
    return g


def train_eval(name, graphs_tr, graphs_va, graphs_te, epochs=150):
    print(f"\n=== {name} ===")
    trl = DataLoader(graphs_tr, batch_size=32, shuffle=True)
    val = DataLoader(graphs_va, batch_size=64, shuffle=False)
    tel = DataLoader(graphs_te, batch_size=64, shuffle=False)
    torch.manual_seed(42); np.random.seed(42)
    model = HeteroFootballGNN(node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3)
    tr = GNNTrainer(model, lr=1e-3, patience=20, device="cpu")
    t1 = time.time()
    tr.train(trl, val, n_epochs=epochs, save_path=str(DATA / "gnn" / f"strict_{name}.pt"))
    yc, pc, yr, pr = A.gnn_predict(model, tel, "cpu")
    m = A.full_metrics(yc, pc, yr, pr)
    print(f"  {name}: acc={m['Acc']:.3f} macroF1={m['macro_F1']:.3f} "
          f"kappa={m['kappa']:.3f} mae={m['MAE']:.3f}  ({time.time()-t1:.0f}s)")
    return m


def main():
    graph_dir = DATA / "networks" / "heterogeneous"
    ds = FootballHeteroDataset(root=str(ROOT))
    train_ids, val_ids, test_ids = split_dataset(ds, 0.7, 0.1, 0.2, 42)
    print("preloading ...")
    g_tr = A.preload(train_ids, graph_dir)
    g_va = A.preload(val_ids, graph_dir)
    g_te = A.preload(test_ids, graph_dir)
    print(f"  {len(g_tr)+len(g_va)+len(g_te)} graphs")

    results = {}

    # HetGNN: stored model, full graphs
    print("\n=== HetGNN (stored) ===")
    model = HeteroFootballGNN(node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3)
    model.load_state_dict(torch.load(DATA / "gnn" / "gnn_model.pt", weights_only=True))
    yc, pc, yr, pr = A.gnn_predict(model, DataLoader(g_te, batch_size=64), "cpu")
    results["HetGNN"] = A.full_metrics(yc, pc, yr, pr)
    print(f"  HetGNN: acc={results['HetGNN']['Acc']:.3f} "
          f"macroF1={results['HetGNN']['macro_F1']:.3f} kappa={results['HetGNN']['kappa']:.3f}")

    # HomGNN: pass edges only, full features
    results["HomGNN"] = train_eval(
        "HomGNN",
        [A.mask_graph(d, A.PASS) for d in g_tr],
        [A.mask_graph(d, A.PASS) for d in g_va],
        [A.mask_graph(d, A.PASS) for d in g_te])

    # AdvGNN: adv/turnover edges only AND passing-derived node features zeroed
    def adv(d):
        return mask_features(A.mask_graph(d, A.ADV_TURN), COOP_FEAT_IDX)
    results["AdvGNN"] = train_eval(
        "AdvGNN", [adv(d) for d in g_tr], [adv(d) for d in g_va], [adv(d) for d in g_te])

    # ---- tabular baselines (NaN-safe) ----
    print("\n=== baselines ===")
    mm = pd.read_parquet(DATA / "processed" / "matches_meta.parquet").set_index("match_id")
    ns = pd.read_parquet(DATA / "processed" / "network_stats.parquet")
    hom = pd.read_parquet(DATA / "motifs" / "homogeneous_motifs.parquet")
    hom = hom[hom.motif_order_k == 3]
    RM = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]

    def motif_feats():
        parts = []
        for side in ("home", "away"):
            p = (hom[hom.team_side == side].pivot_table(index="match_id", columns="motif_id",
                 values="count", aggfunc="sum", fill_value=0).reindex(columns=RM, fill_value=0))
            p.columns = [f"{side}_{i}" for i in RM]
            parts.append(p)
        return parts[0].join(parts[1])

    def topo_feats():
        n2 = ns[ns.w0 == 2]
        cols = ["density", "transitivity", "pass_diversity", "mean_outdegree",
                "mean_betweenness", "mean_eigenvector", "n_nodes", "n_edges"]
        parts = []
        for side in ("home", "away"):
            t = n2[n2.team_side == side].set_index("match_id")[cols]
            t.columns = [f"{side}_{c}" for c in cols]
            parts.append(t)
        return parts[0].join(parts[1])

    mfeat = motif_feats(); tfeat = topo_feats()
    common = [i for i in mm.index if i in mfeat.index and i in tfeat.index]
    y = mm.loc[common, "goal_diff"].astype(float).values
    ycls = np.where(y > 0, 0, np.where(y == 0, 1, 2))
    tr_mask = np.array([c in set(train_ids) for c in common])
    te_mask = np.array([c in set(test_ids) for c in common])

    def run_tab(X, clf, reg):
        X = np.nan_to_num(X, nan=np.nanmedian(X))
        sc = StandardScaler().fit(X[tr_mask]); Xs = sc.transform(X)
        clf.fit(Xs[tr_mask], ycls[tr_mask]); reg.fit(Xs[tr_mask], y[tr_mask])
        return A.full_metrics(ycls[te_mask], clf.predict(Xs[te_mask]),
                              y[te_mask], reg.predict(Xs[te_mask]))

    results["Linear_topology"] = run_tab(tfeat.loc[common].values,
        LogisticRegression(max_iter=2000), LinearRegression())
    results["Linear_motifs"] = run_tab(mfeat.loc[common].values,
        LogisticRegression(max_iter=2000), LinearRegression())
    results["XGBoost_motifs"] = run_tab(mfeat.loc[common].values,
        GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42))
    maj = np.bincount(ycls[tr_mask]).argmax()
    results["Majority"] = A.full_metrics(ycls[te_mask], np.full(te_mask.sum(), maj),
                                         y[te_mask], np.full(te_mask.sum(), y[tr_mask].mean()))

    json.dump(results, open(DATA / "analysis" / "ablation_strict.json", "w"), indent=2)
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
