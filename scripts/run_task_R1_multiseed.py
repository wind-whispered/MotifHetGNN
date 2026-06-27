"""
TASK R1 (Revision): Multi-seed model evaluation.

Re-run all 7 model variants with N_SEEDS=10 different random seeds for
both weight initialisation and train/val/test split.
Produces ablation_results.parquet (70 rows) and
ablation_results_summary.parquet (7 rows, mean±std).

This addresses the reviewer concern that a single seed is insufficient when
HomGNN vs HetGNN accuracy differences are smaller than one match.
"""
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

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             r2_score, mean_absolute_error, mean_squared_error)
from scipy.stats import pearsonr

DATA = ROOT / "data"
SEEDS = [42, 7, 13, 21, 37, 55, 89, 144, 233, 377]
ADV_TURN = [("home_player", "adversarial", "away_player"),
            ("away_player", "adversarial", "home_player"),
            ("home_player", "turnover", "away_player"),
            ("away_player", "turnover", "home_player")]
PASS = [("home_player", "pass", "home_player"),
        ("away_player", "pass", "away_player")]
RM_ORDER = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]


def full_metrics(y_cls, p_cls, y_reg, p_reg):
    return {
        "Acc": accuracy_score(y_cls, p_cls),
        "macro_F1": f1_score(y_cls, p_cls, average="macro", zero_division=0),
        "wtd_F1": f1_score(y_cls, p_cls, average="weighted", zero_division=0),
        "F1_H": f1_score(y_cls, p_cls, labels=[0], average="macro", zero_division=0),
        "F1_D": f1_score(y_cls, p_cls, labels=[1], average="macro", zero_division=0),
        "F1_A": f1_score(y_cls, p_cls, labels=[2], average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_cls, p_cls),
        "MAE": mean_absolute_error(y_reg, p_reg),
        "RMSE": float(np.sqrt(mean_squared_error(y_reg, p_reg))),
        "R2": r2_score(y_reg, p_reg),
        "r": float(pearsonr(y_reg, p_reg)[0]) if np.std(p_reg) > 0 else 0.0,
    }


def mask_graph(data, keep):
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
        yc.append(torch.where(gd > 0, 0, torch.where(gd == 0, 1, 2)).cpu())
        pc.append(cls_logits.argmax(dim=-1).cpu())
        yr.append(gd.cpu())
        pr.append(reg_out.squeeze(-1).cpu())
    return (torch.cat(yc).numpy(), torch.cat(pc).numpy(),
            torch.cat(yr).float().numpy(), torch.cat(pr).float().numpy())


def train_gnn(graphs_tr, graphs_va, graphs_te, seed, edge_filter=None, epochs=150):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HeteroFootballGNN(node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if edge_filter is not None:
        graphs_tr = [mask_graph(d, edge_filter) for d in graphs_tr]
        graphs_va = [mask_graph(d, edge_filter) for d in graphs_va]
        graphs_te = [mask_graph(d, edge_filter) for d in graphs_te]
    trl = DataLoader(graphs_tr, batch_size=32, shuffle=True)
    val = DataLoader(graphs_va, batch_size=64, shuffle=False)
    tel = DataLoader(graphs_te, batch_size=64, shuffle=False)
    trainer = GNNTrainer(model, lr=1e-3, patience=20, device=device)
    trainer.train(trl, val, n_epochs=epochs)
    yc, pc, yr, pr = gnn_predict(model, tel, device)
    return full_metrics(yc, pc, yr, pr)


def load_graphs(ids, graph_dir):
    from src.gnn.dataset import FootballHeteroDataset
    graphs = []
    for mid in ids:
        p = graph_dir / f"het_graph_{mid}.pt"
        if p.exists():
            graphs.append(torch.load(p, weights_only=False))
    return graphs


def main():
    graph_dir = DATA / "networks" / "heterogeneous"
    ds = FootballHeteroDataset(root=str(ROOT))
    if len(ds) == 0:
        print("No heterogeneous graphs found. Run Task 3 first.")
        return

    mm = pd.read_parquet(DATA / "processed" / "matches_meta.parquet").set_index("match_id")
    ns = pd.read_parquet(DATA / "processed" / "network_stats.parquet")
    hom = pd.read_parquet(DATA / "motifs" / "homogeneous_motifs.parquet")
    hom = hom[hom.motif_order_k == 3]
    analysis_dir = DATA / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def motif_feats(common_ids):
        parts = []
        for side in ("home", "away"):
            p = (hom[hom.team_side == side].pivot_table(
                index="match_id", columns="motif_id", values="count",
                aggfunc="sum", fill_value=0).reindex(columns=RM_ORDER, fill_value=0))
            p.columns = [f"{side}_{i}" for i in RM_ORDER]
            parts.append(p)
        return parts[0].join(parts[1]).reindex(common_ids).fillna(0)

    def topo_feats(common_ids):
        n2 = ns[ns.w0 == 2]
        cols = ["density", "transitivity", "pass_diversity", "mean_outdegree",
                "mean_betweenness", "mean_eigenvector", "n_nodes", "n_edges"]
        parts = []
        for side in ("home", "away"):
            t = n2[n2.team_side == side].set_index("match_id")[cols]
            t.columns = [f"{side}_{c}" for c in cols]
            parts.append(t)
        return parts[0].join(parts[1]).reindex(common_ids).fillna(0)

    all_rows = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED = {seed}")
        t_seed = time.time()

        train_ids, val_ids, test_ids = split_dataset(ds, 0.7, 0.1, 0.2, seed)
        all_ids = list(ds._match_ids) if hasattr(ds, "_match_ids") else list(mm.index)
        common = [i for i in all_ids if i in mm.index]

        y_all = mm.loc[common, "goal_diff"].astype(float).values
        ycls_all = np.where(y_all > 0, 0, np.where(y_all == 0, 1, 2))
        tr_mask = np.array([c in set(train_ids) for c in common])
        te_mask = np.array([c in set(test_ids) for c in common])

        mfeat = motif_feats(common)
        tfeat = topo_feats(common)

        def run_tabular(X, clf_cls, clf_reg, name):
            np.random.seed(seed)
            X = np.nan_to_num(X.astype(float), nan=0.0)
            sc = StandardScaler().fit(X[tr_mask])
            Xs = sc.transform(X)
            clf_cls.fit(Xs[tr_mask], ycls_all[tr_mask])
            clf_reg.fit(Xs[tr_mask], y_all[tr_mask])
            m = full_metrics(ycls_all[te_mask], clf_cls.predict(Xs[te_mask]),
                             y_all[te_mask], clf_reg.predict(Xs[te_mask]))
            m["model_name"] = name
            m["seed"] = seed
            print(f"  {name}: acc={m['Acc']:.3f}")
            return m

        # Majority/mean baseline
        maj = np.bincount(ycls_all[tr_mask]).argmax()
        m_maj = full_metrics(ycls_all[te_mask], np.full(te_mask.sum(), maj),
                             y_all[te_mask], np.full(te_mask.sum(), y_all[tr_mask].mean()))
        m_maj["model_name"] = "Majority"; m_maj["seed"] = seed
        all_rows.append(m_maj)
        print(f"  Majority: acc={m_maj['Acc']:.3f}")

        # Linear topology
        all_rows.append(run_tabular(
            tfeat.values,
            LogisticRegression(max_iter=2000, random_state=seed),
            LinearRegression(), "Linear_topology"))

        # Linear motifs k=3
        all_rows.append(run_tabular(
            mfeat.values,
            LogisticRegression(max_iter=2000, random_state=seed),
            LinearRegression(), "Linear_motifs_k3"))

        # XGBoost motifs k=3
        all_rows.append(run_tabular(
            mfeat.values,
            GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=seed),
            GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=seed), "XGBoost_motifs_k3"))

        # GNN variants
        g_tr = load_graphs(train_ids, graph_dir)
        g_va = load_graphs(val_ids, graph_dir)
        g_te = load_graphs(test_ids, graph_dir)

        if g_tr:
            print(f"  Training AdvGNN (seed={seed})...")
            m_adv = train_gnn(g_tr, g_va, g_te, seed, edge_filter=ADV_TURN)
            m_adv["model_name"] = "AdvGNN"; m_adv["seed"] = seed
            all_rows.append(m_adv)
            print(f"  AdvGNN: acc={m_adv['Acc']:.3f}")

            print(f"  Training HomGNN (seed={seed})...")
            m_hom = train_gnn(g_tr, g_va, g_te, seed, edge_filter=PASS)
            m_hom["model_name"] = "HomGNN"; m_hom["seed"] = seed
            all_rows.append(m_hom)
            print(f"  HomGNN: acc={m_hom['Acc']:.3f}")

            print(f"  Training HetGNN (seed={seed})...")
            m_het = train_gnn(g_tr, g_va, g_te, seed, edge_filter=None)
            m_het["model_name"] = "HetGNN"; m_het["seed"] = seed
            all_rows.append(m_het)
            print(f"  HetGNN: acc={m_het['Acc']:.3f}")
        else:
            print("  WARNING: No graphs loaded; skipping GNN variants for this seed.")

        print(f"  Seed {seed} done in {time.time()-t_seed:.0f}s")

    df = pd.DataFrame(all_rows)
    df.to_parquet(analysis_dir / "ablation_results.parquet", index=False)
    print(f"\nSaved ablation_results.parquet ({len(df)} rows)")

    # Summarise
    metric_cols = ["Acc", "macro_F1", "wtd_F1", "F1_H", "F1_D", "F1_A",
                   "kappa", "MAE", "RMSE", "R2", "r"]
    rows = []
    for model_name, grp in df.groupby("model_name"):
        row = {"model_name": model_name}
        for c in metric_cols:
            if c in grp.columns:
                row[f"{c}_mean"] = grp[c].mean()
                row[f"{c}_std"] = grp[c].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_parquet(analysis_dir / "ablation_results_summary.parquet", index=False)
    print(f"Saved ablation_results_summary.parquet ({len(summary)} rows)")

    # Print table
    print("\n=== Ablation Summary (mean ± std over 10 seeds) ===")
    order = ["Majority", "Linear_topology", "Linear_motifs_k3",
             "XGBoost_motifs_k3", "AdvGNN", "HomGNN", "HetGNN"]
    for m in order:
        row = summary[summary.model_name == m]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"{m:20} Acc={r.Acc_mean:.3f}±{r.Acc_std:.3f} "
              f"mF1={r.macro_F1_mean:.3f}±{r.macro_F1_std:.3f} "
              f"kap={r.kappa_mean:.3f}±{r.kappa_std:.3f}")


if __name__ == "__main__":
    main()
