"""
Assemble the final ablation table from real experiments:
  * GNN rows  -> from ablation_strict.json (stricter ablation)
  * baselines -> recomputed here on the same test split, with the
                 gradient-boosting model tuned by 5-fold CV on the train set.
Emits the LaTeX tabular body with the best value per column bolded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gnn.dataset import FootballHeteroDataset, split_dataset
import run_ablation_table as A

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

DATA = ROOT / "data"
RM = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]


def main():
    strict = json.load(open(DATA / "analysis" / "ablation_strict.json"))
    ds = FootballHeteroDataset(root=str(ROOT))
    train_ids, val_ids, test_ids = split_dataset(ds, 0.7, 0.1, 0.2, 42)

    mm = pd.read_parquet(DATA / "processed" / "matches_meta.parquet").set_index("match_id")
    ns = pd.read_parquet(DATA / "processed" / "network_stats.parquet")
    hom = pd.read_parquet(DATA / "motifs" / "homogeneous_motifs.parquet")
    hom = hom[hom.motif_order_k == 3]

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

    def fit_eval(X, clf, reg):
        X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
        sc = StandardScaler().fit(X[tr_mask]); Xs = sc.transform(X)
        clf.fit(Xs[tr_mask], ycls[tr_mask]); reg.fit(Xs[tr_mask], y[tr_mask])
        return A.full_metrics(ycls[te_mask], clf.predict(Xs[te_mask]),
                              y[te_mask], reg.predict(Xs[te_mask]))

    res = {}
    res["Majority"] = A.full_metrics(ycls[te_mask], np.full(te_mask.sum(), np.bincount(ycls[tr_mask]).argmax()),
                                     y[te_mask], np.full(te_mask.sum(), y[tr_mask].mean()))
    res["Linear_topology"] = fit_eval(tfeat.loc[common].values,
        LogisticRegression(max_iter=2000), LinearRegression())
    res["Linear_motifs"] = fit_eval(mfeat.loc[common].values,
        LogisticRegression(max_iter=2000), LinearRegression())

    # XGBoost-style: tune HistGradientBoosting depth by CV accuracy on train
    Xm = np.nan_to_num(mfeat.loc[common].values, nan=0.0)
    Xms = StandardScaler().fit(Xm[tr_mask]).transform(Xm)
    best_d, best_cv = None, -1
    for d in (2, 3, 4):
        cv = cross_val_score(HistGradientBoostingClassifier(max_depth=d, max_iter=300,
             learning_rate=0.05, l2_regularization=1.0, random_state=42),
             Xms[tr_mask], ycls[tr_mask], cv=5, scoring="accuracy").mean()
        print(f"  GBM depth={d} cv_acc={cv:.3f}")
        if cv > best_cv:
            best_cv, best_d = cv, d
    res["XGBoost_motifs"] = fit_eval(mfeat.loc[common].values,
        HistGradientBoostingClassifier(max_depth=best_d, max_iter=300, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=42),
        HistGradientBoostingRegressor(max_depth=best_d, max_iter=300, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=42))

    for k in ("AdvGNN", "HomGNN", "HetGNN"):
        res[k] = strict[k]

    json.dump(res, open(DATA / "analysis" / "ablation_final.json", "w"), indent=2)

    # ---- bold map: best per column ----
    order = ["Majority", "Linear_topology", "Linear_motifs", "XGBoost_motifs",
             "AdvGNN", "HomGNN", "HetGNN"]
    cols = ["Acc", "macro_F1", "wtd_F1", "F1_H", "F1_D", "F1_A", "kappa",
            "MAE", "RMSE", "R2", "r"]
    lower_better = {"MAE", "RMSE"}
    best = {}
    for c in cols:
        vals = {k: res[k][c] for k in order if not (k == "Majority" and c in ("R2", "r"))}
        best[c] = (min if c in lower_better else max)(vals, key=vals.get)

    disp = {"Majority": "Majority / mean", "Linear_topology": "Linear, topology",
            "Linear_motifs": r"Linear, motifs $k\!=\!3$",
            "XGBoost_motifs": r"GBM, motifs $k\!=\!3$",
            "AdvGNN": "AdvGNN", "HomGNN": "HomGNN", "HetGNN": r"\textbf{HetGNN}"}
    edges = {"Majority": "---", "Linear_topology": "---", "Linear_motifs": "---",
             "XGBoost_motifs": "---",
             "AdvGNN": r"$\mathcal{E}^{\mathrm{a}}\cup\mathcal{E}^{\mathrm{t}}$",
             "HomGNN": r"$\mathcal{E}^{\mathrm{p}}$",
             "HetGNN": r"$\mathcal{E}^{\mathrm{p}}\!\cup\!\mathcal{E}^{\mathrm{a}}\!\cup\!\mathcal{E}^{\mathrm{t}}$"}

    def fmt(k, c):
        v = res[k][c]
        if k == "Majority" and c in ("R2", "r"):
            return "---"
        s = f"{v:.3f}"
        if best.get(c) == k:
            s = r"\textbf{" + s + "}"
        return s

    print("\n% ---- ablation table body (real strict-ablation numbers) ----")
    for k in order:
        cls = " & ".join(fmt(k, c) for c in ["Acc", "macro_F1", "wtd_F1", "F1_H", "F1_D", "F1_A", "kappa"])
        reg = " & ".join(fmt(k, c) for c in ["MAE", "RMSE", "R2", "r"])
        print(f"{disp[k]}\n& {edges[k]}\n& {cls}\n& & {reg} \\\\")
        if k == "XGBoost_motifs":
            print(r"\midrule")
    print("\n% summary")
    hdr = f"{'model':16}" + "".join(f"{c:>8}" for c in cols)
    print(hdr)
    for k in order:
        print(f"{k:16}" + "".join(f"{res[k][c]:8.3f}" for c in cols))


if __name__ == "__main__":
    main()
