"""
TASK R3 (Revision): Structure vs. feature ablation.

Isolates graph topology contribution by comparing three HetGNN variants:
  HetGNN          -- full node features + all edges           (from Task R1)
  HetGNN-ConstFeat -- all-ones node features + all edges
  HetGNN-ZoneFeat  -- zone one-hot only (dims 19-28) + all edges

Run all 3 variants with N_SEEDS=10. Results appended to ablation_results.parquet.
Also produces outputs/figures/fig_N_structure_vs_feature.pdf/png.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import FootballHeteroDataset, split_dataset
from src.gnn.model import HeteroFootballGNN
from src.gnn.trainer import GNNTrainer
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             r2_score, mean_absolute_error, mean_squared_error)
from scipy.stats import pearsonr

DATA = ROOT / "data"
FIG = ROOT.parent / "足球分析研究" / "els-cas-templates" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 13, 21, 37, 55, 89, 144, 233, 377]
MM_INCH = 1.0 / 25.4

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
    "grid.color": "#CCCCCC", "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.titlesize": 10,
})


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


def apply_feat_variant(data, variant):
    g = data.clone()
    for nt in g.node_types:
        if hasattr(g[nt], "x") and g[nt].x is not None:
            n, d = g[nt].x.shape
            if variant == "const":
                g[nt].x = torch.ones(n, d)
            elif variant == "zone":
                # dims 19-28 are zone one-hot (9 zones); zero everything else
                new_x = torch.zeros(n, d)
                new_x[:, 19:28] = g[nt].x[:, 19:28]
                g[nt].x = new_x
    return g


def load_graphs(ids, graph_dir):
    graphs = []
    for mid in ids:
        p = graph_dir / f"het_graph_{mid}.pt"
        if p.exists():
            graphs.append(torch.load(p, weights_only=False))
    return graphs


def train_eval_variant(graphs_tr, graphs_va, graphs_te, seed, feat_variant):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeteroFootballGNN(node_feature_dim=39, hidden_dim=64, num_layers=3, dropout=0.3)
    tr_g = [apply_feat_variant(d, feat_variant) for d in graphs_tr]
    va_g = [apply_feat_variant(d, feat_variant) for d in graphs_va]
    te_g = [apply_feat_variant(d, feat_variant) for d in graphs_te]
    trl = DataLoader(tr_g, batch_size=32, shuffle=True)
    val = DataLoader(va_g, batch_size=64, shuffle=False)
    tel = DataLoader(te_g, batch_size=64, shuffle=False)
    trainer = GNNTrainer(model, lr=1e-3, patience=20, device=device)
    trainer.train(trl, val, n_epochs=150)
    yc, pc, yr, pr = gnn_predict(model, tel, device)
    return full_metrics(yc, pc, yr, pr)


def make_fig_N(summary):
    """FIG-N: structure vs feature ablation bar chart."""
    variants = ["HetGNN-ConstFeat", "HetGNN-ZoneFeat", "HetGNN"]
    colors = {"HetGNN-ConstFeat": "#DDDDDD", "HetGNN-ZoneFeat": "#AEC7E8",
              "HetGNN": "#1F77B4"}
    sub = summary[summary.model_name.isin(variants)].set_index("model_name")

    fig, ax = plt.subplots(figsize=(88 * MM_INCH, 65 * MM_INCH))
    ypos = range(len(variants))
    for yi, name in enumerate(variants):
        if name not in sub.index:
            continue
        acc_m = sub.loc[name, "Acc_mean"]
        acc_s = sub.loc[name, "Acc_std"]
        ax.barh(yi, acc_m, color=colors[name], xerr=acc_s,
                error_kw={"ecolor": "#555555", "elinewidth": 1.2, "capsize": 3,
                          "capthick": 1.2},
                height=0.55, edgecolor="black" if name == "HetGNN" else "none",
                linewidth=1.5 if name == "HetGNN" else 0)
        ax.text(acc_m + acc_s + 0.003, yi,
                f"{acc_m:.3f}±{acc_s:.3f}", va="center", fontsize=7.5)
    # dashed line at ConstFeat ceiling
    if "HetGNN-ConstFeat" in sub.index:
        x_ceil = sub.loc["HetGNN-ConstFeat", "Acc_mean"]
        ax.axvline(x_ceil, color="#888888", lw=0.8, ls="--")
        ax.text(x_ceil + 0.001, -0.5, "Feature-only ceiling",
                fontsize=7, color="#555555", va="bottom")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(variants, fontsize=9)
    ax.set_xlim(0.40, 0.68)
    ax.set_xlabel("Test accuracy (mean ± std)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_N_structure_vs_feature.pdf", format="pdf")
    fig.savefig(FIG / "fig_N_structure_vs_feature.png", dpi=600)
    plt.close(fig)
    print("  wrote fig_N_structure_vs_feature")


def main():
    graph_dir = DATA / "networks" / "heterogeneous"
    ds = FootballHeteroDataset(root=str(ROOT))
    if len(ds) == 0:
        print("No heterogeneous graphs found. Run Task 3 first.")
        return

    analysis_dir = DATA / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    variant_map = {"HetGNN-ConstFeat": "const", "HetGNN-ZoneFeat": "zone"}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_ids, val_ids, test_ids = split_dataset(ds, 0.7, 0.1, 0.2, seed)
        g_tr = load_graphs(train_ids, graph_dir)
        g_va = load_graphs(val_ids, graph_dir)
        g_te = load_graphs(test_ids, graph_dir)
        if not g_tr:
            print("  No graphs loaded, skipping.")
            continue
        for model_name, feat_var in variant_map.items():
            print(f"  Training {model_name} ...")
            t0 = time.time()
            m = train_eval_variant(g_tr, g_va, g_te, seed, feat_var)
            m["model_name"] = model_name
            m["seed"] = seed
            new_rows.append(m)
            print(f"  {model_name}: acc={m['Acc']:.3f}  ({time.time()-t0:.0f}s)")

    new_df = pd.DataFrame(new_rows)

    abl_path = analysis_dir / "ablation_results.parquet"
    if abl_path.exists():
        existing = pd.read_parquet(abl_path)
        existing = existing[~existing.model_name.isin(list(variant_map.keys()))]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(abl_path, index=False)
    print(f"Updated ablation_results.parquet ({len(combined)} rows)")

    # Rebuild summary
    metric_cols = ["Acc", "macro_F1", "wtd_F1", "F1_H", "F1_D", "F1_A",
                   "kappa", "MAE", "RMSE", "R2", "r"]
    rows = []
    for model_name, grp in combined.groupby("model_name"):
        row = {"model_name": model_name}
        for c in metric_cols:
            if c in grp.columns:
                row[f"{c}_mean"] = grp[c].mean()
                row[f"{c}_std"] = grp[c].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_parquet(analysis_dir / "ablation_results_summary.parquet", index=False)
    print(f"Updated ablation_results_summary.parquet ({len(summary)} rows)")

    make_fig_N(summary)

    print("\n=== Structure vs Feature Ablation ===")
    for name in list(variant_map.keys()) + ["HetGNN"]:
        row = summary[summary.model_name == name]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {name}: Acc={r.Acc_mean:.3f}±{r.Acc_std:.3f}")

    const_row = summary[summary.model_name == "HetGNN-ConstFeat"]
    full_row = summary[summary.model_name == "HetGNN"]
    if not const_row.empty and not full_row.empty:
        delta = full_row.iloc[0].Acc_mean - const_row.iloc[0].Acc_mean
        if delta > 0.01:
            print("\nConclusion: Graph topology is the primary signal "
                  "(HetGNN >> HetGNN-ConstFeat).")
        else:
            print("\nConclusion: Node features drive performance; "
                  "clarify attribution interpretation.")


if __name__ == "__main__":
    main()
