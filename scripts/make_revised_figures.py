"""
Supplementary figures (B, K, L, N, O).

Output: outputs/figures/<name>.{pdf,png}
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TEX_FIG = None  # set via --tex-fig <dir> to also copy figures there

sys.path.insert(0, str(ROOT))

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "grid.color": "#CCCCCC",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.titlesize": 10,
})

COLOR = {
    "home": "#1F77B4", "away": "#FF7F0E", "green": "#2CA02C",
    "red": "#D6604D", "blue": "#2166AC", "lblue": "#92C5DE",
    "gray": "#888888", "lgray": "#F0F0F0",
}
MM = 1.0 / 25.4
R_M = {6: 0.00, 12: 0.00, 14: 0.67, 36: 0.00, 38: 0.00, 46: 0.33, 74: 0.67,
       78: 0.67, 98: 0.00, 102: 0.00, 108: 0.33, 110: 0.33, 238: 1.00}
RM_ORDER = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]


def save(fig, name):
    for ext in ("pdf", "png"):
        path = FIG / f"{name}.{ext}"
        kw = dict(format=ext, dpi=600) if ext == "png" else dict(format=ext)
        fig.savefig(path, **kw)
        if TEX_FIG is not None:
            shutil.copy2(path, TEX_FIG / f"{name}.{ext}")
    plt.close(fig)
    print("  wrote", name)


# ============================================================ data loading ===
print("loading data ...")
mm = pd.read_parquet(DATA / "processed" / "matches_meta.parquet")
ns = pd.read_parquet(DATA / "processed" / "network_stats.parquet")


def reciprocity_table():
    ep = pd.read_parquet(DATA / "processed" / "events_pass.parquet",
                         columns=["match_id", "team_id", "player_id", "recipient_id"])
    ep = ep.dropna(subset=["player_id", "recipient_id"])
    ep = ep.merge(mm[["match_id", "home_team_id", "away_team_id"]], on="match_id", how="left")
    ep["team_side"] = np.where(ep.team_id == ep.home_team_id, "home",
                               np.where(ep.team_id == ep.away_team_id, "away", None))
    ep = ep.dropna(subset=["team_side"])
    w = ep.groupby(["match_id", "team_side", "player_id", "recipient_id"]).size().reset_index(name="w")
    out = []
    for w0 in (0, 2, 10):
        f = w[w.w > w0].copy()
        a = np.minimum(f.player_id, f.recipient_id)
        b = np.maximum(f.player_id, f.recipient_id)
        f["key"] = list(zip(a, b))
        g = (f.groupby(["match_id", "team_side", "key"]).size()
             .reset_index(name="d"))
        per = g.groupby(["match_id", "team_side"]).agg(
            n_pairs=("d", "size"),
            n_recip=("d", lambda s: int((s == 2).sum()))).reset_index()
        per["rho"] = per.n_recip / per.n_pairs
        per["w0"] = w0
        out.append(per)
    return pd.concat(out, ignore_index=True)


print("computing reciprocity ...")
recip = reciprocity_table()


# ================================================================= FIG-B =====
def fig_B():
    print("FIG-B reciprocity distribution boxplot")
    w0_2_ns = ns[(ns.w0 == 2)]
    rho_home = recip[(recip.team_side == "home") & (recip.w0 == 2)]["rho"].values
    rho_away = recip[(recip.team_side == "away") & (recip.w0 == 2)]["rho"].values
    D_home = w0_2_ns[w0_2_ns.team_side == "home"]["density"].values
    D_away = w0_2_ns[w0_2_ns.team_side == "away"]["density"].values

    data = [D_home, D_away, rho_home, rho_away]
    labels = ["D (home)", "D (away)", r"$\rho$ (home)", r"$\rho$ (away)"]
    colors = ["#AEC7E8", "#FFBB78", COLOR["home"], COLOR["away"]]

    fig, ax = plt.subplots(figsize=(72 * MM, 70 * MM))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    whis=1.5, showfliers=True,
                    flierprops={"marker": ".", "alpha": 0.3, "markersize": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    for med in bp["medians"]:
        med.set_color("#333333")
        med.set_linewidth(2)

    # Dashed connector between median(D_home) and median(rho_home)
    med_D_home = np.median(D_home)
    med_rho_home = np.median(rho_home)
    ax.plot([1, 3], [med_D_home, med_rho_home], color="#888888", ls="--", lw=0.8)
    mid_y = (med_D_home + med_rho_home) / 2
    delta = med_rho_home - med_D_home
    ax.text(2, mid_y, rf"$\rho - D \approx {delta:.2f}$",
            ha="center", va="bottom", fontsize=8, style="italic", color="#333333")

    ax.set_ylim(0.0, 0.90)
    ax.set_ylabel("Value")
    ax.grid(axis="x")
    fig.tight_layout()
    save(fig, "fig_B_reciprocity_boxplot")


# ================================================================= FIG-K =====
def fig_K():
    print("FIG-K ablation bar chart (9 models, multi-seed)")
    abl_path = DATA / "analysis" / "ablation_results_summary.parquet"
    if not abl_path.exists():
        print("  ablation_results_summary.parquet not found; skipping FIG-K")
        return

    summary = pd.read_parquet(abl_path)

    # Model display order (bottom to top = worst to best)
    model_order = [
        "Majority",
        "Linear_topology",
        "Linear_motifs_k3",
        "XGBoost_motifs_k3",
        "Strength_only",
        "AdvGNN",
        "Strength_Motifs_k3",
        "HomGNN",
        "HetGNN",
    ]
    display_labels = {
        "Majority": "Majority baseline",
        "Linear_topology": "Linear topology",
        "Linear_motifs_k3": "Linear motifs k=3",
        "XGBoost_motifs_k3": "XGBoost motifs k=3",
        "Strength_only": "Strength only",
        "AdvGNN": "AdvGNN",
        "Strength_Motifs_k3": "Strength + Motifs k=3",
        "HomGNN": "HomGNN",
        "HetGNN": "HetGNN (full)",
    }
    bar_colors = {
        "Majority": "#BBBBBB",
        "Linear_topology": "#AEC7E8",
        "Linear_motifs_k3": "#AEC7E8",
        "XGBoost_motifs_k3": "#98DF8A",
        "Strength_only": "#C5B0D5",
        "AdvGNN": "#FFBB78",
        "Strength_Motifs_k3": "#C5B0D5",
        "HomGNN": "#FF7F0E",
        "HetGNN": "#1F77B4",
    }

    # Filter to models present in summary
    present = [m for m in model_order if m in summary.model_name.values]
    sub = summary.set_index("model_name")

    fig, ax = plt.subplots(figsize=(95 * MM, 80 * MM))
    ypos = range(len(present))
    for yi, name in enumerate(present):
        acc_m = sub.loc[name, "Acc_mean"]
        acc_s = sub.loc[name, "Acc_std"]
        edge_kw = {}
        if name == "HetGNN":
            edge_kw = {"edgecolor": "#0A3D6B", "linewidth": 2}
        ax.barh(yi, acc_m,
                color=bar_colors.get(name, "#888888"),
                xerr=acc_s,
                error_kw={"ecolor": "#555555", "elinewidth": 1.2,
                          "capsize": 3, "capthick": 1.2},
                height=0.62,
                **edge_kw)
        ax.text(acc_m + acc_s + 0.003, yi,
                f"{acc_m:.3f}±{acc_s:.3f}",
                va="center", fontsize=7.5)

    # Majority baseline reference line
    if "Majority" in sub.index:
        x_maj = sub.loc["Majority", "Acc_mean"]
        ax.axvline(x_maj, color="#888888", lw=0.8, ls="--")

    # Bracket between HomGNN and HetGNN
    if "HomGNN" in present and "HetGNN" in present:
        yi_hom = present.index("HomGNN")
        yi_het = present.index("HetGNN")
        x_right = max(sub.loc["HomGNN", "Acc_mean"] + sub.loc["HomGNN", "Acc_std"],
                      sub.loc["HetGNN", "Acc_mean"] + sub.loc["HetGNN", "Acc_std"]) + 0.065
        ax.annotate("", xy=(x_right, yi_het), xytext=(x_right, yi_hom),
                    arrowprops={"arrowstyle": "|-|", "color": "#333333", "lw": 0.8})
        ax.text(x_right + 0.003, (yi_hom + yi_het) / 2,
                "Statistically\nequivalent\n(within seed\nvariance)",
                va="center", fontsize=6, color="#333333")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([display_labels.get(m, m) for m in present], fontsize=8.5)
    ax.set_xlim(0.40, 0.70)
    ax.set_xlabel("Test accuracy (mean ± std, 10 seeds)")
    fig.tight_layout()
    save(fig, "fig_K_ablation_bar")


# ================================================================= FIG-L =====
def fig_L():
    print("FIG-L attribution triple scatter (3-panel)")
    attr_path = DATA / "gnn" / "motif_attribution.parquet"
    zsc_path = DATA / "motifs" / "homogeneous_zscore.parquet"
    if not attr_path.exists() or not zsc_path.exists():
        print("  attribution or zscore data missing; skipping FIG-L")
        return

    attr = pd.read_parquet(attr_path)
    zsc = pd.read_parquet(zsc_path)

    beta_reg_path = DATA / "analysis" / "regression_results.parquet"
    beta_df = None
    if beta_reg_path.exists():
        beta_reg = pd.read_parquet(beta_reg_path)
        beta_df = beta_reg[beta_reg.panel == "A_k3_homo"].copy()

    A_home, A_away, z_home, z_away, beta_home, beta_away = [], [], [], [], [], []
    for mid in RM_ORDER:
        ah = attr[(attr.motif_id == mid) & (attr.team_side == "home")]["mean_attribution"]
        aa = attr[(attr.motif_id == mid) & (attr.team_side == "away")]["mean_attribution"]
        A_home.append(ah.mean() if len(ah) > 0 else np.nan)
        A_away.append(aa.mean() if len(aa) > 0 else np.nan)
        zh = zsc[(zsc.motif_id == mid) & (zsc.team_side == "home")]["z"]
        za = zsc[(zsc.motif_id == mid) & (zsc.team_side == "away")]["z"]
        z_home.append(zh.mean() if len(zh) > 0 else np.nan)
        z_away.append(za.mean() if len(za) > 0 else np.nan)
        if beta_df is not None:
            bh = beta_df[(beta_df.motif_id == float(mid)) & (beta_df.team_side == "home")]["beta"]
            ba = beta_df[(beta_df.motif_id == float(mid)) & (beta_df.team_side == "away")]["beta"]
            beta_home.append(bh.mean() if len(bh) > 0 else np.nan)
            beta_away.append(ba.mean() if len(ba) > 0 else np.nan)
        else:
            beta_home.append(np.nan); beta_away.append(np.nan)

    A_home = np.array(A_home, dtype=float); A_away = np.array(A_away, dtype=float)
    z_home = np.array(z_home, dtype=float); z_away = np.array(z_away, dtype=float)
    absbeta = np.abs(np.array(beta_home, dtype=float))
    absbeta_a = np.abs(np.array(beta_away, dtype=float))
    absz_home = np.abs(z_home); absz_away = np.abs(z_away)

    def safe_spearman(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 4:
            return np.nan, 1.0
        return spearmanr(x[mask], y[mask])

    rho_Az, _ = safe_spearman(np.concatenate([A_home, A_away]),
                               np.concatenate([absz_home, absz_away]))
    rho_Ab, _ = safe_spearman(np.concatenate([A_home, A_away]),
                               np.concatenate([absbeta, absbeta_a]))
    rho_zb, _ = safe_spearman(np.concatenate([absz_home, absz_away]),
                               np.concatenate([absbeta, absbeta_a]))

    def ols_line(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            return None, None
        c = np.polyfit(x[mask], y[mask], 1)
        xr = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        return xr, np.polyval(c, xr)

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM))

    ax = axes[0]
    ax.scatter(absz_home, A_home, s=36, c=COLOR["home"], zorder=3, label="Home")
    ax.scatter(absz_away, A_away, s=36, facecolors="none",
               edgecolors=COLOR["away"], linewidths=1.2, zorder=3, label="Away")
    xr, yr = ols_line(np.concatenate([absz_home, absz_away]), np.concatenate([A_home, A_away]))
    if xr is not None:
        ax.plot(xr, yr, "k--", lw=1.0, zorder=1)
    ax.text(0.05, 0.95, fr"$\rho_S = {rho_Az:.2f}$" if not np.isnan(rho_Az) else r"$\rho_S$ = N/A",
            transform=ax.transAxes, fontsize=9, style="italic", va="top")
    ax.set_xlabel(r"Structural effect size $|z_m|$")
    ax.set_ylabel(r"GNN attribution $A_m$")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[1]
    ax.scatter(absbeta, absz_home, s=36, c=COLOR["home"], zorder=3, label="Home")
    ax.scatter(absbeta_a, absz_away, s=36, facecolors="none",
               edgecolors=COLOR["away"], linewidths=1.2, zorder=3, label="Away")
    xr, yr = ols_line(np.concatenate([absbeta, absbeta_a]), np.concatenate([absz_home, absz_away]))
    if xr is not None:
        ax.plot(xr, yr, "k--", lw=1.0, zorder=1)
    ax.text(0.05, 0.95, fr"$\rho_S = {rho_zb:.2f}$" if not np.isnan(rho_zb) else r"$\rho_S$ = N/A",
            transform=ax.transAxes, fontsize=9, style="italic", va="top")
    ax.set_xlabel(r"Linear coeff.\ $|\beta_m|$")
    ax.set_ylabel(r"Structural effect size $|z_m|$")

    ax = axes[2]
    ax.scatter(absbeta, A_home, s=36, c=COLOR["home"], zorder=3)
    ax.scatter(absbeta_a, A_away, s=36, facecolors="none",
               edgecolors=COLOR["away"], linewidths=1.2, zorder=3)
    xr, yr = ols_line(np.concatenate([absbeta, absbeta_a]), np.concatenate([A_home, A_away]))
    if xr is not None:
        ax.plot(xr, yr, "k--", lw=1.0, zorder=1)
    ax.text(0.05, 0.95, fr"$\rho_S = {rho_Ab:.2f}$" if not np.isnan(rho_Ab) else r"$\rho_S$ = N/A",
            transform=ax.transAxes, fontsize=9, style="italic", va="top")
    ax.set_xlabel(r"Linear coeff.\ $|\beta_m|$")
    ax.set_ylabel(r"GNN attribution $A_m$")

    fig.tight_layout()
    save(fig, "fig_L_attribution")


# ================================================================= FIG-N =====
def fig_N():
    """Delegates to run_task_R3_struct_feat.py if summary already exists."""
    print("FIG-N structure vs feature ablation")
    abl_path = DATA / "analysis" / "ablation_results_summary.parquet"
    if not abl_path.exists():
        print("  ablation_results_summary.parquet not found; skipping FIG-N")
        return
    summary = pd.read_parquet(abl_path)
    variants = ["HetGNN-ConstFeat", "HetGNN-ZoneFeat", "HetGNN"]
    if not any(v in summary.model_name.values for v in variants[:2]):
        print("  ConstFeat/ZoneFeat variants not in summary; "
              "run run_task_R3_struct_feat.py first")
        return

    colors = {"HetGNN-ConstFeat": "#DDDDDD", "HetGNN-ZoneFeat": "#AEC7E8",
              "HetGNN": "#1F77B4"}
    sub = summary[summary.model_name.isin(variants)].set_index("model_name")

    fig, ax = plt.subplots(figsize=(88 * MM, 65 * MM))
    present = [v for v in variants if v in sub.index]
    for yi, name in enumerate(present):
        acc_m = sub.loc[name, "Acc_mean"]
        acc_s = sub.loc[name, "Acc_std"]
        ax.barh(yi, acc_m, color=colors[name], xerr=acc_s,
                error_kw={"ecolor": "#555555", "elinewidth": 1.2, "capsize": 3,
                          "capthick": 1.2},
                height=0.55,
                edgecolor="black" if name == "HetGNN" else "none",
                linewidth=1.5 if name == "HetGNN" else 0)
        ax.text(acc_m + acc_s + 0.003, yi,
                f"{acc_m:.3f}±{acc_s:.3f}", va="center", fontsize=7.5)
    if "HetGNN-ConstFeat" in sub.index:
        x_ceil = sub.loc["HetGNN-ConstFeat", "Acc_mean"]
        ax.axvline(x_ceil, color="#888888", lw=0.8, ls="--")
        ax.text(x_ceil + 0.001, -0.5, "Feature-only ceiling",
                fontsize=7, color="#555555", va="bottom")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=9)
    ax.set_xlim(0.40, 0.68)
    ax.set_xlabel("Test accuracy (mean ± std)")
    fig.tight_layout()
    save(fig, "fig_N_structure_vs_feature")


# ================================================================= FIG-O =====
def fig_O():
    print("FIG-O CPA robustness comparison")
    comp_path = DATA / "analysis" / "robustness_CPA_comparison.parquet"
    if not comp_path.exists():
        print("  robustness_CPA_comparison.parquet not found; skipping FIG-O")
        return
    comp = pd.read_parquet(comp_path)
    groups = ["Reciprocal", "Chain"]
    col_keys = ["mean_rS_reciprocal", "mean_rS_chain"]
    methods = comp.method.tolist()
    colors = ["#1F77B4", "#FF7F0E"]
    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(70 * MM, 55 * MM))
    for mi, method in enumerate(methods):
        row = comp[comp.method == method].iloc[0]
        vals = [row[k] for k in col_keys]
        bars = ax.bar(x + (mi - 0.5) * w, vals, w, color=colors[mi],
                      label=method.replace("_", " ").title(), alpha=0.85)
        for bar, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}", ha="center", va=va, fontsize=7)
    ax.axhline(0, color="#888888", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(-0.40, 0.20)
    ax.set_ylabel(r"Mean Spearman $\bar{r}_S$ (C$^{\mathrm{PA}}$ block)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "fig_O_robustness_CPA")


# ===================================================================== main ==
if __name__ == "__main__":
    import sys as _sys
    sel = _sys.argv[1:] if len(_sys.argv) > 1 else ["all"]
    todo = {"B": fig_B, "K": fig_K, "L": fig_L, "N": fig_N, "O": fig_O}
    if sel == ["all"]:
        sel = list(todo.keys())
    for key in sel:
        if key in todo:
            todo[key]()
        else:
            print(f"Unknown figure key: {key}")
    print("done.")
