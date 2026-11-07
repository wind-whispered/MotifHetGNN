"""
Generate all paper figures (Tasks A–G, J, M, fig09, fig11).

Output: outputs/figures/<name>.{pdf,png}
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from src.paths import figure_directory
from src.motifs.reciprocity import (TRIAD_RECIPROCITY, TRIAD_DISPLAY_ORDER,
    HIGH_RECIPROCITY_TRIADS, ZERO_RECIPROCITY_TRIADS, reciprocal_link_fraction)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TEX_FIG = figure_directory()
TEX_FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style ----
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
CMAP_DIV = "RdBu_r"
CMAP_SEQ = "YlOrRd"
CMAP_CORR = "RdBu_r"
MM = 1.0 / 25.4  # mm -> inch

R_M = TRIAD_RECIPROCITY
# Stable feature order; reciprocity groups are selected by values, not slices.
RM_ORDER = list(TRIAD_DISPLAY_ORDER)

LONG = ["defensive", "middle", "attacking"]     # rows top->bottom (Def/Mid/Att)
LAT = ["left", "center", "right"]               # cols left->right (L/C/R)
ZONE_ORDER = [f"{lo}_{la}" for lo in LONG for la in LAT]   # row-major 9 zones
ZONE_ABBR = [f"{lo[0].upper()}{la[0].upper()}" for lo in LONG for la in LAT]

ADV_TYPES = {"Interception", "Tackle", "BallRecovery"}
TURN_TYPES = {"Miscontrol", "Dispossessed"}


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
zsc = pd.read_parquet(DATA / "motifs" / "homogeneous_zscore.parquet")
dyad = pd.read_parquet(DATA / "analysis" / "dyad_census.parquet")
census = pd.read_parquet(DATA / "analysis" / "census_sizes_allk.parquet")
hom3 = pd.read_parquet(DATA / "motifs" / "homogeneous_motifs.parquet")
hom3 = hom3[hom3.motif_order_k == 3]
sstate = pd.read_parquet(DATA / "analysis" / "motif_score_state.parquet")

side_map = mm.set_index("match_id")[["home_team_id", "away_team_id"]]


def reciprocity_table():
    """rho (fraction of directed links with a reverse link) per match/side/w0."""
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
            n_pairs=("d", "size"), n_recip=("d", lambda s: int((s == 2).sum()))).reset_index()
        per["rho"] = reciprocal_link_fraction(per.n_pairs, per.n_recip)
        per["w0"] = w0
        out.append(per)
    return pd.concat(out, ignore_index=True)


print("computing reciprocity from pass log ...")
recip = reciprocity_table()


# ============================================ per-match home feature frame ===
def build_feature_frame():
    """Per-match home-perspective feature frame for the correlation figures."""
    # P1: 13 triadic counts
    p1 = (hom3[hom3.team_side == "home"]
          .pivot_table(index="match_id", columns="motif_id", values="count",
                       aggfunc="sum", fill_value=0))
    p1 = p1.reindex(columns=RM_ORDER, fill_value=0)
    p1.columns = [f"n_{i}" for i in RM_ORDER]

    # dyads (k=2)
    dh = dyad[dyad.team_side == "home"].set_index("match_id")
    dyads = pd.DataFrame({"n_recip_dyad": dh.n_recip, "n_unidirec_dyad": dh.n_asym})

    # P2: N(4..7)
    ch = census[(census.team_side == "home") & (census.k.isin([4, 5, 6, 7]))]
    p2 = ch.pivot_table(index="match_id", columns="k", values="n_instances",
                        aggfunc="mean")
    p2.columns = [f"N{int(k)}" for k in p2.columns]

    # P3: rho, D, T, phi
    nh = ns[(ns.team_side == "home") & (ns.w0 == 2)].set_index("match_id")
    rh = recip[(recip.team_side == "home") & (recip.w0 == 2)].set_index("match_id")["rho"]
    p3 = pd.DataFrame({"rho": rh, "D": nh.density, "T": nh.transitivity,
                       "phi": nh.pass_diversity})

    # adversarial / turnover events for the home team, by zone and period
    ea = pd.read_parquet(DATA / "processed" / "events_adversarial.parquet",
                         columns=["match_id", "team_id", "event_type", "zone", "minute"])
    ea = ea.merge(mm[["match_id", "home_team_id"]], on="match_id", how="left")
    ea = ea[ea.team_id == ea.home_team_id]
    ea = ea[ea.zone.isin(ZONE_ORDER)]
    adv = ea[ea.event_type.isin(ADV_TYPES)]
    tur = ea[ea.event_type.isin(TURN_TYPES)]

    def zone_counts(df, prefix):
        t = (df.groupby(["match_id", "zone"]).size().unstack(fill_value=0)
             .reindex(columns=ZONE_ORDER, fill_value=0))
        t.columns = [f"{prefix}_{z}" for z in ZONE_ORDER]
        return t

    a1 = zone_counts(adv, "advz")
    a3 = zone_counts(tur, "turz")

    bins = [-1, 30, 60, 90, 1e9]
    labels = ["p0_30", "p30_60", "p60_90", "p90p"]
    adv = adv.assign(period=pd.cut(adv.minute, bins=bins, labels=labels))
    a2 = (adv.groupby(["match_id", "period"], observed=False).size().unstack(fill_value=0)
          .reindex(columns=labels, fill_value=0))
    a2.columns = [f"adv_{c}" for c in labels]

    frame = p1.join([dyads, p2, p3, a1, a2, a3], how="left")
    frame = frame.reindex(mm.match_id.values)
    frame = frame.fillna(frame.median(numeric_only=True))
    return frame


print("building feature frame ...")
F = build_feature_frame()


# ================================================================= FIG-A =====
def fig_A():
    print("FIG-A reciprocity vs w0")
    w0s = [0, 2, 10]
    fig, ax = plt.subplots(figsize=(80 * MM, 68 * MM))
    for side, col, mk in [("home", COLOR["home"], "o"), ("away", COLOR["away"], "o")]:
        rmean, rstd, dmean, dstd = [], [], [], []
        for w0 in w0s:
            r = recip[(recip.team_side == side) & (recip.w0 == w0)]["rho"]
            d = ns[(ns.team_side == side) & (ns.w0 == w0)]["density"]
            rmean.append(r.mean()); rstd.append(r.std())
            dmean.append(d.mean()); dstd.append(d.std())
        ax.errorbar(w0s, rmean, yerr=rstd, color=col, marker="o", lw=1.5,
                    capsize=3, label=fr"$\rho$ ({side})")
        ax.errorbar(w0s, dmean, yerr=dstd, color=col, marker="s", lw=1.5,
                    ls="--", capsize=3, label=f"$D$ ({side})")
        ax.fill_between(w0s, rmean, dmean, alpha=0.12, color=col)
    ax.annotate(r"$\rho-D$ surplus", xy=(2, 0.43), xytext=(3.6, 0.56),
                fontsize=8, style="italic",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(w0s)
    ax.set_xlabel(r"Link-weight threshold $w_0$")
    ax.set_ylabel("Reciprocity / Density")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "Figure_02a_Reciprocity_vs_Pass_Threshold")


# ================================================================= FIG-C =====
def fig_C():
    print("FIG-C z_m vs r_m")
    zh = zsc[zsc.team_side == "home"].set_index("motif_id")["z"]
    za = zsc[zsc.team_side == "away"].set_index("motif_id")["z"]
    ids = list(R_M.keys())
    rm = np.array([R_M[i] for i in ids])
    zhv = np.array([zh[i] for i in ids])
    zav = np.array([za[i] for i in ids])

    fig, ax = plt.subplots(figsize=(96 * MM, 67 * MM))
    ax.axhline(0, color=COLOR["gray"], ls="--", lw=0.8, zorder=0)
    ax.axhline(1.96, color="#B2182B", ls=":", lw=0.8, zorder=0)
    ax.axhline(-1.96, color="#B2182B", ls=":", lw=0.8, zorder=0)
    fit = np.polyfit(rm, zhv, 1)
    xr = np.linspace(-0.05, 1.05, 50)
    ax.plot(xr, np.polyval(fit, xr), "k-", lw=1.0, zorder=1)
    offset = 0.012
    ax.scatter(rm - offset, zhv, s=42, c=COLOR["home"], zorder=3, label="Home")
    ax.scatter(rm + offset, zav, s=42, facecolors="none", edgecolors=COLOR["away"],
               linewidths=1.2, zorder=3, label="Away")
    label_offsets = {78: (-24, 7), 98: (5, 5), 238: (-26, 7)}
    for i, x, y in zip(ids, rm, zhv):
        if i in label_offsets:
            ax.annotate(f"id {i}", (x - offset, y), textcoords="offset points",
                        xytext=label_offsets[i], fontsize=7, color="0.25")
    rh, ra = spearmanr(rm, zhv).statistic, spearmanr(rm, zav).statistic
    ax.text(0.05, 0.95, rf"$\rho_S$: H {rh:.3f}, A {ra:.3f}", transform=ax.transAxes,
            fontsize=9, style="italic", va="top")
    ax.text(1.04, 1.96, r"$|z|=1.96$", fontsize=6.5, color="#B2182B",
            ha="right", va="bottom")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xticks([0.00, 0.50, 2/3, 0.80, 1.00])
    ax.set_xlabel(r"Reciprocity proportion $r_m$")
    ax.set_ylabel(r"Population $z$-score $z_m$")
    ax.legend(loc="lower right", frameon=False, fontsize=8,
              markerscale=.85, handletextpad=.5)
    fig.tight_layout()
    save(fig, "Figure_04_Triad_Reciprocity_Enrichment")


# ============================================ helper: heatmap with annotations
def annotated_heatmap(ax, M, cmap, vmin, vmax, fmt="{:.2f}", fs=8,
                      white_thresh=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    nr, nc = M.shape
    if white_thresh is not None:
        for i in range(nr):
            for j in range(nc):
                v = M[i, j]
                if np.isnan(v):
                    continue
                col = "white" if abs(v) > white_thresh else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fs, color=col)
    return im


# ================================================================= FIG-D =====
def corr_order_block(C, labels, blocks):
    """Dendrogram leaf order applied within-display while keeping block order;
    here we simply return identity (keep block structure)."""
    return np.arange(len(labels))


def _zone_grid(series_by_zone):
    g = np.full((3, 3), np.nan)
    for idx, z in enumerate(ZONE_ORDER):
        g[idx // 3, idx % 3] = series_by_zone.get(z, 0.0)
    return g


# ================================================================= FIG-F =====
def fig_F():
    print("FIG-F 43x43 cross-modal correlation matrix")
    cols = ([f"n_{i}" for i in RM_ORDER]                       # P1 0-12
            + ["N4", "N5", "N6", "N7"]                         # P2 13-16
            + ["rho", "D", "T", "phi"]                         # P3 17-20
            + [f"advz_{z}" for z in ZONE_ORDER]                # A1 21-29
            + ["adv_p0_30", "adv_p30_60", "adv_p60_90", "adv_p90p"]   # A2 30-33
            + [f"turz_{z}" for z in ZONE_ORDER])               # A3 34-42
    X = F[cols].values
    C, _ = spearmanr(X)

    tick = ([str(i) for i in RM_ORDER]
            + ["N(4)", "N(5)", "N(6)", "N(7)"]
            + [r"$\rho$", "D", "T", r"$\phi$"]
            + ZONE_ABBR
            + ["0-30", "30-60", "60-90", "90+"]
            + ZONE_ABBR)

    fig, ax = plt.subplots(figsize=(170 * MM, 158 * MM))
    im = ax.imshow(C, cmap=CMAP_CORR, vmin=-1, vmax=1, aspect="equal")
    # thick separator between cooperative (P) and adversarial (A) blocks
    ax.axvline(20.5, color="white", lw=2.0)
    ax.axhline(20.5, color="white", lw=2.0)
    for pos in (12.5, 16.5, 29.5, 33.5):
        ax.axvline(pos, color="#AAAAAA", lw=0.8, ls="--")
        ax.axhline(pos, color="#AAAAAA", lw=0.8, ls="--")
    # gold rectangle on the C^PA off-diagonal block
    ax.add_patch(Rectangle((20.5, -0.5), 22, 21, fill=False,
                           edgecolor="#D4A017", lw=1.5))
    ax.set_xticks(range(len(tick)))
    ax.set_yticks(range(len(tick)))
    ax.set_xticklabels(tick, rotation=90, fontsize=6)
    ax.set_yticklabels(tick, fontsize=6)
    ax.grid(False)
    blocks = [(6, "P1\nTriadic"), (14.5, "P2\nHigh-ord"), (18.5, "P3\nTopo"),
              (25, "A1\nAdv.zones"), (31.5, "A2\nPeriods"), (38, "A3\nTO zones")]
    for pos, lbl in blocks:
        ax.text(pos, -1.8, lbl, ha="center", va="bottom", fontsize=7,
                fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
    cb.set_label("Spearman correlation coefficient", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "Figure_13_Full_Feature_Correlation_Atlas")




# ================================================================= FIG-J =====
def _away_feature_frame():
    """Away-side motif/dyad/profile counts, aligned to F.index."""
    p1 = (hom3[hom3.team_side == "away"]
          .pivot_table(index="match_id", columns="motif_id", values="count",
                       aggfunc="sum", fill_value=0)
          .reindex(columns=RM_ORDER, fill_value=0))
    p1.columns = [f"a_n_{i}" for i in RM_ORDER]
    da = dyad[dyad.team_side == "away"].set_index("match_id")
    dd = pd.DataFrame({"a_recip": da.n_recip, "a_asym": da.n_asym})
    ca = census[(census.team_side == "away") & (census.k.isin([4, 5, 6]))]
    cc = ca.pivot_table(index="match_id", columns="k", values="n_instances")
    cc.columns = [f"a_N{int(k)}" for k in cc.columns]
    out = p1.join([dd, cc]).reindex(F.index)
    return out.fillna(out.median(numeric_only=True))


def fig_J():
    print("FIG-J cumulative information gain")
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # The cumulative explained variance R^2 is taken from the pipeline's stored
    # incremental regression (which had access to the full per-class k>=4 motif
    # counts that are no longer materialised); K=2 is recomputed from the dyad
    # design.  The cross-validation std band and the accuracy curve are computed
    # on the buildable cumulative feature set (dyads + triadic classes + N(k)).
    A = _away_feature_frame()
    G = F.join(A)
    feat = {2: ["n_recip_dyad", "n_unidirec_dyad", "a_recip", "a_asym"]}
    feat[3] = feat[2] + [f"n_{i}" for i in RM_ORDER] + [f"a_n_{i}" for i in RM_ORDER]
    feat[4] = feat[3] + ["N4", "a_N4"]
    feat[5] = feat[4] + ["N5", "a_N5"]
    feat[6] = feat[5] + ["N6", "a_N6"]

    inc = pd.read_parquet(DATA / "analysis" / "incremental_r2_k7.parquet")
    r2_stored = {int(r.k): float(r.r_squared_cumulative) for _, r in inc.iterrows()}

    y_reg = mm.set_index("match_id").loc[G.index, "goal_diff"].values.astype(float)
    y_cls = np.sign(y_reg).astype(int)
    Ks = [2, 3, 4, 5, 6]
    r2m, r2s, accm, accs = [], [], [], []
    for K in Ks:
        X = StandardScaler().fit_transform(G[feat[K]].values)
        r2cv = cross_val_score(LinearRegression(), X, y_reg, cv=5, scoring="r2")
        ac = cross_val_score(LogisticRegression(max_iter=2000), X, y_cls, cv=5,
                             scoring="accuracy")
        if K == 2:
            r2line = LinearRegression().fit(X, y_reg).score(X, y_reg)
        else:
            r2line = r2_stored.get(K, r2cv.mean())
        r2m.append(r2line); r2s.append(r2cv.std())
        accm.append(ac.mean()); accs.append(ac.std())
    r2m, r2s = np.array(r2m), np.array(r2s)
    accm, accs = np.array(accm), np.array(accs)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(130 * MM, 60 * MM))
    for ax, m, s, ylab, ylim in [
            (axl, r2m, r2s, r"Explained variance $R^2$", (0.10, 0.32)),
            (axr, accm, accs, "Classification accuracy", (0.50, 0.62))]:
        ax.plot(Ks, m, color=COLOR["home"], marker="o", ms=6, lw=1.5)
        ax.fill_between(Ks, m - s, m + s, color=COLOR["home"], alpha=0.2)
        ax.axvspan(5, 6, color=COLOR["lgray"])
        ax.text(5.5, ylim[0] + 0.9 * (ylim[1] - ylim[0]), "Saturation",
                fontsize=8, style="italic", ha="center")
        ax.axvline(5, color=COLOR["gray"], ls="--", lw=0.8)
        ax.set_xticks(Ks); ax.set_xlim(1.7, 6.3); ax.set_ylim(*ylim)
        ax.set_xlabel(r"Maximum motif order $K$")
        ax.set_ylabel(ylab)
    fig.tight_layout()
    save(fig, "fig_J_cuminfo")
    print("   R2:", np.round(r2m, 3), " Acc:", np.round(accm, 3))


# ================================================================= FIG-M =====
def census_size(A, k):
    import itertools
    n = A.shape[0]
    if n < k:
        return 0
    combos = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64)
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    eye = np.eye(k, dtype=np.uint8)
    bits = np.stack([A[combos[:, i], combos[:, j]] for i, j in pairs], axis=1)
    m = bits.shape[0]
    U = np.zeros((m, k, k), dtype=np.uint8)
    for c, (i, j) in enumerate(pairs):
        U[:, i, j] |= bits[:, c]
        U[:, j, i] |= bits[:, c]
    R = U | eye
    p = 1
    while p < k:
        R = (np.matmul(R, R) > 0).astype(np.uint8)
        p *= 2
    return int(R[:, 0, :].all(axis=1).sum())


def net_from_passes(sub):
    if sub.empty:
        return np.zeros((0, 0), dtype=np.uint8)
    g = sub.groupby(["player_id", "recipient_id"]).size()
    g = g[g > 0]
    nodes = sorted(set(g.index.get_level_values(0)) | set(g.index.get_level_values(1)))
    idx = {u: i for i, u in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    for (u, v) in g.index:
        A[idx[u], idx[v]] = 1
    return A


def fig_M(n_sample=300):
    print("FIG-M mean temporal growth (subsample of %d matches)" % n_sample)
    rng = np.random.default_rng(42)
    ids = rng.choice(mm.match_id.values, size=min(n_sample, len(mm)), replace=False)
    ep = pd.read_parquet(DATA / "processed" / "events_pass.parquet",
                         columns=["match_id", "team_id", "player_id", "recipient_id",
                                  "minute", "second"])
    # StatsBomb minutes are already cumulative across periods; derive the
    # second-level event clock here instead of trusting legacy cached columns.
    ep["continuous_minute"] = (
        pd.to_numeric(ep["minute"], errors="coerce").fillna(0)
        + pd.to_numeric(ep["second"], errors="coerce").fillna(0) / 60.0
    )
    ep = ep[ep.match_id.isin(ids)].dropna(subset=["player_id", "recipient_id"])
    ep = ep.merge(mm[["match_id", "home_team_id", "away_team_id"]], on="match_id", how="left")

    grid = np.arange(3, 94, 3)
    curves = {k: np.full((len(ids), len(grid)), np.nan) for k in (3, 4, 5)}
    for mi, mid in enumerate(ids):
        P = ep[ep.match_id == mid]
        if P.empty:
            continue
        home = P.home_team_id.iloc[0]
        for ti, t in enumerate(grid):
            up = P[P.continuous_minute < t]
            Ah = net_from_passes(up[up.team_id == home])
            Aa = net_from_passes(up[up.team_id != home])
            for k in (3, 4, 5):
                curves[k][mi, ti] = census_size(Ah, k) + census_size(Aa, k)

    growth_rows = []
    for k in (3, 4, 5):
        for ti, minute in enumerate(grid):
            values = curves[k][:, ti]
            valid = np.isfinite(values)
            growth_rows.append({
                "motif_order_k": k,
                "cutoff_minute": int(minute),
                "mean_count": float(np.nanmean(values)),
                "sd_count": float(np.nanstd(values)),
                "n_matches": int(valid.sum()),
                "cutoff_rule": "minute + second/60 < cutoff",
            })
    growth_path = DATA / "information_sciences" / "motif_growth_timecourse.csv"
    growth_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(growth_rows).to_csv(growth_path, index=False)

    fig, ax = plt.subplots(figsize=(88 * MM, 68 * MM))
    kc = {3: COLOR["home"], 4: COLOR["away"], 5: COLOR["green"]}
    for k in (3, 4, 5):
        mu = np.nanmean(curves[k], axis=0)
        sd = np.nanstd(curves[k], axis=0)
        ax.plot(grid, mu, color=kc[k], lw=1.6, label=f"$k={k}$")
        ax.fill_between(grid, np.maximum(mu - sd, 1e-1), mu + sd, color=kc[k], alpha=0.15)
    ax.axvline(45, color="#555555", ls="--", lw=1.0)
    ax.text(46, 0.96, "Half-time", rotation=90, fontsize=8,
            va="top", ha="left", color="#555555",
            transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlim(0, 93)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Match minute $t$")
    ax.set_ylabel(r"Mean subgraph count $N(k;t)$")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "fig_M_temporal")


def fig09_w0():
    """Six global topology measures vs the link-weight threshold w0."""
    print("fig09 w0 robustness")
    measures = [("density", r"Density $D$"), ("transitivity", r"Transitivity $T$"),
                ("pass_diversity", r"Diversity $\phi$"),
                ("mean_outdegree", r"Out-degree $k^{\mathrm{out}}$"),
                ("mean_betweenness", r"Betweenness $b$"),
                ("mean_eigenvector", r"Eigenvector $e$")]
    w0s = [0, 2, 10]
    fig, axes = plt.subplots(2, 3, figsize=(120 * MM, 78 * MM))
    for ax, (col, lab) in zip(axes.ravel(), measures):
        for side, c in (("home", COLOR["home"]), ("away", COLOR["away"])):
            m, s = [], []
            for w0 in w0s:
                v = ns[(ns.team_side == side) & (ns.w0 == w0)][col]
                m.append(v.mean()); s.append(v.std())
            ax.errorbar(w0s, m, yerr=s, color=c, marker="o", ms=4, lw=1.4,
                        capsize=2.5, label=side.capitalize())
        ax.set_xticks(w0s); ax.set_xlabel(r"$w_0$")
        ax.set_ylabel(lab)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "Figure_12_Threshold_Robustness")


def fig11_zheat():
    """FIG-EX-2: triadic z-score heatmap, rows by z_H asc, with r_m column."""
    print("fig11 z-score heatmap")
    zh = zsc[zsc.team_side == "home"].set_index("motif_id")["z"]
    za = zsc[zsc.team_side == "away"].set_index("motif_id")["z"]
    order = zh.sort_values().index.tolist()
    M = np.array([[zh[i], za[i]] for i in order])

    fig, ax = plt.subplots(figsize=(70 * MM, 92 * MM))
    im = ax.imshow(M, cmap=CMAP_DIV, vmin=-2.5, vmax=2.5, aspect="equal")
    for r, i in enumerate(order):
        for c in range(2):
            v = M[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 1.5 else "black")
        ax.text(2.05, r, f"$r$={R_M[i]:.2f}", va="center", ha="left", fontsize=7,
                color="0.3")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Home", "Away"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(i) for i in order], fontsize=8)
    ax.set_ylabel("Motif id (ordered by $z_{\\mathrm{H}}$)")
    ax.set_xlim(-0.5, 2.7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.16)
    cb.set_label("$z$-score", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig11_zscore_heatmap")


# ======================================================= COMBINED SPEC FIGURES =
def fig_D_allorder():
    """Combined: N(k) growth profile (left) + 23x23 cross-order corrmat (right)."""
    print("Figure_05_Higher_Order_Motif_Atlas (combined N(k) + corrmat)")
    import json
    summ = json.load(open(DATA / "analysis" / "allorder_summary.json"))
    size = (census[census.k <= 14].groupby(["k", "team_side"])["n_instances"]
            .mean().unstack())
    ks = size.index.values
    mobs = {2: 2, 3: 13}
    for r in summ["table"]:
        mobs[int(r["k"])] = r["classes_obs"]
    occ_ks = [k for k in ks if k in mobs]
    occ = {side: [size.loc[k, side] / mobs[k] for k in occ_ks] for side in ("home", "away")}

    cols = (["n_recip_dyad", "n_unidirec_dyad"]
            + [f"n_{i}" for i in RM_ORDER]
            + ["N4", "N5", "N6", "N7", "rho", "D", "T", "phi"])
    X = F[cols].values
    C, _ = spearmanr(X)
    labels = (["Rec.dyad", "Uni.dyad"]
              + [f"id{i}" for i in RM_ORDER]
              + ["N(4)", "N(5)", "N(6)", "N(7)", r"$\rho$", "D", "T", r"$\phi$"])

    fig = plt.figure(figsize=(180 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.45, 0.55], wspace=0.35)

    # Left: N(k) + occupancy
    ax_l = fig.add_subplot(gs[0])
    ax_l.plot(ks, size["home"], "o-", ms=4, lw=1.5, color=COLOR["home"], label=r"$N(k)$ home")
    ax_l.plot(ks, size["away"], "o-", ms=4, lw=1.5, color=COLOR["away"], label=r"$N(k)$ away")
    ax_l.axvline(7, color="0.5", ls="--", lw=0.9)
    ax_l.text(7.3, size.values.max() * 0.55, r"$k^*=7$", fontsize=8, color="0.35")
    ax_l.set_yscale("log"); ax_l.set_ylim(2, 2000)
    ax_l.set_xlabel(r"Motif order $k$"); ax_l.set_ylabel(r"Mean subgraph count $N(k)$")
    ax_l.set_xticks(range(2, 15, 2))
    ax2 = ax_l.twinx()
    ax2.plot(occ_ks, occ["home"], "s--", ms=3.5, lw=1.2, color="#7FB3D5", label="Occ. home")
    ax2.plot(occ_ks, occ["away"], "s--", ms=3.5, lw=1.2, color="#F5B377", label="Occ. away")
    ax2.set_yscale("log"); ax2.set_ylim(1e-4, 20)
    ax2.set_ylabel("Mean occupancy per class"); ax2.grid(False)
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    h1, l1 = ax_l.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="lower center", frameon=False, fontsize=6.5, ncol=2)

    # Cluster rows AND columns in the leaf order of the displayed tree.
    # Ward linkage is inappropriate for this non-Euclidean dissimilarity.
    gs2 = gs[1].subgridspec(2, 1, height_ratios=[0.13, 1.0], hspace=0.015)
    ax_top = fig.add_subplot(gs2[0])
    ax_r = fig.add_subplot(gs2[1])
    dist_mat = 1 - np.abs(C); np.fill_diagonal(dist_mat, 0)
    Z = linkage(squareform(dist_mat, checks=False), method="average",
                optimal_ordering=True)
    tree = dendrogram(Z, no_plot=True)
    order = np.asarray(tree["leaves"])
    clustered = C[np.ix_(order, order)]
    clustered_labels = [labels[i] for i in order]
    # scipy uses terminal coordinates 5,15,...; imshow uses 0,1,... .
    for xs, ys in zip(tree["icoord"], tree["dcoord"]):
        ax_top.plot(np.asarray(xs) / 10 - .5, ys, color="#696969", lw=.65)
    ax_top.set_xlim(-.5, len(labels) - .5)
    ax_top.set_ylim(0, max(Z[:, 2]) * 1.06)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values(): s.set_visible(False)
    ax_top.grid(False)
    im = ax_r.imshow(clustered, cmap=CMAP_CORR, vmin=-1, vmax=1,
                     aspect="equal", interpolation="nearest")
    ax_r.set_box_aspect(1)
    ax_r.set_xticks(range(len(labels))); ax_r.set_yticks(range(len(labels)))
    ax_r.set_xticklabels(clustered_labels, rotation=90, fontsize=5.5)
    ax_r.set_yticklabels(clustered_labels, fontsize=5.5); ax_r.grid(False)
    cb = fig.colorbar(im, ax=ax_r, fraction=0.046, pad=0.02)
    cb.set_label("Spearman r", fontsize=7); cb.ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.canvas.draw()
    # A shared physical width is essential: the colorbar shrinks the heatmap.
    pos = ax_r.get_position()
    ax_top.set_position([pos.x0, pos.y1 + .012, pos.width, .10])
    tree_tips = sorted(x / 10 - .5 for xs, ys in
                       zip(tree["icoord"], tree["dcoord"])
                       for x, y in zip(xs, ys) if y == 0)
    assert np.allclose(tree_tips, np.arange(len(labels)))
    assert np.allclose(np.sort(clustered.ravel()), np.sort(C.ravel()))
    save(fig, "Figure_05_Higher_Order_Motif_Atlas")


def fig_E_het_coupling():
    """Spatial evidence and the cross-relation correlation distribution."""
    print("Figure_07_Structural_Coupling (spatial + cross-relation evidence)")
    ea = pd.read_parquet(
        DATA / "processed" / "events_adversarial.parquet",
        columns=["event_type", "zone"])
    ea = ea[ea.event_type.isin(ADV_TYPES) & ea.zone.isin(ZONE_ORDER)]
    dens = ea.zone.value_counts(normalize=True)
    density_map = _zone_grid(dens)

    cooperative_cols = ([f"n_{i}" for i in RM_ORDER]
                        + ["N4", "N5", "N6", "N7"]
                        + ["rho", "D", "T", "phi"])
    opposition_cols = ([f"advz_{z}" for z in ZONE_ORDER]
                       + ["adv_p0_30", "adv_p30_60", "adv_p60_90", "adv_p90p"]
                       + [f"turz_{z}" for z in ZONE_ORDER])
    C, _ = spearmanr(F[cooperative_cols + opposition_cols].values)
    cross_block = C[:len(cooperative_cols), len(cooperative_cols):]
    high_rows = [i for i, mid in enumerate(RM_ORDER) if mid in HIGH_RECIPROCITY_TRIADS]
    zero_rows = [i for i, mid in enumerate(RM_ORDER) if mid in ZERO_RECIPROCITY_TRIADS]
    reciprocal = cross_block[high_rows].ravel()
    chain = cross_block[zero_rows].ravel()
    # The cached matrices use this same immutable column order. Their rho
    # column is rank-equivalent under the dyad-to-directed-link conversion.
    robustness = []
    for suffix in ["", "_v2"]:
        matrix = np.load(DATA / "analysis" / f"spearman_corrmat_43x43{suffix}.npy")
        block = matrix[:21, 21:43]
        robustness.append((block[high_rows].mean(), block[zero_rows].mean()))

    fig = plt.figure(figsize=(155 * MM, 74 * MM))
    # Equal physical panel sizes, not just equal GridSpec slot widths. The
    # square heatmap used to shrink inside a narrower slot than the boxplot.
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.52,
                          left=.075, right=.985, bottom=.22, top=.88)

    ax0 = fig.add_subplot(gs[0])
    annotated_heatmap(ax0, density_map, CMAP_SEQ, 0,
                      np.nanmax(density_map), fmt="{:.2f}", fs=7,
                      white_thresh=np.nanmax(density_map) * 0.75)
    ax0.set_title("(a) Adversarial-event share", fontsize=8, fontweight="bold")
    ax0.set_xticks(range(3)); ax0.set_xticklabels(["L", "C", "R"])
    ax0.set_yticks(range(3)); ax0.set_yticklabels(["Def", "Mid", "Att"])
    ax0.grid(False)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_box_aspect(1)
    ax0.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False,
                            edgecolor="black", lw=1.0))

    ax1 = fig.add_subplot(gs[1])
    ax1.set_box_aspect(1)
    box = ax1.boxplot([reciprocal, chain], positions=[1, 2], widths=.48,
                      patch_artist=True, showfliers=False,
                      medianprops={"color": "black", "lw": 1.0},
                      whiskerprops={"color": "#555555", "lw": .8},
                      capprops={"color": "#555555", "lw": .8})
    for patch, color in zip(box["boxes"], ["#4C78A8", "#F58518"]):
        patch.set_facecolor(color); patch.set_alpha(.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(20260905)
    for xpos, values, color in [(1, reciprocal, "#4C78A8"),
                                (2, chain, "#F58518")]:
        keep = rng.choice(len(values), size=min(70, len(values)), replace=False)
        jitter = rng.uniform(-.16, .16, size=len(keep))
        ax1.scatter(xpos + jitter, values[keep], s=7, alpha=.30,
                    color=color, edgecolors="none", zorder=2)
    for offset, means in zip([-.07, .07], robustness):
        marker = "D" if offset < 0 else "X"
        label = "original grid" if offset < 0 else "shifted grid"
        ax1.scatter([1 + offset, 2 + offset],
                    means,
                    s=28, marker=marker, facecolor="white",
                    edgecolor="#202020", linewidth=.8, zorder=4,
                    label=label)
    ax1.axhline(0, color="#555555", ls="--", lw=.8)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["High reciprocity\n" + r"($r_m\geq2/3$)",
                         "No reciprocity\n" + r"($r_m=0$)"])
    ax1.set_ylabel(r"Cross-relation Spearman $r_S$")
    ax1.set_title("(b) Cross-relation associations", fontsize=8,
                  fontweight="bold")
    ax1.set_ylim(-.46, .40)
    ax1.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax1.grid(axis="x")

    fig.canvas.draw()
    extents = [a.get_window_extent() for a in (ax0, ax1)]
    assert abs(extents[0].width - extents[1].width) < .5
    assert abs(extents[0].height - extents[1].height) < .5
    save(fig, "Figure_07_Structural_Coupling")


def fig_G_context():
    """Score-state motif contrasts and their reciprocity ordering."""
    print("fig_G_context (score-state contrast + reciprocity association)")
    state_col = {"trailing": "Trailing", "drawing": "Level", "leading": "Leading"}
    states = ["trailing", "drawing", "leading"]
    M = np.zeros((len(RM_ORDER), 3))
    for r, mid in enumerate(RM_ORDER):
        for c, st in enumerate(states):
            h = sstate[(sstate.motif_id == mid) & (sstate.team_side == "home")
                       & (sstate.score_state == st)]["mean_count"].mean()
            a = sstate[(sstate.motif_id == mid) & (sstate.team_side == "away")
                       & (sstate.score_state == st)]["mean_count"].mean()
            M[r, c] = (0 if np.isnan(h) else h) - (0 if np.isnan(a) else a)
    rowmax = np.nanmax(np.abs(M), axis=1, keepdims=True); rowmax[rowmax == 0] = 1
    Mn = M / rowmax

    contrast = M[:, 2] - M[:, 0]
    rho, pval = spearmanr([R_M[mid] for mid in RM_ORDER], contrast)

    fig, axes = plt.subplots(1, 2, figsize=(148 * MM, 66 * MM),
                             gridspec_kw={"width_ratios": [0.57, 0.43]})

    ax = axes[0]
    cmap_b = plt.cm.RdBu_r.copy(); cmap_b.set_bad("#DDDDDD")
    ax.imshow(Mn, cmap=cmap_b, vmin=-1, vmax=1, aspect="equal")
    for ri in range(len(RM_ORDER)):
        for ci in range(3):
            v = Mn[ri, ci]
            col = "white" if abs(v) > 0.6 else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels([state_col[s] for s in states])
    ax.set_yticks(range(len(RM_ORDER)))
    ax.set_yticklabels([f"{i} (r={R_M[i]:.2f})" for i in RM_ORDER], fontsize=7.5)
    ax.set_title("(a) Score-state motif contrast", fontsize=8,
                 fontweight="bold")
    ax.grid(False)

    ax = axes[1]
    x = np.asarray([R_M[mid] for mid in RM_ORDER])
    ax.axhline(0, color="#777777", ls="--", lw=.8)
    ax.scatter(x, contrast, s=32, color="#7A5195", edgecolor="white",
               linewidth=.5, zorder=3)
    for mid, xv, yv in zip(RM_ORDER, x, contrast):
        if mid in {78, 98, 238}:
            ax.annotate(f"id {mid}", (xv, yv), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color="#333333")
    fit = np.polyfit(x, contrast, 1)
    xr = np.linspace(-.03, 1.03, 50)
    ax.plot(xr, np.polyval(fit, xr), color="#7A5195", lw=1.0,
            alpha=.75)
    ax.text(.04, .95, rf"$\rho_S={rho:.3f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlim(-.06, 1.08)
    ax.set_xticks([0, .33, .67, 1.0])
    ax.set_xlabel(r"Reciprocity proportion $r_m$")
    ax.set_ylabel("Leading minus trailing contrast")
    ax.set_title("(b) Reciprocity-aligned adaptation", fontsize=8,
                 fontweight="bold")

    fig.tight_layout()
    save(fig, "fig_G_context")


# ===================================================================== main ==
if __name__ == "__main__":
    sel = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    # Only canonical spec-named outputs (figure_specs_revised.txt / programming_task_revised.txt)
    todo = {
        "A":     fig_A,              # Figure_02a_Reciprocity_vs_Pass_Threshold
        "C":     fig_C,              # Figure_04_Triad_Reciprocity_Enrichment
        "D":     fig_D_allorder,     # Figure_05_Higher_Order_Motif_Atlas  (N(k) + cross-order corrmat)
        "E":     fig_E_het_coupling, # Figure_07_Structural_Coupling
        "F":     fig_F,              # Figure_13_Full_Feature_Correlation_Atlas
        "G":     fig_G_context,      # fig_G_context  (spatial + score-state + condcov)
        "J":     fig_J,              # fig_J_cuminfo
        "M":     fig_M,              # fig_M_temporal
        "fig09": fig09_w0,           # Figure_12_Threshold_Robustness  (FIG-EX-1)
        "fig11": fig11_zheat,        # fig11_zscore_heatmap  (FIG-EX-2)
    }
    if sel == ["all"]:
        sel = list(todo.keys())
    for key in sel:
        todo[key]()
    print("done.")
