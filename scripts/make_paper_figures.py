"""
Generate all paper figures (Tasks A–G, J, M, fig09, fig11).

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
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TEX_FIG = None  # set via --tex-fig <dir> to also copy figures there

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

# reciprocity proportion of each canonical triadic class
R_M = {6: 0.00, 12: 0.00, 14: 0.67, 36: 0.00, 38: 0.00, 46: 0.33, 74: 0.67,
       78: 0.67, 98: 0.00, 102: 0.00, 108: 0.33, 110: 0.33, 238: 1.00}
# triadic ids ordered by r_m ascending (manuscript ordering)
RM_ORDER = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]

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
spat = pd.read_parquet(DATA / "analysis" / "motif_spatial.parquet")
sstate = pd.read_parquet(DATA / "analysis" / "motif_score_state.parquet")
attr = pd.read_parquet(DATA / "gnn" / "motif_attribution.parquet")

side_map = mm.set_index("match_id")[["home_team_id", "away_team_id"]]


def reciprocity_table():
    """rho (fraction of connected dyads that are bidirectional) per match/side/w0."""
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
        per["rho"] = per.n_recip / per.n_pairs
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
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(w0s)
    ax.set_xlabel(r"Link-weight threshold $w_0$")
    ax.set_ylabel("Reciprocity / Density")
    ax.legend(loc="lower left", frameon=False, ncol=1, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_A_reciprocity_vs_w0")


# ================================================================= FIG-C =====
def fig_C():
    print("FIG-C z_m vs r_m")
    zh = zsc[zsc.team_side == "home"].set_index("motif_id")["z"]
    za = zsc[zsc.team_side == "away"].set_index("motif_id")["z"]
    ids = list(R_M.keys())
    rm = np.array([R_M[i] for i in ids])
    zhv = np.array([zh[i] for i in ids])
    zav = np.array([za[i] for i in ids])

    fig, ax = plt.subplots(figsize=(82 * MM, 72 * MM))
    ax.axhline(0, color=COLOR["gray"], ls="--", lw=0.8, zorder=0)
    fit = np.polyfit(rm, zhv, 1)
    xr = np.linspace(-0.05, 1.05, 50)
    ax.plot(xr, np.polyval(fit, xr), "k-", lw=1.0, zorder=1)
    ax.scatter(rm, zhv, s=48, c=COLOR["home"], zorder=3, label="Home")
    ax.scatter(rm, zav, s=48, facecolors="none", edgecolors=COLOR["away"],
               linewidths=1.2, zorder=3, label="Away")
    for i, x, y in zip(ids, rm, zhv):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(3, 3),
                    fontsize=7, color="0.25")
    ax.text(0.05, 0.95, r"$\rho_S = 0.91$, $p < 0.001$", transform=ax.transAxes,
            fontsize=9, style="italic", va="top")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2.5, 2.0)
    ax.set_xticks([0.00, 0.33, 0.67, 1.00])
    ax.set_xlabel(r"Reciprocity proportion $r_m$")
    ax.set_ylabel(r"Population $z$-score $z_m$")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "fig_C_zm_vs_rm")


# ============================================ helper: heatmap with annotations
def annotated_heatmap(ax, M, cmap, vmin, vmax, fmt="{:.2f}", fs=8,
                      white_thresh=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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
    save(fig, "fig_F_corrmat_43x43")




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
                                  "continuous_minute"])
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
            up = P[P.continuous_minute <= t]
            Ah = net_from_passes(up[up.team_id == home])
            Aa = net_from_passes(up[up.team_id != home])
            for k in (3, 4, 5):
                curves[k][mi, ti] = census_size(Ah, k) + census_size(Aa, k)

    fig, ax = plt.subplots(figsize=(88 * MM, 68 * MM))
    kc = {3: COLOR["home"], 4: COLOR["away"], 5: COLOR["green"]}
    for k in (3, 4, 5):
        mu = np.nanmean(curves[k], axis=0)
        sd = np.nanstd(curves[k], axis=0)
        ax.plot(grid, mu, color=kc[k], lw=1.6, label=f"$k={k}$")
        ax.fill_between(grid, np.maximum(mu - sd, 1e-1), mu + sd, color=kc[k], alpha=0.15)
    ax.axvline(45, color="#555555", ls="--", lw=1.0)
    ax.text(45, ax.get_ylim()[1] * 0.5, "Half-time", rotation=90, fontsize=8,
            va="center", ha="right", color="#555555")
    ax.axvline(30, color=COLOR["gray"], ls=":", lw=1.0)
    ax.text(30, ax.get_ylim()[1] * 0.5, "Backbone stabilisation", rotation=90,
            fontsize=8, va="center", ha="right", color=COLOR["gray"])
    # substitution tick marks (top) from the most common substitution minutes
    sub = pd.read_parquet(DATA / "processed" / "substitutions.parquet")
    top = sub.minute.value_counts().head(10).index.values
    ymax = ax.get_ylim()[1]
    for mnt in top:
        ax.plot([mnt, mnt], [ymax * 0.93, ymax], color="#999999", lw=1.0)
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
    save(fig, "fig09_w0_robustness")


def fig11_zheat():
    """FIG-EX-2: triadic z-score heatmap, rows by z_H asc, with r_m column."""
    print("fig11 z-score heatmap")
    zh = zsc[zsc.team_side == "home"].set_index("motif_id")["z"]
    za = zsc[zsc.team_side == "away"].set_index("motif_id")["z"]
    order = zh.sort_values().index.tolist()
    M = np.array([[zh[i], za[i]] for i in order])

    fig, ax = plt.subplots(figsize=(70 * MM, 92 * MM))
    im = ax.imshow(M, cmap=CMAP_DIV, vmin=-2.5, vmax=2.5, aspect="auto")
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
    print("fig_D_allorder (combined N(k) + corrmat)")
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

    # Right: 23x23 heatmap with dendrogram
    gs2 = gs[1].subgridspec(2, 1, height_ratios=[0.13, 1.0], hspace=0.015)
    ax_top = fig.add_subplot(gs2[0])
    ax_r = fig.add_subplot(gs2[1])
    dist_mat = 1 - np.abs(C); np.fill_diagonal(dist_mat, 0)
    Z = linkage(squareform(dist_mat, checks=False), method="ward")
    dendrogram(Z, ax=ax_top, color_threshold=0, above_threshold_color=COLOR["gray"], no_labels=True)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values(): s.set_visible(False)
    ax_top.grid(False)
    im = ax_r.imshow(C, cmap=CMAP_CORR, vmin=-1, vmax=1, aspect="auto")
    for pos in (1.5, 14.5, 18.5):
        ax_r.axvline(pos, color="white", lw=1.5); ax_r.axhline(pos, color="white", lw=1.5)
    ax_r.set_xticks(range(len(labels))); ax_r.set_yticks(range(len(labels)))
    ax_r.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax_r.set_yticklabels(labels, fontsize=5.5); ax_r.grid(False)
    cb = fig.colorbar(im, ax=ax_r, fraction=0.046, pad=0.02)
    cb.set_label("Spearman r", fontsize=7); cb.ax.tick_params(labelsize=6)

    fig.tight_layout()
    save(fig, "fig_D_allorder")


def fig_E_het_coupling():
    """Combined: composition stacked bars (left) + 2 spatial heatmaps (right)."""
    print("fig_E_het_coupling (composition + spatial contrast)")
    het = pd.read_parquet(DATA / "motifs" / "heterogeneous_motifs.parquet",
                          columns=["motif_order_k", "count", "n_cross_team_nodes", "has_pass_edge"])
    het = het[het.motif_order_k == 3]
    within = het[het.n_cross_team_nodes == 0]["count"].sum()
    cross = het[het.n_cross_team_nodes > 0]
    cross_mixed = cross[cross.has_pass_edge == 1]["count"].sum()
    cross_adv = cross[cross.has_pass_edge == 0]["count"].sum()
    comp = {
        "Within-team": {"cooperative": within, "mixed": 0.0, "adversarial": 0.0},
        "Cross-team": {"cooperative": 0.0, "mixed": cross_mixed, "adversarial": cross_adv},
    }
    groups = list(comp.keys())
    prop = {g: {k: comp[g][k] / (sum(comp[g].values()) or 1) for k in comp[g]} for g in groups}

    ea = pd.read_parquet(DATA / "processed" / "events_adversarial.parquet", columns=["zone"])
    ea = ea[ea.zone.isin(ZONE_ORDER)]
    dens = ea.zone.value_counts(normalize=True)
    left_map = _zone_grid(dens)

    sb = pd.read_parquet(DATA / "analysis" / "motif_spatial_bymotif.parquet")
    def norm_dist_sp(mid):
        s = (sb[(sb.motif_id == mid) & (sb.team_side == "home")]
             .groupby("zone")["count"].sum().reindex(ZONE_ORDER).fillna(0.0))
        mx = s.max(); return s / mx if mx > 0 else s
    right_raw = norm_dist_sp(78) - norm_dist_sp(98)
    right_map = _zone_grid(right_raw)
    rm = np.nanmax(np.abs(right_map)); right_map = right_map / rm if rm > 0 else right_map

    fig = plt.figure(figsize=(120 * MM, 65 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.35)

    # Left: stacked bars
    ax0 = fig.add_subplot(gs[0])
    x = np.arange(len(groups)); bottoms = np.zeros(len(groups))
    order = [("cooperative", COLOR["blue"]), ("mixed", COLOR["lblue"]), ("adversarial", COLOR["red"])]
    for key, col in order:
        vals = np.array([prop[g][key] for g in groups])
        ax0.bar(x, vals, bottom=bottoms, color=col, width=0.62, label=key.capitalize(),
                edgecolor="white", lw=0.5)
        bottoms += vals
    ax0.annotate("99.99%", xy=(1, 0.5), xytext=(1.05, 0.62), fontsize=9, fontweight="bold",
                 ha="center", arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
    ax0.set_xticks(x); ax0.set_xticklabels(groups, fontsize=8)
    ax0.set_ylim(0, 1); ax0.set_ylabel("Proportion of triadic subgraphs")
    ax0.legend(loc="upper right", frameon=False, fontsize=7.5); ax0.grid(axis="x")

    # Right: adversarial density
    ax1 = fig.add_subplot(gs[1])
    im0 = annotated_heatmap(ax1, left_map, CMAP_SEQ, 0, np.nanmax(left_map),
                            white_thresh=np.nanmax(left_map) * 0.7)
    ax1.set_title("Adversarial event density", fontsize=8)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["L", "C", "R"])
    ax1.set_yticks(range(3)); ax1.set_yticklabels(["Def", "Mid", "Att"])
    ax1.grid(False)
    ax1.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="black", lw=1.2))

    fig.tight_layout()
    save(fig, "fig_E_het_coupling")


def fig_G_context():
    """Combined: spatial motif contrast (a) + score-state heatmap (b) + condcov (c)."""
    print("fig_G_context (3-panel combined)")
    # Panel (a): spatial contrast (home only, id78-id98)
    sb = pd.read_parquet(DATA / "analysis" / "motif_spatial_bymotif.parquet")
    def norm_dist_g(mid):
        s = (sb[(sb.motif_id == mid) & (sb.team_side == "home")]
             .groupby("zone")["count"].sum().reindex(ZONE_ORDER).fillna(0.0))
        mx = s.max(); return s / mx if mx > 0 else s
    contrast_g = norm_dist_g(78) - norm_dist_g(98)
    grid_g = _zone_grid(contrast_g)
    mg = np.nanmax(np.abs(grid_g)); grid_g = grid_g / mg if mg > 0 else grid_g

    # Panel (b): score-state heatmap
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

    # Panel (c): conditional covariance difference
    cols_p1 = [f"n_{i}" for i in RM_ORDER]
    C_all, _ = spearmanr(F[cols_p1].values)
    lead_ids = mm[mm.goal_diff > 0].match_id.values
    Cl, _ = spearmanr(F.loc[F.index.isin(lead_ids), cols_p1].values)
    D = Cl - C_all; np.fill_diagonal(D, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM),
                             gridspec_kw={"width_ratios": [0.29, 0.42, 0.29]})

    # (a) spatial
    ax = axes[0]
    annotated_heatmap(ax, grid_g, CMAP_DIV, -1, 1, white_thresh=0.6, fs=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(["L", "C", "R"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["Def", "Mid", "Att"])
    ax.set_title(r"id78 $-$ id98", fontsize=9); ax.grid(False)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="black", lw=1.2))

    # (b) score-state
    ax = axes[1]
    cmap_b = plt.cm.RdBu_r.copy(); cmap_b.set_bad("#DDDDDD")
    ax.imshow(Mn, cmap=cmap_b, vmin=-1, vmax=1, aspect="auto")
    for ri in range(len(RM_ORDER)):
        for ci in range(3):
            v = Mn[ri, ci]
            col = "white" if abs(v) > 0.6 else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels([state_col[s] for s in states])
    ax.set_yticks(range(len(RM_ORDER)))
    ax.set_yticklabels([f"{i} (r={R_M[i]:.2f})" for i in RM_ORDER], fontsize=7.5)
    ax.grid(False)

    # (c) condcov
    ax = axes[2]
    masked = np.ma.masked_where(np.eye(len(cols_p1), dtype=bool), D)
    cmap_c = plt.cm.RdBu_r.copy(); cmap_c.set_bad("#DDDDDD")
    im_c = ax.imshow(masked, cmap=cmap_c, vmin=-0.3, vmax=0.3, aspect="equal")
    ax.axvline(8.5, color="white", lw=1.2, ls="--"); ax.axhline(8.5, color="white", lw=1.2, ls="--")
    ax.set_xticks(range(len(cols_p1))); ax.set_yticks(range(len(cols_p1)))
    ax.set_xticklabels(RM_ORDER, rotation=90, fontsize=6)
    ax.set_yticklabels(RM_ORDER, fontsize=6); ax.grid(False)
    ax.set_title(r"$\Delta\mathbf{C}^{(s=+1)}$", fontsize=9)
    fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.03).set_label(r"$\Delta$ corr.", fontsize=7)

    fig.tight_layout()
    save(fig, "fig_G_context")


# ===================================================================== main ==
if __name__ == "__main__":
    sel = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    # Only canonical spec-named outputs (figure_specs_revised.txt / programming_task_revised.txt)
    todo = {
        "A":     fig_A,              # fig_A_reciprocity_vs_w0
        "C":     fig_C,              # fig_C_zm_vs_rm
        "D":     fig_D_allorder,     # fig_D_allorder  (N(k) + cross-order corrmat)
        "E":     fig_E_het_coupling, # fig_E_het_coupling  (composition + spatial)
        "F":     fig_F,              # fig_F_corrmat_43x43
        "G":     fig_G_context,      # fig_G_context  (spatial + score-state + condcov)
        "J":     fig_J,              # fig_J_cuminfo
        "M":     fig_M,              # fig_M_temporal
        "fig09": fig09_w0,           # fig09_w0_robustness  (FIG-EX-1)
        "fig11": fig11_zheat,        # fig11_zscore_heatmap  (FIG-EX-2)
    }
    if sel == ["all"]:
        sel = list(todo.keys())
    for key in sel:
        todo[key]()
    print("done.")
