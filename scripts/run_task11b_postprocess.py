"""
Task 11b - Post-process the extended higher-order census:
  * extended decay curve (k=3..7) -> regenerate fig09_w0_robustness
  * extended incremental R^2 (k=3..7, TOP_K=25 per order) -> regenerate fig_D_allorder
  * sparsity / occupancy / concentration summary for k=2..7
  * all-order census-size curve summary
Prints a JSON summary and writes figures to outputs/figures/.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, ".")
from src.analysis.saturation import compute_decay_curve
from src.analysis.regression import compute_incremental_r2
from src.visualization.decay_plots import plot_decay_curve, plot_saturation_curve

ANALYSIS = Path("data/analysis")
MOTIFS = Path("data/motifs")
OUT_FIG = Path("outputs/figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---- extended homogeneous motif table (k=3..7) ----
homo = pd.read_parquet(MOTIFS / "homogeneous_motifs.parquet")
hi = pd.read_parquet(ANALYSIS / "highorder_census_k67.parquet")
hi = hi.rename(columns={"k": "motif_order_k", "canon_id": "motif_id"})
hi["count"] = hi["count"].astype(float)
ext = pd.concat([homo[["match_id", "team_side", "motif_order_k", "motif_id", "count"]],
                 hi[["match_id", "team_side", "motif_order_k", "motif_id", "count"]]],
                ignore_index=True)

# ---- decay curve ----
decay = compute_decay_curve(ext)
hetero_summary = pd.read_parquet(MOTIFS / "hetero_order_summary.parquet")
plot_decay_curve(decay, hetero_summary, output_path=str(OUT_FIG / "fig09_w0_robustness.pdf"))

# ---- incremental R^2 ----
meta = pd.read_parquet("data/processed/matches_meta.parquet")
incr = compute_incremental_r2(ext, meta, k_max=7)
incr.to_parquet(ANALYSIS / "incremental_r2_k7.parquet", index=False)
plot_saturation_curve(incr, output_path=str(OUT_FIG / "fig_D_allorder.pdf"))

# ---- sparsity / concentration summary ----
summary = {"incremental_r2": incr.to_dict("records"), "orders": {}}
for k in (3, 4, 5, 6, 7):
    sub = ext[ext.motif_order_k == k]
    per_class = sub.groupby("motif_id")["count"].sum().sort_values(ascending=False)
    n_networks = sub.groupby(["match_id", "team_side"]).ngroups
    census_net = float(sub["count"].sum() / n_networks)
    per_side = {}
    for side in ("home", "away"):
        ss = sub[sub.team_side == side]
        tot = ss.groupby("match_id")["count"].sum()
        per_side[side] = dict(obs_classes=int(ss.motif_id.nunique()),
                              mean_census=float(tot.mean()),
                              std_census=float(tot.std()))
    summary["orders"][k] = dict(
        classes_observed=int(per_class.size),
        census_per_network=census_net,
        occupancy_per_class=float(census_net / per_class.size),
        top10_share=float(per_class.head(10).sum() / per_class.sum()),
        per_side=per_side,
    )

# dyads for table completeness
dy = pd.read_parquet(ANALYSIS / "dyad_census.parquet")
dy["total"] = dy.n_recip + dy.n_asym
summary["dyads"] = {
    side: dict(mean_census=float(g.total.mean()), std_census=float(g.total.std()))
    for side, g in dy.groupby("team_side")
}

# ---- all-order census-size curve ----
cs = pd.read_parquet(ANALYSIS / "census_sizes_allk.parquet")
curve = cs.groupby("k")["n_instances"].mean()
summary["allorder_curve"] = {int(k): float(v) for k, v in curve.items()}
peak_k = int(curve.idxmax())
summary["allorder_peak"] = dict(k=peak_k, mean_instances=float(curve.max()))
nn = cs.groupby(["match_id", "team_side"])["k"].max()
summary["mean_max_order"] = float(nn.mean())

with open(ANALYSIS / "highorder_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary["orders"], indent=1))
print("incremental:", summary["incremental_r2"])
print("all-order curve:", summary["allorder_curve"])
print("peak:", summary["allorder_peak"], "mean max order:", summary["mean_max_order"])
print("dyads:", summary["dyads"])
