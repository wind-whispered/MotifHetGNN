"""
TASK R2 (Revision): Team-strength baseline.

Builds match-outcome predictors from team rolling-window strength features only,
then tests whether structural (motif) features add incremental predictive information
beyond team quality.

Two models:
  StrengthOnly        -- LogisticRegression on 4-dim strength vector
  Strength+Motifs_k3  -- LogisticRegression on [strength, motif_counts_k3] (30-dim)

Results appended to ablation_results.parquet and ablation_results_summary.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             r2_score, mean_absolute_error, mean_squared_error)
from scipy.stats import pearsonr

DATA = ROOT / "data"
SEEDS = [42, 7, 13, 21, 37, 55, 89, 144, 233, 377]
RM_ORDER = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]
WINDOW = 20


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


def compute_strength_features(mm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling strength features for every match (ordered by match_date).

    Features (home-team perspective):
      home_roll_winrate : fraction of last WINDOW home-team matches won
      home_roll_gd      : mean home_score - away_score over last WINDOW home matches
      away_roll_winrate : same for away team as away side
      away_roll_gd      : same

    For early matches with < WINDOW history, use corpus mean as fallback.
    """
    mm = mm.sort_values("match_date").copy()
    mm["goal_diff"] = mm["home_score"] - mm["away_score"]
    mm["home_win"] = (mm["goal_diff"] > 0).astype(float)

    # Build per-team match history (each team appears as home or away)
    home_hist = mm[["match_id", "match_date", "home_team_id", "goal_diff", "home_win"]].copy()
    home_hist.columns = ["match_id", "match_date", "team_id", "gd", "win"]
    away_hist = mm[["match_id", "match_date", "away_team_id", "goal_diff", "home_win"]].copy()
    away_hist["gd"] = -away_hist["goal_diff"]
    away_hist["win"] = (away_hist["goal_diff"] < 0).astype(float)
    away_hist = away_hist[["match_id", "match_date", "away_team_id", "gd", "win"]]
    away_hist.columns = ["match_id", "match_date", "team_id", "gd", "win"]

    team_hist = pd.concat([home_hist, away_hist], ignore_index=True)
    team_hist = team_hist.sort_values(["team_id", "match_date"])

    global_win_mean = team_hist["win"].mean()
    global_gd_mean = team_hist["gd"].mean()

    roll_win = {}
    roll_gd = {}
    for team_id, grp in team_hist.groupby("team_id"):
        grp = grp.reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp.iloc[max(0, i - WINDOW):i]
            if len(past) == 0:
                roll_win[(row.match_id, team_id)] = global_win_mean
                roll_gd[(row.match_id, team_id)] = global_gd_mean
            else:
                roll_win[(row.match_id, team_id)] = past["win"].mean()
                roll_gd[(row.match_id, team_id)] = past["gd"].mean()

    records = []
    for _, row in mm.iterrows():
        mid = row["match_id"]
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        records.append({
            "match_id": mid,
            "home_roll_winrate": roll_win.get((mid, htid), global_win_mean),
            "home_roll_gd": roll_gd.get((mid, htid), global_gd_mean),
            "away_roll_winrate": roll_win.get((mid, atid), global_win_mean),
            "away_roll_gd": roll_gd.get((mid, atid), global_gd_mean),
        })
    return pd.DataFrame(records).set_index("match_id")


def main():
    analysis_dir = DATA / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mm = pd.read_parquet(DATA / "processed" / "matches_meta.parquet")
    hom = pd.read_parquet(DATA / "motifs" / "homogeneous_motifs.parquet")
    hom = hom[hom.motif_order_k == 3]

    print("Computing rolling strength features...")
    strength_df = compute_strength_features(mm)
    strength_df.reset_index().to_parquet(
        analysis_dir / "strength_features.parquet", index=False)
    print(f"Saved strength_features.parquet ({len(strength_df)} rows)")

    mm_idx = mm.set_index("match_id")
    y_all = mm_idx["goal_diff"].astype(float)
    ycls_all = np.where(y_all > 0, 0, np.where(y_all == 0, 1, 2))

    def motif_feat_frame():
        parts = []
        for side in ("home", "away"):
            p = (hom[hom.team_side == side].pivot_table(
                index="match_id", columns="motif_id", values="count",
                aggfunc="sum", fill_value=0).reindex(columns=RM_ORDER, fill_value=0))
            p.columns = [f"{side}_{i}" for i in RM_ORDER]
            parts.append(p)
        return parts[0].join(parts[1])

    mfeat = motif_feat_frame()
    common = [i for i in mm_idx.index if i in strength_df.index and i in mfeat.index]
    strength_X = strength_df.reindex(common).fillna(0).values
    motif_X = mfeat.reindex(common).fillna(0).values
    y_reg = y_all.reindex(common).values
    y_cls = ycls_all[mm_idx.index.get_indexer(common)]

    new_rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        n = len(common)
        idx = rng.permutation(n)
        n_tr = int(0.7 * n); n_va = int(0.1 * n)
        tr = idx[:n_tr]; te = idx[n_tr + n_va:]

        for name, X in [("Strength_only", strength_X),
                        ("Strength_Motifs_k3", np.hstack([strength_X, motif_X]))]:
            X_clean = np.nan_to_num(X.astype(float), nan=0.0)
            sc = StandardScaler().fit(X_clean[tr])
            Xs = sc.transform(X_clean)
            clf = LogisticRegression(max_iter=2000, random_state=seed)
            reg = LinearRegression()
            clf.fit(Xs[tr], y_cls[tr])
            reg.fit(Xs[tr], y_reg[tr])
            m = full_metrics(y_cls[te], clf.predict(Xs[te]),
                             y_reg[te], reg.predict(Xs[te]))
            m["model_name"] = name
            m["seed"] = seed
            new_rows.append(m)
            print(f"  {name} seed={seed}: acc={m['Acc']:.3f}")

    new_df = pd.DataFrame(new_rows)

    # Merge with existing ablation results if present
    abl_path = analysis_dir / "ablation_results.parquet"
    if abl_path.exists():
        existing = pd.read_parquet(abl_path)
        # Remove any old Strength rows to avoid duplication
        existing = existing[~existing.model_name.isin(
            ["Strength_only", "Strength_Motifs_k3"])]
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

    print("\n=== Strength baseline results (mean ± std) ===")
    for name in ["Strength_only", "Strength_Motifs_k3"]:
        row = summary[summary.model_name == name]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {name}: Acc={r.Acc_mean:.3f}±{r.Acc_std:.3f}")


if __name__ == "__main__":
    main()
