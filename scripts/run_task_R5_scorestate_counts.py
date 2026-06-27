"""
TASK R5 (Revision): Score-state cell sample counts.

Adds per-cell match counts (n_matches) to the score-state motif heatmap data,
so that FIG-G(b) can flag low-count cells (n < 50) as gray.

Reads motif_score_state.parquet, computes n_matches per (motif_id, score_state)
cell, flags low-count cells, and writes:
  data/analysis/motif_score_state.parquet   (updated with n_matches column)
  data/analysis/scorestate_cell_counts.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
LOW_COUNT_THRESHOLD = 50
RM_ORDER = [98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78]


def main():
    analysis_dir = DATA / "analysis"
    score_state_path = analysis_dir / "motif_score_state.parquet"

    if not score_state_path.exists():
        print(f"motif_score_state.parquet not found at {score_state_path}")
        print("Run run_task7_spatiotemporal.py first.")
        return

    print("Loading motif_score_state.parquet...")
    df = pd.read_parquet(score_state_path)
    print(f"  Loaded {len(df)} records, columns: {list(df.columns)}")

    # Normalise score_state labels (may be stored as -1/0/+1 or 'trailing'/'drawing'/'leading')
    if df["score_state"].dtype == object:
        state_map = {"trailing": "trailing", "drawing": "level",
                     "leading": "leading", "level": "level",
                     "level": "level"}
        df["score_state_str"] = df["score_state"].map(
            lambda x: {"trailing": "Trailing", "drawing": "Level",
                       "level": "Level", "leading": "Leading"}.get(str(x).lower(), str(x)))
    else:
        df["score_state_str"] = df["score_state"].map(
            {-1: "Trailing", 0: "Level", 1: "Leading"})

    # Count number of distinct matches contributing to each (motif_id, score_state) cell
    # A match contributes to a cell if any motif of that id was observed in that score state
    if "match_id" in df.columns:
        cell_counts = (df.groupby(["motif_id", "score_state_str"])["match_id"]
                       .nunique().reset_index(name="n_matches"))
    else:
        # Fallback: use count of rows as proxy
        cell_counts = (df.groupby(["motif_id", "score_state_str"])
                       .size().reset_index(name="n_matches"))

    cell_counts["flag_low_count"] = cell_counts["n_matches"] < LOW_COUNT_THRESHOLD
    cell_counts.rename(columns={"score_state_str": "score_state"}, inplace=True)

    # Save cell counts table
    cell_counts.to_parquet(analysis_dir / "scorestate_cell_counts.parquet", index=False)
    print(f"Saved scorestate_cell_counts.parquet ({len(cell_counts)} rows)")

    n_low = cell_counts["flag_low_count"].sum()
    print(f"  Low-count cells (n<{LOW_COUNT_THRESHOLD}): {n_low} / {len(cell_counts)}")

    # Merge n_matches back into score_state df
    df_merged = df.merge(
        cell_counts[["motif_id", "score_state", "n_matches", "flag_low_count"]],
        left_on=["motif_id", "score_state_str"],
        right_on=["motif_id", "score_state"],
        how="left"
    )
    # Drop redundant column if it appeared
    if "score_state_x" in df_merged.columns:
        df_merged.rename(columns={"score_state_x": "score_state"}, inplace=True)
    if "score_state_y" in df_merged.columns:
        df_merged.drop(columns=["score_state_y"], inplace=True)
    if "score_state_str" in df_merged.columns:
        df_merged.drop(columns=["score_state_str"], inplace=True)

    df_merged.to_parquet(score_state_path, index=False)
    print(f"Updated motif_score_state.parquet ({len(df_merged)} rows)")

    # Print summary of low-count cells
    if n_low > 0:
        print("\nLow-count cells:")
        low_cells = cell_counts[cell_counts["flag_low_count"]]
        print(low_cells.to_string(index=False))

    print("\nTask R5 complete.")


if __name__ == "__main__":
    main()
