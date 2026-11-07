"""Recompute corrected reciprocity summaries; no GNN result cache is changed."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.motifs.reciprocity import (TRIAD_RECIPROCITY, TRIAD_DISPLAY_ORDER,
                                  HIGH_RECIPROCITY_TRIADS, ZERO_RECIPROCITY_TRIADS)


def main():
    data = ROOT / "data"
    out = data / "information_sciences/round3_structure_audit"
    out.mkdir(parents=True, exist_ok=True)
    zpath = data / "motifs/homogeneous_zscore.parquet"
    z = pd.read_parquet(zpath)
    ids = list(TRIAD_RECIPROCITY)
    stats = {}
    for side in ["home", "away"]:
        values = z[(z.team_side == side) & (z.motif_order_k == 3)].set_index("motif_id").loc[ids, "z"]
        stats[side] = float(spearmanr([TRIAD_RECIPROCITY[i] for i in ids], values).statistic)
    high = [j for j, mid in enumerate(TRIAD_DISPLAY_ORDER) if mid in HIGH_RECIPROCITY_TRIADS]
    zero = [j for j, mid in enumerate(TRIAD_DISPLAY_ORDER) if mid in ZERO_RECIPROCITY_TRIADS]
    sources = [zpath]
    blocks = []
    for suffix, label in [("", "original_grid"), ("_v2", "shifted_grid")]:
        path = data / "analysis" / f"spearman_corrmat_43x43{suffix}.npy"
        matrix = np.load(path)
        assert matrix.shape == (43, 43)
        blocks.append(dict(grid=label, high_reciprocity=float(matrix[np.ix_(high, range(21, 43))].mean()),
                           no_reciprocity=float(matrix[np.ix_(zero, range(21, 43))].mean())))
        sources.append(path)
    result = dict(revision="adjacency_reciprocity_v1", reciprocal_link_definition="sum Aij Aji / sum Aij",
        triad_reciprocity=TRIAD_RECIPROCITY, spearman=stats, coupling=blocks,
        high_ids=sorted(HIGH_RECIPROCITY_TRIADS), zero_ids=sorted(ZERO_RECIPROCITY_TRIADS),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        note="Descriptive ranks of 13 dependent motif templates; no motif-class p-value or target-resolution claim.")
    # Reuse the plotting data assembly to verify that Figure 3 and the text
    # use directed-link rather than connected-dyad reciprocity.
    from scripts import make_paper_figures as figures
    matched = figures.recip.merge(figures.ns[figures.ns.w0 == 2][
        ["match_id", "team_side", "density"]], on=["match_id", "team_side"])
    matched = matched[matched.w0 == 2].copy()
    matched["surplus"] = matched.rho-matched.density
    result["network_reciprocity_w0_2"] = matched.groupby("team_side")[["rho", "density", "surplus"]].mean().to_dict("index")
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    pd.DataFrame([dict(motif_id=mid, reciprocal_link_fraction=TRIAD_RECIPROCITY[mid]) for mid in ids]).to_csv(
        out / "template_reciprocity.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
