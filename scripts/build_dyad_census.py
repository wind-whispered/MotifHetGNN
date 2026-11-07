"""Rebuild the descriptive dyad table from canonical passing graphs."""
from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.networks.homogeneous import load_network


def dyad_counts(graph):
    edges = {(u, v) for u, v in graph.edges() if u != v}
    pairs = {frozenset((u, v)) for u, v in edges}
    mutual = sum((v, u) in edges for u, v in edges)//2
    return dict(n_recip=mutual, n_pairs=len(pairs), n_asym=len(pairs)-mutual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"data/analysis/dyad_census.parquet")
    args = parser.parse_args()
    rows = []
    for path in sorted((ROOT/"data/networks/homogeneous").glob("*.gpickle")):
        match, side = path.stem.rsplit("_", 1)
        rows.append(dict(match_id=int(match), team_side=side, **dyad_counts(load_network(str(path)))))
    if not rows:
        raise FileNotFoundError("Build canonical passing graphs with run_task2_homogeneous first")
    frame = pd.DataFrame(rows).sort_values(["match_id", "team_side"]).reset_index(drop=True)
    assert not frame.duplicated(["match_id", "team_side"]).any()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"Rebuilt {len(frame)} network dyad counts -> {args.out}")


if __name__ == "__main__":
    main()
