"""Audit completeness and internal consistency of the exact high-order census.

The audit distinguishes two representations produced by
``run_task11_highorder.py``:

* ``census_sizes_allk.parquet`` contains one row for every graph order
  ``2 <= k <= n`` even when the connected-subgraph count is zero;
* ``highorder_census_k67.parquet`` is sparse and contains only positive
  canonical-class counts for k=6 and k=7.

Consequently, absence from the sparse table is valid only when the matching
all-order total is zero (or the graph has fewer than k nodes).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.homogeneous import load_network


NETWORK_DIR = ROOT / "data" / "networks" / "homogeneous"
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUTPUT = ANALYSIS_DIR / "highorder_coverage_audit.json"


def _network_index() -> pd.DataFrame:
    rows: list[tuple[int, str, int]] = []
    for path in sorted(NETWORK_DIR.glob("*.gpickle")):
        match_id, team_side = path.stem.rsplit("_", 1)
        graph = load_network(str(path))
        rows.append((int(match_id), team_side, graph.number_of_nodes()))
    return pd.DataFrame(rows, columns=["match_id", "team_side", "n_nodes"])


def _records(frame: pd.DataFrame, limit: int = 20) -> list[dict]:
    return frame.head(limit).to_dict(orient="records")


def audit_census() -> dict:
    networks = _network_index()
    sizes = pd.read_parquet(ANALYSIS_DIR / "census_sizes_allk.parquet")
    high = pd.read_parquet(ANALYSIS_DIR / "highorder_census_k67.parquet")
    low = pd.read_parquet(
        ROOT / "data" / "motifs" / "homogeneous_motifs.parquet"
    ).rename(columns={"motif_order_k": "k", "motif_id": "canon_id"})
    dyad = pd.read_parquet(ANALYSIS_DIR / "dyad_census.parquet")

    key_cols = ["match_id", "team_side"]
    size_key_cols = key_cols + ["k"]
    high_key_cols = size_key_cols + ["canon_id"]

    duplicate_networks = networks.duplicated(key_cols, keep=False)
    duplicate_sizes = sizes.duplicated(size_key_cols, keep=False)
    duplicate_high = high.duplicated(high_key_cols, keep=False)
    duplicate_low = low.duplicated(high_key_cols, keep=False)
    duplicate_dyad = dyad.duplicated(key_cols, keep=False)

    expected_rows: list[tuple[int, str, int]] = []
    for row in networks.itertuples(index=False):
        expected_rows.extend(
            (row.match_id, row.team_side, k)
            for k in range(2, row.n_nodes + 1)
        )
    expected = pd.DataFrame(expected_rows, columns=size_key_cols)

    coverage = expected.merge(
        sizes[size_key_cols + ["n_instances"]],
        on=size_key_cols,
        how="outer",
        indicator=True,
    )
    missing_size_rows = coverage[coverage["_merge"] == "left_only"]
    extra_size_rows = coverage[coverage["_merge"] == "right_only"]

    high_totals = (
        high.groupby(size_key_cols, as_index=False, observed=True)["count"]
        .sum()
        .rename(columns={"count": "canonical_total"})
    )
    eligible = (
        networks.assign(_join=1)
        .merge(pd.DataFrame({"k": [6, 7], "_join": [1, 1]}), on="_join")
        .drop(columns="_join")
    )
    eligible = eligible[eligible.n_nodes >= eligible.k]
    comparison = eligible.merge(
        sizes[size_key_cols + ["n_instances"]], on=size_key_cols, how="left"
    ).merge(high_totals, on=size_key_cols, how="left")
    comparison["canonical_total"] = comparison.canonical_total.fillna(0)
    comparison["difference"] = (
        comparison.canonical_total - comparison.n_instances
    )
    mismatched_totals = comparison[
        comparison.n_instances.isna() | comparison.difference.ne(0)
    ]

    low_totals = (
        low.groupby(size_key_cols, as_index=False, observed=True)["count"]
        .sum()
        .rename(columns={"count": "canonical_total"})
    )
    low_comparison = sizes[sizes.k.isin([3, 4, 5])].merge(
        low_totals, on=size_key_cols, how="outer", indicator=True
    )
    low_comparison["canonical_total"] = low_comparison.canonical_total.fillna(0)
    low_comparison["difference"] = (
        low_comparison.canonical_total - low_comparison.n_instances
    )
    mismatched_low_totals = low_comparison[
        low_comparison.n_instances.isna() | low_comparison.difference.ne(0)
    ]

    dyad_comparison = sizes[sizes.k.eq(2)].merge(
        dyad.assign(canonical_total=dyad.n_recip + dyad.n_asym)[
            key_cols + ["canonical_total"]
        ],
        on=key_cols,
        how="outer",
        indicator=True,
    )
    dyad_comparison["difference"] = (
        dyad_comparison.canonical_total - dyad_comparison.n_instances
    )
    mismatched_dyad_totals = dyad_comparison[
        (dyad_comparison._merge != "both") | dyad_comparison.difference.ne(0)
    ]

    ineligible_high = high.merge(networks, on=key_cols, how="left")
    ineligible_high = ineligible_high[
        ineligible_high.n_nodes.isna() | (ineligible_high.k > ineligible_high.n_nodes)
    ]
    invalid_high_rows = high[
        (~high.k.isin([6, 7]))
        | high["count"].isna()
        | high["count"].le(0)
        | high["canon_id"].isna()
    ]
    invalid_size_rows = sizes[
        sizes.n_instances.isna() | sizes.n_instances.lt(0)
    ]

    checks = {
        "network_keys_unique": not duplicate_networks.any(),
        "size_keys_unique": not duplicate_sizes.any(),
        "canonical_keys_unique": not duplicate_high.any(),
        "loworder_canonical_keys_unique": not duplicate_low.any(),
        "dyad_keys_unique": not duplicate_dyad.any(),
        "all_expected_size_rows_present": missing_size_rows.empty,
        "no_extra_size_rows": extra_size_rows.empty,
        "k2_dyad_sums_equal_allorder_totals": mismatched_dyad_totals.empty,
        "k3_k5_canonical_sums_equal_allorder_totals": mismatched_low_totals.empty,
        "k67_canonical_sums_equal_allorder_totals": mismatched_totals.empty,
        "no_ineligible_highorder_rows": ineligible_high.empty,
        "highorder_rows_valid": invalid_high_rows.empty,
        "allorder_rows_valid": invalid_size_rows.empty,
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "network_files": int(len(networks)),
            "network_keys": int(networks[key_cols].drop_duplicates().shape[0]),
            "allorder_rows_expected": int(len(expected)),
            "allorder_rows_observed": int(len(sizes)),
            "highorder_sparse_rows": int(len(high)),
            "loworder_sparse_rows": int(len(low)),
            "eligible_network_orders_k67": int(len(eligible)),
            "positive_network_orders_k67": int(comparison.n_instances.gt(0).sum()),
            "zero_network_orders_k67": int(comparison.n_instances.eq(0).sum()),
            "k_min": int(sizes.k.min()),
            "k_max": int(sizes.k.max()),
        },
        "failures": {
            "duplicate_networks": _records(networks[duplicate_networks]),
            "duplicate_size_keys": _records(sizes[duplicate_sizes]),
            "duplicate_canonical_keys": _records(high[duplicate_high]),
            "duplicate_loworder_canonical_keys": _records(low[duplicate_low]),
            "duplicate_dyad_keys": _records(dyad[duplicate_dyad]),
            "missing_size_rows": _records(missing_size_rows),
            "extra_size_rows": _records(extra_size_rows),
            "mismatched_k2_totals": _records(mismatched_dyad_totals),
            "mismatched_k3_k5_totals": _records(mismatched_low_totals),
            "mismatched_k67_totals": _records(mismatched_totals),
            "ineligible_highorder_rows": _records(ineligible_high),
            "invalid_highorder_rows": _records(invalid_high_rows),
            "invalid_allorder_rows": _records(invalid_size_rows),
        },
    }
    return report


def main() -> None:
    report = audit_census()
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
