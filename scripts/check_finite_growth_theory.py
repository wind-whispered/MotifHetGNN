"""Exhaustive finite-graph sanity checks; not a substitute for the proof.

Enumerate all labelled simple graphs through six vertices. Connectivity is
checked on vertex masks; extensions and removable vertices are counted
independently. All comparisons use integer arithmetic or exact fractions.
"""
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path
import argparse
import json
import time


def connected(mask, neighbours):
    reached = mask & -mask
    frontier = reached
    while frontier:
        bit = frontier & -frontier
        frontier ^= bit
        new = neighbours[bit.bit_length()-1] & mask & ~reached
        reached |= new
        frontier |= new
    return reached == mask


def check_graph(n, neighbours):
    sets = {k: [] for k in range(1, n+1)}
    for mask in range(1, 1 << n):
        if connected(mask, neighbours):
            sets[mask.bit_count()].append(mask)
    counts = {k: len(v) for k, v in sets.items()}
    growth = 0
    for k in range(1, n):
        smaller, larger = counts[k], counts[k+1]
        left = sum(connected(mask | (1 << v), neighbours)
                   for mask in sets[k] for v in range(n) if not mask & (1 << v))
        removable = [sum(connected(mask ^ (1 << v), neighbours)
                         for v in range(n) if mask & (1 << v)) for mask in sets[k+1]]
        right = sum(removable)
        assert left == right
        if not larger:
            assert left == 0
            if not smaller:
                assert not any(counts[j] for j in range(k+1, n+1))
            continue
        assert smaller > 0 and all(2 <= value <= k+1 for value in removable)
        extension = Fraction(left, (n-k)*smaller)
        mu = Fraction(right, larger)
        q_ratio = Fraction(larger*comb(n, k), smaller*comb(n, k+1))
        assert Fraction(larger, smaller) == (n-k)*extension/mu
        assert q_ratio == (k+1)*extension/mu
        assert 0 < extension <= 1
        gamma = Fraction(2*larger, (n-k)*smaller)
        assert gamma <= extension
        assert (gamma == extension) == (mu == 2)
        assert (larger > smaller) == (extension > mu/(n-k))
        assert (q_ratio > 1) == (extension > mu/(k+1))
        regime = larger > smaller and q_ratio <= 1
        assert regime == (mu/(n-k) < extension <= mu/(k+1))
        if regime:
            assert n-k > k+1
            growth += 1
    return growth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]
                        / "outputs/audits/theory_checks.json")
    args = parser.parse_args()
    started = time.perf_counter()
    totals, growth = {}, 0
    for n in range(2, args.max_n+1):
        edges = list(combinations(range(n), 2))
        for edge_mask in range(1 << len(edges)):
            neighbours = [0]*n
            for j, (u, v) in enumerate(edges):
                if edge_mask & (1 << j):
                    neighbours[u] |= 1 << v
                    neighbours[v] |= 1 << u
            growth += check_graph(n, neighbours)
        totals[n] = 1 << len(edges)
        print(f"PASS n={n}: {totals[n]} labelled graphs", flush=True)
    result = dict(graph_counts=totals, total_graphs=sum(totals.values()),
        capacity_dominated_cases=growth, arithmetic="integer and rational",
        checks=["double count", "growth ratios", "extension and removal bounds",
                "zero counts", "corollary interval", "lower-bound equality"],
        status="passed", seconds=time.perf_counter()-started,
        interpretation="Exhaustive weak projections through max_n only; mathematical proof is in the manuscript.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
