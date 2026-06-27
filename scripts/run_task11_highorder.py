"""
Task 11 - Exact higher-order census extension.

For every thresholded passing network (w0=2):
  * exact canonical (Milo-id) census for k = 6 and k = 7
    (k = 3..5 already stored in data/motifs/homogeneous_motifs.parquet);
  * census SIZE (number of weakly connected induced subgraphs) for every
    order 2 <= k <= n, i.e. the complete all-order census curve.

Outputs (data/analysis/):
  highorder_census_k67.parquet : match_id, team_side, k, canon_id, count
  census_sizes_allk.parquet    : match_id, team_side, k, n_instances
"""
import numpy as np
import pandas as pd
import itertools
import time
from pathlib import Path

NET_DIR = Path("data/networks/homogeneous")
OUT_DIR = Path("data/analysis")

PERMCACHE = {}


def perm_tables(k):
    if k in PERMCACHE:
        return PERMCACHE[k]
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    pairpos = {p: c for c, p in enumerate(pairs)}
    perms = list(itertools.permutations(range(k)))
    gather = np.array([[pairpos[(p[i], p[j])] for (i, j) in pairs] for p in perms],
                      dtype=np.int64)
    weights = np.array([1 << (k * k - 1 - (i * k + j)) for (i, j) in pairs],
                       dtype=np.int64)
    PERMCACHE[k] = (pairs, gather, weights)
    return PERMCACHE[k]


def load_A(path):
    E = [tuple(map(int, l.split()[:2])) for l in open(path)]
    if not E:
        return None
    nodes = sorted({u for e in E for u in e})
    idx = {u: i for i, u in enumerate(nodes)}
    A = np.zeros((len(nodes),) * 2, dtype=np.uint8)
    for u, v in E:
        A[idx[u], idx[v]] = 1
    return A

def connected_bits(A, k, chunk=30000):
    """Yield bit-matrices (m, k*(k-1)) of weakly connected induced k-subgraphs."""
    n = A.shape[0]
    if n < k:
        return
    pairs, _, _ = perm_tables(k) if k <= 7 else ((
        [(i, j) for i in range(k) for j in range(k) if i != j]), None, None)
    if k > 7:
        pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    combos = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64)
    eye = np.eye(k, dtype=np.uint8)
    for s in range(0, len(combos), chunk):
        cb = combos[s:s + chunk]
        bits = np.stack([A[cb[:, i], cb[:, j]] for i, j in pairs], axis=1)
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
        conn = R[:, 0, :].all(axis=1)
        yield bits[conn]


def census_canonical(A, k):
    """Exact canonical census dict {milo_id: count}."""
    pairs, gather, weights = perm_tables(k)
    out = {}
    for bits in connected_bits(A, k):
        if not len(bits):
            continue
        bits = bits.astype(np.int64)
        canon = bits @ weights
        for p_i in range(1, len(gather)):
            np.minimum(canon, bits[:, gather[p_i]] @ weights, out=canon)
        uniq, cnt = np.unique(canon, return_counts=True)
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            out[u] = out.get(u, 0) + c
    return out


def census_size(A, k):
    total = 0
    for bits in connected_bits(A, k):
        total += len(bits)
    return total


def main():
    files = sorted(NET_DIR.glob("*.edgelist"))
    print(f"{len(files)} networks", flush=True)
    canon_rows, size_rows = [], []
    t0 = time.time()
    for fi, f in enumerate(files):
        match_id, side = f.stem.split("_")
        A = load_A(f)
        if A is None:
            continue
        n = A.shape[0]
        for k in (6, 7):
            for cid, cnt in census_canonical(A, k).items():
                canon_rows.append((int(match_id), side, k, cid, cnt))
        for k in range(2, n + 1):
            size_rows.append((int(match_id), side, k, census_size(A, k)))
        if (fi + 1) % 200 == 0:
            el = time.time() - t0
            print(f"{fi+1}/{len(files)}  {el:.0f}s  eta {el/(fi+1)*(len(files)-fi-1):.0f}s",
                  flush=True)

    pd.DataFrame(canon_rows,
                 columns=["match_id", "team_side", "k", "canon_id", "count"]
                 ).to_parquet(OUT_DIR / "highorder_census_k67.parquet", index=False)
    pd.DataFrame(size_rows,
                 columns=["match_id", "team_side", "k", "n_instances"]
                 ).to_parquet(OUT_DIR / "census_sizes_allk.parquet", index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
