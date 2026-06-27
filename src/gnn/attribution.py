"""
Task 9 - Part E: Gradient-based motif attribution (Integrated Gradients).
Quantifies contribution of each motif type to GNN output.
"""
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .model import HeteroFootballGNN

logger = logging.getLogger(__name__)


def compute_integrated_gradients(
    model: HeteroFootballGNN,
    data,
    target_output: str = "reg",  # "reg" | "cls_0" | "cls_1" | "cls_2"
    n_steps: int = 50,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Compute Integrated Gradients for node features.

    IG(x) = (x - x') * integral_0^1 [dF/dx at x' + alpha*(x-x')] dalpha

    Baseline x' = zero tensor.

    Returns dict: node_type -> IG tensor (same shape as node features).
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)

    NODE_TYPES = ["home_player", "away_player"]

    # Baseline: zero features
    baseline_x = {
        nt: torch.zeros_like(data[nt].x)
        for nt in NODE_TYPES
        if hasattr(data[nt], "x") and data[nt].x is not None
    }

    original_x = {
        nt: data[nt].x.clone()
        for nt in NODE_TYPES
        if hasattr(data[nt], "x") and data[nt].x is not None
    }

    # Accumulate gradients along interpolation path
    grad_accum = {nt: torch.zeros_like(original_x[nt]) for nt in original_x}

    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        # Interpolated input
        interp_x = {
            nt: baseline_x[nt] + alpha * (original_x[nt] - baseline_x[nt])
            for nt in original_x
        }

        # Set interpolated features
        for nt in original_x:
            data[nt].x = interp_x[nt].requires_grad_(True)

        cls_logits, reg_out = model(data)

        # Select target
        if target_output == "reg":
            target = reg_out.sum()
        elif target_output.startswith("cls_"):
            cls_idx = int(target_output.split("_")[1])
            target = cls_logits[:, cls_idx].sum()
        else:
            target = reg_out.sum()

        target.backward()

        for nt in original_x:
            if data[nt].x.grad is not None:
                grad_accum[nt] += data[nt].x.grad.detach()
            data[nt].x.grad = None

    # Integrated gradients: average gradient * (input - baseline)
    ig = {}
    for nt in original_x:
        ig[nt] = (original_x[nt] - baseline_x[nt]) * (grad_accum[nt] / n_steps)

    # Restore original features
    for nt in original_x:
        data[nt].x = original_x[nt]

    return ig


def enumerate_triads_with_membership(G) -> List[Tuple[int, Tuple]]:
    """
    Enumerate every connected induced 3-node subgraph of the directed graph ``G``
    and return a list of ``(canonical_motif_id, (node_a, node_b, node_c))`` pairs.

    Unlike a bare census this keeps the *identity* of the three participating
    nodes, which is what lets us attach a genuine, node-level attribution to each
    motif occurrence (rather than the circular ``IG * |z|`` proxy).
    """
    from itertools import combinations
    from src.motifs.enumerator import canonical_motif_id

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 3:
        return []
    succ = {u: set(G.successors(u)) for u in nodes}
    und = {u: set(G.successors(u)) | set(G.predecessors(u)) for u in nodes}

    out: List[Tuple[int, Tuple]] = []
    for trio in combinations(nodes, 3):
        cset = set(trio)
        # weak-connectivity check
        seen, stack = {trio[0]}, [trio[0]]
        while stack:
            x = stack.pop()
            for y in und[x]:
                if y in cset and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != 3:
            continue
        local = {node: i for i, node in enumerate(trio)}
        edges = tuple(
            (local[a], local[b]) for a in trio for b in succ[a] if b in cset
        )
        out.append((canonical_motif_id(edges, 3), trio))
    return out


def aggregate_attribution_by_motif(
    ig_scores: Dict[str, torch.Tensor],
    player_ids: Dict[str, "torch.Tensor"],
    homo_nets: Dict[str, "object"],
    match_id: int,
) -> pd.DataFrame:
    """
    Genuine, participation-based motif attribution.

    For each side we map every player node to its integrated-gradients magnitude
    (mean |IG| over the node-feature dimensions), enumerate the triadic motifs of
    the thresholded passing network *with node membership*, and attribute to each
    motif occurrence the mean IG magnitude of its three constituent players. The
    per-motif attribution is therefore derived purely from the trained GNN and is
    independent of the z-score, so a subsequent comparison with the structural
    significance is a genuine, non-circular validation.

    Args:
        ig_scores:  {"home": IG tensor [n_home, F], "away": IG tensor [n_away, F]}
        player_ids: {"home": LongTensor [n_home], "away": LongTensor [n_away]}
        homo_nets:  {"home": DiGraph, "away": DiGraph} with player-id nodes (w0=2)
        match_id:   match identifier (kept for traceability)

    Returns DataFrame: match_id, motif_id, motif_order_k, team_side,
                       ig_attr (sum over instances), n_instances.
    """
    # The IG dict is keyed by node type ("home_player"/"away_player"); normalise
    # to the "home"/"away" side labels used by the networks and player_ids.
    ig_by_side = {("home" if "home" in k else "away"): v for k, v in ig_scores.items()}

    records = []
    for side in ("home", "away"):
        ig = ig_by_side.get(side)
        pid = player_ids.get(side)
        G = homo_nets.get(side)
        if ig is None or pid is None or G is None or G.number_of_nodes() < 3:
            continue
        node_ig = ig.abs().mean(dim=1).detach().cpu().numpy()
        pid_arr = pid.detach().cpu().numpy()
        ig_of_player = {int(p): float(node_ig[i]) for i, p in enumerate(pid_arr)}

        agg: Dict[int, list] = {}
        for mid, trio in enumerate_triads_with_membership(G):
            vals = [ig_of_player.get(int(p)) for p in trio]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            agg.setdefault(mid, []).append(float(np.mean(vals)))

        for mid, vals in agg.items():
            records.append({
                "match_id": match_id,
                "motif_id": mid,
                "motif_order_k": 3,
                "team_side": side,
                "ig_attr": float(np.sum(vals)),
                "n_instances": len(vals),
            })
    return pd.DataFrame(records)


def _load_homo_nets(homo_net_dir, match_id: int) -> Dict[str, "object"]:
    """Load the thresholded (w0=2) home/away passing networks for a match."""
    import pickle
    from pathlib import Path
    out = {}
    for side in ("home", "away"):
        p = Path(homo_net_dir) / f"{match_id}_{side}.gpickle"
        if p.exists():
            with open(p, "rb") as f:
                out[side] = pickle.load(f)
    return out


def compute_population_attribution(
    model: HeteroFootballGNN,
    test_loader,
    homo_net_dir: str,
    zscore_df: Optional[pd.DataFrame] = None,
    regression_df: Optional[pd.DataFrame] = None,
    device: str = "cpu",
    n_samples: int = 10_000,
    n_steps: int = 25,
) -> pd.DataFrame:
    """
    Genuine motif attribution averaged across the test graphs.

    For every test match the integrated gradients of the node features are
    aggregated onto the triadic motifs through explicit node participation
    (:func:`aggregate_attribution_by_motif`). The per-motif attribution is then
    averaged over all occurrences across the test set and merged with the
    structural z-score and the OLS coefficient so that Fig. 17 can compare the
    three independent rankings honestly.

    Returns a summary DataFrame for Fig. 17.
    """
    model.eval()
    all_records = []
    sample_count = 0

    for batch in test_loader:
        n_in_batch = batch.num_graphs
        for i in range(n_in_batch):
            if sample_count >= n_samples:
                break
            try:
                single = batch.get_example(i)
                match_id = int(single.match_id) if hasattr(single, "match_id") else -1
                ig_scores = compute_integrated_gradients(
                    model, single, target_output="reg", device=device, n_steps=n_steps
                )
                player_ids = {
                    "home": single["home_player"].player_ids,
                    "away": single["away_player"].player_ids,
                }
                homo_nets = _load_homo_nets(homo_net_dir, match_id)
                rec = aggregate_attribution_by_motif(
                    ig_scores, player_ids, homo_nets, match_id
                )
                if not rec.empty:
                    all_records.append(rec)
                sample_count += 1
            except Exception as e:
                logger.warning(f"Attribution failed for sample {sample_count}: {e}")
        if sample_count >= n_samples:
            break

    if not all_records:
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    grp = combined.groupby(["motif_id", "motif_order_k", "team_side"])
    summary = grp.agg(
        ig_attr_sum=("ig_attr", "sum"),
        n_instances=("n_instances", "sum"),
        std_attribution=("ig_attr", "std"),
    ).reset_index()
    # Instance-weighted mean integrated-gradients magnitude per motif class.
    summary["mean_attribution"] = summary["ig_attr_sum"] / summary["n_instances"]

    if zscore_df is not None and "z" in zscore_df.columns:
        summary = summary.merge(
            zscore_df[["motif_id", "motif_order_k", "team_side", "z"]]
            .rename(columns={"z": "mean_z"}),
            on=["motif_id", "motif_order_k", "team_side"], how="left",
        )
    if regression_df is not None and "beta" in regression_df.columns:
        reg = regression_df[regression_df.get("panel", "") == "A_k3_homo"].copy()
        if not reg.empty:
            reg = reg.dropna(subset=["motif_id"])
            reg["motif_id"] = reg["motif_id"].astype(int)
            summary = summary.merge(
                reg[["motif_id", "team_side", "beta"]],
                on=["motif_id", "team_side"], how="left",
            )
    return summary
