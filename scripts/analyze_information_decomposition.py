"""Summarise shared/private/availability/conflict components of SPARC-GNN."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             FootballHeteroDataset,
                             split_dataset_by_metadata)
from src.gnn.mcar_model import MCARFootballGNN, SPARC_REVISION
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.trainer import CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from scripts.checkpoint_provenance import validated_ablation_checkpoint

SEEDS = (42, 7, 13, 21, 37)
RELATIONS = ("pass", "adversarial", "turnover")
PAIRS = ("pass-adversarial", "pass-turnover", "adversarial-turnover")


def main():
    out = ROOT / "data" / "information_sciences"
    base = FootballHeteroDataset(
        root=str(ROOT), use_motif_context=True,
        required_graph_revision=HETEROGENEOUS_GRAPH_REVISION)
    _, _, test_ids = split_dataset_by_metadata(
        base, str(ROOT / "data/processed/matches_meta.parquet"),
        strategy="chronological")
    graph_by_id = {mid: base[i] for i, mid in enumerate(base.match_ids)}
    loader = DataLoader([graph_by_id[mid] for mid in test_ids if mid in graph_by_id],
                        batch_size=64, shuffle=False)
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in SEEDS:
        # Diagnostics describe the fixed-loss main specification; comparative
        # claims are made only after the current-revision ablation is complete.
        path = validated_ablation_checkpoint(
            out, "MCAR-Decomp-FixedLoss", seed, SPARC_REVISION,
            CHRONOLOGICAL_SPLIT_REVISION,
            PRIMARY_FULL_MATCH_FEATURE_REVISION,
            HETEROGENEOUS_GRAPH_REVISION,
            FIXED_LOSS_REVISION,
            CLASS_WEIGHT_REVISION)
        if path is None:
            continue
        model = MCARFootballGNN(hidden_dim=64, num_layers=3, dropout=.3,
                               use_info_decomposition=True,
                               use_assist_flags=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device,
                                         weights_only=True))
        model.eval()
        accum = {nt: {"shared": [], "private": [], "availability": [],
                      "pair_available": [], "conflict": [],
                      "private_conflict": [], "gate": []}
                 for nt in model.NODE_TYPES}
        with torch.no_grad():
            for batch in loader:
                model(batch.to(device))
                for nt in model.NODE_TYPES:
                    dec = model.last_decomposition[nt]
                    accum[nt]["shared"].append(dec["shared_norm"].cpu())
                    accum[nt]["private"].append(dec["private_norm"].cpu())
                    accum[nt]["availability"].append(
                        dec["availability"].cpu())
                    accum[nt]["pair_available"].append(
                        dec["pair_available"].cpu())
                    accum[nt]["conflict"].append(dec["conflict"].cpu())
                    accum[nt]["private_conflict"].append(
                        dec["private_conflict"].cpu())
                    accum[nt]["gate"].append(model.last_gate_weights[nt].cpu())
        for nt in model.NODE_TYPES:
            shared = torch.cat(accum[nt]["shared"]).numpy()
            private = torch.cat(accum[nt]["private"]).numpy()
            availability = torch.cat(accum[nt]["availability"]).numpy()
            pair_available = torch.cat(
                accum[nt]["pair_available"]).numpy().astype(bool)
            conflict = torch.cat(accum[nt]["conflict"]).numpy()
            private_conflict = torch.cat(
                accum[nt]["private_conflict"]).numpy()
            gate = torch.cat(accum[nt]["gate"]).numpy()
            observed_any = availability.any(axis=1)
            shared_observed = shared[observed_any]
            rows.append({"seed": seed, "node_type": nt, "quantity": "shared_norm",
                         "component": "shared",
                         "mean": (shared_observed.mean()
                                  if len(shared_observed) else np.nan),
                         "std": (shared_observed.std()
                                 if len(shared_observed) else np.nan),
                         "n": len(shared_observed)})
            denominator = shared + private.sum(axis=1)
            shared_fraction = np.divide(
                shared, denominator, out=np.zeros_like(shared),
                where=denominator > 0)[observed_any]
            rows.append({"seed": seed, "node_type": nt,
                         "quantity": "shared_fraction",
                         "component": "shared",
                         "mean": (shared_fraction.mean()
                                  if len(shared_fraction) else np.nan),
                         "std": (shared_fraction.std()
                                 if len(shared_fraction) else np.nan),
                         "n": len(shared_fraction)})
            safe_gate = np.clip(gate[observed_any], 1e-12, 1.0)
            gate_entropy = -(gate[observed_any] * np.log(safe_gate)).sum(
                axis=1)
            for quantity, values in (
                    ("gate_entropy", gate_entropy),
                    ("effective_relation_count", np.exp(gate_entropy))):
                rows.append({"seed": seed, "node_type": nt,
                             "quantity": quantity,
                             "component": "observed_relations",
                             "mean": values.mean() if len(values) else np.nan,
                             "std": values.std() if len(values) else np.nan,
                             "n": len(values)})
            multi_observed = availability.sum(axis=1) >= 2
            safe_multi_gate = np.clip(gate[multi_observed], 1e-12, 1.0)
            multi_entropy = -(
                gate[multi_observed] * np.log(safe_multi_gate)).sum(axis=1)
            for quantity, values in (
                    ("gate_entropy_multi", multi_entropy),
                    ("effective_relation_count_multi",
                     np.exp(multi_entropy))):
                rows.append({"seed": seed, "node_type": nt,
                             "quantity": quantity,
                             "component": "multi_relation_nodes",
                             "mean": values.mean() if len(values) else np.nan,
                             "std": values.std() if len(values) else np.nan,
                             "n": len(values)})
            for i, rel in enumerate(RELATIONS):
                observed = availability[:, i].astype(bool)
                for quantity, values in (
                        ("private_norm", private[observed, i]),
                        ("gate_weight", gate[observed, i]),
                        ("gate_weight_multi_relation_nodes",
                         gate[multi_observed & observed, i]),
                        ("gate_mass_multi_relation_nodes",
                         gate[multi_observed, i]),
                        ("gate_mass_all_nodes", gate[:, i]),
                        ("relation_availability", availability[:, i])):
                    rows.append({"seed": seed, "node_type": nt,
                                 "quantity": quantity, "component": rel,
                                 "mean": values.mean() if len(values) else np.nan,
                                 "std": values.std() if len(values) else np.nan,
                                 "n": len(values)})
            for i, pair in enumerate(PAIRS):
                observed = pair_available[:, i]
                values = conflict[observed, i]
                rows.append({"seed": seed, "node_type": nt,
                             "quantity": "conflict", "component": pair,
                             "mean": values.mean() if len(values) else np.nan,
                             "std": values.std() if len(values) else np.nan,
                             "n": len(values)})
                values = private_conflict[observed, i]
                rows.append({"seed": seed, "node_type": nt,
                             "quantity": "private_conflict",
                             "component": pair,
                             "mean": values.mean() if len(values) else np.nan,
                             "std": values.std() if len(values) else np.nan,
                             "n": len(values)})
    frame = pd.DataFrame(rows)
    frame["model_revision"] = SPARC_REVISION
    frame["split_revision"] = CHRONOLOGICAL_SPLIT_REVISION
    frame["feature_revision"] = PRIMARY_FULL_MATCH_FEATURE_REVISION
    frame["graph_revision"] = HETEROGENEOUS_GRAPH_REVISION
    frame["loss_revision"] = FIXED_LOSS_REVISION
    frame["class_weight_revision"] = CLASS_WEIGHT_REVISION
    frame.to_parquet(out / "decomposition_diagnostics.parquet", index=False)
    print(frame.groupby(["quantity", "component"])["mean"].agg(
        ["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
