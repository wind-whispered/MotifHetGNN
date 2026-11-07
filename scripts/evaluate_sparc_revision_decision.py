"""Apply the validation-only, predeclared SPARC v3/v4 decision rule."""
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.dataset import CHRONOLOGICAL_SPLIT_REVISION
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.gnn.mcar_model import SPARC_REVISION
from src.gnn.trainer import CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION


SEEDS = {42, 7, 13, 21, 37}
FULL = "MCAR-Decomp-FixedLoss"
CONTROL = "MotifGate-ObservedOnly"


def evaluate():
    output_dir = ROOT / "data" / "information_sciences"
    final = output_dir / "ablation_results.parquet"
    partial = output_dir / "ablation_results_partial.parquet"
    source = final if final.exists() else partial
    if not source.exists():
        payload = {"status": "PENDING", "reason": "missing ablation results"}
    else:
        frame = pd.read_parquet(source)
        required = {
            "variant", "seed", "model_revision", "split_revision",
            "feature_revision", "graph_revision", "loss_revision",
            "class_weight_revision", "best_val_task_loss",
        }
        missing_columns = sorted(required.difference(frame.columns))
        if missing_columns:
            payload = {
                "status": "PENDING",
                "reason": f"missing columns: {missing_columns}",
            }
        else:
            selected = frame[
                frame["variant"].isin({FULL, CONTROL})
                & frame["seed"].isin(SEEDS)
                & (frame["split_revision"] == CHRONOLOGICAL_SPLIT_REVISION)
                & (frame["feature_revision"] ==
                   PRIMARY_FULL_MATCH_FEATURE_REVISION)
                & (frame["graph_revision"] ==
                   HETEROGENEOUS_GRAPH_REVISION)
                & (frame["loss_revision"] == FIXED_LOSS_REVISION)
                & (frame["class_weight_revision"] == CLASS_WEIGHT_REVISION)
            ].copy()
            selected = selected[
                (selected["variant"] == CONTROL)
                | (selected["model_revision"] == SPARC_REVISION)]
            pivot = selected.pivot_table(
                index="seed", columns="variant",
                values="best_val_task_loss", aggfunc="last")
            complete_seeds = (set(pivot.dropna().index)
                              if {FULL, CONTROL}.issubset(pivot.columns)
                              else set())
            missing_seeds = sorted(SEEDS - complete_seeds)
            if missing_seeds:
                payload = {
                    "status": "PENDING",
                    "evaluated_model_revision": SPARC_REVISION,
                    "missing_seeds": missing_seeds,
                    "complete_seeds": sorted(complete_seeds),
                }
            else:
                pivot = pivot.loc[sorted(SEEDS)]
                delta = pivot[FULL] - pivot[CONTROL]
                worse_seeds = int((delta > 0).sum())
                mean_delta = float(delta.mean())
                trigger = worse_seeds >= 4 and mean_delta > 0
                payload = {
                    "status": "READY",
                    "evaluated_model_revision": SPARC_REVISION,
                    "class_weight_revision": CLASS_WEIGHT_REVISION,
                    "metric": "best_val_task_loss_at_selected_checkpoint",
                    "test_metrics_used": False,
                    "criterion": (
                        "trigger v4 iff full decomposition is worse in at "
                        "least 4/5 paired seeds and mean paired delta > 0"),
                    "worse_seed_count": worse_seeds,
                    "mean_delta_full_minus_control": mean_delta,
                    "trigger_v4": bool(trigger),
                    "paired_rows": [
                        {
                            "seed": int(seed),
                            "full": float(pivot.loc[seed, FULL]),
                            "control": float(pivot.loc[seed, CONTROL]),
                            "delta": float(delta.loc[seed]),
                        }
                        for seed in sorted(SEEDS)
                    ],
                }
    output = output_dir / "sparc_revision_decision.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output


def main():
    payload, output = evaluate()
    print(json.dumps(payload, indent=2))
    print(output)


if __name__ == "__main__":
    main()
