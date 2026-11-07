"""Revision checks for checkpoints reused across Information Sciences tasks."""
from pathlib import Path
from typing import Optional

import pandas as pd


def validated_ablation_checkpoint(
    output_dir: Path,
    variant: str,
    seed: int,
    model_revision: str,
    split_revision: str,
    feature_revision: str,
    graph_revision: str,
    loss_revision: str,
    class_weight_revision: str,
) -> Optional[Path]:
    """Return a checkpoint only when a matching result row proves provenance.

    State-dict files do not contain experiment metadata.  Requiring a unique
    result row with the expected model and split revisions prevents an older
    same-shaped checkpoint from being silently labelled as a newer method.
    """
    output_dir = Path(output_dir)
    result_path = output_dir / "ablation_results.parquet"
    if not result_path.exists():
        result_path = output_dir / "ablation_results_partial.parquet"
    checkpoint = (output_dir / "checkpoints_ablation" /
                  f"{variant}_seed{int(seed)}.pt")
    if not result_path.exists() or not checkpoint.exists():
        return None

    frame = pd.read_parquet(result_path)
    required = {"variant", "seed", "model_revision", "split_revision",
                "feature_revision", "graph_revision", "loss_revision",
                "class_weight_revision"}
    if not required.issubset(frame.columns):
        return None
    rows = frame[
        frame["variant"].astype(str).eq(variant)
        & pd.to_numeric(frame["seed"], errors="coerce").eq(int(seed))
        & frame["model_revision"].astype(str).eq(model_revision)
        & frame["split_revision"].astype(str).eq(split_revision)
        & frame["feature_revision"].astype(str).eq(feature_revision)
        & frame["graph_revision"].astype(str).eq(graph_revision)
        & frame["loss_revision"].astype(str).eq(loss_revision)
        & frame["class_weight_revision"].astype(str).eq(
            class_weight_revision)
    ]
    return checkpoint if len(rows) == 1 else None


def validated_temporal_checkpoint(
    output_dir: Path,
    cutoff: int,
    model: str,
    seed: int,
    model_revision: str,
    split_revision: str,
    graph_revision: str,
    base_graph_revision: str,
    score_timeline_revision: str,
    score_context_revision: str,
    loss_revision: str,
    class_weight_revision: str,
) -> Optional[Path]:
    """Return a temporal checkpoint only with exact result-row provenance."""
    output_dir = Path(output_dir)
    partial = output_dir / "temporal_results_partial.parquet"
    final = output_dir / "temporal_results.parquet"
    result_path = partial if partial.exists() else final
    checkpoint = (output_dir / "checkpoints_temporal" /
                  f"t{int(cutoff)}_{model}_seed{int(seed)}.pt")
    if not result_path.exists() or not checkpoint.exists():
        return None

    frame = pd.read_parquet(result_path)
    if "model" in frame:
        frame["model"] = frame["model"].replace({
            "MCAR-Decomp": "SPARC-GNN",
            "MCAR-Decomp+Score": "SPARC-GNN+Score",
        })
    required = {
        "cutoff", "model", "seed", "model_revision", "split_revision",
        "graph_revision", "base_graph_revision", "score_timeline_revision",
        "score_context_revision", "loss_revision", "class_weight_revision",
    }
    if not required.issubset(frame.columns):
        return None
    rows = frame[
        pd.to_numeric(frame["cutoff"], errors="coerce").eq(int(cutoff))
        & frame["model"].astype(str).eq(model)
        & pd.to_numeric(frame["seed"], errors="coerce").eq(int(seed))
        & frame["model_revision"].astype(str).eq(model_revision)
        & frame["split_revision"].astype(str).eq(split_revision)
        & frame["graph_revision"].astype(str).eq(graph_revision)
        & frame["base_graph_revision"].astype(str).eq(base_graph_revision)
        & frame["score_timeline_revision"].astype(str).eq(
            score_timeline_revision)
        & frame["score_context_revision"].astype(str).eq(
            score_context_revision)
        & frame["loss_revision"].astype(str).eq(loss_revision)
        & frame["class_weight_revision"].astype(str).eq(
            class_weight_revision)
    ]
    return checkpoint if len(rows) == 1 else None
