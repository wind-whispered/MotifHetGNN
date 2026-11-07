from pathlib import Path

import pandas as pd

from scripts.checkpoint_provenance import validated_temporal_checkpoint


EXPECTED = {
    "cutoff": 45,
    "model": "SPARC-GNN+Score",
    "seed": 7,
    "model_revision": "sparc-v3",
    "split_revision": "chrono-v2",
    "graph_revision": "prefix-v5",
    "base_graph_revision": "base-v2",
    "score_timeline_revision": "score-v2",
    "score_context_revision": "context-v2",
    "loss_revision": "loss-v2",
    "class_weight_revision": "weights-v1",
}


def call_validator(root: Path, **overrides):
    values = {**EXPECTED, **overrides}
    return validated_temporal_checkpoint(
        root,
        values["cutoff"], values["model"], values["seed"],
        values["model_revision"], values["split_revision"],
        values["graph_revision"], values["base_graph_revision"],
        values["score_timeline_revision"],
        values["score_context_revision"], values["loss_revision"],
        values["class_weight_revision"],
    )


def test_temporal_checkpoint_requires_exact_revision_row(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints_temporal"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "t45_SPARC-GNN+Score_seed7.pt"
    checkpoint.touch()
    pd.DataFrame([EXPECTED]).to_parquet(
        tmp_path / "temporal_results_partial.parquet", index=False)

    assert call_validator(tmp_path) == checkpoint
    assert call_validator(tmp_path, graph_revision="stale") is None


def test_temporal_checkpoint_rejects_duplicate_provenance_rows(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints_temporal"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "t45_SPARC-GNN+Score_seed7.pt").touch()
    pd.DataFrame([EXPECTED, EXPECTED]).to_parquet(
        tmp_path / "temporal_results_partial.parquet", index=False)

    assert call_validator(tmp_path) is None
