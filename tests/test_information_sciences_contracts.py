"""Regression tests for the Information Sciences experiment contract."""

import pandas as pd
import torch

from src.gnn.dataset import (CHRONOLOGICAL_SPLIT_REVISION,
                             split_dataset_by_metadata)

from scripts.run_information_sciences_generalization import (
    DEFAULT_GENERALIZATION_MODELS, NON_TEMPORAL_SPLIT_REVISION,
    audit_split,
    load_existing_rows,
    reuse_chronological_rows,
)
from scripts.run_information_sciences_models import (
    INVALID_FULL_MATCH_MODELS,
    MODELS,
    _clean_rows,
)
from scripts.run_information_sciences_ablation import VARIANTS
from scripts.run_information_sciences_relation_controls import (
    degree_marginal_permutation,
)
from scripts.run_temporal_cutoff_experiment import (
    DEFAULT_TEMPORAL_MODELS, SCORE_LOSS_REVISION, TEMPORAL_GRAPH_REVISION,
    load_existing_rows as load_temporal_rows,
)
from scripts.checkpoint_provenance import validated_ablation_checkpoint
from src.gnn.mcar_model import (SPARC_REVISION, masked_pairwise_conflict,
                                masked_relation_softmax,
                                observed_relation_mean)
from src.gnn.score_context import SCORE_CONTEXT_REVISION
from src.gnn.evaluator import evaluate_on_test
from src.gnn.trainer import (
    CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION,
    UNCERTAINTY_LOSS_REVISION, GNNTrainer, training_class_weights,
)
from src.gnn.feature_contracts import PRIMARY_FULL_MATCH_FEATURE_REVISION
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.temporal import SCORE_TIMELINE_REVISION


def test_stress_tests_retain_strongest_graph_and_nested_score_baseline():
    assert "HeteroSAGE-sum" in DEFAULT_GENERALIZATION_MODELS
    assert "HeteroSAGE-sum" in DEFAULT_TEMPORAL_MODELS
    assert "HeteroSAGE-sum+Score" in DEFAULT_TEMPORAL_MODELS
    assert "TeamPool-MLP+Score" in DEFAULT_TEMPORAL_MODELS


def test_checkpoint_reuse_requires_matching_result_provenance(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints_ablation"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "MCAR-Decomp-FixedLoss_seed42.pt"
    torch.save({}, checkpoint)
    pd.DataFrame([{
        "variant": "MCAR-Decomp-FixedLoss",
        "seed": 42,
        "model_revision": SPARC_REVISION,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
        "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
    }]).to_parquet(tmp_path / "ablation_results_partial.parquet", index=False)

    assert validated_ablation_checkpoint(
        tmp_path, "MCAR-Decomp-FixedLoss", 42, SPARC_REVISION,
        CHRONOLOGICAL_SPLIT_REVISION,
        PRIMARY_FULL_MATCH_FEATURE_REVISION,
        HETEROGENEOUS_GRAPH_REVISION,
        FIXED_LOSS_REVISION,
        CLASS_WEIGHT_REVISION) == checkpoint
    assert validated_ablation_checkpoint(
        tmp_path, "MCAR-Decomp-FixedLoss", 42, "stale_revision",
        CHRONOLOGICAL_SPLIT_REVISION,
        PRIMARY_FULL_MATCH_FEATURE_REVISION,
        HETEROGENEOUS_GRAPH_REVISION,
        FIXED_LOSS_REVISION,
        CLASS_WEIGHT_REVISION) is None


def test_training_class_weights_are_train_only_sqrt_inverse_frequency():
    # Home/draw/away counts are 9/1/4. The mean-normalised weights must
    # preserve inverse square-root ordering without using validation labels.
    weights = training_class_weights([1.0] * 9 + [0.0] + [-1.0] * 4)
    expected = torch.sqrt(torch.tensor(14.0) / (
        3.0 * torch.tensor([9.0, 1.0, 4.0])))
    expected = expected / expected.mean()

    assert torch.allclose(weights, expected)
    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert weights[1] > weights[2] > weights[0]


def test_trainer_uses_declared_class_weights():
    weights = torch.tensor([0.8, 1.2, 1.0])
    trainer = GNNTrainer(
        torch.nn.Linear(1, 1), device="cpu", class_weights=weights)
    assert torch.equal(trainer.cls_criterion.weight, weights)


def test_absent_relation_is_not_labelled_as_pairwise_conflict():
    messages = torch.tensor([
        [[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
    ])
    availability = torch.tensor([
        [True, False, True],
        [True, True, True],
    ])

    conflict, pair_available = masked_pairwise_conflict(
        messages, availability)

    # Row 0: pass-adversarial and adversarial-turnover are unavailable, not
    # maximally conflicting. The observed pass-turnover pair is opposing.
    assert torch.equal(pair_available[0], torch.tensor([False, True, False]))
    assert torch.allclose(conflict[0], torch.tensor([0.0, 2.0, 0.0]))
    # Row 1 contains all pairs: aligned, opposing, opposing.
    assert torch.allclose(conflict[1], torch.tensor([0.0, 2.0, 2.0]))


def test_absent_relation_receives_zero_fusion_weight():
    logits = torch.tensor([[1.0, 8.0, 3.0], [1.0, 2.0, 3.0]])
    availability = torch.tensor([[True, False, True],
                                 [False, False, False]])
    weights = masked_relation_softmax(logits, availability)
    assert weights[0, 1].item() == 0.0
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0))
    assert torch.equal(weights[1], torch.zeros(3))


def test_observed_only_gate_is_a_matched_semantic_absence_control():
    flags = VARIANTS["MotifGate-ObservedOnly"]
    assert flags["use_motif_context"] is True
    assert flags["adaptive_gate"] is True
    assert flags["mask_unavailable_relations"] is True
    assert not flags.get("use_info_decomposition", False)


def test_no_orthogonality_control_changes_only_auxiliary_weight():
    full = VARIANTS["MCAR-Decomp-FixedLoss"]
    no_orth = VARIANTS["MCAR-Decomp-NoOrthogonality"]
    structural_keys = {
        "use_edge_attr", "use_motif_context", "adaptive_gate",
        "use_info_decomposition", "loss_mode",
    }
    assert all(full[key] == no_orth[key] for key in structural_keys)
    assert no_orth["auxiliary_weight"] == 0.0


def test_primary_full_match_variants_mask_retrospective_assist_flags():
    for name, flags in VARIANTS.items():
        if name == "MCAR-Decomp-WithAssistProxy":
            assert flags["use_assist_flags"] is True
        else:
            assert flags["use_assist_flags"] is False


def test_absent_biased_messages_are_removed_before_decomposition():
    messages = torch.tensor([
        [[3.0, 0.0], [9.0, 9.0], [1.0, 2.0]],
        [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
    ])
    availability = torch.tensor([
        [True, False, True],
        [False, False, False],
    ])

    masked, mean, has_relation = observed_relation_mean(
        messages, availability)

    assert torch.equal(masked[0, 1], torch.zeros(2))
    assert torch.allclose(mean[0], torch.tensor([2.0, 1.0]))
    assert torch.equal(masked[1], torch.zeros((3, 2)))
    assert torch.equal(mean[1], torch.zeros(2))
    assert torch.equal(has_relation, torch.tensor([[True], [False]]))


def test_shared_private_projections_do_not_inject_affine_offsets():
    from src.gnn.mcar_model import MCARFootballGNN

    model = MCARFootballGNN(
        hidden_dim=8, num_layers=1, use_info_decomposition=True)
    projections = [
        *model.shared_projs[0].values(),
        *[projection for group in model.private_projs[0].values()
          for projection in group],
        *model.conflict_projs[0].values(),
        *model.availability_projs[0].values(),
    ]
    assert projections
    assert all(projection.bias is None for projection in projections)


def test_no_assist_control_masks_only_pass_assist_channels():
    from src.gnn.mcar_model import MCARFootballGNN

    model = MCARFootballGNN(hidden_dim=8, num_layers=1)
    assert model.use_assist_flags is False
    original = torch.ones((2, model.EDGE_DIMS["pass"]))
    prepared = model._prepare_edge_attributes("pass", original)

    assert torch.equal(prepared[:, :10], torch.ones((2, 10)))
    assert torch.equal(prepared[:, 10:12], torch.zeros((2, 2)))
    assert torch.equal(prepared[:, 12:], torch.ones((2, 1)))
    assert torch.equal(original, torch.ones_like(original))


def test_relation_empty_nodes_have_zero_shared_and_private_evidence():
    from torch_geometric.data import HeteroData
    from src.gnn.mcar_model import MCARFootballGNN

    model = MCARFootballGNN(
        hidden_dim=8, num_layers=1, dropout=0.0,
        use_info_decomposition=True).eval()
    graph = HeteroData()
    for node_type in model.NODE_TYPES:
        graph[node_type].x = torch.randn(2, 39)
        graph[node_type].motif_x = torch.zeros(2, 7)
    for edge_type in model.EDGE_TYPES:
        relation = edge_type[1]
        graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph[edge_type].edge_attr = torch.empty(
            (0, model.EDGE_DIMS[relation]))

    model.encode(graph)

    for diagnostics in model.last_decomposition.values():
        assert torch.equal(
            diagnostics["shared_norm"],
            torch.zeros_like(diagnostics["shared_norm"]))
        assert torch.equal(
            diagnostics["private_norm"],
            torch.zeros_like(diagnostics["private_norm"]))


def test_uncertainty_task_weights_are_optimised_and_reportable():
    model = torch.nn.Linear(1, 1)
    trainer = GNNTrainer(model, device="cpu", loss_mode="uncertainty")
    optimiser_parameters = {
        id(parameter) for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(trainer.task_log_vars) in optimiser_parameters

    logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    regression = torch.tensor([[0.25]], requires_grad=True)
    target = torch.tensor([[1.0]])
    loss, _, _ = trainer._compute_loss(logits, regression, target)
    loss.backward()
    assert trainer.task_log_vars.grad is not None
    assert torch.isfinite(trainer.task_log_vars.grad).all()


def test_validation_task_loss_excludes_auxiliary_penalty():
    class AuxiliaryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return (torch.tensor([[1.0, 0.0, -1.0]]) + self.anchor,
                    torch.tensor([[0.25]]) + self.anchor)

        def auxiliary_loss(self):
            return torch.tensor(2.0)

    class Batch:
        y = torch.tensor([[1.0]])

        def to(self, device):
            return self

    trainer = GNNTrainer(
        AuxiliaryModel(), device="cpu", auxiliary_weight=0.01)
    metrics = trainer.validate_epoch([Batch()])
    assert abs(metrics["loss"] - metrics["task_loss"] - 0.02) < 1e-7
    assert metrics["auxiliary_loss"] == 2.0


def test_uncertainty_loss_starts_at_the_fixed_loss_scale():
    fixed = GNNTrainer(
        torch.nn.Linear(1, 1), device="cpu", loss_mode="fixed")
    uncertainty = GNNTrainer(
        torch.nn.Linear(1, 1), device="cpu", loss_mode="uncertainty")
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    regression = torch.tensor([[0.25]])
    target = torch.tensor([[1.0]])

    fixed_loss, _, _ = fixed._compute_loss(logits, regression, target)
    uncertainty_loss, _, _ = uncertainty._compute_loss(
        logits, regression, target)

    assert torch.allclose(uncertainty_loss, fixed_loss)
    assert FIXED_LOSS_REVISION != UNCERTAINTY_LOSS_REVISION


def test_single_match_batch_is_valid_for_evaluation():
    class Batch:
        y = torch.tensor([[0.0]])
        match_id = torch.tensor([123])

        def to(self, device):
            return self

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return (torch.tensor([[0.0, 1.0, -1.0]]),
                    torch.tensor([[0.25]]))

    result = evaluate_on_test(Model(), [Batch()], device="cpu")
    assert len(result["predictions_df"]) == 1
    assert result["predictions_df"].iloc[0]["match_id"] == 123


def test_pass_permutation_preserves_degrees_without_creating_self_loops():
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ])
    order, accepted = degree_marginal_permutation(
        edge_index, torch.Generator().manual_seed(42))
    permuted = torch.stack(
        [edge_index[0], edge_index[1, order]], dim=0)

    assert torch.equal(torch.sort(order).values, torch.arange(order.numel()))
    assert torch.equal(
        torch.bincount(permuted[0]), torch.bincount(edge_index[0]))
    assert torch.equal(
        torch.bincount(permuted[1]), torch.bincount(edge_index[1]))
    assert not bool((permuted[0] == permuted[1]).any())
    assert len(set(map(tuple, permuted.t().tolist()))) == permuted.size(1)
    assert not torch.equal(permuted, edge_index)
    assert accepted > 0


def test_full_match_model_registry_excludes_score_leakage():
    """Final scores are labels for complete-match graphs, not predictors."""
    assert INVALID_FULL_MATCH_MODELS.isdisjoint(MODELS)


def test_main_result_cleaner_removes_score_rows_and_canonicalises_method():
    rows = [
        {"model": "CurrentScore", "seed": -1, "accuracy": 1.0},
        {"model": "MCAR-Decomp", "seed": 42, "accuracy": 0.6,
         "model_revision": SPARC_REVISION,
         "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
         "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
         "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION},
        {"model": "HGT", "seed": 42, "accuracy": 0.5,
         "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
         "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
         "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION},
    ]
    cleaned = _clean_rows(rows)
    assert {row["model"] for row in cleaned} == {"SPARC-GNN", "HGT"}


def test_generalization_resume_rows_are_canonical_and_unique(tmp_path):
    path = tmp_path / "partial.parquet"
    pd.DataFrame([
        {"protocol": "random", "model": "MCAR-Decomp", "seed": 42,
         "accuracy": 0.5, "model_revision": SPARC_REVISION,
         "split_revision": NON_TEMPORAL_SPLIT_REVISION,
         "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
         "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION},
        {"protocol": "random", "model": "SPARC-GNN", "seed": 42,
         "accuracy": 0.6, "model_revision": SPARC_REVISION,
         "split_revision": NON_TEMPORAL_SPLIT_REVISION,
         "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
         "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION},
        {"protocol": "random", "model": "CurrentScore", "seed": -1,
         "accuracy": 1.0,
         "split_revision": NON_TEMPORAL_SPLIT_REVISION},
    ]).to_parquet(path, index=False)

    cleaned = load_existing_rows(path)

    assert len(cleaned) == 1
    assert cleaned[0]["model"] == "SPARC-GNN"
    assert cleaned[0]["accuracy"] == 0.6


def test_generalization_reuses_main_chronological_rows(tmp_path):
    pd.DataFrame([{
        "model": "HGT", "seed": 42, "best_val_loss": 1.2,
        "accuracy": 0.5, "macro_f1": 0.4, "draw_f1": 0.1,
        "mae": 1.5, "r2": 0.2,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
        "feature_revision": PRIMARY_FULL_MATCH_FEATURE_REVISION,
        "graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
        "class_weight_home": 0.8,
        "class_weight_draw": 1.2,
        "class_weight_away": 1.0,
        "n_train": 2426, "n_validation": 346, "n_test": 692,
    }]).to_parquet(tmp_path / "model_results.parquet", index=False)

    rows = reuse_chronological_rows([], tmp_path)

    assert len(rows) == 1
    assert rows[0]["protocol"] == "chronological"
    assert rows[0]["model"] == "HGT"
    assert rows[0]["n_test"] == 692
    assert rows[0]["class_weight_home"] == 0.8


def test_team_grouped_audit_rejects_identity_overlap():
    metadata = pd.DataFrame([
        {"match_id": 1, "home_team_id": 10, "away_team_id": 11,
         "competition_name": "A", "match_date": "2020-01-01"},
        {"match_id": 2, "home_team_id": 11, "away_team_id": 12,
         "competition_name": "A", "match_date": "2020-01-02"},
        {"match_id": 3, "home_team_id": 20, "away_team_id": 21,
         "competition_name": "B", "match_date": "2020-01-03"},
    ]).set_index("match_id")

    try:
        audit_split("team_grouped", ([1], [2], [3]), metadata)
    except ValueError as exc:
        assert "team identity leakage" in str(exc)
    else:
        raise AssertionError("overlapping team identity was not rejected")


def test_chronological_split_keeps_complete_dates_together(tmp_path):
    class DatasetStub:
        match_ids = list(range(1, 13))

    metadata = pd.DataFrame({
        "match_id": range(1, 13),
        "match_date": (
            ["2020-01-01"] * 3 + ["2020-01-02"] * 2
            + ["2020-01-03"] * 4 + ["2020-01-04"] * 3),
        "competition_name": ["A"] * 12,
        "home_team_id": range(101, 113),
        "away_team_id": range(201, 213),
    })
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)

    split = split_dataset_by_metadata(
        DatasetStub(), str(path), strategy="chronological",
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    dates = metadata.set_index("match_id")["match_date"]
    date_sets = [set(dates.loc[ids]) for ids in split]

    assert all(date_sets[i].isdisjoint(date_sets[j])
               for i in range(3) for j in range(i + 1, 3))
    assert max(date_sets[0]) < min(date_sets[1])
    assert max(date_sets[1]) < min(date_sets[2])


def test_temporal_resume_canonicalises_legacy_method_names(tmp_path):
    path = tmp_path / "temporal.parquet"
    pd.DataFrame([
        {"cutoff": 15, "model": "MCAR-Decomp", "seed": 42,
         "macro_f1": 0.4, "model_revision": SPARC_REVISION,
         "graph_revision": TEMPORAL_GRAPH_REVISION,
         "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "score_timeline_revision": SCORE_TIMELINE_REVISION,
         "score_context_revision": SCORE_CONTEXT_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION,
         "split_revision": CHRONOLOGICAL_SPLIT_REVISION},
        {"cutoff": 15, "model": "SPARC-GNN", "seed": 42,
         "macro_f1": 0.5, "model_revision": SPARC_REVISION,
         "graph_revision": TEMPORAL_GRAPH_REVISION,
         "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "score_timeline_revision": SCORE_TIMELINE_REVISION,
         "score_context_revision": SCORE_CONTEXT_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION,
         "split_revision": CHRONOLOGICAL_SPLIT_REVISION},
        {"cutoff": 15, "model": "MCAR-Decomp+Score", "seed": 42,
         "macro_f1": 0.6, "model_revision": SPARC_REVISION,
         "graph_revision": TEMPORAL_GRAPH_REVISION,
         "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
         "score_timeline_revision": SCORE_TIMELINE_REVISION,
         "score_context_revision": SCORE_CONTEXT_REVISION,
         "loss_revision": FIXED_LOSS_REVISION,
         "class_weight_revision": CLASS_WEIGHT_REVISION,
         "split_revision": CHRONOLOGICAL_SPLIT_REVISION},
    ]).to_parquet(path, index=False)

    rows = load_temporal_rows(path)

    assert len(rows) == 2
    assert {row["model"] for row in rows} == {
        "SPARC-GNN", "SPARC-GNN+Score"}


def test_stale_sparc_revision_is_not_resumed(tmp_path):
    path = tmp_path / "temporal.parquet"
    pd.DataFrame([{
        "cutoff": 15, "model": "SPARC-GNN", "seed": 42,
        "macro_f1": 0.9, "model_revision": "unmasked_conflict_v1",
        "graph_revision": TEMPORAL_GRAPH_REVISION,
        "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "score_context_revision": SCORE_CONTEXT_REVISION,
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
    }]).to_parquet(path, index=False)
    assert load_temporal_rows(path) == []


def test_stale_chronological_split_is_not_resumed(tmp_path):
    path = tmp_path / "temporal.parquet"
    pd.DataFrame([{
        "cutoff": 15, "model": "HGT", "seed": 42,
        "macro_f1": 0.9, "model_revision": "baseline_v1",
        "graph_revision": TEMPORAL_GRAPH_REVISION,
        "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "score_context_revision": SCORE_CONTEXT_REVISION,
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
        "split_revision": "index_boundary_v1",
    }]).to_parquet(path, index=False)
    assert load_temporal_rows(path) == []


def test_stale_score_timeline_is_not_resumed(tmp_path):
    path = tmp_path / "temporal.parquet"
    pd.DataFrame([{
        "cutoff": 15, "model": "HGT", "seed": 42,
        "macro_f1": 0.9, "model_revision": "baseline_v1",
        "graph_revision": TEMPORAL_GRAPH_REVISION,
        "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "score_timeline_revision": "double_counted_own_goals_v1",
        "score_context_revision": SCORE_CONTEXT_REVISION,
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
    }]).to_parquet(path, index=False)
    assert load_temporal_rows(path) == []


def test_stale_score_context_is_not_resumed(tmp_path):
    path = tmp_path / "temporal.parquet"
    pd.DataFrame([{
        "cutoff": 15, "model": "HeteroSAGE-sum+Score", "seed": 42,
        "macro_f1": 0.9, "model_revision": "baseline_v1",
        "graph_revision": TEMPORAL_GRAPH_REVISION,
        "base_graph_revision": HETEROGENEOUS_GRAPH_REVISION,
        "score_timeline_revision": SCORE_TIMELINE_REVISION,
        "score_context_revision": "clipped_at_five_v1",
        "loss_revision": FIXED_LOSS_REVISION,
        "class_weight_revision": CLASS_WEIGHT_REVISION,
        "split_revision": CHRONOLOGICAL_SPLIT_REVISION,
    }]).to_parquet(path, index=False)
    assert load_temporal_rows(path) == []
