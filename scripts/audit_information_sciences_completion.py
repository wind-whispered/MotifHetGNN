"""Machine-checkable evidence audit for the Information Sciences revision."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn.causal_context import (
    MATCH_STATE_FEATURE_NAMES, MATCH_STATE_REVISION,
    TEAM_STRENGTH_FEATURE_NAMES, TEAM_STRENGTH_REVISION,
)
from src.gnn.dataset import CHRONOLOGICAL_SPLIT_REVISION
from src.gnn.feature_contracts import (
    ASSIST_PROXY_FEATURE_REVISION, PRIMARY_FULL_MATCH_FEATURE_REVISION,
)
from src.gnn.mcar_model import SPARC_REVISION
from src.gnn.score_context import SCORE_CONTEXT_REVISION
from src.gnn.trainer import (
    CLASS_WEIGHT_REVISION, FIXED_LOSS_REVISION,
    UNCERTAINTY_LOSS_REVISION,
)
from src.networks.edge_features import PASS_EDGE_DIM
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION
from src.networks.temporal import SCORE_TIMELINE_REVISION
from scripts.run_information_sciences_generalization import (
    NON_TEMPORAL_SPLIT_REVISION,
)
from scripts.run_information_sciences_relation_controls import (
    RELATION_CONTROL_REVISION,
)
from scripts.run_relation_subset_retraining import SUBSET_REVISION
from scripts.run_temporal_cutoff_experiment import (
    SCORE_LOSS_REVISION, TEMPORAL_GRAPH_REVISION,
)
from scripts.evaluate_temporal_causal_baselines import (
    BASELINE_REVISION as CAUSAL_BASELINE_REVISION, SCORE_FEATURES,
)
from scripts.evaluate_temporal_context_graph_blends import (
    BLEND_REVISION as CONTEXT_GRAPH_BLEND_REVISION,
)
from scripts.evaluate_temporal_validation_blends import (
    BLEND_REVISION as SCORE_GRAPH_BLEND_REVISION,
)
from scripts.analyze_temporal_controlled_effects import (
    METRICS as TEMPORAL_CONTROLLED_METRICS,
    _contrasts as temporal_controlled_contrasts,
)
from scripts.audit_highorder_census_coverage import audit_census
from scripts.run_finite_size_motif_analysis import (
    FINITE_SIZE_GRAPH_REVISION, _network_digest,
)

OUT = ROOT / "data" / "information_sciences"
from src.paths import paper_directory
PAPER = paper_directory()
SEEDS = {42, 7, 13, 21, 37}
GRAPH_MODELS = {
    "TeamPool-MLP", "HeteroSAGE-sum", "R-GCN", "HGT", "MotifGate-ObservedOnly",
    "EdgeAware-Fixed", "SPARC-GNN",
}
GENERALIZATION_MODELS = {
    "TeamPool-MLP", "HeteroSAGE-sum", "HGT", "EdgeAware-Fixed",
    "SPARC-GNN",
}
GENERALIZATION_BASELINES = GENERALIZATION_MODELS - {"SPARC-GNN"}
TEMPORAL_NEURAL = {
    "TeamPool-MLP", "HeteroSAGE-sum", "HGT", "EdgeAware-Fixed",
    "SPARC-GNN", "TeamPool-MLP+Score", "HeteroSAGE-sum+Score",
    "SPARC-GNN+Score",
}
SCORE_MODELS = {"CurrentScore", "CurrentScore-Calibrated"}
CAUSAL_TEMPORAL_MODELS = {"Score+Strength", "CausalMatchState-HGB"}
TEMPORAL_BLEND_BASE_MODELS = {
    "TeamPool-MLP+Score", "HeteroSAGE-sum+Score", "SPARC-GNN+Score",
}
TEMPORAL_EFFECT_CONTRASTS = {
    "sparc_over_edgeaware", "sparc_over_team_pool", "sparc_over_heterosage",
    "score_augmented_over_graph_only",
    "heterosage_score_augmented_over_graph_only",
    "team_pool_score_augmented_over_graph_only",
    "sparc_score_over_heterosage_score",
    "sparc_score_over_team_pool_score",
    "heterosage_score_over_team_pool_score",
    "score_augmented_over_calibrated_score",
    "heterosage_score_over_calibrated_score",
    "team_pool_score_over_calibrated_score",
    "graph_only_over_calibrated_score",
    "heterosage_graph_only_over_calibrated_score",
}
PROTOCOLS = {"random", "chronological", "competition", "team_grouped"}
CUTOFFS = {15, 30, 45, 60, 75}
STRICT_CUTOFF_RULE = "0 <= minute + second/60 < cutoff"
ABLATIONS = {
    "NoEdgeAttr-Fixed", "EdgeAware-Fixed", "Adaptive-NoMotif", "MotifGate",
    "MotifGate-ObservedOnly",
    "MCAR-Decomp-FixedLoss", "MCAR-Decomp-NoConflict",
    "MCAR-Decomp-NoAvailabilityFeature", "MCAR-Decomp-NoMotif",
    "MCAR-Decomp-NoOrthogonality", "MCAR-Decomp-WithAssistProxy",
    "MCAR-Decomp-Uncertainty",
}
PRIMARY_METRICS = {"accuracy", "macro_f1", "draw_f1", "mae", "r2"}
ABLATION_COMPONENTS = {
    "edge_attributes", "adaptive_relation_gate", "motif_conditioned_gate",
    "shared_private_conflict_decomposition", "conflict_channel",
    "availability_feature", "motif_context_in_full_model",
    "orthogonality_regularizer", "uncertainty_task_weighting",
    "retrospective_assist_proxy_upper_bound",
}
RELATION_SUBSETS = {
    "pass", "adversarial", "turnover", "pass+adversarial",
    "pass+turnover", "adversarial+turnover", "all",
}
RELATION_SUBSET_CONTRASTS = {
    "all_over_pass", "all_over_adversarial", "all_over_turnover",
    "all_over_pass_adversarial", "all_over_pass_turnover",
    "all_over_adversarial_turnover", "pass_adversarial_over_pass",
    "pass_adversarial_over_adversarial", "pass_turnover_over_pass",
    "pass_turnover_over_turnover",
    "adversarial_turnover_over_adversarial",
    "adversarial_turnover_over_turnover",
}
RELATION_CONTROL_CONTRASTS = {
    "pass_availability", "adversarial_availability",
    "turnover_availability", "pass_topology_degree_preserving",
    "adversarial_anchor_identity", "turnover_anchor_identity",
}


def _result(name, passed, detail):
    return {"check": name, "status": "PASS" if passed else "FAIL",
            "detail": detail}


def _keys(frame, columns):
    return {tuple(row) for row in frame[list(columns)].itertuples(index=False,
                                                                  name=None)}


def _class_weight_values_valid(frame):
    """Check positive, finite, mean-one class weights on neural rows."""
    columns = ["class_weight_home", "class_weight_draw",
               "class_weight_away"]
    if not set(columns).issubset(frame.columns) or frame.empty:
        return False
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    return bool(
        np.isfinite(values).all()
        and (values > 0).all()
        and np.allclose(values.mean(axis=1), 1.0, atol=1e-6)
    )


def audit_full_graph_features():
    """Require a fully rebuilt, dimension- and multiplicity-audited corpus."""
    graph_dir = ROOT / "data" / "networks" / "heterogeneous"
    manifest_path = graph_dir / "manifest.json"
    feature_path = OUT / "full_graph_feature_audit.json"
    checks = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ok = (
            manifest.get("graph_revision") == HETEROGENEOUS_GRAPH_REVISION
            and int(manifest.get("expected_matches", -1)) == 3464
            and int(manifest.get("graph_files", -1)) == 3464
            and int(manifest.get("successful_builds", -1)) == 3464
            and int(manifest.get("failed_builds", -1)) == 0
            and manifest.get("edge_dimensions", {}).get("pass")
            == PASS_EDGE_DIM
        )
        detail = (
            f"graphs={manifest.get('graph_files')}/"
            f"{manifest.get('expected_matches')}, "
            f"failed={manifest.get('failed_builds')}, "
            f"pass_dim={manifest.get('edge_dimensions', {}).get('pass')}"
        )
    else:
        manifest_ok = False
        detail = "missing data/networks/heterogeneous/manifest.json"
    checks.append(_result("full-match graph rebuild manifest",
                          manifest_ok, detail))

    if feature_path.exists():
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        counts = payload.get("counts", {})
        feature_ok = (
            payload.get("status") == "PASS"
            and payload.get("graph_revision") ==
            HETEROGENEOUS_GRAPH_REVISION
            and int(payload.get("pass_feature_dimension", -1))
            == PASS_EDGE_DIM
            and int(counts.get("graphs", -1)) == 3464
            and int(counts.get("revision_mismatches", 0)) == 0
            and int(counts.get("pass_shape_mismatches", 0)) == 0
            and int(counts.get("nonfinite_pass_feature_values", 0)) == 0
            and int(counts.get(
                "nonpositive_pass_intensity_values", 0)) == 0
            and int(counts.get("pass_pair_set_mismatches", 0)) == 0
            and int(counts.get("pass_multiplicity_mismatches", 0)) == 0
            and float(payload.get(
                "maximum_pass_intensity_absolute_error", float("inf")))
            <= 1e-6
        )
        detail = (
            f"graphs={counts.get('graphs')}, pass_edges={counts.get('pass_edges')}, "
            f"pair mismatches={counts.get('pass_pair_set_mismatches', 0)}, "
            f"multiplicity mismatches="
            f"{counts.get('pass_multiplicity_mismatches', 0)}, "
            f"max_error={payload.get('maximum_pass_intensity_absolute_error')}"
        )
    else:
        feature_ok = False
        detail = "missing full_graph_feature_audit.json"
    checks.append(_result("full-match graph feature provenance",
                          feature_ok, detail))
    return checks


def audit_main():
    path = OUT / "model_results.parquet"
    if not path.exists():
        return [_result("main comparison", False, "missing model_results.parquet")]
    frame = pd.read_parquet(path)
    expected = {(model, seed) for model in GRAPH_MODELS for seed in SEEDS}
    actual = _keys(frame, ("model", "seed"))
    invalid = actual.intersection({(model, -1) for model in SCORE_MODELS})
    class_weights_ok = _class_weight_values_valid(frame)
    revisions_ok = (
        "split_revision" in frame
        and (frame["split_revision"] == CHRONOLOGICAL_SPLIT_REVISION).all()
        and "model_revision" in frame
        and (frame.loc[frame["model"] == "SPARC-GNN", "model_revision"]
             == SPARC_REVISION).all()
        and set(frame.get("n_test", [])) == {692}
        and set(frame.get("feature_revision", [])) == {
            PRIMARY_FULL_MATCH_FEATURE_REVISION}
        and set(frame.get("graph_revision", [])) == {
            HETEROGENEOUS_GRAPH_REVISION}
        and set(frame.get("loss_revision", [])) == {FIXED_LOSS_REVISION}
        and set(frame.get("class_weight_revision", [])) == {
            CLASS_WEIGHT_REVISION}
        and class_weights_ok
    )
    return [
        _result("main comparison coverage", actual == expected,
                f"{len(actual)}/{len(expected)} model-seed rows; "
                f"missing={sorted(expected-actual)} extra={sorted(actual-expected)}"),
        _result("main comparison score leakage", not invalid,
                f"invalid score rows={sorted(invalid)}"),
        _result("main comparison revision contract", revisions_ok,
                f"split={CHRONOLOGICAL_SPLIT_REVISION}, "
                f"SPARC={SPARC_REVISION}, "
                f"features={PRIMARY_FULL_MATCH_FEATURE_REVISION}, "
                f"graph={HETEROGENEOUS_GRAPH_REVISION}, "
                f"loss={FIXED_LOSS_REVISION}, "
                f"class_weights={CLASS_WEIGHT_REVISION}/"
                f"values_valid={class_weights_ok}, n_test=692"),
    ]


def audit_statistical_outputs():
    checks = []
    predictions_path = OUT / "predictions.parquet"
    prediction_mtime = (predictions_path.stat().st_mtime
                        if predictions_path.exists() else float("inf"))
    expected_prediction_keys = {
        (model, seed) for model in GRAPH_MODELS for seed in SEEDS}
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        counts = predictions.groupby(["model", "seed"])["match_id"].agg(
            ["size", "nunique"])
        prediction_keys = set(counts.index.tolist())
        prediction_ok = (
            prediction_keys == expected_prediction_keys
            and len(predictions) == len(expected_prediction_keys) * 692
            and (counts["size"] == 692).all()
            and (counts["nunique"] == 692).all())
        detail = (f"rows={len(predictions)}/"
                  f"{len(expected_prediction_keys) * 692}, "
                  f"keys={len(prediction_keys)}/{len(expected_prediction_keys)}")
    else:
        prediction_ok = False
        detail = "missing predictions.parquet"
    checks.append(_result("main prediction-level coverage", prediction_ok,
                          detail))

    hierarchical_path = OUT / "hierarchical_paired_bootstrap.csv"
    expected_hierarchical = {
        (baseline, metric)
        for baseline in GRAPH_MODELS - {"SPARC-GNN"}
        for metric in PRIMARY_METRICS}
    if hierarchical_path.exists():
        hierarchical = pd.read_csv(hierarchical_path)
        keys = _keys(hierarchical, ("baseline", "metric"))
        hierarchical_ok = (
            keys == expected_hierarchical
            and set(hierarchical["target"]) == {"SPARC-GNN"}
            and set(hierarchical["n_seeds"]) == {5}
            and set(hierarchical["n_matches"]) == {692}
            and int(hierarchical["repeats"].min()) >= 5000
            and hierarchical_path.stat().st_mtime >= prediction_mtime)
        detail = f"rows={len(hierarchical)}/{len(expected_hierarchical)}"
    else:
        hierarchical_ok = False
        detail = "missing hierarchical_paired_bootstrap.csv"
    checks.append(_result("hierarchical paired bootstrap", hierarchical_ok,
                          detail))

    seed_path = OUT / "paired_bootstrap.csv"
    expected_seed = {
        (baseline, seed, metric)
        for baseline in GRAPH_MODELS - {"SPARC-GNN"}
        for seed in SEEDS for metric in PRIMARY_METRICS}
    if seed_path.exists():
        seed_frame = pd.read_csv(seed_path)
        keys = _keys(seed_frame, ("baseline", "seed", "metric"))
        seed_ok = (keys == expected_seed and seed_path.stat().st_mtime >=
                   prediction_mtime)
        detail = f"rows={len(seed_frame)}/{len(expected_seed)}"
    else:
        seed_ok = False
        detail = "missing paired_bootstrap.csv"
    checks.append(_result("within-seed paired bootstrap", seed_ok, detail))

    ablation_path = OUT / "ablation_paired_effects.csv"
    ablation_results_path = OUT / "ablation_results.parquet"
    ablation_results_mtime = (
        ablation_results_path.stat().st_mtime
        if ablation_results_path.exists() else float("inf"))
    expected_ablation = {
        (component, metric) for component in ABLATION_COMPONENTS
        for metric in PRIMARY_METRICS}
    if ablation_path.exists():
        paired = pd.read_csv(ablation_path)
        keys = _keys(paired, ("component", "metric"))
        ablation_ok = (
            keys == expected_ablation
            and set(paired["n_seeds"]) == {5}
            and int(paired["bootstrap_repeats"].min()) >= 1000
            and paired["exact_two_sided_sign_flip_p"].between(0, 1).all()
            and ablation_path.stat().st_mtime >= ablation_results_mtime)
        detail = f"rows={len(paired)}/{len(expected_ablation)}"
    else:
        ablation_ok = False
        detail = "missing ablation_paired_effects.csv"
    checks.append(_result("paired ablation effect audit", ablation_ok,
                          detail))
    return checks


def audit_generalization():
    path = OUT / "generalization_results.parquet"
    if not path.exists():
        return [_result("four-level generalization", False,
                        "missing generalization_results.parquet")]
    frame = pd.read_parquet(path)
    frame["model"] = frame["model"].replace({"MCAR-Decomp": "SPARC-GNN"})
    expected = {(protocol, model, seed) for protocol in PROTOCOLS
                for model in GENERALIZATION_MODELS for seed in SEEDS}
    actual = _keys(frame, ("protocol", "model", "seed"))
    checks = [_result(
        "four-level generalization coverage", actual == expected,
        f"{len(actual)}/{len(expected)} protocol-model-seed rows; "
        f"missing={len(expected-actual)} extra={len(actual-expected)}")]
    leaked = frame[frame["model"].isin(SCORE_MODELS)]
    checks.append(_result("generalization score leakage", leaked.empty,
                          f"invalid rows={len(leaked)}"))
    expected_revisions = frame["protocol"].map(
        lambda value: (CHRONOLOGICAL_SPLIT_REVISION
                       if value == "chronological"
                       else NON_TEMPORAL_SPLIT_REVISION))
    class_weights_ok = _class_weight_values_valid(frame)
    revision_ok = (
        "split_revision" in frame
        and (frame["split_revision"] == expected_revisions).all()
        and (frame.loc[frame["model"] == "SPARC-GNN", "model_revision"]
             == SPARC_REVISION).all()
        and set(frame.get("feature_revision", [])) == {
            PRIMARY_FULL_MATCH_FEATURE_REVISION}
        and set(frame.get("graph_revision", [])) == {
            HETEROGENEOUS_GRAPH_REVISION}
        and set(frame.get("loss_revision", [])) == {FIXED_LOSS_REVISION}
        and set(frame.get("class_weight_revision", [])) == {
            CLASS_WEIGHT_REVISION}
        and class_weights_ok
    )
    checks.append(_result("generalization revision contract", revision_ok,
                          "protocol-specific split, SPARC and feature revisions"))

    manifest = OUT / "generalization_splits.json"
    if not manifest.exists():
        checks.append(_result("generalization split manifest", False, "missing"))
        return checks
    splits = json.loads(manifest.read_text(encoding="utf-8"))
    expected_keys = {f"{protocol}:{seed}" for protocol in PROTOCOLS for seed in SEEDS}
    manifest_keys = set(splits)
    audit_present = all("audit" in record for record in splits.values())
    manifest_revision_ok = all(
        record.get("graph_revision") == HETEROGENEOUS_GRAPH_REVISION
        and record.get("feature_revision") ==
        PRIMARY_FULL_MATCH_FEATURE_REVISION
        and record.get("loss_revision") == FIXED_LOSS_REVISION
        and record.get("class_weight_revision") == CLASS_WEIGHT_REVISION
        and set(record.get("class_weights", {})) == {
            "home", "draw", "away"}
        and np.isclose(np.mean(list(record.get(
            "class_weights", {}).values())), 1.0, atol=1e-6)
        for record in splits.values())
    team_zero = all(
        not any(record.get("audit", {}).get("team_overlap", {"missing": 1}).values())
        for key, record in splits.items() if key.startswith("team_grouped:")
    )
    match_zero = all(
        not any(record.get("audit", {}).get("match_overlap", {"missing": 1}).values())
        for record in splits.values()
    )
    chronology_strict = all(
        pd.Timestamp(record["audit"]["date_ranges"]["train"]["max"])
        < pd.Timestamp(record["audit"]["date_ranges"]["validation"]["min"])
        and pd.Timestamp(record["audit"]["date_ranges"]["validation"]["max"])
        < pd.Timestamp(record["audit"]["date_ranges"]["test"]["min"])
        for key, record in splits.items() if key.startswith("chronological:")
    )
    memberships = {}
    for protocol in PROTOCOLS:
        signatures = {
            tuple(tuple(sorted(record[partition]))
                  for partition in ("train", "validation", "test"))
            for key, record in splits.items()
            if key.startswith(f"{protocol}:")}
        memberships[protocol] = len(signatures)
    membership_ok = (
        memberships.get("random") == len(SEEDS)
        and all(memberships.get(protocol) == 1 for protocol in
                ("chronological", "competition", "team_grouped")))
    checks.extend([
        _result("generalization split manifest",
                manifest_keys == expected_keys and audit_present
                and manifest_revision_ok,
                f"keys={len(manifest_keys)}/{len(expected_keys)}, "
                f"audit={audit_present}, revisions={manifest_revision_ok}"),
        _result("split match identity disjointness", match_zero,
                "all pairwise match intersections are zero" if match_zero else "violation/missing audit"),
        _result("team-disjoint identity constraint", team_zero,
                "all pairwise team intersections are zero" if team_zero else "violation/missing audit"),
        _result("complete-date chronological boundaries", chronology_strict,
                "strict date ordering" if chronology_strict else "shared/missing boundary date"),
        _result("protocol membership across seeds", membership_ok,
                f"distinct split signatures={memberships}"),
    ])
    effect_path = OUT / "generalization_paired_effects.csv"
    if effect_path.exists():
        effects = pd.read_csv(effect_path)
        expected_effects = {
            (protocol, baseline, metric) for protocol in PROTOCOLS
            for baseline in GENERALIZATION_BASELINES
            for metric in PRIMARY_METRICS}
        actual_effects = _keys(
            effects, ("protocol", "baseline", "metric"))
        effects_ok = (
            actual_effects == expected_effects
            and set(effects["target"]) == {"SPARC-GNN"}
            and set(effects["n_seeds"]) == {5}
            and int(effects["bootstrap_repeats"].min()) >= 1000
            and effects["exact_two_sided_sign_flip_p"].between(0, 1).all()
            and effect_path.stat().st_mtime >= path.stat().st_mtime)
        checks.append(_result(
            "generalization paired effects", effects_ok,
            f"rows={len(actual_effects)}/{len(expected_effects)}"))
    else:
        checks.append(_result("generalization paired effects", False,
                              "missing"))
    return checks


def audit_temporal():
    path = OUT / "temporal_results.parquet"
    neural_expected = {(cutoff, model, seed) for cutoff in CUTOFFS
                       for model in TEMPORAL_NEURAL for seed in SEEDS}
    score_expected = {(cutoff, model, -1) for cutoff in CUTOFFS
                      for model in SCORE_MODELS}
    expected = neural_expected | score_expected
    source = path if path.exists() else OUT / "temporal_results_partial.parquet"
    if source.exists():
        frame = pd.read_parquet(source)
        frame["model"] = frame["model"].replace({
            "MCAR-Decomp": "SPARC-GNN",
            "MCAR-Decomp+Score": "SPARC-GNN+Score",
        })
        actual = _keys(frame, ("cutoff", "model", "seed"))
        neural = frame[~frame["model"].isin(SCORE_MODELS)].copy()
        class_weights_ok = _class_weight_values_valid(neural)
        revision_ok = (
            "split_revision" in frame
            and (frame["split_revision"] == CHRONOLOGICAL_SPLIT_REVISION).all()
            and "graph_revision" in frame
            and (frame["graph_revision"] == TEMPORAL_GRAPH_REVISION).all()
            and "base_graph_revision" in frame
            and (frame["base_graph_revision"] ==
                 HETEROGENEOUS_GRAPH_REVISION).all()
            and "score_timeline_revision" in frame
            and (frame["score_timeline_revision"] ==
                 SCORE_TIMELINE_REVISION).all()
            and "score_context_revision" in frame
            and (frame["score_context_revision"] ==
                 SCORE_CONTEXT_REVISION).all()
            and (frame.loc[frame["model"].isin(
                {"SPARC-GNN", "SPARC-GNN+Score"}), "model_revision"]
                 == SPARC_REVISION).all()
            and set(frame.get("n_test", [])) == {692}
            and "loss_revision" in frame
            and (frame["loss_revision"] == frame["model"].map(
                lambda name: SCORE_LOSS_REVISION
                if name in SCORE_MODELS else FIXED_LOSS_REVISION)).all()
            and "class_weight_revision" in frame
            and (frame["class_weight_revision"] == frame["model"].map(
                lambda name: SCORE_LOSS_REVISION
                if name in SCORE_MODELS else CLASS_WEIGHT_REVISION)).all()
            and class_weights_ok
        )
    else:
        actual = set()
        revision_ok = False
    coverage_ok = path.exists() and actual == expected
    source_label = "final" if path.exists() else (
        "partial" if source.exists() else "missing")
    manifest_path = OUT / "temporal_graphs" / "manifest.json"
    manifest_ok = False
    manifest_detail = "missing"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = {int(key): int(value) for key, value in manifest.get("counts", {}).items()}
        cutoff_rule = manifest.get("cutoff_rule")
        graph_revision = manifest.get("graph_revision")
        assist_masked = manifest.get("assist_flags_masked") is True
        base_revision = manifest.get("base_graph_revision")
        score_revision = manifest.get("score_timeline_revision")
        pass_dimension = manifest.get("edge_dimensions", {}).get("pass")
        manifest_ok = set(counts) == CUTOFFS and min(counts.values(), default=0) == 3464 \
                      and not manifest.get("failures") \
                      and cutoff_rule == STRICT_CUTOFF_RULE \
                      and graph_revision == TEMPORAL_GRAPH_REVISION \
                      and base_revision == HETEROGENEOUS_GRAPH_REVISION \
                      and score_revision == SCORE_TIMELINE_REVISION \
                      and pass_dimension == PASS_EDGE_DIM \
                      and assist_masked
        manifest_detail = (
            f"counts={counts}, failures={len(manifest.get('failures', []))}, "
            f"cutoff_rule={cutoff_rule!r}, revision={graph_revision!r}, "
            f"base={base_revision!r}, pass_dim={pass_dimension}, "
            f"score_timeline={score_revision!r}, "
            f"assist_flags_masked={assist_masked}"
        )
    checks = [
        _result("strict temporal cutoff coverage", coverage_ok,
                f"{source_label}: {len(actual)}/{len(expected)} "
                f"cutoff-model-seed rows; "
                f"missing={len(expected-actual)} extra={len(actual-expected)}"),
        _result("event-prefix graph manifest", manifest_ok, manifest_detail),
        _result("temporal revision contract", revision_ok,
                f"split={CHRONOLOGICAL_SPLIT_REVISION}, "
                f"graph={TEMPORAL_GRAPH_REVISION}, "
                f"base={HETEROGENEOUS_GRAPH_REVISION}, "
                f"score_timeline={SCORE_TIMELINE_REVISION}, "
                f"score_context={SCORE_CONTEXT_REVISION}, n_test=692"),
    ]
    feature_path = OUT / "temporal_graph_feature_audit.json"
    if feature_path.exists():
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        feature_counts = feature.get("counts", {})
        feature_ok = (
            feature.get("status") == "PASS"
            and feature.get("graph_revision") == TEMPORAL_GRAPH_REVISION
            and feature.get("base_graph_revision") ==
            HETEROGENEOUS_GRAPH_REVISION
            and feature.get("score_timeline_revision") ==
            SCORE_TIMELINE_REVISION
            and int(feature_counts.get("graphs", 0)) == 17320
            and int(feature_counts.get("nonzero_assist_values", 0)) == 0
            and int(feature_counts.get(
                "nonpositive_pass_intensity_values", 0)) == 0
            and int(feature_counts.get(
                "nonfinite_pass_feature_values", 0)) == 0
            and int(feature.get("pass_feature_dimension", 0)) == PASS_EDGE_DIM
        )
        checks.append(_result(
            "temporal graph feature provenance", feature_ok,
            f"graphs={feature_counts.get('graphs', 0)}, "
            f"nonzero assists={feature_counts.get('nonzero_assist_values', 0)}, "
            f"nonpositive intensity={feature_counts.get('nonpositive_pass_intensity_values', 0)}"))
    else:
        checks.append(_result("temporal graph feature provenance", False,
                              "missing temporal_graph_feature_audit.json"))

    score_audit_path = OUT / "score_timeline_audit.json"
    if score_audit_path.exists():
        score_audit = json.loads(score_audit_path.read_text(encoding="utf-8"))
        score_ok = (
            score_audit.get("status") == "PASS"
            and score_audit.get("score_timeline_revision") ==
            SCORE_TIMELINE_REVISION
            and int(score_audit.get("matches_expected", 0)) == 3464
            and int(score_audit.get("matches_audited", 0)) == 3464
            and int(score_audit.get("missing_event_files", -1)) == 0
            and int(score_audit.get("official_score_mismatches", -1)) == 0
            and int(score_audit.get(
                "superseded_logic_score_mismatches", 0)) > 0
            and int(score_audit.get("paired_own_goal_timestamps", 0)) > 0
        )
        checks.append(_result(
            "official score-timeline reconstruction", score_ok,
            f"audited={score_audit.get('matches_audited')}/3464, "
            f"mismatches={score_audit.get('official_score_mismatches')}, "
            f"superseded_mismatches={score_audit.get('superseded_logic_score_mismatches')}, "
            f"paired_own_goals={score_audit.get('paired_own_goal_timestamps')}"))
    else:
        checks.append(_result("official score-timeline reconstruction", False,
                              "missing score_timeline_audit.json"))

    leakage_path = OUT / "temporal_assist_leakage_audit.json"
    if leakage_path.exists():
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        leakage_counts = leakage.get("counts", {})
        crossing = int(leakage_counts.get("cross_boundary_passes", -1))
        linked = int(leakage_counts.get("linked_assists", -1))
        flagged = int(leakage_counts.get("flagged_passes", -1))
        leakage_ok = crossing == 154 and linked == flagged == 61994
        checks.append(_result(
            "retrospective assist annotation audit", leakage_ok,
            f"linked={linked}/{flagged}, cross-boundary={crossing}"))
    else:
        checks.append(_result("retrospective assist annotation audit", False,
                              "missing temporal_assist_leakage_audit.json"))
    effect_path = OUT / "temporal_paired_effects.csv"
    final_path = OUT / "temporal_results.parquet"
    if effect_path.exists():
        effects = pd.read_csv(effect_path)
        expected_effects = {
            (cutoff, contrast, metric) for cutoff in CUTOFFS
            for contrast in TEMPORAL_EFFECT_CONTRASTS
            for metric in PRIMARY_METRICS}
        actual_effects = _keys(effects, ("cutoff", "contrast", "metric"))
        effects_ok = (
            actual_effects == expected_effects
            and set(effects["n_target_seeds"]) == {5}
            and set(effects["n_reference_seeds"]).issubset({1, 5})
            and set(effects["pairing"]).issubset({
                "matched_neural_seeds",
                "neural_seeds_vs_fixed_score_baseline"})
            and int(effects["bootstrap_repeats"].min()) >= 1000
            and effects["exact_two_sided_sign_flip_p"].between(0, 1).all()
            and final_path.exists()
            and effect_path.stat().st_mtime >= final_path.stat().st_mtime)
        checks.append(_result("temporal paired effects", effects_ok,
                              f"rows={len(actual_effects)}/{len(expected_effects)}"))
    else:
        checks.append(_result("temporal paired effects", False, "missing"))
    return checks


def audit_temporal_context_controls():
    """Audit causal state controls and validation-only temporal calibration."""
    checks = []

    state_path = OUT / "temporal_match_state.parquet"
    state_manifest_path = OUT / "temporal_match_state_manifest.json"
    if state_path.exists() and state_manifest_path.exists():
        state = pd.read_parquet(state_path)
        state_manifest = json.loads(
            state_manifest_path.read_text(encoding="utf-8"))
        state_keys = _keys(state, ("match_id", "cutoff"))
        feature_matrix = state.loc[:, list(MATCH_STATE_FEATURE_NAMES)].to_numpy(
            dtype=float)
        state_ok = (
            len(state) == 3464 * len(CUTOFFS)
            and len(state_keys) == len(state)
            and set(state["cutoff"].astype(int)) == CUTOFFS
            and set(state["match_state_revision"]) == {MATCH_STATE_REVISION}
            and set(state["score_timeline_revision"]) == {
                SCORE_TIMELINE_REVISION}
            and np.isfinite(feature_matrix).all()
            and state_manifest.get("match_state_revision") ==
            MATCH_STATE_REVISION
            and state_manifest.get("score_timeline_revision") ==
            SCORE_TIMELINE_REVISION
            and state_manifest.get("cutoff_rule") == STRICT_CUTOFF_RULE
            and state_manifest.get("period_5_excluded") is True
            and state_manifest.get("retrospective_pass_annotations_used") is False
            and int(state_manifest.get("feature_count", -1)) ==
            len(MATCH_STATE_FEATURE_NAMES)
            and int(state_manifest.get("row_count", -1)) == len(state)
            and set(map(int, state_manifest.get("cutoffs", []))) == CUTOFFS
        )
        state_detail = (
            f"rows={len(state)}/{3464 * len(CUTOFFS)}, "
            f"features={len(MATCH_STATE_FEATURE_NAMES)}, "
            f"revision={set(state.get('match_state_revision', []))}"
        )
    else:
        state_ok = False
        state_detail = "missing final state cache or manifest"
    checks.append(_result(
        "strict causal match-state cache", state_ok, state_detail))

    result_path = OUT / "temporal_causal_baselines.parquet"
    prediction_path = OUT / "temporal_causal_baseline_predictions.parquet"
    causal_manifest_path = OUT / "temporal_causal_baselines_manifest.json"
    if (result_path.exists() and prediction_path.exists()
            and causal_manifest_path.exists()):
        result = pd.read_parquet(result_path)
        prediction = pd.read_parquet(prediction_path)
        causal_manifest = json.loads(
            causal_manifest_path.read_text(encoding="utf-8"))
        expected = {(cutoff, model, -1) for cutoff in CUTOFFS
                    for model in CAUSAL_TEMPORAL_MODELS}
        actual = _keys(result, ("cutoff", "model", "seed"))
        expected_prediction = {(cutoff, model, match_id)
                               for cutoff in CUTOFFS
                               for model in CAUSAL_TEMPORAL_MODELS
                               for match_id in prediction.loc[
                                   (prediction["cutoff"] == cutoff)
                                   & (prediction["model"] == model),
                                   "match_id"].astype(int)}
        prediction_counts = prediction.groupby(
            ["cutoff", "model"]).size()
        feature_counts = {
            model: set(group["feature_count"].astype(int))
            for model, group in result.groupby("model")
        }
        causal_ok = (
            actual == expected
            and len(prediction) == 692 * len(expected)
            and len(expected_prediction) == len(prediction)
            and set(prediction_counts) == {692}
            and set(result["model_revision"]) == {
                CAUSAL_BASELINE_REVISION}
            and set(result["match_state_revision"]) == {
                MATCH_STATE_REVISION}
            and set(result["team_strength_revision"]) == {
                TEAM_STRENGTH_REVISION}
            and set(result["score_timeline_revision"]) == {
                SCORE_TIMELINE_REVISION}
            and set(result["split_revision"]) == {
                CHRONOLOGICAL_SPLIT_REVISION}
            and set(result["n_test"].astype(int)) == {692}
            and feature_counts.get("Score+Strength") == {
                len(SCORE_FEATURES) + len(TEAM_STRENGTH_FEATURE_NAMES)}
            and feature_counts.get("CausalMatchState-HGB") == {
                len(SCORE_FEATURES) + len(TEAM_STRENGTH_FEATURE_NAMES)
                + len(MATCH_STATE_FEATURE_NAMES)}
            and causal_manifest.get("model_revision") ==
            CAUSAL_BASELINE_REVISION
            and causal_manifest.get("match_state_revision") ==
            MATCH_STATE_REVISION
            and causal_manifest.get("team_strength_revision") ==
            TEAM_STRENGTH_REVISION
            and causal_manifest.get("score_timeline_revision") ==
            SCORE_TIMELINE_REVISION
            and causal_manifest.get("split_revision") ==
            CHRONOLOGICAL_SPLIT_REVISION
            and causal_manifest.get("selection_partition") == "validation"
            and causal_manifest.get("test_used_for_selection") is False
            and set(map(int, causal_manifest.get("cutoffs", []))) == CUTOFFS
            and set(causal_manifest.get("models", [])) ==
            CAUSAL_TEMPORAL_MODELS
        )
        causal_detail = (
            f"results={len(actual)}/{len(expected)}, "
            f"predictions={len(prediction)}/{692 * len(expected)}, "
            f"selection={causal_manifest.get('selection_partition')}"
        )
    else:
        causal_ok = False
        causal_detail = "missing causal results, predictions or manifest"
    checks.append(_result(
        "causal temporal baselines", causal_ok, causal_detail))

    blend_specs = (
        (
            "validation-only score--graph blends",
            OUT / "temporal_validation_blends.parquet",
            OUT / "temporal_validation_blends_manifest.json",
            SCORE_GRAPH_BLEND_REVISION,
            False,
        ),
        (
            "validation-only context--graph blends",
            OUT / "temporal_context_graph_blends.parquet",
            OUT / "temporal_context_graph_blends_manifest.json",
            CONTEXT_GRAPH_BLEND_REVISION,
            True,
        ),
    )
    expected_blends = {(cutoff, model, seed) for cutoff in CUTOFFS
                       for model in TEMPORAL_BLEND_BASE_MODELS for seed in SEEDS}
    for label, path, manifest_path, revision, uses_context in blend_specs:
        if path.exists() and manifest_path.exists():
            blend = pd.read_parquet(path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual = _keys(blend, ("cutoff", "base_model", "seed"))
            weights_ok = (
                blend["classification_graph_weight"].between(0, 1).all()
                and blend["regression_graph_weight"].between(0, 1).all())
            revision_ok = (
                set(blend["blend_revision"]) == {revision}
                and set(blend["graph_revision"]) == {TEMPORAL_GRAPH_REVISION}
                and set(blend["base_graph_revision"]) == {
                    HETEROGENEOUS_GRAPH_REVISION}
                and set(blend["score_timeline_revision"]) == {
                    SCORE_TIMELINE_REVISION}
                and set(blend["score_context_revision"]) == {
                    SCORE_CONTEXT_REVISION}
                and set(blend["split_revision"]) == {
                    CHRONOLOGICAL_SPLIT_REVISION}
                and set(blend["n_test"].astype(int)) == {692}
                and blend.get("checkpoint_provenance_verified",
                              pd.Series(False, index=blend.index)).eq(True).all()
                and manifest.get("blend_revision") == revision
                and manifest.get("selection_partition") == "validation"
                and manifest.get("test_used_for_selection") is False
                and manifest.get("checkpoint_provenance_required") is True
                and set(map(int, manifest.get("cutoffs", []))) == CUTOFFS
                and set(manifest.get("base_models", [])) ==
                TEMPORAL_BLEND_BASE_MODELS
            )
            if uses_context:
                revision_ok = revision_ok and (
                    set(blend["match_state_revision"]) == {
                        MATCH_STATE_REVISION}
                    and set(blend["team_strength_revision"]) == {
                        TEAM_STRENGTH_REVISION}
                    and manifest.get("match_state_revision") ==
                    MATCH_STATE_REVISION
                    and manifest.get("team_strength_revision") ==
                    TEAM_STRENGTH_REVISION
                    and manifest.get("score_timeline_revision") ==
                    SCORE_TIMELINE_REVISION
                    and manifest.get("split_revision") ==
                    CHRONOLOGICAL_SPLIT_REVISION)
            blend_ok = actual == expected_blends and weights_ok and revision_ok
            detail = (
                f"rows={len(actual)}/{len(expected_blends)}, "
                f"weights_in_[0,1]={weights_ok}, revision={revision}"
            )
        else:
            blend_ok = False
            detail = "missing final blend rows or manifest"
        checks.append(_result(label, blend_ok, detail))

    effect_path = OUT / "temporal_controlled_effects.csv"
    if effect_path.exists():
        effects = pd.read_csv(effect_path)
        contrast_names = {row[0] for row in temporal_controlled_contrasts()}
        expected_effects = {
            (cutoff, contrast, metric)
            for cutoff in CUTOFFS
            for contrast in contrast_names
            for metric in TEMPORAL_CONTROLLED_METRICS
        }
        actual_effects = _keys(effects, ("cutoff", "contrast", "metric"))
        numeric_columns = [
            "raw_delta_target_minus_reference", "mean_improvement",
            "ci_low", "ci_high", "exact_two_sided_sign_flip_p",
        ]
        required_columns = {
            "target", "reference", "pairing", "direction",
            "target_better_seeds", "ties", "n_target_seeds",
            "n_reference_seeds", "bootstrap_repeats",
            "seedwise_improvements", *numeric_columns,
        }
        columns_ok = required_columns.issubset(effects.columns)
        numeric_ok = columns_ok and np.isfinite(
            effects[numeric_columns].to_numpy(float)).all()
        direction_ok = columns_ok and (
            effects.loc[effects.metric == "mae", "direction"]
            .eq("reference_minus_target").all()
            and effects.loc[effects.metric != "mae", "direction"]
            .eq("target_minus_reference").all())
        provenance_ok = columns_ok and (
            effects["n_target_seeds"].eq(5).all()
            and effects["n_reference_seeds"].isin([1, 5]).all()
            and effects["target_better_seeds"].between(0, 5).all()
            and effects["exact_two_sided_sign_flip_p"].between(0, 1).all()
            and effects["bootstrap_repeats"].ge(1000).all())
        source_paths = [
            OUT / "temporal_results.parquet",
            OUT / "temporal_validation_blends.parquet",
            OUT / "temporal_context_graph_blends.parquet",
            OUT / "temporal_causal_baselines.parquet",
        ]
        current_ok = (
            all(path.exists() for path in source_paths)
            and effect_path.stat().st_mtime >= max(
                path.stat().st_mtime for path in source_paths))
        effects_ok = (
            actual_effects == expected_effects and len(effects) == 425
            and numeric_ok and direction_ok and provenance_ok and current_ok)
        detail = (
            f"rows={len(actual_effects)}/{len(expected_effects)}, "
            f"contrasts={len(contrast_names)}, finite={numeric_ok}, "
            f"current={current_ok}")
    else:
        effects_ok = False
        detail = "missing temporal_controlled_effects.csv"
    checks.append(_result(
        "controlled temporal calibration effects", effects_ok, detail))
    return checks


def audit_components():
    checks = []
    path = OUT / "ablation_results.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        expected = {(variant, seed) for variant in ABLATIONS for seed in SEEDS}
        actual = _keys(frame, ("variant", "seed"))
        revision_ok = (
            "split_revision" in frame
            and (frame["split_revision"] == CHRONOLOGICAL_SPLIT_REVISION).all()
            and (frame.loc[frame["variant"].str.startswith("MCAR-Decomp"),
                       "model_revision"] == SPARC_REVISION).all()
            and set(frame.get("n_test", [])) == {692}
        )
        expected_feature = frame["variant"].map(
            lambda name: (ASSIST_PROXY_FEATURE_REVISION
                          if name == "MCAR-Decomp-WithAssistProxy"
                          else PRIMARY_FULL_MATCH_FEATURE_REVISION))
        feature_ok = (
            "feature_revision" in frame
            and (frame["feature_revision"] == expected_feature).all())
        graph_ok = (
            "graph_revision" in frame
            and set(frame["graph_revision"]) == {
                HETEROGENEOUS_GRAPH_REVISION})
        expected_loss = frame["variant"].map(
            lambda name: (UNCERTAINTY_LOSS_REVISION
                          if name == "MCAR-Decomp-Uncertainty"
                          else FIXED_LOSS_REVISION))
        loss_ok = (
            "loss_revision" in frame
            and (frame["loss_revision"] == expected_loss).all())
        class_weight_ok = (
            "class_weight_revision" in frame
            and set(frame["class_weight_revision"]) == {
                CLASS_WEIGHT_REVISION}
            and _class_weight_values_valid(frame))
        auxiliary_ok = (
            "auxiliary_weight" in frame
            and set(frame.loc[
                frame.variant == "MCAR-Decomp-NoOrthogonality",
                "auxiliary_weight"]) == {0.0}
            and (frame.loc[
                frame.variant.str.startswith("MCAR-Decomp")
                & frame.variant.ne("MCAR-Decomp-NoOrthogonality"),
                "auxiliary_weight"] == 0.01).all()
        )
        checks.append(_result("component/loss ablations", actual == expected,
                              f"{len(actual)}/{len(expected)} rows"))
        checks.append(_result("ablation revision contract",
                              revision_ok and auxiliary_ok and feature_ok
                              and graph_ok and loss_ok and class_weight_ok,
                              "strict split, current decomposition revision, "
                              f"orthogonality weights={auxiliary_ok}, "
                               f"feature={feature_ok}, graph={graph_ok}, "
                               f"loss={loss_ok}, class_weights={class_weight_ok}"))
    else:
        checks.append(_result("component/loss ablations", False, "missing"))

    decision_path = OUT / "sparc_revision_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        evaluated = decision.get("evaluated_model_revision")
        triggered = decision.get("trigger_v4") is True
        decision_ok = (
            decision.get("status") == "READY"
            and decision.get("test_metrics_used") is False
            and decision.get("class_weight_revision") ==
            CLASS_WEIGHT_REVISION
            and ((not triggered and evaluated == SPARC_REVISION)
                 or (triggered and evaluated != SPARC_REVISION))
        )
        checks.append(_result(
            "validation-only SPARC revision decision", decision_ok,
            f"evaluated={evaluated}, current={SPARC_REVISION}, "
            f"trigger_v4={decision.get('trigger_v4')}, "
            f"worse_seeds={decision.get('worse_seed_count')}"))
    else:
        checks.append(_result(
            "validation-only SPARC revision decision", False, "missing"))

    path = OUT / "relation_controls_decomp.parquet"
    expected_conditions = {
        ("all", "all"), ("delete", "pass"), ("delete", "adversarial"),
        ("delete", "turnover"), ("permute", "pass"),
        ("anchor_shift", "adversarial"), ("anchor_shift", "turnover"),
    }
    if path.exists():
        frame = pd.read_parquet(path)
        expected = {(condition, relation, seed)
                    for condition, relation in expected_conditions for seed in SEEDS}
        actual = _keys(frame, ("condition", "relation", "seed"))
        revision_ok = (
            set(frame.get("model_revision", [])) == {SPARC_REVISION}
            and set(frame.get("split_revision", [])) == {
                CHRONOLOGICAL_SPLIT_REVISION}
            and set(frame.get("control_revision", [])) == {
                RELATION_CONTROL_REVISION}
            and set(frame.get("feature_revision", [])) == {
                PRIMARY_FULL_MATCH_FEATURE_REVISION}
            and set(frame.get("graph_revision", [])) == {
                HETEROGENEOUS_GRAPH_REVISION}
            and set(frame.get("loss_revision", [])) == {
                FIXED_LOSS_REVISION}
            and set(frame.get("class_weight_revision", [])) == {
                CLASS_WEIGHT_REVISION}
            and "control_changed_graphs" in frame
            and (pd.to_numeric(frame.loc[
                (frame["condition"] == "permute")
                & (frame["relation"] == "pass"),
                "control_changed_graphs"], errors="coerce") > 0).all()
        )
        checks.append(_result("relation deletion/permutation controls",
                              actual == expected, f"{len(actual)}/{len(expected)} rows"))
        checks.append(_result("relation-control revision contract", revision_ok,
                              RELATION_CONTROL_REVISION))
    else:
        checks.append(_result("relation deletion/permutation controls", False, "missing"))

    control_effect_path = OUT / "relation_control_paired_effects.csv"
    if control_effect_path.exists():
        effects = pd.read_csv(control_effect_path)
        expected = {(contrast, metric)
                    for contrast in RELATION_CONTROL_CONTRASTS
                    for metric in PRIMARY_METRICS}
        actual = _keys(effects, ("contrast", "metric"))
        effects_ok = (
            actual == expected and set(effects["n_seeds"]) == {5}
            and int(effects["bootstrap_repeats"].min()) >= 1000
            and effects["exact_two_sided_sign_flip_p"].between(0, 1).all()
            and path.exists()
            and control_effect_path.stat().st_mtime >= path.stat().st_mtime)
        checks.append(_result("relation-control paired effects", effects_ok,
                              f"rows={len(actual)}/{len(expected)}"))
    else:
        checks.append(_result("relation-control paired effects", False,
                              "missing"))

    subset_path = OUT / "relation_subset_results.parquet"
    if subset_path.exists():
        subset = pd.read_parquet(subset_path)
        expected = {(relation_subset, seed)
                    for relation_subset in RELATION_SUBSETS for seed in SEEDS}
        actual = _keys(subset, ("relation_subset", "seed"))
        subset_revisions_ok = (
            set(subset.get("model_revision", [])) == {SPARC_REVISION}
            and set(subset.get("subset_revision", [])) == {SUBSET_REVISION}
            and set(subset.get("split_revision", [])) == {
                CHRONOLOGICAL_SPLIT_REVISION}
            and set(subset.get("feature_revision", [])) == {
                PRIMARY_FULL_MATCH_FEATURE_REVISION}
            and set(subset.get("graph_revision", [])) == {
                HETEROGENEOUS_GRAPH_REVISION}
            and set(subset.get("loss_revision", [])) == {
                FIXED_LOSS_REVISION}
            and set(subset.get("class_weight_revision", [])) == {
                CLASS_WEIGHT_REVISION}
            and _class_weight_values_valid(subset))
        checks.append(_result(
            "retrained relation-subset coverage",
            actual == expected and subset_revisions_ok,
            f"rows={len(actual)}/{len(expected)}, revisions={subset_revisions_ok}"))
    else:
        checks.append(_result("retrained relation-subset coverage", False,
                              "missing"))

    subset_effect_path = OUT / "relation_subset_paired_effects.csv"
    if subset_effect_path.exists():
        effects = pd.read_csv(subset_effect_path)
        expected = {(contrast, metric)
                    for contrast in RELATION_SUBSET_CONTRASTS
                    for metric in PRIMARY_METRICS}
        actual = _keys(effects, ("contrast", "metric"))
        effects_ok = (
            actual == expected and set(effects["n_seeds"]) == {5}
            and int(effects["bootstrap_repeats"].min()) >= 1000
            and subset_path.exists()
            and subset_effect_path.stat().st_mtime >=
            subset_path.stat().st_mtime)
        checks.append(_result("relation-subset paired effects", effects_ok,
                              f"rows={len(actual)}/{len(expected)}"))
    else:
        checks.append(_result("relation-subset paired effects", False,
                              "missing"))

    path = OUT / "decomposition_diagnostics.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        checks.append(_result("shared/private/conflict diagnostics",
                              len(frame) == 300 and set(frame.seed) == SEEDS
                              and {"relation_availability",
                                   "gate_mass_all_nodes", "gate_weight",
                                   "gate_entropy", "effective_relation_count",
                                   "gate_entropy_multi",
                                   "effective_relation_count_multi",
                                   "gate_weight_multi_relation_nodes",
                                   "gate_mass_multi_relation_nodes",
                                   "shared_fraction", "shared_norm",
                                   "private_norm", "conflict",
                                   "private_conflict"}.issubset(
                                       set(frame.quantity))
                              and set(frame.get("model_revision", [])) == {
                                  SPARC_REVISION}
                              and set(frame.get("split_revision", [])) == {
                                  CHRONOLOGICAL_SPLIT_REVISION}
                              and set(frame.get("feature_revision", [])) == {
                                  PRIMARY_FULL_MATCH_FEATURE_REVISION}
                              and set(frame.get("graph_revision", [])) == {
                                  HETEROGENEOUS_GRAPH_REVISION}
                              and set(frame.get("loss_revision", [])) == {
                                  FIXED_LOSS_REVISION}
                              and set(frame.get("class_weight_revision", [])) == {
                                  CLASS_WEIGHT_REVISION},
                              f"rows={len(frame)}, seeds={sorted(set(frame.seed))}"))
    else:
        checks.append(_result("shared/private/conflict diagnostics", False, "missing"))

    path = OUT / "relation_observation_support.json"
    if path.exists():
        support = json.loads(path.read_text(encoding="utf-8"))
        pooled = support.get("pooled", {})
        valid = (
            support.get("revision") == "target_node_coobservation_v1"
            and support.get("graph_revision") ==
            HETEROGENEOUS_GRAPH_REVISION
            and pooled.get("matches") == 3464
            and pooled.get("graph_sides") == 6928
            and pooled.get("nodes") == 131901
            and 0 < pooled.get("targets_at_least_two_node_rate", 0) < 0.1
        )
        checks.append(_result(
            "node-level relation co-observation support", valid,
            f"nodes={pooled.get('nodes')}, at-least-two="
            f"{pooled.get('targets_at_least_two_node_rate', float('nan')):.4f}, "
            f"all-three={pooled.get('targets_all_three_node_rate', float('nan')):.4f}",
        ))
    else:
        checks.append(_result("node-level relation co-observation support",
                              False, "missing"))
    return checks


def audit_complexity():
    path = OUT / "model_complexity.parquet"
    expected = GRAPH_MODELS | {"CurrentScore"}
    if not path.exists():
        return [_result("model complexity benchmark", False, "missing")]
    frame = pd.read_parquet(path)
    required = {
        "model", "model_revision", "benchmark_revision", "parameters",
        "trainable_parameters", "batch_size", "device",
        "inference_ms_per_graph", "feature_revision",
        "graph_revision",
    }
    columns_ok = required.issubset(frame.columns)
    models_ok = set(frame.get("model", [])) == expected and len(frame) == len(expected)
    sparc_ok = columns_ok and set(frame.loc[
        frame["model"] == "SPARC-GNN", "model_revision"]) == {SPARC_REVISION}
    benchmark_ok = columns_ok and set(frame["benchmark_revision"]) == {
        "real_graph_batch_128_v2"}
    feature_ok = columns_ok and (
        set(frame.loc[frame["model"] != "CurrentScore", "feature_revision"])
        == {PRIMARY_FULL_MATCH_FEATURE_REVISION}
        and set(frame.loc[frame["model"] == "CurrentScore", "feature_revision"])
        == {"cutoff_score_only_v1"})
    graph_ok = columns_ok and set(frame["graph_revision"]) == {
        HETEROGENEOUS_GRAPH_REVISION}
    numeric_ok = columns_ok and (
        pd.to_numeric(frame["inference_ms_per_graph"], errors="coerce") > 0).all()
    passed = (columns_ok and models_ok and sparc_ok and benchmark_ok
              and feature_ok and graph_ok and numeric_ok)
    return [_result(
        "model complexity benchmark", passed,
        f"models={len(frame)}/{len(expected)}, SPARC={SPARC_REVISION}, "
        f"revision={'real_graph_batch_128_v2' if benchmark_ok else 'invalid'}, "
        f"feature={feature_ok}, graph={graph_ok}")]


def audit_feature_leakage(require_archives=True):
    path = OUT / "full_match_assist_proxy_audit.json"
    if not path.exists():
        return [_result("full-match assist proxy audit", False, "missing")]
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = payload.get("assist_only_calibrated", {})
    correlations = payload.get("test_correlations", {})
    archives = (
        OUT / "archive_fullmatch_assist_proxy_v3_20260905",
        OUT / "archive_fullmatch_assist_proxy_generalization_20260905",
    )
    passed = (
        payload.get("feature_revision") == ASSIST_PROXY_FEATURE_REVISION
        and payload.get("graph_revision") == HETEROGENEOUS_GRAPH_REVISION
        and payload.get("split_sizes") == {
            "train": 2426, "validation": 346, "test": 692}
        and float(metric.get("macro_f1", 0)) > 0.7
        and float(metric.get("r2", 0)) > 0.7
        and abs(float(correlations.get(
            "goal_assist_pair_difference", 0))) > 0.8
        and (not require_archives or all(directory.exists() for directory in archives))
    )
    return [_result(
        "full-match assist proxy audit", passed,
        f"macro-F1={metric.get('macro_f1')}, R2={metric.get('r2')}, "
        f"archives={sum(directory.exists() for directory in archives)}/2, "
        f"historical_archives_required={require_archives}")]


def audit_finite_size():
    checks = []
    observed_summary_path = OUT / "finite_size_observed_summary.csv"
    weak_summary_path = OUT / "finite_size_controlled_summary.csv"
    if observed_summary_path.exists() and weak_summary_path.exists():
        observed_summary = pd.read_csv(observed_summary_path).set_index("k")
        weak_summary = pd.read_csv(weak_summary_path).set_index("k")
        interpretation_ok = (
            int(observed_summary.raw_count_mean.idxmax()) == 7
            and observed_summary.loc[7, "q_mean"]
            > observed_summary.loc[3, "q_mean"]
            and (weak_summary.loc[3:7, "q_excess_mean"] < 0).all()
            and (weak_summary.loc[3:7, "q_excess_ci_high"] < 0).all()
            and abs(weak_summary.loc[7, "q_excess_cohen_dz"])
            < abs(weak_summary.loc[3, "q_excess_cohen_dz"])
            and set(observed_summary.n_graphs_total) == {6928}
        )
        checks.append(_result(
            "finite-size interpretation contract", interpretation_ok,
            f"raw peak={int(observed_summary.raw_count_mean.idxmax())}, "
            f"Q3={observed_summary.loc[3, 'q_mean']:.3f}, "
            f"Q7={observed_summary.loc[7, 'q_mean']:.3f}, "
            f"weak d3={weak_summary.loc[3, 'q_excess_cohen_dz']:.3f}, "
            f"d7={weak_summary.loc[7, 'q_excess_cohen_dz']:.3f}",
        ))
    else:
        checks.append(_result("finite-size interpretation contract", False,
                              "missing observed/control summary"))
    growth_path = OUT / "finite_size_growth_decomposition.parquet"
    if growth_path.exists():
        growth = pd.read_parquet(growth_path)
        maximum = max(float(growth["growth_identity_error"].abs().max()),
                      float(growth["occupancy_identity_error"].abs().max()))
        checks.append(_result("finite-size growth identities", maximum < 1e-10,
                              f"maximum absolute error={maximum:.3e}"))
        required = {
            "raw_growth_ratio", "occupancy_ratio",
            "raw_growth_threshold", "occupancy_growth_threshold",
            "capacity_dominated_growth",
        }
        if required.issubset(growth.columns):
            raw_condition = ((growth["raw_growth_ratio"] > 1.0)
                             == (growth["extension_probability"]
                                 > growth["raw_growth_threshold"]))
            occupancy_condition = ((growth["occupancy_ratio"] > 1.0)
                                   == (growth["extension_probability"]
                                       > growth["occupancy_growth_threshold"]))
            declared = ((growth["raw_growth_ratio"] > 1.0)
                        & (growth["occupancy_ratio"] < 1.0))
            corollary_ok = (raw_condition.all() and occupancy_condition.all()
                            and (declared == growth[
                                "capacity_dominated_growth"]).all())
        else:
            corollary_ok = False
        checks.append(_result(
            "finite-size growth-threshold corollary", corollary_ok,
            "raw and occupancy inequalities verified row-wise"
            if corollary_ok else f"missing/inconsistent={sorted(required-set(growth.columns))}"))
    else:
        checks.append(_result("finite-size growth identities", False, "missing"))

    for suffix, label, weak_preserved in (
            ("", "weak degree/density null", True),
            ("_directed_degree_diagnostic", "directed-degree sensitivity null", False)):
        path = OUT / f"finite_size_rewiring_diagnostics{suffix}.parquet"
        if not path.exists():
            checks.append(_result(label, False, "missing"))
            continue
        frame = pd.read_parquet(path)
        completion = bool(np.allclose(frame["swap_completion"], 1.0))
        invariant = bool((frame["rewired_edges"] ==
                          (frame["original_weak_edges"] if weak_preserved
                           else frame["original_directed_edges"])).all())
        manifest_path = OUT / f"finite_size_null{suffix}_manifest.json"
        controlled_path = OUT / f"finite_size_controlled{suffix}.parquet"
        manifest_ok = False
        if manifest_path.exists() and controlled_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            controlled = pd.read_parquet(
                controlled_path, columns=["match_id", "team_side"])
            keys = (controlled.drop_duplicates()
                    .sort_values(["match_id", "team_side"])
                    .reset_index(drop=True))
            manifest_ok = (
                manifest.get("graph_source_revision")
                == FINITE_SIZE_GRAPH_REVISION
                and manifest.get("allorder_repair_revision")
                == "skip_gtrie_header_v2"
                and manifest.get("max_attempts_per_target_swap") == 300
                and manifest.get("selected_network_sha256")
                == _network_digest(keys)
            )
        checks.append(_result(label, len(frame) == 20000 and completion
                              and invariant and manifest_ok,
                              f"rows={len(frame)}, all swaps complete={completion}, "
                              f"declared edge invariant={invariant}, "
                              f"canonical manifest={manifest_ok}"))
    return checks


def audit_highorder(require_repair=True):
    try:
        report = audit_census()
    except Exception as error:
        return [_result("exact motif census coverage", False,
                        f"audit raised {type(error).__name__}: {error}")]
    counts = report.get("counts", {})
    repair_path = ROOT / "data" / "analysis" / "highorder_header_repair.json"
    if repair_path.exists():
        repair = json.loads(repair_path.read_text(encoding="utf-8"))
        repair_ok = (
            repair.get("repair_revision") == "skip_gtrie_header_v2"
            and repair.get("network_graphs") == 6928
            and repair.get("corrected_rows") == counts.get(
                "allorder_rows_observed")
        )
        repair_detail = (
            f"revision={repair.get('repair_revision')}, "
            f"rows={repair.get('corrected_rows')}"
        )
    else:
        repair_ok = False
        repair_detail = "missing highorder_header_repair.json"
    return [
        _result(
            "exact motif census coverage",
            report.get("status") == "PASS",
            f"networks={counts.get('network_files')}, "
            f"all-order={counts.get('allorder_rows_observed')}, "
            f"k6/7 sparse rows={counts.get('highorder_sparse_rows')}",
        ),
    ] + ([_result("gtrie-header repair contract", repair_ok, repair_detail)]
         if require_repair else [])


def audit_event_clock(require_legacy_growth=True):
    checks = []
    pass_path = ROOT / "data" / "processed" / "events_pass.parquet"
    backup = (ROOT / "data" / "processed" /
              "archive_pre_clock_fix_20260903" / "events_pass.parquet")
    report = OUT / "statsbomb_clock_repair.json"
    if pass_path.exists():
        frame = pd.read_parquet(
            pass_path, columns=["minute", "second", "continuous_minute"])
        expected = (pd.to_numeric(frame["minute"], errors="coerce").fillna(0)
                    + pd.to_numeric(frame["second"], errors="coerce").fillna(0) / 60.0)
        mismatches = int((~np.isclose(
            frame["continuous_minute"], expected, equal_nan=True)).sum())
        checks.append(_result(
            "StatsBomb cumulative event clock", mismatches == 0,
            f"rows={len(frame)}, mismatches={mismatches}, "
            f"backup={backup.exists()}, report={report.exists()}"))
    else:
        checks.append(_result("StatsBomb cumulative event clock", False,
                              "missing events_pass.parquet"))

    growth_path = OUT / "motif_growth_timecourse.csv"
    if growth_path.exists():
        growth = pd.read_csv(growth_path)
        expected_orders = {3, 4, 5}
        monotone = all(
            group.sort_values("cutoff_minute")["mean_count"].diff().dropna().ge(-1e-12).all()
            for _, group in growth.groupby("motif_order_k")
        )
        rule_ok = set(growth["cutoff_rule"]) == {
            "minute + second/60 < cutoff"}
        valid = (set(growth["motif_order_k"]) == expected_orders
                 and monotone and rule_ok and np.isfinite(growth["mean_count"]).all())
        checks.append(_result(
            "temporal motif-growth reconstruction", valid,
            f"rows={len(growth)}, orders={sorted(set(growth['motif_order_k']))}, "
            f"monotone={monotone}, strict_rule={rule_ok}"))
    elif require_legacy_growth:
        checks.append(_result("temporal motif-growth reconstruction", False,
                              "missing motif_growth_timecourse.csv"))
    return checks


def audit_artifacts():
    basic_script = ROOT / "scripts" / "make_information_sciences_figures.py"
    extended_script = (ROOT / "scripts" /
                       "make_information_sciences_extended_figures.py")
    figure_specs = {
        PAPER / "figures" / "Figure_08_Model_Comparison.pdf": [
            OUT / "model_results.parquet", basic_script],
        PAPER / "figures" / "Figure_09_Generalization.pdf": [
            OUT / "generalization_results.parquet", extended_script],
        PAPER / "figures" / "Figure_10_Temporal_Cutoff.pdf": [
            OUT / "temporal_results.parquet",
            OUT / "temporal_causal_baselines.parquet",
            OUT / "temporal_validation_blends.parquet",
            extended_script],
        PAPER / "figures" / "Figure_11_Relation_Information_Decomposition.pdf": [
            OUT / "decomposition_diagnostics.parquet", extended_script],
        PAPER / "figures" / "Figure_03_Finite_Size_Controls.pdf": [
            OUT / "finite_size_observed_summary.csv",
            OUT / "finite_size_controlled_summary.csv", basic_script],
        PAPER / "figures" / "Figure_01_Method_Overview.pdf": [
            ROOT / "scripts/draw_sparc_architecture.py"],
    }
    problems = []
    valid_figures = []
    for figure, sources in figure_specs.items():
        missing_sources = [source for source in sources if not source.exists()]
        if missing_sources:
            problems.append(
                f"{figure.name}: missing sources "
                f"{[source.name for source in missing_sources]}")
            continue
        if not figure.exists() or figure.stat().st_size == 0:
            problems.append(f"{figure.name}: missing/empty")
            continue
        newest_source = max(source.stat().st_mtime for source in sources)
        if figure.stat().st_mtime < newest_source:
            problems.append(f"{figure.name}: stale")
            continue
        valid_figures.append(figure)

    verification_pdf = PAPER / "main.pdf"
    manuscript = PAPER / "main.tex"
    pdf_sources = [manuscript, *figure_specs]
    missing_pdf_sources = [source for source in pdf_sources
                           if not source.exists()]
    if missing_pdf_sources:
        problems.append("main.pdf: figure/manuscript dependency missing")
    elif (not verification_pdf.exists()
          or verification_pdf.stat().st_size == 0):
        problems.append("main.pdf: missing/empty")
    elif verification_pdf.stat().st_mtime < max(
            source.stat().st_mtime for source in pdf_sources):
        problems.append("main.pdf: stale")

    return [_result(
        "publication figures and verification PDF", not problems,
        "all dependencies current" if not problems
        else "; ".join(problems))]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-manuscript", action="store_true",
                        help="Audit data/results without the separately distributed TeX manuscript")
    parser.add_argument("--fresh-rebuild", action="store_true",
                        help="Do not require historical archives or the non-paper legacy growth cache; all main-paper numerical checks remain active")
    args = parser.parse_args()
    checks = (audit_full_graph_features() + audit_main()
              + audit_statistical_outputs()
              + audit_generalization() + audit_temporal()
              + audit_temporal_context_controls()
              + audit_components() + audit_complexity()
              + audit_feature_leakage(require_archives=not args.fresh_rebuild)
              + audit_highorder(require_repair=not args.fresh_rebuild) + audit_finite_size()
              + audit_event_clock(require_legacy_growth=not args.fresh_rebuild)
              + ([] if args.skip_manuscript else audit_artifacts()))
    payload = {
        "overall": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
        "provenance_mode": "fresh_rebuild" if args.fresh_rebuild else "historical_migration",
        "optional_legacy_growth_not_required": args.fresh_rebuild,
    }
    (OUT / "completion_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Information Sciences completion audit", "",
             f"Overall: **{payload['overall']}**", "",
             "| Check | Status | Evidence |", "|---|---:|---|"]
    for item in checks:
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['check']} | {item['status']} | {detail} |")
    report = ROOT / "INFORMATION_SCIENCES_COMPLETION_AUDIT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(f"{item['status']:4s} {item['check']}: {item['detail']}"
                    for item in checks))
    print(f"OVERALL {payload['overall']} -> {report}")
    raise SystemExit(0 if payload["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
