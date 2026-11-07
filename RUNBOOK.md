# Experiment runbook

This runbook organizes the experiments used by the Information Sciences study.
Prepare the environment, data and fixed splits with the
[reproducibility guide](REPRODUCIBILITY.md) first. Run commands from the
repository root, in the order shown within each stage.

Outputs are written under `data/information_sciences/` unless a different path
is stated. The default neural protocol uses seeds `42, 7, 13, 21, 37`. Run one
neural experiment at a time on a single GPU. In particular, finish temporal
training before starting the match-state builder or calibration stages, which
also consume substantial CPU, memory and graph-loading resources.

## Prerequisites

- Rebuild the 3,464-match cohort and verify the chronological split.
- Generate heterogeneous graphs for the neural stages.
- For stage 1 and the full completion audit, also finish the structural and
  high-order census stages in the reproducibility guide.
- Keep the default protocol settings for comparison with the packaged results.
  Reduced runs are useful for exploration but do not reproduce the full study.

Check the complete-match graph features before training:

```bash
python -m scripts.audit_full_graph_features
```

## 1. Finite-size structural controls

Requires the passing graphs and `data/analysis/census_sizes_allk.parquet`.

```bash
python -m scripts.audit_highorder_census_coverage
python -m scripts.run_finite_size_motif_analysis --null-model both
python -m scripts.analyze_finite_size_growth_decomposition
```

The default null experiment selects 200 networks and generates 100
randomizations per network through order seven. The primary weak-projection
null preserves node count, weak-edge count and the weak-degree sequence. The
directed-degree diagnostic preserves in/out degrees, but does not fix
weak-projection density; interpret the two controls according to these distinct
constraints.

## 2. Complete-match model comparison

```bash
python -m scripts.run_information_sciences_models
python -m scripts.analyze_information_sciences_results
```

This stage evaluates TeamPool-MLP, HeteroSAGE-sum, R-GCN, HGT,
EdgeAware-Fixed, MotifGate-ObservedOnly and SPARC-GNN on the chronological split.
It writes per-seed metrics, per-match predictions and aggregate comparisons.

Complete-match graphs describe the whole match, so these results measure
structural retrodiction. The primary feature contract masks assist annotations
and excludes final-score inputs. Compare regenerated metrics with
[results/reference/](results/reference/).

## 3. Ablations and relation diagnostics

```bash
python -m scripts.run_information_sciences_ablation
python -m scripts.analyze_ablation_paired_effects
python -m scripts.evaluate_sparc_revision_decision
python -m scripts.analyze_information_decomposition
python -m scripts.audit_relation_observation_support
python -m scripts.audit_full_match_assist_proxy
```

The ablation runner creates the provenance-matched checkpoints used by the
subsequent decomposition and relation-control analyses. The observation audit
measures how often multiple incoming relations coexist at a node. The
assist-proxy audit measures the signal in retrospective annotations and records
it separately from the primary comparison.

## 4. Relation perturbation and subset retraining

```bash
python -m scripts.run_information_sciences_relation_controls
python -m scripts.analyze_relation_control_paired_effects
python -m scripts.run_relation_subset_retraining
python -m scripts.analyze_relation_subset_effects
```

Perturbation controls evaluate relation deletion, passing-endpoint permutation
and opposing-team-anchor shifts. Subset experiments retrain all seven non-empty
combinations of the three relation types over matched seeds. These address
separate questions: sensitivity of a fitted model and usefulness of a relation
when the model is trained with that subset.

## 5. Generalization

```bash
python -m scripts.run_information_sciences_generalization
python -m scripts.analyze_generalization_paired_effects
```

The four protocols are random, chronological, competition-held-out and
team-disjoint. The default held-out competition is the Premier League. The
team-disjoint protocol assigns connected components of the team co-occurrence
graph as whole groups. The default model set is TeamPool-MLP, HeteroSAGE-sum,
HGT, EdgeAware-Fixed and SPARC-GNN.

## 6. Temporal graph construction and prediction

```bash
python -m scripts.audit_score_timeline
python -m scripts.build_temporal_cutoff_graphs
python -m scripts.audit_temporal_graph_features
python -m scripts.audit_temporal_assist_leakage
python -m scripts.run_temporal_cutoff_experiment
```

Cutoffs are 15, 30, 45, 60 and 75 minutes. Every prefix uses the half-open
interval `[0, cutoff)`: an event at the cutoff is excluded. Node statistics,
edges, substitutions, tactical context and typed-triad features are rebuilt
from the available prefix. Final outcomes remain the targets.

The temporal comparison includes current-score baselines, graph models and
score-aware TeamPool-MLP, HeteroSAGE-sum and SPARC-GNN variants. Score-aware
inputs contain the observed cutoff score, not the final score.

## 7. Causal context and validation-selected calibration

Start this stage after temporal training finishes:

```bash
python -m scripts.build_temporal_match_state
python -m scripts.evaluate_temporal_causal_baselines
python -m scripts.evaluate_temporal_validation_blends
python -m scripts.evaluate_temporal_context_graph_blends
python -m scripts.analyze_temporal_paired_effects
python -m scripts.analyze_temporal_controlled_effects
```

The match-state builder computes 97 pre-cutoff summaries of score, expected
goals, shots and event activity. Score+Strength updates team strength after a
complete calendar-date batch; CausalMatchState-HGB uses cutoff-available state.

Blend weights and calibration choices are selected on validation data. Report
these extensions separately from the equal-budget neural comparison, and keep
test metrics out of configuration selection.

## 8. Consolidate results and verify completeness

```bash
python -m scripts.benchmark_model_complexity
python -m scripts.finalize_information_sciences_results --scope all
python -m scripts.audit_information_sciences_completion --fresh-rebuild --skip-manuscript
```

The finalizer consolidates result tables and quarantines any legacy
complete-match current-score rows it encounters. The completion audit writes
`data/information_sciences/completion_audit.json` and
`INFORMATION_SCIENCES_COMPLETION_AUDIT.md` with individual checks and an overall
status. A full run should finish with `PASS`; inspect failed checks before
reporting a completed reproduction.

The flags above are appropriate for a standalone checkout. They leave the main
numerical and provenance checks enabled while excluding historical migration
artifacts and a separately compiled manuscript. See the
[reproducibility guide](REPRODUCIBILITY.md#5-train-analyze-and-audit) for details.

To create plots, follow the
[figure reconstruction steps](REPRODUCIBILITY.md#6-reconstruct-figures), including
the all-order class-count lookup required by the higher-order motif atlas.

## Output reference

Paths below are relative to `data/information_sciences/`.

| Output | Purpose |
| --- | --- |
| `model_results.parquet`, `predictions.parquet` | Primary complete-match metrics and per-match predictions |
| `ablation_results.parquet` | Component and training-objective ablations |
| `decomposition_diagnostics.parquet` | Shared/private norms, fusion weights and conflict diagnostics |
| `relation_controls_decomp.parquet` | Fitted-model relation perturbations |
| `relation_subset_results.parquet`, `relation_subset_paired_effects.csv` | Retrained relation subsets and paired effects |
| `generalization_results.parquet` | Four generalization protocols |
| `temporal_results.parquet` | Event-prefix models and score baselines |
| `temporal_match_state.parquet` | Cutoff-available state summaries |
| `temporal_causal_baselines.parquet` | Score+Strength and CausalMatchState-HGB controls |
| `temporal_validation_blends.parquet`, `temporal_context_graph_blends.parquet` | Validation-selected calibrated extensions |
| `temporal_controlled_effects.csv` | Paired temporal effects against the context controls |
| `finite_size_controlled.parquet`, `finite_size_growth_decomposition.parquet` | Primary finite-size controls and extension/removal factors |
| `finite_size_controlled_directed_degree_diagnostic.parquet` | Directed-degree-preserving sensitivity analysis |
| `finite_size_rewiring_diagnostics*.parquet`, `finite_size_null*_manifest.json` | Swap diagnostics, seeds, invariants and network hashes |
| `model_complexity.parquet` | Parameter counts and batched inference timing |
| `quarantined_full_match_score_leakage.parquet` | Audit-only legacy rows, when present |

Frozen hashes for the original primary outputs are recorded in
[reproducibility/reference_provenance.json](reproducibility/reference_provenance.json).
The packaged CSVs provide reference values; they do not replace the Parquet
files and checkpoints required by downstream analyses.

## Configuration and resuming

[config.yaml](config.yaml) controls shared data and analysis settings. Several
study scripts also define their own fixed constants; editing the YAML file
does not override every training setting.

| Variable | Supported use |
| --- | --- |
| `IS_SEEDS` | Seed selection in the main, ablation, generalization, temporal and relation experiment runners |
| `IS_VARIANTS`, `IS_RUN_TAG` | Variant selection and tagged output names in the ablation runner |
| `IS_MODELS` | Model selection in the generalization and temporal runners |
| `IS_PROTOCOLS`, `IS_HOLDOUT_COMPETITION` | Generalization protocols and held-out competition |
| `IS_SPLIT_AUDIT_ONLY=1` | Write generalization split records without training |
| `IS_CUTOFFS` | Cutoff selection in the temporal experiment runner; the graph builder uses its fixed five cutoffs |
| `IS_RELATION_SUBSETS` | Subset selection in the relation-subset runner |

Set overrides explicitly in the shell and clear exploratory overrides before
running the full protocol. Post-processing and audit scripts can require all
five seeds even when a training runner supports a subset.

Several training runners save `*_partial.parquet` files and resume completed
model/seed, protocol or cutoff units. An interrupted training unit may need to
run again. Reuse partial outputs only with compatible provenance and inspect
the runner's resume messages. The exact high-order census has a different
execution path and does not provide the same model/seed resume mechanism.
