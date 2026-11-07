# runbook

Run commands from the repository root. Every long neural experiment writes a
partial Parquet file and resumes by model/seed rather than restarting completed
work.

Run the temporal neural matrix by itself. The match-state builder and both
calibration scripts perform substantial JSON, graph-loading or CPU inference
work and measurably slow the GPU training process when launched concurrently.
After the neural matrix finishes, the listed order reuses its provenance-
matched checkpoints and resumes the corrected state cache safely.

```powershell
python scripts/audit_full_graph_features.py
python scripts/audit_highorder_census_coverage.py
python scripts/run_finite_size_motif_analysis.py --null-model both
python scripts/analyze_finite_size_growth_decomposition.py
python scripts/run_information_sciences_models.py
python scripts/analyze_information_sciences_results.py
python scripts/run_information_sciences_ablation.py
python scripts/analyze_ablation_paired_effects.py
python scripts/evaluate_sparc_revision_decision.py
python scripts/analyze_information_decomposition.py
python scripts/audit_relation_observation_support.py
python scripts/audit_full_match_assist_proxy.py
python scripts/run_information_sciences_relation_controls.py
python scripts/analyze_relation_control_paired_effects.py
python scripts/run_relation_subset_retraining.py
python scripts/analyze_relation_subset_effects.py
python scripts/run_information_sciences_generalization.py
python scripts/analyze_generalization_paired_effects.py
python scripts/audit_score_timeline.py
python scripts/build_temporal_cutoff_graphs.py
python scripts/audit_temporal_graph_features.py
python scripts/audit_temporal_assist_leakage.py
python scripts/run_temporal_cutoff_experiment.py
python scripts/build_temporal_match_state.py
python scripts/evaluate_temporal_causal_baselines.py
python scripts/evaluate_temporal_validation_blends.py
python scripts/evaluate_temporal_context_graph_blends.py
python scripts/analyze_temporal_paired_effects.py
python scripts/analyze_temporal_controlled_effects.py
python scripts/benchmark_model_complexity.py
python scripts/finalize_information_sciences_results.py --scope all
python scripts/make_information_sciences_figures.py
python scripts/make_information_sciences_extended_figures.py
python scripts/audit_information_sciences_completion.py
```

## Claim-bearing outputs

- `data/information_sciences/ablation_results.parquet`: component and loss
  ablations on the chronological split.
- `data/information_sciences/generalization_results.parquet`: random,
  chronological, competition-held-out and team-disjoint evaluation.
- `data/information_sciences/temporal_results.parquet`: strict event-prefix
  evaluation, score baselines, score-aware edge-free TeamPool-MLP,
  HeteroSAGE-sum and SPARC-GNN. Every row records the corrected score-timeline
  and exact un-clipped score-context revisions.
- `data/information_sciences/temporal_match_state.parquet`: 97 strictly
  pre-cutoff score/xG/shot/event-state summaries for each match and cutoff;
  the manifest records the half-open clock rule and confirms that retrospective
  assist annotations are not read.
- `data/information_sciences/temporal_causal_baselines.parquet`: validation-
  selected Score+Strength and CausalMatchState-HGB controls. Pre-match team
  strength is updated only after a complete calendar-date batch.
- `data/information_sciences/temporal_validation_blends.parquet` and
  `temporal_context_graph_blends.parquet`: explicitly labelled validation-only
  calibrated extensions. They are not replacements for the equal-budget main
  comparison and must not be selected by test performance.
- `data/information_sciences/temporal_controlled_effects.csv`: 425 deterministic
  seed-paired or neural-seed-versus-fixed-control effects covering both
  calibration layers, all five cutoffs and five oriented metrics.
- `data/information_sciences/relation_subset_results.parquet` and
  `relation_subset_paired_effects.csv`: all seven non-empty relation subsets
  retrained over matched seeds and their conditional marginal utilities.
- `data/information_sciences/relation_controls_decomp.parquet`: deletion,
  passing-endpoint permutation and opposing-team-anchor shift controls.
- `data/information_sciences/decomposition_diagnostics.parquet`: shared/private
  norms, gate weights and raw/private conflict diagnostics.
- `data/information_sciences/finite_size_controlled.parquet` and
  `finite_size_growth_decomposition.parquet`: weak-projection degree/density
  controls and exact finite-size growth factors.
- `data/information_sciences/finite_size_controlled_directed_degree_diagnostic.parquet`:
  sensitivity null preserving every directed in/out degree. It does not
  constrain weak-projection density and is therefore not reported as a second
  density-matched control.
- `data/information_sciences/finite_size_rewiring_diagnostics*.parquet` and
  `finite_size_null*_manifest.json`: accepted-swap diagnostics, deterministic
  seeds, declared invariants and selected-network content hashes for both null
  ensembles.
- `data/information_sciences/model_complexity.parquet`: parameter counts and
  batched inference timing.
- `data/information_sciences/quarantined_full_match_score_leakage.parquet`:
  audit-only rows removed from the complete-match comparison because the
  metadata scores are final labels. Current-score baselines are valid only in
  `temporal_results.parquet`, where they are observed at an explicit cutoff.

