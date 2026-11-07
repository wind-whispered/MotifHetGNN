# Reproducibility

## Data and cohort

Clone the public corpus into `StatsBomb`, not `StatsBomb/data`:

```bash
git clone https://github.com/statsbomb/open-data.git StatsBomb
python -m scripts.run_task1_load --match-ids reproducibility/match_ids.json
python -m scripts.run_task2_homogeneous
python -m scripts.build_dyad_census
python -m scripts.run_task3_heterogeneous
```

`config.yaml` expects `StatsBomb/data/competitions.json`. The explicit cohort
manifest fixes membership to 3,464 IDs and fails if an upstream match is missing.
The original upstream Git commit was not recorded in the supplied local tree;
cohort membership alone does not freeze the contents of event files. Record the
data checkout commit and retain its files when repeating the study. The package
does not claim byte-for-byte corpus reproducibility without that snapshot.

Copy `reproducibility/chronological_split.json` to
`data/information_sciences/chronological_split.json` after creating the parent
directory. The primary split has 2,426 training, 346 validation and 692 test
matches; a calendar date belongs wholly to one partition. The packaged
`generalization_splits.json` records the other original splits for comparison.
Check regenerated split memberships against those manifests before fitting.

## Structural prerequisites

Build the passing and heterogeneous motif summaries before plotting or training
with motif context:

```bash
python -m scripts.run_task4_homo_motifs
python -m scripts.run_task5_hetero_motifs
python -m scripts.run_task6_zscore
python -m scripts.run_task7_spatiotemporal
python -m scripts.run_task8_regression
```

The exact directed census can use `gtrieScanner`, obtained separately from its
upstream authors. Export `GTRIE_SCANNER_BIN` in your shell to point to the
executable. `.env.example` is documentation; the scripts do not auto-load `.env`.
Inspect the enumerator's selected backend before an expensive run.

`run_task11_highorder.py` is a legacy exact canonical-class extension and can be
very expensive. The finite-size occupancy analysis counts connected subsets
separately with `run_finite_size_motif_analysis.py`; it does not require a complete
high-order canonical motif vocabulary for prediction. Do not confuse all-order
connected-subset counts with exhaustive class-resolved directed census results.

For the full paper (including class-resolved orders six and seven), this legacy-
named census stage is still required before `audit_highorder_census_coverage`.
It is not a prerequisite for the GNN-only experiments. Also run the documented
dyad, all-order and zone-grid analysis stages needed by the corresponding plots;
missing cached analyses are not supplied by the reference metric CSVs.

```bash
python -m scripts.run_task11_highorder
python -m scripts.run_task11b_postprocess
python -m scripts.run_task_R4_robustness
```

The first command is the expensive canonical census, not a quick check. The
second supplies `incremental_r2_k7.parquet`; the third supplies both descriptive
spatial-grid correlation matrices. These scripts are retained under their
historical names, but `run_task_R4_robustness` now labels its actual binning
operation accurately. The main structural plots do not require legacy GNN
attribution outputs.

## Neural experiments and audits

Follow `INFORMATION_SCIENCES_RUNBOOK.md` after preparing these inputs. Neural
experiments and calibration should not be launched concurrently on one GPU.
Some scripts retain their own frozen protocol constants; changing `config.yaml`
does not override every experiment's seeds or split rules. Read the script's
documented `IS_*` environment overrides when requesting a subset.

Primary contracts include `no_assist_pass_intensity_v2`,
`pass_multiplicity_edge13_v2`, `complete_date_boundaries_v2` and
`availability_masked_messages_v3`. Reusing a checkpoint requires its matching
feature, graph, split and loss provenance. Never combine clean and assist-enabled
rows or substitute current-score labels into complete-match inputs.

Run the final result audit without a separately distributed manuscript using:

```bash
python -m scripts.audit_information_sciences_completion --skip-manuscript --fresh-rebuild
```

Omit `--skip-manuscript` only when `MOTIF_PAPER_DIR` points to a compiled matching TeX
project. A failed missing-data audit in a fresh checkout does not mean the
synthetic unit tests failed or that empirical results have been reproduced.
`--fresh-rebuild` omits requirements for historical migration archives, the old
header-repair receipt and a non-paper legacy growth-curve cache. That 300-match
legacy plot does not have a retained sample manifest and is not claimed to be
reproducible or included in the submission. If its cache exists it is still
checked. The flag does not disable any main-paper numerical, feature, split,
graph, census or event-clock check. The default audit retains the original
project's history requirements.

## Figure reconstruction

```bash
python -m scripts.make_paper_figures
python -m scripts.make_revised_figures B
python -m scripts.make_information_sciences_figures
python -m scripts.make_information_sciences_extended_figures
python -m scripts.make_review_supplementary_figures
```

These commands read the saved analyses; they are not stand-alone data generators.
Keep scalar values fixed when modifying presentation. Figure 7's two axes are
equal-sized squares and the clustering in Figure 5 follows the actual leaf order.
Figure 1 is original vector artwork, generated by `draw_sparc_architecture.py`.

Template reciprocity is computed from adjacency integers in
`src/motifs/reciprocity.py`, not a hand-entered label table. Network reciprocity
counts directed links with a reverse link; it is not mutual/connected dyads.
`python -m scripts.audit_round3_structure` verifies the corrected structural
summaries. The two spatial-grid correlation matrices are a bin-boundary
sensitivity analysis, not a test of graph-edge destination reconstruction.

## Development search

```bash
python -m scripts.run_round3_hpo
python -m scripts.summarize_round3_hpo
```

This writes a new protocol and source hashes before fitting, loads only training
and validation graphs, redacts assist flags for every model, screens eight
settings per model at seed 7 and repeats each model's top two at seeds 13 and 21.
Selection uses mean normalized validation loss, not the best seed or test score.
The known historical test set is not advertised as a fresh untouched holdout.
Development CSVs must not replace the primary Table 3 values.

The runtime snapshot records tested package versions, not a universal CUDA
installation recipe. CUDA scatter operations may differ across environments.
Keep per-seed predictions and report uncertainty rather than claiming exact
bitwise training reproducibility across hardware.
