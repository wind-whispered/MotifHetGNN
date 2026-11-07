# Reproducibility guide

This guide explains how to rebuild the inputs used by the MotifHetGNN study.
Start with the [installation instructions](README.md#quick-start), and run all
commands from the repository root with that Python environment active.

The repository includes cohort and split manifests, reference metrics and
environment metadata. Raw StatsBomb files, processed data, trained weights and
the manuscript are obtained or generated separately.

## Choose a workflow

| Goal | Required work |
| --- | --- |
| Check the implementation | Run the synthetic tests and finite-graph check in the README |
| Reproduce the GNN comparison | Prepare data and heterogeneous graphs, verify splits, then follow the neural stages in the runbook |
| Reproduce the structural analyses | Also build passing networks, motif statistics and the high-order census |
| Reconstruct empirical figures | Complete the analyses required by each plot, then run the figure commands below |

The GNN's seven-dimensional typed-triad context is computed directly from each
heterogeneous graph. It does not require the class-resolved high-order census.
The complete study and its completion audit require both structural and neural
outputs.

## 1. Obtain and record the data

Download [StatsBomb Open Data](https://github.com/hudl/open-data) into a directory
named `StatsBomb`:

```bash
git clone https://github.com/hudl/open-data.git StatsBomb
git -C StatsBomb rev-parse HEAD
```

Save the printed commit with your experiment records. The default
`data.raw_dir` in [config.yaml](config.yaml) is `StatsBomb/data`, so the expected
layout is:

```text
StatsBomb/
└── data/
    ├── competitions.json
    ├── matches/
    ├── events/
    └── lineups/
```

The historical `statsbomb/open-data` URL redirects to `hudl/open-data`. Data
attribution and licensing information is linked in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#statsbomb-open-data).

The supplied [cohort manifest](reproducibility/match_ids.json) fixes membership
to **3,464 matches**. The original upstream data commit was not recorded; the
manifest therefore fixes match identities but cannot pin the exact event-file
contents. Retain your data checkout and its commit to make a new run traceable.

## 2. Parse events and build graphs

```bash
python -m scripts.run_task1_load --match-ids reproducibility/match_ids.json
python -m scripts.run_task2_homogeneous
python -m scripts.build_dyad_census
python -m scripts.run_task3_heterogeneous
```

The loader checks that every requested match exists in the upstream match
metadata. Keep `data.max_matches` at least `3464`, or set it to `null`: a smaller
value truncates the selected cohort. Use a fresh generated-data directory when
changing cohorts so old graph files cannot enter the new experiment.

Expected outputs include:

| Directory or file | Contents |
| --- | --- |
| `data/processed/` | Match metadata, lineups, event tables and per-match shards |
| `data/networks/homogeneous/` | Passing graphs at the primary threshold `w0 = 2` |
| `data/analysis/dyad_census.parquet` | Reciprocal and asymmetric dyad counts |
| `data/networks/heterogeneous/` | One PyTorch Geometric graph per match |

Passing edges in the primary homogeneous graphs require more than two passes.
The threshold-sensitivity statistics also use `w0 = 0` and `w0 = 10`.

Optionally precompute the GNN context to reuse it across runs:

```bash
python -m scripts.prepare_motif_context
```

Otherwise, the dataset computes typed-triad context when loading each graph.

## 3. Verify the fixed splits

The primary chronological split keeps each calendar date wholly within one
partition:

| Partition | Matches |
| --- | ---: |
| Training | 2,426 |
| Validation | 346 |
| Test | 692 |

Before training, check that the rebuilt graphs produce the packaged split. Run
the following in a Python session started from the repository root:

```python
import json
from pathlib import Path
from src.gnn.dataset import FootballHeteroDataset, split_dataset_by_metadata
from src.networks.heterogeneous import HETEROGENEOUS_GRAPH_REVISION

cohort = json.loads(Path("reproducibility/match_ids.json").read_text())
expected = json.loads(Path("reproducibility/chronological_split.json").read_text())
dataset = FootballHeteroDataset(
    root=".", required_graph_revision=HETEROGENEOUS_GRAPH_REVISION
)
assert set(dataset.match_ids) == set(cohort), "Graph cohort differs from manifest"
actual = split_dataset_by_metadata(
    dataset, "data/processed/matches_meta.parquet", strategy="chronological"
)
for name, ids in zip(("train", "validation", "test"), actual):
    assert set(ids) == set(expected[name]), f"Split mismatch: {name}"
print("Cohort and chronological split match the packaged manifests.")
```

The main model script writes its computed split to
`data/information_sciences/chronological_split.json`. For a workflow that needs
this file before the main comparison, copy the packaged manifest after the
checks above:

```bash
python -c "from pathlib import Path; import shutil; p = Path('data/information_sciences'); p.mkdir(parents=True, exist_ok=True); shutil.copy2('reproducibility/chronological_split.json', p / 'chronological_split.json')"
```

[generalization_splits.json](reproducibility/generalization_splits.json) records
the random, chronological, competition-held-out and team-disjoint memberships
for all five seeds. Compare regenerated generalization splits with this file
before interpreting them as reproductions of the reference protocols.

## 4. Build structural analyses

For the structural study, run:

```bash
python -m scripts.run_task4_homo_motifs
python -m scripts.run_task5_hetero_motifs
python -m scripts.run_task6_zscore
python -m scripts.run_task7_spatiotemporal
python -m scripts.run_task8_regression
```

### Motif enumeration backend

[src/motifs/enumerator.py](src/motifs/enumerator.py) uses the included exact
Python enumerator by default. To select the optional external `gtrieScanner`
backend, obtain its executable separately and set `GTRIE_SCANNER_BIN` to its
full path before starting Python.

Linux or macOS example:

```bash
export GTRIE_SCANNER_BIN=/path/to/gtrieScanner
```

Windows PowerShell example:

```powershell
$env:GTRIE_SCANNER_BIN = "C:\tools\gtrieScanner.exe"
```

[.env.example](.env.example) documents these settings; scripts do not load an
`.env` file automatically. Confirm the chosen backend in the enumeration logs
before a large run.

### High-order census and spatial controls

The following stage is computationally expensive. It enumerates directed
canonical classes at orders six and seven and counts weakly connected induced
subsets at every order from two to the number of nodes:

```bash
python -m scripts.run_task11_highorder
python -m scripts.run_task11b_postprocess
python -m scripts.run_task_R4_robustness
python -m scripts.audit_highorder_census_coverage
python -m scripts.audit_round3_structure
```

This supplies `census_sizes_allk.parquet`, `highorder_census_k67.parquet`,
`incremental_r2_k7.parquet` and the two spatial-grid correlation matrices.
The finite-size analysis in the runbook reads the all-order census-size table.
Counting connected subsets and resolving their directed isomorphism classes
are separate operations.

For the connection to node-reliability theory, see
[Brown and Mol (2018)](https://doi.org/10.1016/j.dam.2017.11.035).
The finite-size analysis studies extension and removal factors; it treats the
connected-set upper bound as existing theory.

Network reciprocity is the fraction of directed links with a reverse link.
Template reciprocity is derived from adjacency matrices in
[src/motifs/reciprocity.py](src/motifs/reciprocity.py). The two spatial grids
measure sensitivity to bin boundaries, not the accuracy of inferred opposition
destinations.

## 5. Train, analyze and audit

Use the [experiment runbook](INFORMATION_SCIENCES_RUNBOOK.md) for the full command
sequence. It covers the primary comparison, ablations, relation controls,
generalization, event-prefix prediction and causal context baselines.

The reference protocol uses these revision identifiers:

| Component | Identifier |
| --- | --- |
| Primary features | `no_assist_pass_intensity_v2` |
| Heterogeneous graphs | `pass_multiplicity_edge13_v2` |
| Chronological split | `complete_date_boundaries_v2` |
| SPARC-GNN messages | `availability_masked_messages_v3` |
| Fixed multitask loss | `sqrt_invfreq_ce_fixed_half_v2` |
| Training-derived class weights | `train_sqrt_inverse_frequency_mean1_v1` |

Checkpoint reuse requires compatible model, graph, feature, split and loss
provenance. Preserve these identifiers when reporting results. Temporal runs
also record their event-prefix graph, score-timeline and score-context versions.

For a fresh checkout, the final numerical audit is:

```bash
python -m scripts.audit_information_sciences_completion --fresh-rebuild --skip-manuscript
```

`--fresh-rebuild` removes requirements for historical migration receipts and an
optional legacy plot cache. `--skip-manuscript` excludes checks of a separately
compiled TeX project. The main numerical, feature, split, census and event-clock
checks remain active. Run this audit after generating its inputs; missing-data
failures in a new checkout are expected.

## 6. Reconstruct figures

Generate the architecture schematic without downloading the dataset:

```bash
python -m scripts.draw_sparc_architecture
```

For the higher-order motif atlas, the plotting script expects an
`allorder_summary.json` lookup. After `run_task11b_postprocess`, derive the lookup
from its computed class counts with this Python snippet:

```python
import json
from pathlib import Path

directory = Path("data/analysis")
summary = json.loads((directory / "highorder_summary.json").read_text())
lookup = {"table": [
    {"k": int(k), "classes_obs": values["classes_observed"]}
    for k, values in sorted(summary["orders"].items(), key=lambda item: int(item[0]))
]}
(directory / "allorder_summary.json").write_text(
    json.dumps(lookup, indent=2), encoding="utf-8"
)
```

After the corresponding structural and neural analyses have completed:

```bash
python -m scripts.make_paper_figures
python -m scripts.make_revised_figures B
python -m scripts.make_information_sciences_figures
python -m scripts.make_information_sciences_extended_figures
python -m scripts.make_review_supplementary_figures
```

These commands read saved analyses. The reference metric CSVs alone cannot
supply their graph statistics, per-match predictions or control results.

Publication figures use descriptive stems such as `Figure_01_Method_Overview`.
The architecture script exports PDF, PNG and SVG; available formats for other
plots depend on their generating script. In a standalone checkout, publication
figures go to `outputs/paper/figures/`; a detected sibling manuscript can change
this default. Set `MOTIF_PAPER_DIR` explicitly to choose a destination, for example:

```powershell
$env:MOTIF_PAPER_DIR = (Join-Path (Get-Location) "outputs/paper")
```

## Development experiments and reporting

The optional validation-search workflow is:

```bash
python -m scripts.run_round3_hpo
python -m scripts.summarize_round3_hpo
```

It records a protocol and source hashes before fitting, screens eight settings
per model at seed 7, and repeats each model's top two settings at seeds 13 and
21. Configuration selection uses mean normalized validation loss. Test graphs
are not loaded by this search. Its results belong to a separate development
comparison, and the historical study test set should not be described as a new
untouched holdout for later model development.

The candidate modules `residual_candidate.py`, `edge_value_candidate.py` and
`anchor_readout_candidate.py` in `src/gnn/` are development variants. The primary
SPARC-GNN comparison uses the configuration linked from the README.

When sharing a reproduction, record the code and data commits, package and
hardware versions, configuration, seeds, split manifests and provenance
identifiers. Retain per-seed metrics and predictions. The recorded environment
is a reference snapshot; floating-point and CUDA operations can vary across
hardware, so use statistical agreement rather than requiring identical model
weights.
