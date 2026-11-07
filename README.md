# SPARC-GNN and finite-size motif controls

Code for a manuscript revision targeting *Information Sciences*. SPARC-GNN
combines edge-aware relation encoders, observed-message shared/private
decomposition and motif/conflict-conditioned fusion. The study uses 3,464
StatsBomb Open Data matches from 21 competitions. This is not an accepted-paper
or universal state-of-the-art claim.

Complete-match experiments perform **structural retrodiction**, not forecasting.
Prospective experiments use strict event prefixes and cutoff-available context.
Goal-assist and shot-assist channels are redacted from primary model inputs.
Opposition events target a fixed opposing-team anchor, not an observed duel opponent.



## Install and check

Use Python 3.11 or later (tested locally with Python 3.11). Create a fresh
environment, install a PyTorch build appropriate for your hardware, then run:

```bash
python -m pip install -e ".[gnn,dev,pitch]"
python -m pytest tests -q
python -m scripts.check_finite_growth_theory
python -m scripts.draw_sparc_architecture
```

These checks use synthetic graphs or exact small finite graphs; they do not
download data or reproduce the empirical results. Figure 1 can be regenerated
without the dataset. Figure output defaults to `outputs/paper/figures` when no
sibling manuscript exists; `MOTIF_PAPER_DIR` overrides the manuscript directory.
Publication assets use `Figure_NN_English_Name` (for example,
`Figure_01_Method_Overview.pdf`); Figure 2 uses the `02a` and `02b` suffixes.
PDF, PNG and SVG versions share the same stem. Legacy exploratory figures keep
their separate names and are not manuscript figure inputs.

## Reproduce the study

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for cohort locking, graph building,
experiment order, version contracts, and the distinction between primary,
legacy and development results. The ordered experiment commands are in
[INFORMATION_SCIENCES_RUNBOOK.md](INFORMATION_SCIENCES_RUNBOOK.md).

The package contains:

- `src/`: graph construction, motif analysis, models and numerical analysis.
- `scripts/`: primary experiments, controls, plotting and provenance checks.
- `tests/`: synthetic regression tests; external motif tooling is optional.
- `reproducibility/`: exact split membership, cohort IDs and environment metadata.
- `results/reference/`: per-seed clean Table 3 metrics and their summaries.
- `results/development/`: separately labelled validation searches, not test results.

Large data caches, raw events, trained weights, local environments and manuscript
backups are intentionally excluded. Reference CSVs are not automatically loaded
as training outputs. Rebuild missing dependencies before invoking a data audit.
Historical cache-migration and revision-stamping utilities are excluded from
the release; they are not steps in a fresh reproduction.

## Research and software status

The frozen paper architecture remains in `src/gnn/mcar_model.py`.
`residual_candidate.py`, `edge_value_candidate.py` and
`anchor_readout_candidate.py` are development variants; their presence does not
mean they were adopted in the manuscript. HPO uses a validation-only protocol and
records its search space before fitting. Test performance must not guide search.

Connected-subset counts have a prior interpretation in node-reliability theory;
see [Brown and Mol (2018)](https://doi.org/10.1016/j.dam.2017.11.035).
The finite-size checks retain extension and removal factors to interpret growth,
rather than claiming the existing connected-set upper bound as a new theorem.

Use `CITATION.cff` to cite the software and record the actual release commit.
No manuscript DOI or acceptance status is invented. The MIT license is preserved
from the existing public repository; StatsBomb data and third-party tools retain
their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This directory is prepared for manual GitHub upload. No upload, push, release,
repository creation or access change was performed during packaging.
