# MotifHetGNN


---

## Overview

This repository provides the full analysis pipeline applied to 3,464 football matches from 21 elite competitions (StatsBomb Open Data). The framework:

1. Represents each match as a **heterogeneous directed graph** (passing, duelling, turnover edges)
2. Performs an **exact motif census** from dyads to order *k* = 14
3. Trains a **heterogeneous graph neural network** (HetGNN) with gradient-based attribution to decode match outcomes

## Project Structure

```
MotifHetGNN/
├── src/
│   ├── data/          # Task 1: loading, parsing, cleaning
│   ├── networks/      # Tasks 2–3: homogeneous/heterogeneous graph construction
│   ├── motifs/        # Tasks 4–6: motif enumeration and z-score testing
│   ├── analysis/      # Tasks 7–8: spatiotemporal stratification and regression
│   ├── gnn/           # Task 9: HetGNN training and Integrated Gradients attribution
│   └── visualization/ # Tasks 10–11: all paper figures and LaTeX tables
├── scripts/           # One script per pipeline task (run_task*.py)
├── tests/             # Unit tests for all modules
├── outputs/
│   └── tables/        # Generated LaTeX tables (figures reproduced by pipeline)
├── config.yaml        # All hyperparameters and paths (edit here, not in source)
└── .env.example       # Environment variable template
```

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/wind-whispered/MotifHetGNN.git
cd MotifHetGNN
```

### 2. Install Python dependencies

```bash
pip install -e ".[gnn,dev]"
```

Requires Python ≥ 3.9. Key packages: `torch`, `torch_geometric`, `networkx`, `statsmodels`, `matplotlib`, `pandas`, `pyarrow`.

### 3. Download StatsBomb Open Data

```bash
git clone https://github.com/statsbomb/open-data.git StatsBomb/data
```

### 4. Install gtrieScanner (required for motif enumeration)

Download and compile from: https://www.dcc.fc.up.pt/~pribeiro/asd/gtriescanner/

Then set the binary path in `.env`:

```bash
cp .env.example .env
# edit .env: set GTRIE_SCANNER_BIN=/path/to/gtrieScanner
```

### 5. Run the pipeline

```bash
# Full pipeline (Tasks 1–11)
bash scripts/run_pipeline.sh

# Or run individual tasks:
python scripts/run_task1_load.py
python scripts/run_task2_homogeneous.py
python scripts/run_task3_heterogeneous.py
python scripts/run_task4_homo_motifs.py
python scripts/run_task5_hetero_motifs.py
python scripts/run_task6_zscore.py
python scripts/run_task7_spatiotemporal.py
python scripts/run_task8_regression.py
python scripts/run_task9_gnn.py
python scripts/run_task9b_attribution.py

# Generate all paper figures
python scripts/make_paper_figures.py
python scripts/make_revised_figures.py
```

## Configuration

All parameters (thresholds, GNN hyperparameters, path settings) are in `config.yaml`. Edit this file rather than modifying source code.

## Running Tests

```bash
pytest tests/ -v
```

## Data Flow

```
StatsBomb JSON
  └─ Task 1  → data/processed/      (events, lineups, match metadata)
  └─ Task 2  → data/networks/       (homogeneous passing networks, w0 ∈ {0,2,10})
  └─ Task 3  → data/networks/       (heterogeneous graphs)
  └─ Task 4  → data/motifs/         (homogeneous motif counts, k = 3–14)
  └─ Task 5  → data/motifs/         (heterogeneous motif counts)
  └─ Task 6  → data/motifs/         (z-scores against random baseline)
  └─ Task 7  → data/analysis/       (spatiotemporal stratification)
  └─ Task 8  → data/analysis/       (OLS regression, 43-dim feature vector)
  └─ Task 9  → data/gnn/            (HetGNN model weights, predictions)
  └─ Task 9b → data/gnn/            (Integrated Gradients attribution)
  └─ Task 10 → outputs/             (paper figures and LaTeX tables)
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Deep learning backend |
| `torch_geometric` | ≥ 2.3 | Heterogeneous GNN |
| `networkx` | ≥ 3.0 | Graph construction and analysis |
| `statsmodels` | ≥ 0.14 | OLS regression |
| `matplotlib` | ≥ 3.7 | Figure generation |
| `pandas` / `pyarrow` | — | Data processing |
| `captum` | ≥ 0.6 | Integrated Gradients attribution |
| `gtrieScanner` | — | Exact motif enumeration (external binary) |

## Citation

If you use this code, please cite:

```
Tian, L., Li, X., Liang, H., Cai, Y., & Li, Z. (2025).
Cooperative and Adversarial Interaction Structure in Competitive Complex Networks:
Interpretable All-Order Motif Characterisation with Heterogeneous Graph Decoding.
https://github.com/wind-whispered/MotifHetGNN
```

Data source: StatsBomb Open Data (https://github.com/statsbomb/open-data).
Please include the StatsBomb logo when publishing results based on this data.

## License

MIT
