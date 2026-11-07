"""
Master script for all revision tasks (R1-R5) and revised figures.

Run order:
  1. R5: Add n_matches to score-state data (fast, data-only)
  2. R2: Team-strength baseline (fast tabular)
  3. R3: Structure vs feature ablation (needs GNN training, ~2h)
  4. R1: Multi-seed evaluation of all 7 models (needs GNN, ~10h)
  5. R4: Robustness check (fast correlation computation)
  6. Revised figures: B, K, L, N, O

Usage:
  python run_revision_tasks.py          # all tasks
  python run_revision_tasks.py R5 R2    # specific tasks
  python run_revision_tasks.py figs     # figures only
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASK_SCRIPTS = {
    "R5": ROOT / "run_task_R5_scorestate_counts.py",
    "R2": ROOT / "run_task_R2_strength.py",
    "R3": ROOT / "run_task_R3_struct_feat.py",
    "R1": ROOT / "run_task_R1_multiseed.py",
    "R4": ROOT / "run_task_R4_robustness.py",
    "figs": ROOT / "make_revised_figures.py",
}

DEFAULT_ORDER = ["R5", "R2", "R3", "R1", "R4", "figs"]


def run(script: Path):
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT.parent),
    )
    if result.returncode != 0:
        print(f"WARNING: {script.name} exited with code {result.returncode}")
    return result.returncode


def main():
    sel = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ORDER
    print(f"Revision tasks to run: {sel}")
    exit_codes = {}
    for key in sel:
        if key not in TASK_SCRIPTS:
            print(f"Unknown task: {key}. Available: {list(TASK_SCRIPTS.keys())}")
            continue
        exit_codes[key] = run(TASK_SCRIPTS[key])

    print(f"\n{'='*60}")
    print("REVISION TASKS COMPLETE")
    print(f"{'='*60}")
    for k, code in exit_codes.items():
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {k}: {status}")


if __name__ == "__main__":
    main()
