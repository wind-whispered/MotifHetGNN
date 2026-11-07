"""
Cross-platform pipeline runner.
Works on Windows, macOS, and Linux.

Usage:
    python scripts/run_pipeline.py              # run all tasks
    python scripts/run_pipeline.py --from 4     # resume from Task 4
    python scripts/run_pipeline.py --only 8     # run only Task 8
    python scripts/run_pipeline.py --skip 9     # skip Task 9 (GNN)
    python scripts/run_pipeline.py --list       # list all tasks
"""
import argparse
import subprocess
import sys
import os
import io
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Windows: force UTF-8 for stdout/stderr so unicode chars (✓ ✗) don't crash
# on GBK/CP936 consoles. Must run before any print() calls.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASKS = [
    (1,  "task1_load",           "scripts/run_task1_load.py",
     "Data loading and parsing"),
    (2,  "task2_homogeneous",    "scripts/run_task2_homogeneous.py",
     "Homogeneous network construction"),
    (3,  "task3_heterogeneous",  "scripts/run_task3_heterogeneous.py",
     "Heterogeneous graph construction  [requires torch_geometric]"),
    (4,  "task4_homo_motifs",    "scripts/run_task4_homo_motifs.py",
     "Homogeneous motif enumeration     [requires gtrieScanner]"),
    (5,  "task5_hetero_motifs",  "scripts/run_task5_hetero_motifs.py",
     "Heterogeneous motif enumeration"),
    (6,  "task6_zscore",         "scripts/run_task6_zscore.py",
     "z-score significance testing      [slow: ~100 random networks each]"),
    (7,  "task7_spatiotemporal", "scripts/run_task7_spatiotemporal.py",
     "Spatiotemporal stratification"),
    (8,  "task8_regression",     "scripts/run_task8_regression.py",
     "OLS regression analysis"),
    (9,  "task9_gnn",            "scripts/run_task9_gnn.py",
     "GNN training and attribution      [requires torch_geometric + GPU recommended]"),
    (10, "task10_figures",       "scripts/run_task10_figures.py",
     "Figure and table generation"),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _color(text: str, code: str) -> str:
    """ANSI color — disabled on Windows unless ANSICON / Windows Terminal."""
    if sys.platform == "win32" and not os.environ.get("WT_SESSION"):
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _color(t, "32")
def red(t):    return _color(t, "31")
def yellow(t): return _color(t, "33")
def bold(t):   return _color(t, "1")


def _ensure_log_dir() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _run_task(task_num: int, name: str, script: str, log_dir: Path) -> bool:
    """
    Run a single task as a subprocess.
    Streams output to console AND writes to log file simultaneously.
    Returns True on success.
    """
    log_path = log_dir / f"{name}.log"
    start = time.time()

    print()
    print(bold(f">>> Task {task_num}: {name}"))
    print(f"    Script : {script}")
    print(f"    Log    : {log_path}")

    # Use sys.executable to ensure same Python environment
    cmd = [sys.executable, script]

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== {name} started at {datetime.now()} ===\n\n")
            log_file.flush()

            # Pass PYTHONIOENCODING so child scripts also output UTF-8 on Windows
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(Path.cwd()),
                env=child_env,
            )

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

            process.wait()

        elapsed = time.time() - start

        if process.returncode == 0:
            print(green(f"    [OK] Done in {elapsed:.1f}s"))
            return True
        else:
            print(red(f"    [FAILED] exit code {process.returncode} after {elapsed:.1f}s"))
            print(red(f"      Check log: {log_path}"))
            return False

    except FileNotFoundError:
        print(red(f"    [ERROR] Script not found: {script}"))
        return False
    except KeyboardInterrupt:
        print(yellow("\n    Interrupted by user."))
        raise


def _list_tasks():
    print(bold("\nAvailable tasks:"))
    print(f"  {'#':<4} {'Name':<26} Description")
    print("  " + "-" * 70)
    for num, name, script, desc in TASKS:
        print(f"  {num:<4} {name:<26} {desc}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform pipeline runner for football-motif-analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--from", dest="from_task", type=int, default=1,
                        metavar="N", help="Resume pipeline from task N (default: 1)")
    parser.add_argument("--to", dest="to_task", type=int, default=10,
                        metavar="N", help="Stop pipeline after task N (default: 10)")
    parser.add_argument("--only", type=int, metavar="N",
                        help="Run only task N (overrides --from/--to)")
    parser.add_argument("--skip", type=int, action="append", metavar="N",
                        default=[], help="Skip task N (can repeat: --skip 3 --skip 9)")
    parser.add_argument("--list", action="store_true",
                        help="List all tasks and exit")
    args = parser.parse_args()

    if args.list:
        _list_tasks()
        return

    # Determine which tasks to run
    if args.only is not None:
        selected = [t for t in TASKS if t[0] == args.only]
        if not selected:
            print(red(f"ERROR: Unknown task number {args.only}"))
            sys.exit(1)
    else:
        selected = [
            t for t in TASKS
            if args.from_task <= t[0] <= args.to_task
            and t[0] not in args.skip
        ]

    if not selected:
        print(yellow("No tasks selected. Use --list to see available tasks."))
        return

    # Header
    print(bold("\n" + "=" * 48))
    print(bold("  Football Motif Analysis Pipeline"))
    print(bold("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(bold("=" * 48))
    print(f"  Python  : {sys.executable}")
    print(f"  Tasks   : {', '.join(str(t[0]) for t in selected)}")
    if args.skip:
        print(f"  Skipped : {', '.join(str(s) for s in args.skip)}")

    log_dir = _ensure_log_dir()
    pipeline_start = time.time()
    failed_tasks = []

    for task_num, name, script, desc in selected:
        try:
            ok = _run_task(task_num, name, script, log_dir)
        except KeyboardInterrupt:
            print(yellow("\nPipeline interrupted."))
            sys.exit(130)

        if not ok:
            failed_tasks.append(task_num)
            print(red(f"\nPipeline stopped at Task {task_num}."))
            print(red(f"Fix the error and resume with:"))
            print(red(f"    python scripts/run_pipeline.py --from {task_num}"))
            sys.exit(1)

    # Summary
    elapsed = time.time() - pipeline_start
    print()
    print(bold("=" * 48))
    if not failed_tasks:
        print(green("  Pipeline complete!"))
        print(f"  Total time : {elapsed:.1f}s")
        print(f"  Figures    : outputs{os.sep}figures{os.sep}")
        print(f"  Tables     : outputs{os.sep}tables{os.sep}")
        print(f"  Logs       : logs{os.sep}")
    else:
        print(red(f"  Pipeline failed at tasks: {failed_tasks}"))
    print(bold("=" * 48))


if __name__ == "__main__":
    main()
