#!/bin/bash
# Full pipeline execution script (Linux / macOS).
# For Windows: use  python scripts/run_pipeline.py
# Run from project root: bash scripts/run_pipeline.sh
set -e

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "========================================"
echo " Football Motif Analysis Pipeline"
echo "========================================"

run_step() {
    STEP=$1
    SCRIPT=$2
    LOG="$LOG_DIR/${STEP}.log"
    echo ""
    echo ">>> $STEP"
    echo "    Script: $SCRIPT"
    echo "    Log:    $LOG"
    python $SCRIPT 2>&1 | tee $LOG
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: $STEP failed. Check $LOG"
        echo "Resume with: python scripts/run_pipeline.py --from <task_number>"
        exit 1
    fi
    echo "    Done: $STEP"
}

run_step "task1_load"           "scripts/run_task1_load.py"
run_step "task2_homogeneous"    "scripts/run_task2_homogeneous.py"
run_step "task3_heterogeneous"  "scripts/run_task3_heterogeneous.py"
run_step "task4_homo_motifs"    "scripts/run_task4_homo_motifs.py"
run_step "task5_hetero_motifs"  "scripts/run_task5_hetero_motifs.py"
run_step "task6_zscore"         "scripts/run_task6_zscore.py"
run_step "task7_spatiotemporal" "scripts/run_task7_spatiotemporal.py"
run_step "task8_regression"     "scripts/run_task8_regression.py"
run_step "task9_gnn"            "scripts/run_task9_gnn.py"
run_step "task10_figures"       "scripts/run_task10_figures.py"

echo ""
echo "========================================"
echo " Pipeline complete."
echo " Figures: outputs/figures/"
echo " Tables:  outputs/tables/"
echo "========================================"
