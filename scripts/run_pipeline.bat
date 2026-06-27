@echo off
REM =========================================
REM Football Motif Analysis Pipeline
REM Windows batch script
REM Run from project root: scripts\run_pipeline.bat
REM =========================================

setlocal enabledelayedexpansion

set LOG_DIR=logs
if not exist %LOG_DIR% mkdir %LOG_DIR%

echo ========================================
echo  Football Motif Analysis Pipeline
echo ========================================

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ and add to PATH.
    exit /b 1
)

REM ---- Helper: run one task ----
REM Usage: call :run_step STEP_NAME SCRIPT_PATH
goto :main

:run_step
    set STEP=%~1
    set SCRIPT=%~2
    set LOG=%LOG_DIR%\%STEP%.log
    echo.
    echo ^>^>^> %STEP%
    echo     Script: %SCRIPT%
    echo     Log:    %LOG%
    python %SCRIPT% > %LOG% 2>&1
    if errorlevel 1 (
        echo ERROR: %STEP% failed. Check %LOG%
        type %LOG%
        exit /b 1
    )
    echo     Done: %STEP%
    exit /b 0

:main
call :run_step "task1_load"           "scripts\run_task1_load.py"
if errorlevel 1 exit /b 1

call :run_step "task2_homogeneous"    "scripts\run_task2_homogeneous.py"
if errorlevel 1 exit /b 1

call :run_step "task3_heterogeneous"  "scripts\run_task3_heterogeneous.py"
if errorlevel 1 exit /b 1

call :run_step "task4_homo_motifs"    "scripts\run_task4_homo_motifs.py"
if errorlevel 1 exit /b 1

call :run_step "task5_hetero_motifs"  "scripts\run_task5_hetero_motifs.py"
if errorlevel 1 exit /b 1

call :run_step "task6_zscore"         "scripts\run_task6_zscore.py"
if errorlevel 1 exit /b 1

call :run_step "task7_spatiotemporal" "scripts\run_task7_spatiotemporal.py"
if errorlevel 1 exit /b 1

call :run_step "task8_regression"     "scripts\run_task8_regression.py"
if errorlevel 1 exit /b 1

call :run_step "task9_gnn"            "scripts\run_task9_gnn.py"
if errorlevel 1 exit /b 1

call :run_step "task10_figures"       "scripts\run_task10_figures.py"
if errorlevel 1 exit /b 1

echo.
echo ========================================
echo  Pipeline complete.
echo  Figures: outputs\figures\
echo  Tables:  outputs\tables\
echo ========================================
endlocal
