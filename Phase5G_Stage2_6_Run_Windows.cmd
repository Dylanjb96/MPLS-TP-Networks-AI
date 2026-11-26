@echo off
setlocal EnableExtensions
cd /d "%~dp0"
REM Exactly six declared experiments. --resume is safe for new or existing runs.
REM If an individual run fails, this batch stops without running another.
for %%M in (mlp gnn) do (
    for %%S in (7 17 27) do (
        echo ==== Phase 5G Stage 2 model=%%M seed=%%S ====
        python run_phase5g_stage2.py train --model %%M --agent-seed %%S --episodes 48 --validate-every 12 --out outputs\phase5g\stage2_train\%%M_seed%%S --cache-root outputs\phase5g\stage2_cohort_cache --resume
        if errorlevel 1 (
            echo ERROR: Stage 2 training failed on %%M seed %%S. Fix the issue and rerun this batch to resume.
            exit /b 1
        )
    )
)
echo PHASE5G_STAGE2_ALL_SIX_RUNS_OK
endlocal
