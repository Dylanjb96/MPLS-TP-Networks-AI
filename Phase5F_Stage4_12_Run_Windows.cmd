@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Phase 5F Stage 4 - twelve planned DEVELOPMENT-only runs.
REM Run in the MPLS-AI project root, after tests and preflight.
REM This batch never runs Stage 5 final heldout seeds.
cd /d "%~dp0"
python run_phase5f_stage4.py preflight --stage3-manifest outputs\phase5f\stage3_smoke_dev\stage3_smoke_manifest.json
if errorlevel 1 (
    echo ERROR: Stage 4 preflight failed. Review installation/source hashes.
    exit /b 1
)
for %%C in (A_standard B_guided C_enhanced D_full_hybrid) do (
    for %%S in (7 17 27) do (
        set "RUN=outputs\phase5f\stage4_train\%%C_seed%%S"
        if exist "!RUN!\training_provenance.json" (
            echo SKIP completed: !RUN!
        ) else (
            if exist "!RUN!\latest_state.pt" (
                echo RESUME: !RUN!
                python run_phase5f_stage4.py train --config %%C --agent-seed %%S --out "!RUN!" --resume
            ) else (
                echo START: !RUN!
                python run_phase5f_stage4.py train --config %%C --agent-seed %%S --out "!RUN!"
            )
            if errorlevel 1 (
                echo ERROR: training failed; stop and preserve output for diagnosis.
                exit /b 1
            )
        )
    )
)
echo TRAINING DONE. Verify twelve complete training_provenance.json files.
echo NEXT: python run_phase5f_stage4.py select --runs-root outputs\phase5f\stage4_train --out outputs\phase5f\stage4_selection_FINAL
exit /b 0
