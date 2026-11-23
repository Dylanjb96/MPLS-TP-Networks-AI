# Phase 5F Stage 4 — Candidate-Aware PPO Full Training Add-On

Read **`PHASE5F_STAGE4_START_TO_FINISH_WINDOWS_GUIDE.md`** before doing anything.

This package installs **only new Stage 4 source files** into an existing MPLS-AI project with Phase 5F Stage 1, 2 and 3 installed. It does not overwrite earlier phases and does not include the earlier simulator sources.

Windows fast path after extraction into MPLS-AI root:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5f_stage4.py
python -m pytest -q
python run_phase5f_stage4.py preflight --stage3-manifest outputs\phase5f\stage3_smoke_dev\stage3_smoke_manifest.json
python run_phase5f_stage4.py protocol
```

Follow the guide for the **required new-directory engineering quickcheck**. Once preflight, tests and quickcheck pass, initiate the **12 full experiments**:

```cmd
Phase5F_Stage4_12_Run_Windows.cmd
```

After all twelve runs are finished, **audit before selecting or opening Stage 5 heldout data**. The validation-only selector is:

```cmd
python run_phase5f_stage4.py select --runs-root outputs\phase5f\stage4_train --out outputs\phase5f\stage4_selection_FINAL
```

Selection is only a *development* checkpoint freeze. There are **no Stage 5 held-out evaluation commands** in this package. Reserved topology seeds 401–420 and traffic seeds 73000–73002 remain inaccessible to the training workflow. A successful training run does not imply GG-TA-RL outperforms greedy or footprint.
