# Phase 5F Stage 5 — Reporting-only recovery from the final CSV-schema exception

**Scope:** This procedure repairs only the *post-experiment CSV serialization* after all **480 distinct `HELDOUT_COHORT_OK`** lines were printed. It does **not** rerun held-out experiments, modify the frozen Stage 5 source, retrain models, select a new checkpoint or change the research protocol.

## Diagnosis

`phase5f_stage5.statistics.analyze()` generates one preregistered **primary** contrast and three **secondary** contrasts. Only the three secondary p-values receive Holm adjustment. Their dictionaries consequently have a `p_holm_adjusted` field, while the primary contrast does not. The original strict CSV writer in `phase5f_stage5/workflow.py` requires identical columns in every row, resulting in:

```
ValueError: inconsistent CSV row keys
```

This happens **after** all 480 cohort files have been saved and after `heldout_runs.csv` and `heldout_cell_descriptives.csv` were written, but **before** `stage5_final_manifest.json` could be produced. No new experimental result is required.

The recovery script supplies a **blank/not-applicable** Holm field to the primary CSV row at *serialization time only*; it does not alter any statistics. It reuses the original locked analysis, enforces all 480 cohort file identities, verifies hashes before and after, and runs the unchanged official read-only verifier.

## Windows instructions

1. **Do not run `final` or `final --resume` yet.** Do not delete or rename `outputs\phase5f\stage5_final` or edit any `phase5f_stage5/` `.py` files.
2. Download `recover_phase5f_stage5_reporting.py` and put it in the root of your existing project, next to `run_phase5f_stage5.py`. This is a new helper outside every frozen source directory.
3. Open Windows Command Prompt, then run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python recover_phase5f_stage5_reporting.py --root outputs\phase5f\stage5_final
```

Expected terminal markers:

```
STAGE5_RECOVERY_LOCK_OK cohorts=480 policies_per_cohort=5 ...
STAGE5_POSTPROCESS_RECOVERY_OK rows=2400 cohorts=480 topologies=20 cohort_sha256_unchanged=True
STAGE5_VERIFY_OK ... all_five_tables_sha256_verified=True primary_RL_traces_verified=True
```

4. Then independently run the **original unmodified** verifier:

```cmd
python run_phase5f_stage5.py verify --root outputs\phase5f\stage5_final
```

This should report `STAGE5_VERIFY_OK`.

5. Back up the **entire** completed directory, including all 480 compressed cohort files. For upload, run:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5f\stage5_final' -DestinationPath 'Phase5F_Stage5_Final_Evidence.zip' -Force"
```

Upload `Phase5F_Stage5_Final_Evidence.zip` for a complete independent audit.

## Important cautions

- Do **not** edit `phase5f_stage5/workflow.py` or `phase5f_stage5/statistics.py`: their SHA-256 hashes are part of the locked protocol; changing them would invalidate the resume/verification fingerprint.
- Do **not** delete any cohorts or rerun an individual topology/stream because of the analysis exception.
- Keep all Stage 4 checkpoints frozen; no retraining or additional held-out model tuning is justified.
- This repair changes only an inapplicable CSV column, not the numerical primary/secondary/factorial results or their statistical interpretations.
- If the helper reports anything other than the expected completion/verification messages, preserve the original folder and paste the **full new error**. Do not experiment with changing source files or deleting data.
