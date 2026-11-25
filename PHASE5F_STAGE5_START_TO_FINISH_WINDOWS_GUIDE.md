# Phase 5F — Stage 5: Full start-to-finish Windows guide

**Your current position:** All twelve Stage 4 training runs are complete, the official model selector has been audited, and the **C_enhanced seed27 episode60** and **D_full_hybrid seed7 episode84** checkpoints are frozen. Stage 5 is the one-time final held-out evaluation. **Do not train or change anything while performing these steps.**

## A. Before the final data are opened

The Stage 5 release ZIP contains **both frozen checkpoints** and the official model-selection manifest. You do **not** need to copy the checkpoint files yourself or move the original `stage4_train` folder. The package adds only `phase5f_stage5/`, `stage5_frozen/`, `run_phase5f_stage5.py`, a new test, a requirements note, and guides. It should not replace any Phase 1–4 or Phase 5F Stage 1–4 files.

Make a backup of the existing project and preserve these folders unchanged:

```text
outputs\phase5f\stage2_60_dev
outputs\phase5f\stage2_150_250_dev
outputs\phase5f\stage2_500_dev
outputs\phase5f\stage3_smoke_dev
outputs\phase5f\stage4_train
outputs\phase5f\stage4_selection_FINAL
```

Review `PHASE5F_STAGE5_PROTOCOL_LOCK.md` **before** executing any `final` command. If your supervisor requires a different protection model, stop before opening the final seeds. The current study is a synthetic end-to-end disjoint-pair protection surrogate, **not MSRP compliance**.

## B. Installation

1. Download the provided `Phase5F_Stage5_Final_Heldout_AddOn.zip`.
2. Use Windows Extract All to extract **into** this existing project (merge directories):

```text
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI
```

3. Open Command Prompt and move into the project:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
```

Do not use WordPad/Notepad to resave source files because that may change file SHA-256 and correctly fail the source-integrity gate.

## C. Check the Python environment

The pre-tested Windows versions were Python **3.11.9**, NetworkX **3.5**, NumPy **2.2.6**, PyTorch **2.7.1+cu118**. These can stay as they are. The Stage 5 statistical routine also needs SciPy; check it without changing anything:

```cmd
python -c "import scipy; print('SciPy:', scipy.__version__)"
```

If the command fails with `ModuleNotFoundError: No module named 'scipy'`, install just this missing analysis dependency:

```cmd
python -m pip install "scipy>=1.12,<2"
```

Do not blindly upgrade Python, NumPy, NetworkX, PyTorch, or old Phase 5F source packages before final evaluation.

## D. Run automated tests and preflight (NO held-out access)

New Stage 5 tests:

```cmd
python -m pytest -q tests/test_phase5f_stage5.py
```

Expected if files unchanged: **21 passed**.

All earlier stages and new Stage 5 tests:

```cmd
python -m pytest -q
```

Expected if your Stage 4 project has the earlier **242 passing tests** (224 from Stage 3 plus 18 Stage 4 tests): **263 passed** after adding 21 Stage 5 tests.

Frozen source, checkpoint, actor-architecture and selected-model integrity:

```cmd
python run_phase5f_stage5.py preflight
```

Expected output includes:

```text
"result": "STAGE5_PREFLIGHT_OK"
"cohorts": 480
"policy_evaluations": 2400
"heldout_data_accessed": false
```

Two exact checkpoint SHA-256 digests printed by preflight must be:

```text
C: e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9
D: e55b7b551b09bffe036d5bb75a471d87bb75568b48fa64aac483f947454e8e44
```

**If preflight does not pass, STOP.** Do not bypass the source or checkpoint hash checks, edit the Stage 5 script, or run the final experiment with a different model.

## E. Development-only integration check (NO held-out access)

This small run uses an **already used Stage 4 validation topology/traffic seed**. It tests all five policies and an independent failure-scenario oracle; it is not new validation or final evidence.

```cmd
python run_phase5f_stage5.py devcheck --out outputs\phase5f\stage5_devcheck
```

Expected:

```text
"result": "STAGE5_DEVCHECK_OK"
"rows": 5
"heldout_data_accessed": false
```

Generated:

```text
outputs\phase5f\stage5_devcheck\stage5_devcheck_results.csv
outputs\phase5f\stage5_devcheck\stage5_devcheck_manifest.json
```

If `stage5_devcheck` already exists, the command **deliberately refuses to overwrite it**. Don't rerun it unless you choose a new *development-only* output name. Do not reinterpret devcheck as an independent statistical result.

**Send me your automated test, preflight and devcheck console results at this point if you would like the final gate reviewed before opening the reserved data.**

## F. Execute the entire frozen held-out experiment **ONCE**

After reviewing the research protocol and successful gates, run **one** command:

```cmd
python run_phase5f_stage5.py final --out outputs\phase5f\stage5_final --confirm-final
```

The command evaluates **all eight A×B×C cells** at load 4.0, for **20 unseen topologies × 3 traffic streams × 5 policies = 2,400** full sequential policy simulations, with shared arrivals and candidate lists within each matched group. It records the 480 completed matched cohorts atomically and prints lines beginning with `HELDOUT_COHORT_OK` as each completes. It does **not** train or modify either frozen model.

This is a substantial computation. Keep Windows powered and avoid moving/deleting outputs while it runs. If an exception occurs, preserve the exact console error and existing output files. **Do not substitute different seeds, loads, models or thresholds.**

### If Windows is interrupted

Provided no source, checkpoint or package dependency versions have changed, continue the **same** run from the saved cohorts:

```cmd
python run_phase5f_stage5.py final --out outputs\phase5f\stage5_final --resume --confirm-final
```

This validates and skips completed matched cohorts. It will refuse to resume if any source/checkpoint/runtime fingerprint differs. Do not edit, manually delete, or selectively repeat completed cohorts. A fully completed Stage 5 refuses to rerun automatically.

### When it finishes

You should see:

```text
STAGE5_FINAL_OK rows=2400 cohorts=480 topologies=20
```

The output folder will contain:

```text
outputs\phase5f\stage5_final\
    locked_final_plan.json
    cohorts\                               (480 compressed per-cohort evidence files)
    heldout_runs.csv                       (2,400 policy evaluations)
    heldout_cell_descriptives.csv           (40 cell/policy summaries)
    heldout_primary_inference.csv           (1 primary + 3 secondary comparisons)
    heldout_factorial_effects.csv           (7 fixed factorial effects)
    heldout_topology_paired_deltas.csv      (paired topology-level differences)
    stage5_final_manifest.json             (integrity and inference provenance)
```

Primary cell C/D traces for every LSP request are preserved in compressed primary cohort files; other cells preserve verified policy-level results and scenario-oracle digests.

## G. Final read-only verification

Run:

```cmd
python run_phase5f_stage5.py verify --root outputs\phase5f\stage5_final
```

Expected:

```text
"result": "STAGE5_VERIFY_OK"
"verified_policy_rows": 2400
"matched_cohorts": 480
"topology_clusters": 20
```

This reads the saved evidence, recomputes the primary paired effect, reconciles the primary decision traces, and checks every results-table SHA-256. It **does not regenerate** held-out networks or retrain a policy.

## H. Upload the complete evidence for independent audit

If feasible, zip the **entire** final output directory (including `cohorts`):

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5f\stage5_final' -DestinationPath 'Phase5F_Stage5_Final_Evidence.zip' -Force"
```

Upload `Phase5F_Stage5_Final_Evidence.zip`. If it exceeds the upload limit, upload these **seven** root files separately first: `locked_final_plan.json`, `heldout_runs.csv`, `heldout_cell_descriptives.csv`, `heldout_primary_inference.csv`, `heldout_factorial_effects.csv`, `heldout_topology_paired_deltas.csv`, `stage5_final_manifest.json`. Then we can arrange how to audit the compressed decision traces and cohort evidence in smaller archives. Include your `STAGE5_FINAL_OK` and `STAGE5_VERIFY_OK` console messages.

## I. Interpretation and what happens after Stage 5

**Do not retrain, rerun or adjust settings to improve the held-out results.** Report the C vs greedy effect and the 95% interval regardless of whether it is positive, negative, or inconclusive. Secondary tests and 7 factorial effects are distinct statistical families, Holm adjusted. The main estimate is based on **20 independent topology clusters**, with three traffic streams per topology (not n=60 independent samples and not n=120,000 independent LSP requests). The selected validation gain **is not** itself a final result.

Following an independent audit of the Stage 5 files, **Stage 6** integrates the findings into the dissertation, including unsuccessful or null outcomes and the fact that this synthetic end-to-end failure model is **not** standards-compliant MPLS-TP MSRP or operational railway protection timing.

**Important:** once you run section F, the final held-out test has been opened. Do not go back and change any Stage 1–4 design or model choices.
