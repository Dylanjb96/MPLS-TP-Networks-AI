# Phase 5G Stage 2 — Start-to-finish Windows instructions

**Only proceed if Phase 5G V1.1 is installed and passes 23 Phase 5G tests, 286 full-suite tests, and `PHASE5G_PREFLIGHT_OK`.** Those checks have already passed in the user's Windows environment. Phase 5F and the Phase 5G Stage 1 outputs must remain untouched.

## 1. Install the Phase 5G Stage 2 add-on

Download `Phase5G_Stage2_Matched_GNN_MLP_Training_AddOn.zip`, then extract **to the project root**:

```text
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI
```

The ZIP **only adds** a new `phase5g_stage2` package, `run_phase5g_stage2.py`, `tests/test_phase5g_stage2.py`, `Phase5G_Stage2_6_Run_Windows.cmd`, and these documents. It does not replace Stage 1 V1.1, any Stage 5 checkpoint, Phase 5F code, or completed experiment files.

Open CMD:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
```

## 2. Test the new implementation

```cmd
python -m pytest -q tests/test_phase5g_stage2.py
python -m pytest -q
```

Expected **16 Stage 2 tests**; assuming the previous 286 tests are unchanged, **302 total**. Any failure must be resolved before training.

## 3. Verify frozen source, checkpoint, candidate scores and parameter counts

```cmd
python run_phase5g_stage2.py preflight
```

Look for `PHASE5G_STAGE2_PREFLIGHT_OK`, plus two `PHASE5G_STAGE2_PREFLIGHT_ARM_OK` lines. The MLP should have **15,758** parameters; the GNN **15,778** parameters. The source-lock guard checks the frozen Phase 5F Enhanced PPO checkpoint.

## 4. Execute a development-only integration check

```cmd
python run_phase5g_stage2.py quickcheck --out outputs\phase5g\stage2_quickcheck_dev
```

Look for `PHASE5G_STAGE2_QUICKCHECK_OK`. This is a small **two-episode 60-node GNN-only** check on already designated Phase 5G training and validation partitions. It is **not** an outcome or a full-training result. The output folder contains training history, validation scores, best checkpoint, the model state journal, compressed decision trace and provenance.

**If tests, preflight or quickcheck fail, stop here** and send the error with the associated files. Do not launch full training.

## 5. Conduct the six fixed-budget, matched-seed training runs

Read `PHASE5G_STAGE2_PROTOCOL_LOCK.md` before executing. **Do not modify the episode count, model seeds, training seeds or validation cell between arms.**

```cmd
Phase5G_Stage2_6_Run_Windows.cmd
```

The script runs `mlp_seed7`, `mlp_seed17`, `mlp_seed27`, `gnn_seed7`, `gnn_seed17`, `gnn_seed27` in sequence, **48 episodes each**. It uses a common disk candidate cache at `outputs\phase5g\stage2_cohort_cache`. The program reports progress after every committed episode. Each run saves a `latest_state.pt` journal, source-locked plan, training history and validation records.

**If Windows training is interrupted:** rerun the **same batch command**. It uses `--resume` and verifies the saved plan, source fingerprints and checkpoint hash. It will skip completed runs. Do not use new seeds or recreate the prior run folders. To inspect an individual run:

```cmd
python run_phase5g_stage2.py status --root outputs\phase5g\stage2_train\gnn_seed7
```

A major source/environment change invalidates safe resumption; resolve that issue before continuing. Do not edit or delete the shared candidate cache while training is running.

## 6. Select the models on the development validation partition

Only after **all six runs** complete, run **once**:

```cmd
python run_phase5g_stage2.py select --runs-root outputs\phase5g\stage2_train --out outputs\phase5g\stage2_selection_FINAL
```

This produces `selection_trials.csv`, `selection_architectures.csv`, and `selection_manifest.json`. The selector chooses the best architecture by mean of **three validation-best checkpoint values**, breaks equal scores in favour of the MLP, and preserves the best individual seed/checkpoint for **both** architectures.

**Do not rerun or overwrite model selection** once it is frozen; if it already exists, simply upload it.

## 7. Upload the evidence for the independent audit

To preserve the runs and model-selection records without including a potentially large candidate cache, run:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5g\stage2_train','outputs\phase5g\stage2_selection_FINAL' -DestinationPath 'Phase5G_Stage2_Training_Evidence.zip' -Force"
```

Upload `Phase5G_Stage2_Training_Evidence.zip`, along with the results of Step 2 (pytest), Step 3 (preflight), and Step 4 (quickcheck). Also retain your original directory and common candidate cache locally.

The independent audit will verify source hashes, all six training plans and checkpoints, matched cohort schedules, validation pairing, intervention counts, no infeasible admissions, and the validation-only architecture selection. It will *not* describe Stage 2 validation as an independent final test.

## 8. After this training programme

Only after its audit will we prospectively freeze Phase 5G Stage 3/4: independent robustness/generalization comparisons on genuinely unseen corridor-ladder and hierarchical topologies, traffic stress tests, and stronger scenario-minimax optimization, plus optional DDQN if justified. **Do not access Phase 5G final seeds `801–820 / 77000–77002` yet.**

### Important scientific limitation

This is a synthetic end-to-end link-disjoint single-link-failure reservation surrogate, not MSRP implementation or real Irish Rail data. Supervisor approval of the abstraction is still required.
