# Phase 5F Stage 4 — Start-to-Finish Windows Guide

**Read this before running the full 12 experiments.** The Stage 4 add-on implements real multi-topology candidate-aware PPO training and *validation-only* model selection. It does **not** contain the Stage 5 held-out evaluation command. The complete full-scale Stage 4 training has **not** been executed by the assistant: that must run on your Windows machine.

**Research system:** a synthetic national-scale transport network, not the actual Iarnród Éireann MPLS-TP topology. Protection remains an end-to-end link-disjoint single-link-failure scenario surrogate, **not** MPLS-TP Shared Ring Protection. Supervisor approval of that modelling abstraction remains outstanding.

## Step 0 — Freeze the older evidence

Preserve `outputs\phase5f\stage2_60_dev`, `stage2_150_250_dev`, `stage2_500_dev`, and `stage3_smoke_dev`, as well as previous Phase 5 and Phase 5E checkpoints. The Stage 4 add-on installs new files only and does not overwrite Stage 1/2/3 implementation. Reserve **physical topology seeds 401–420 and traffic stream seeds 73000–73002** solely for Stage 5 final evaluation. Do not evaluate them to look for better hyperparameters.

The agreed Stage 4 configuration is in `PHASE5F_STAGE4_PROTOCOL_LOCK.md`. Do not make post hoc changes to the four-arm design based on validation/held-out performance. Stage 4 training uses network sizes 60/150/250/500, K=6, static/dynamic request holding, loads 1.6/2.5/4.0, 10% headroom and agent seeds 7/17/27. There are 4 × 3 = **12 training runs**.

## Step 1 — Install the ZIP

1. Download `Phase5F_Stage4_Full_Training_AddOn.zip` from the chat.
2. Extract its contents **into** your existing MPLS-AI root, keeping its folder hierarchy.
3. Do not unpack it under an extra nested folder named `Phase5F_Stage4_Full_Training_AddOn`.
4. Confirm that all these paths exist:

```text
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\run_phase5f_stage4.py
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\phase5f_stage4\workflow.py
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\phase5f_stage4\ablation.py
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\phase5f_stage4\selection.py
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\tests\test_phase5f_stage4.py
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\Phase5F_Stage4_12_Run_Windows.cmd
```

The new module requires the existing `phase5f`, `phase5f_stage2` and `phase5f_stage3` modules. Do not update NetworkX, NumPy or PyTorch merely to install Stage 4. Your previously verified environment is Python 3.11.9 / NetworkX 3.5 / NumPy 2.2.6 / PyTorch 2.7.1+cu118. The main training runner uses the CPU and limits PyTorch to one CPU thread; the CUDA suffix is not required for this runner.

## Step 2 — Run installation tests

Open **Windows Command Prompt** (not Python interactive shell):

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5f_stage4.py
python -m pytest -q
```

The supplied add-on passed **18 new Stage 4 tests** in local development. The combined Stage 1–4 *new* suite passed **56 tests** locally; previous source archives may not contain all your original project tests. If your 224-test Windows suite is unchanged and no tests are disabled or added, the expected total is **242 passed**. If anything fails, stop and share the traceback rather than training anyway.

## Step 3 — Verify source compatibility and seed safety

```cmd
python run_phase5f_stage4.py preflight --stage3-manifest outputs\phase5f\stage3_smoke_dev\stage3_smoke_manifest.json
python run_phase5f_stage4.py protocol
```

Expected preflight prefix: `STAGE4_PREFLIGHT_OK`. The preflight checks frozen Stage 3/Stage 2 source hashes against your uploaded smoke manifest, the synthetic protected candidate feasibility mask, and action selection under all four ablations. It checks the same 60-node Stage 3 topology fingerprint previously audited. If source hashes mismatch, **do not train** until we audit the difference.

The `protocol` command prints the reserved training, validation and untouched final seed partitions. No Stage 4 CLI has an `evaluate --role final` or analogous Stage 5 operation.

## Step 4 — Run one minimal Stage 4 end-to-end engineering test

This verifies the full sequence of training → PPO update → validation (hybrid, greedy, footprint) → checkpoint → manifest using only a fresh development directory. It is **not** a dissertation training run, must not be fed into the 12-run selector, and uses reduced nodes/episodes solely to check integration:

```cmd
python run_phase5f_stage4.py train --out outputs\phase5f\stage4_quickcheck_dev --config D_full_hybrid --agent-seed 7 --episodes 2 --validate-every 1 --patience 0 --train-sizes 60 --train-loads 1.6 --train-modes static --train-topologies 201 --train-streams 71000 --val-size 60 --val-load 1.6 --val-mode static --val-topologies 301 --val-streams 72000
```

Expected success prefix: `STAGE4_TRAIN_OK`. Inspect:

```text
outputs\phase5f\stage4_quickcheck_dev\
  training_plan.json
  training_history.csv
  validation_summary.csv
  validation_checkpoints.csv
  best_model.pt
  latest_state.pt
  training_provenance.json
```

You should see **two** training-history rows and **six** validation rows (2 evaluations × 3 policies) in this engineering test. The `validation_summary.csv` hybrid `decision_records` should correctly reflect request-level decisions, fixing the Stage 3 summary bookkeeping issue. The protected-pair feasibility violation count must be zero. **Do not infer algorithm superiority from this tiny integration test.**

**Directory protection:** Stage 4 refuses to overwrite an existing evidence directory. If you previously ran the quickcheck, use a different new `--out` name rather than deleting or overwriting the earlier run. The full 12-run script uses its own separate folders.

## Step 5 — Run the 12 planned full training experiments

Only do this after Steps 2–4 pass, and after the Stage 4 protocol and protection-model qualification have been recorded for your dissertation.

**Recommended:** double-click or invoke the included batch file from a Command Prompt already in the project root:

```cmd
Phase5F_Stage4_12_Run_Windows.cmd
```

The batch runs these 12 configurations **sequentially** (no simultaneous GPU/CPU jobs), automatically resumes interrupted runs with valid saved state, skips completed runs without retraining, and stops on an error:

| Configuration | Initialization seeds | Training output directories |
|---|---|---|
| `A_standard` | 7, 17, 27 | `stage4_train\A_standard_seed7/17/27` |
| `B_guided` | 7, 17, 27 | `stage4_train\B_guided_seed7/17/27` |
| `C_enhanced` | 7, 17, 27 | `stage4_train\C_enhanced_seed7/17/27` |
| `D_full_hybrid` | 7, 17, 27 | `stage4_train\D_full_hybrid_seed7/17/27` |

**Manual alternative** (Windows **Command Prompt**, typed directly; use `%%C` and `%%S` instead if placed in a `.cmd` file):

```cmd
for %C in (A_standard B_guided C_enhanced D_full_hybrid) do for %S in (7 17 27) do python run_phase5f_stage4.py train --config %C --agent-seed %S --out outputs\phase5f\stage4_train\%C_seed%S
```

The defaults intentionally implement the frozen protocol: 96 maximum PPO episodes, validation every 12 episodes, five non-improving-checkpoint early-stop patience; four training network sizes, three development loads, two traffic modes, 20 training topology seeds, eight training stream seeds; **eight primary validation cohorts** at 500 nodes / K=6 / dynamic / load 4.0; 10% headroom and max 40 explored working paths. Every algorithm's baseline/validation cohorts share the same cached structural protected alternatives.

Avoid modifying these settings partway through the 12 experiments. The selector deliberately rejects a batch with mismatched settings. The local shared candidate inventory cache is placed under:

```text
outputs\phase5f\stage4_shared_candidate_cache
```

**Keep this cache** until all 12 trainings are finished. It is a content-keyed gzip/JSON optimization of deterministic candidate generation (and includes training/validation seeds only), *not* new learned evidence. Candidate generation and 500-node provisioning episodes are computationally substantial: preserve free disk space and do not expect the smoke-test runtime to predict full-training runtime.

### If a run is interrupted

- The runner automatically writes `latest_state.pt` after every **completed** PPO episode. This state contains the model, optimizer, progress counters, training/validation records and Torch/Python/NumPy RNG states.
- Reissue **exactly the original command** with `--resume` on that specific incomplete run, e.g.:

```cmd
python run_phase5f_stage4.py train --config D_full_hybrid --agent-seed 7 --out outputs\phase5f\stage4_train\D_full_hybrid_seed7 --resume
```

- Or invoke the batch file again; it resumes incomplete runs automatically.
- `--resume` rejects mismatched training settings, missing checkpoints, changed Python source hashes, and already completed runs.
- If a process stopped **before** saving its first `latest_state.pt`, leave the evidence folder intact and send the error; do not silently wipe and rerun it.
- **Do not modify the frozen code or train after Stage 5 held-out results have been examined.**

## Step 6 — Check that 12 runs completed

After the batch prints `TRAINING DONE`, the root folder must contain all twelve:

```text
outputs\phase5f\stage4_train\A_standard_seed7\training_provenance.json
outputs\phase5f\stage4_train\A_standard_seed17\training_provenance.json
outputs\phase5f\stage4_train\A_standard_seed27\training_provenance.json
outputs\phase5f\stage4_train\B_guided_seed7\training_provenance.json
outputs\phase5f\stage4_train\B_guided_seed17\training_provenance.json
outputs\phase5f\stage4_train\B_guided_seed27\training_provenance.json
outputs\phase5f\stage4_train\C_enhanced_seed7\training_provenance.json
outputs\phase5f\stage4_train\C_enhanced_seed17\training_provenance.json
outputs\phase5f\stage4_train\C_enhanced_seed27\training_provenance.json
outputs\phase5f\stage4_train\D_full_hybrid_seed7\training_provenance.json
outputs\phase5f\stage4_train\D_full_hybrid_seed17\training_provenance.json
outputs\phase5f\stage4_train\D_full_hybrid_seed27\training_provenance.json
```

To check the count:

```cmd
powershell -NoProfile -Command "(Get-ChildItem 'outputs\phase5f\stage4_train' -Filter training_provenance.json -Recurse -File).Count"
```

Expected: `12`.

Every run also has its `training_history.csv`, `validation_checkpoints.csv`, `validation_summary.csv`, `training_plan.json`, `best_model.pt` and `latest_state.pt`. **Do not inspect any reserved final seed to decide which checkpoint to use.**

## Step 7 — Upload the 12-run evidence for independent audit (before final evaluation)

I recommend uploading a ZIP of the **training** directory, keeping all 12 subfolders rather than manually renaming the 12 similar files:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5f\stage4_train' -DestinationPath 'Phase5F_Stage4_Training_Evidence.zip' -Force"
```

You may alternatively ZIP each of the 12 individual run folders separately. The most important files per folder are: `training_history.csv`, `validation_summary.csv`, `validation_checkpoints.csv`, `training_plan.json`, `training_provenance.json` and `best_model.pt`. The `latest_state.pt` files are recovery artifacts; they are optional for the independent numerical audit and may make the ZIP larger. **Do not ZIP or upload the shared candidate cache** unless a reproducibility discrepancy requires it.

I will check: balanced factorial configuration coverage; training and validation partitions; complete-sequence reward; override counts; zero infeasible assignments; source/stream/candidate hashes; checkpoint episode and file SHA; mean per-cohort weighted admission versus greedy and footprint; and absence of any final held-out evaluation.

## Step 8 — Validation-only model selection

Once the 12-run records are complete and consistent, run the selection command **once**:

```cmd
python run_phase5f_stage4.py select --runs-root outputs\phase5f\stage4_train --out outputs\phase5f\stage4_selection_FINAL
```

Expected success prefix: `STAGE4_SELECTION_OK`.

Outputs:

```text
outputs\phase5f\stage4_selection_FINAL\
  selection_trials.csv           # 12 initialization-level records
  selection_configurations.csv   # 4 configuration group scores
  selection_manifest.json        # authoritative validation-only freeze
```

The selector refuses to proceed without all **12 complete** runs and verifies per-run logs, model hashes, source agreement, identical validation cohorts and consistent baseline results. The winning configuration is chosen by mean best-checkpoint weighted validation admission across three model seeds. Ties are resolved in favour of the simpler arm A → B → C → D; then the highest-scoring individual model, lowest seed on a tie. The separately nominated `D_full_hybrid` checkpoint is also included in the selection manifest, whether or not D wins.

**Do not call a configuration better because it has a higher *training* reward. Do not report validation-only scores as independent evidence of superiority.**

## Step 9 — Upload the selection evidence

Upload:

```text
selection_trials.csv
selection_configurations.csv
selection_manifest.json
```

Also send terminal output from the Stage 4 tests and preflight if not already provided. Preserve the selected `.pt` checkpoint, all twelve training outputs, Stage 3 smoke, Stage 2 calibration, and Stage 4 research protocol. We will audit selection before building and running the separate Stage 5 national-scale held-out evaluator.

## Troubleshooting

| Symptom | What it means / what to do |
|---|---|
| `ModuleNotFoundError: phase5f_stage3` | The earlier Stage 3 add-on is missing or Stage 4 was unpacked under the wrong directory. |
| `Stage 3 source mismatch` | Your existing Stage 3 files differ from the verified smoke model's source fingerprints. Stop and send the exact error. |
| `refusing to overwrite prior run` | The chosen output folder exists. Use `--resume` for that incomplete run, or a different directory only for a new development test. |
| `resume plan/source/schedule mismatch` | The command or source was changed since the run began. Restore the exact frozen configuration; do not restart under the same directory. |
| `... seeds outside reserved ... partition` | An out-of-partition (possibly final) topology or stream seed was entered; do not bypass the guard. |
| `FileNotFoundError` during selection | One or more of the twelve required training runs has not completed. |
| `different source hashes across runs` | The source changed during experiments. Freeze and audit before any selection. |
| High memory/disk use | Run models sequentially; keep the deterministic shared cache in a drive with sufficient space; don't launch parallel training instances sharing the same cache key. |

## Final boundary

**Stage 4 is complete only when all 12 experiment manifests and the selection manifest exist and have passed independent audit.** Stage 5 remains a separate, newly frozen held-out experiment using topology seeds 401–420 and stream seeds 73000–73002. Keep those seeds untouched until the selection is audited. Neither Stage 4 nor the Stage 2/3 protection surrogate demonstrates Irish Rail's actual MPLS-TP topology, MSRP compliance, protection switching times, or railway safety performance.
