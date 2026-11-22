# Phase 5E — Windows Command Prompt start-to-finish implementation and experiment guide

**Project:** Resilience-Aware LSP Provisioning in Interconnected Railway MPLS-TP Networks.

**Scope:** One integrated **Greedy-Guided Residual PPO (GG-RPPO)** hybrid with feature ablation and non-learning controls. The code is included and locally tested. The full 60-node, 6-run training and 60-seed held-out research evaluation **must still be executed on your own computer** using your frozen project and model files. The new held-out test must be opened only after selection is final. Read `PHASE5E_RESEARCH_PROTOCOL.md` before execution.

## 0. Inputs already present in your project

From earlier phases your original repository contains `railway_tp`, `phase3`, `phase4`, `phase5`, `phase5b`, `phase5c`, `phase5d` and `config/primary_60_dual.json`; prior chosen checkpoints and the Phase 5C `selection_manifest.json` are at your local paths.

**Do not overwrite or delete:** the Phase 5C selected `.pt` files, `outputs/phase5c/heldout_51000_FINAL`, Phase 5D evidence, or previous source files. The add-on contains **only new Phase 5E files plus one new test**. It does **not** contain/overwrite the missing Phase 2 `railway_tp` package or any prior models. Check `phase5e_addon_manifest.json` inside the package for included files and SHA256 hashes.

### Install

1. Back up your complete working project and archived results.
2. Extract `MPLS_TP_Phase5E_Hybrid_Greedy_Residual_RL_AddOn.zip` to the root of `C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI` (the same root containing `run_phase5c.py`).
3. Open **Command Prompt** (not PowerShell) and run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python run_phase5e.py --help
python run_phase5e.py preflight --config config/primary_60_dual.json --phase5d-manifest outputs\phase5d\behavior_validation_20000\diagnostic_manifest.json
python -m pytest -q tests/test_phase5e.py
python -m pytest -q
```

Expected in the supplied local integration environment: **19 new Phase 5E tests**, **186 total tests** (167 previously including Phase 5D). Your actual total may differ if your existing project has additional tests. If dependencies are missing, install them using your existing environment and `python -m pip install -r requirements_phase5e.txt`. Do not silently upgrade a working PyTorch setup.

The read-only preflight should report `phase5d_source_provenance_confirmed: true`. If the Phase 5D manifest is stored elsewhere, change that file path. An exact source-hash mismatch must be investigated **before** training; do not rewrite the manifest to silence it.

**Verify that your existing Phase 5D tests still pass.** There is no need to rerun earlier held-out tests.

## 1. Development-only opportunity and candidate inventory probes

A probe uses separate, newly designated **development** streams 23000–23009; it cannot use 61000 or 62000. It tests greedy, random override and a conservative capacity-score control, without training any model.

Primary two-candidate condition:

```cmd
python run_phase5e.py probe --config config/primary_60_dual.json --out outputs\phase5e\probe_dev_k2 --requests 100 --headroom 10 --max-candidates 2 --seed-start 23000 --seed-count 10 --capacity-threshold 0.05
```

Informative candidate-limit probe (only after primary probe; **not** a new final test):

```cmd
python run_phase5e.py probe --config config/primary_60_dual.json --out outputs\phase5e\probe_dev_k4 --requests 100 --headroom 10 --max-candidates 4 --seed-start 23000 --seed-count 10 --capacity-threshold 0.05
```

Files: `probe_runs.csv`, `probe_paired.csv`, `probe_manifest.json`, plus per-seed `decisions.csv` and `summary.json`. **Inspect** `probe_manifest.json` → `candidate_inventory`, `k_above_two_actually_used`. In local development (using older Phase 2 support modules, not supplied in the user's latest source-only ZIP), inspection of 1,000 requests on the 60-node seed-42 topology at k=8 yielded **2 candidates per request**. This must be confirmed against your current `railway_tp` files; increasing only the cap may not introduce new choices. This add-on does not invent extra structurally invalid routes or rewrite the source generator to force a result. We should not treat k=4 as an expanded-choice experiment unless actual candidate counts exceed 2.

## 2. Pre-registered full training — six runs

Two configurations, **three random agent initialization seeds 7, 17, 27** each. Both configurations use: train requests 100, 80 training streams **11000–11079**, validation streams **22000–22019**, headroom 10%, k=2, 160 PPO episodes maximum, validate every 10, early stop patience 8, 0.1 greedy-imitation regularization, 50 greedy-only warm-start optimizer steps, fixed gate margin 0.15.

Difference: `base` uses original 26 features + greedy flag; `enhanced` adds seven advisory congestion features. Everything else is identical. Each output path must be **new**. If a command was partially executed, *archive* its incomplete output and use a fresh path; do not overwrite other evidence. Run from **interactive Windows CMD**, where loop variables use `%S` (inside a `.bat` file use `%%S`).

Configuration A (base):

```cmd
for %S in (7 17 27) do python run_phase5e.py train --config config/primary_60_dual.json --out outputs\phase5e\train\base_seed%S --requests 100 --headroom 10 --max-candidates 2 --features base --agent-seed %S --train-seed-start 11000 --train-seed-count 80 --val-seed-start 22000 --val-seed-count 20 --episodes 160 --validate-every 10 --patience 8 --learning-rate 0.0003 --gamma 1.0 --greedy-bc-coef 0.1 --warmstart-steps 50 --gate-threshold 0.15
```

Configuration B (enhanced):

```cmd
for %S in (7 17 27) do python run_phase5e.py train --config config/primary_60_dual.json --out outputs\phase5e\train\enhanced_seed%S --requests 100 --headroom 10 --max-candidates 2 --features enhanced --agent-seed %S --train-seed-start 11000 --train-seed-count 80 --val-seed-start 22000 --val-seed-count 20 --episodes 160 --validate-every 10 --patience 8 --learning-rate 0.0003 --gamma 1.0 --greedy-bc-coef 0.1 --warmstart-steps 50 --gate-threshold 0.15
```

Each run produces `best_model.pt`, `last_model.pt`, `training_history.csv`, `training_provenance.json`. A `best_model.pt` is validation-selected; **do not substitute `last_model.pt`**.

**Training integrity:** six valid run directories with `training_provenance.json`; two distinct configs x three seeds; matching source/config/validation demand SHA256; no non-train smoke runs inside `outputs\phase5e\train`.

## 3. Validation-only model selection — do before any held-out test

```cmd
python run_phase5e.py select --runs-root outputs\phase5e\train --out outputs\phase5e\selection_FINAL --min-seeds 3
```

Outputs:

- `outputs\phase5e\selection_FINAL\selection_trials.csv` — **2 hyperparameter configuration summaries**.
- `outputs\phase5e\selection_FINAL\selection_manifest.json` — **one selected hybrid checkpoint**, SHA256, config, seed and source provenance.

Review the selection and archive it without edits. **Send me these two files and six training histories/provenances for independent audit before opening the final test.**

The selector verifies model metadata, demand stream hashes, source hashes, training history best-checkpoint flags, and recorded checkpoint SHA256. The choice is by three-seed mean validation **weighted** admission, not held-out metrics.

## 4. Optional development-only dry-run of the frozen selected model

This is *not* the Phase 5E final test. It is a compatibility check on the **23000-series development streams already used for the opportunity probe**. Do not use this result to revise hyperparameters. It includes conventional baselines and both gated/ungated ablations.

```cmd
python run_phase5e.py evaluate --role development --selection outputs\phase5e\selection_FINAL\selection_manifest.json --config config/primary_60_dual.json --out outputs\phase5e\compatibility_dev_23000 --requests 100 --headroom 10 --max-candidates 2 --seed-start 23000 --seed-count 2
```

A development dry-run is optional. Do not report it as an independent held-out evaluation.

## 5. Single pre-specified NEW held-out experiment (seeds 61000–61059)

**Do this only when Step 3 is frozen, code/source hashes are unchanged, checkpoints pass SHA verification, the protocol is signed off, and no experimental choices remain.** This command evaluates the same candidate pool and request streams across all algorithms. It will refuse changes to primary topology, headroom, request count, candidate limit, or test seed range.

```cmd
python run_phase5e.py evaluate --role primary --selection outputs\phase5e\selection_FINAL\selection_manifest.json --config config/primary_60_dual.json --out outputs\phase5e\heldout_61000_FINAL --requests 100 --headroom 10 --max-candidates 2 --seed-start 61000 --seed-count 60 --phase5c-selection outputs\phase5c\selection_FINAL\selection_manifest.json --original-ddqn outputs\phase5\train_60_seed7\best_model.pt --discrete-sac outputs\phase5b\tuning_v1\confirm\discrete_sac_cfg2_seed7\best_model.pt
```

The above includes 15 policy labels: `greedy`, `cspf`, `milp_v2`, `random_feasible`, `random_override`, `capacity_override`, `hybrid_gated`, `hybrid_ungated`, the five frozen Phase 5C algorithms, original DDQN and Discrete SAC. That means **900 matched policy × demand-seed episodes** (= 15 × 60); 6,000 offered requests per algorithm. This is substantial research work. Candidate generation is shared per seed, not counted in the per-request selection-time measurement.

If you do not have all old checkpoints present, **do not replace or fabricate them**. You may run only the new hybrid and conventional controls by omitting the three optional legacy arguments, but decide and document that narrower comparison **before** opening the held-out data. (Eight core labels = 480 policy × seed episodes.)

Outputs: `phase5e_runs.csv`, `phase5e_paired_deltas.csv`, `evaluation_manifest.json`, and one per-policy/per-seed folder containing `requests.csv`/`decisions.csv`/`summary.json`/manifests. For the hybrid and controls, `decisions.csv` records native and residual route choices, feasibility and SHA256 of the reservation state before and after each choice. Add `--save-plans` to archive full accepted protection plans, if you have adequate storage.

**One-shot rule:** do not rerun held-out tests with different thresholds or model seeds, and do not use held-out outcomes to choose a configuration. If there is a software defect, document it transparently, preserve all evidence and define a formally corrected follow-up test before restarting.

## 6. Final statistical report and charts

```cmd
python run_phase5e.py statistics --eval-dir outputs\phase5e\heldout_61000_FINAL --out outputs\phase5e\statistics_FINAL
python run_phase5e.py plots --stats outputs\phase5e\statistics_FINAL --out outputs\phase5e\figures_FINAL
```

Expected summary artifacts:

- `statistics_FINAL\condition_summary.csv`
- `statistics_FINAL\paired_summary.csv`
- `statistics_FINAL\phase5e_statistical_report.md`
- `statistics_FINAL\statistics_manifest.json`
- `figures_FINAL\weighted_admission.png`
- `figures_FINAL\ordinary_admission.png`
- `figures_FINAL\selection_latency.png`

The inferential unit is the **whole demand stream**, NOT the individual LSP. Report paired differences in percentage points, nominal 95% Student-t CIs, and Holm-corrected two-sided p-values across multiple greedy comparisons. Zero observed variance does not demonstrate equivalence. The purpose of this study is an honest test of the proposed hybrid, not a guaranteed win.

## 7. Optional k>2 sensitivity — only if justified

First inspect Step 1 candidate inventories. Since the locally reconstructed fixed topology generated **exactly two candidates per request even with a cap of eight** in a 1,000-request development check, merely increasing `--max-candidates` may have no substantive effect. Do not run a large k=4 experiment purely to change the parameter value.

If an independently validated new scenario actually has >2 distinct feasible policy-valid candidates, pre-register it separately; retrain and select a new k-specific hybrid, rerun matching greedy/CSPF/MILP baselines, and use a distinct untouched sensitivity test range 62000–62059. `--role sensitivity` enforces k>2 and that range and never pools outcomes with the primary k=2 result. Introducing additional path types requires **separate implementation and structural verification**, not supplied by this add-on.

## 8. What to upload after each milestone

After training: all six `training_history.csv` and `training_provenance.json`, `selection_trials.csv`, `selection_manifest.json`. **Do not yet upload test outputs before selection review.**

After the final test: `phase5e_runs.csv`, `phase5e_paired_deltas.csv`, `evaluation_manifest.json`, `condition_summary.csv`, `paired_summary.csv`, `phase5e_statistical_report.md`, `statistics_manifest.json`. Include representative per-seed `decisions.csv` for detailed verification and figures if convenient.

## 9. Dissertation interpretation / limitations

- Distinguish on-policy residual PPO from offline BCQ/CQL/AWAC/IQL. The Phase 5E agent uses **greedy-guided initialization and policy regularization**, not new offline BCQ or learned restoration.
- No voluntary rejection; no LSP teardown/preemption; all committed paths remain fixed after admission.
- Enhancement features include current fault-state scenario budgets; they are **advisory features**, while Phase-3 independent audit remains the sole feasibility authority.
- Confidence gate reflects actor probability preference, not a guarantee that a route will improve future admission.
- A higher `max_candidates` value is only a limit; it does not automatically generate additional valid alternatives.
- Inference timing is not fault-restoration latency. The protection model remains **`RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`**, with `protection_model_approved=false`. Supervisor sign-off is still necessary for any standards-level claim.
- Fixed 60-node topology seed 42 results cannot establish cross-topology generalization.

**Phase 5E implementation is ready; Phase 5E empirical conclusions do not yet exist.**
