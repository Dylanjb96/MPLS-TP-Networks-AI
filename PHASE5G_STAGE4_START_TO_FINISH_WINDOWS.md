# Phase 5G — Stage 4 implementation and development validation (Windows)

**Important:** This is the **development/implementation add-on**, **NOT** permission to run the reserved Phase 5G final evaluation. Do not generate any of the topology seeds **801–820** or traffic seeds **77000–77002** yet. The `final` command intentionally fails without a separate, hash-specific `stage4_AUDIT_RELEASE.json` issued **after** the independent implementation audit.

## What the source handoff established

The supplied `Phase5G_Stage4_Source_Handoff.zip` passed ZIP integrity and **49/49 source-file fingerprint checks**, including the exact frozen Stage 3 protocol SHA `6edad86fb2f49328784c5e80ef60d967413f6bf4f30f64debf57ad3381ce1d43`. It contained **no model weights, training results or new held-out networks**, as expected. The add-on was built against these exact Windows source bytes; it does not replace previous Python modules.

## 1. Install

Extract `Phase5G_Stage4_Development_Evaluator_AddOn.zip` into the existing MPLS-AI repository root:

```text
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI
```

Only new `phase5g_stage4/`, `run_phase5g_stage4.py`, `tests/test_phase5g_stage4.py` and documentation files are added. **Do not overwrite** `phase5g`, `phase5g_stage2`, `run_phase5g_stage3.py`, `outputs\phase5g\stage3_FROZEN`, or earlier datasets. The two selected neural models are loaded **from the existing Stage 3 freeze**; no training takes place.

## 2. Verify tests and model/source hashes

Open **Command Prompt** and run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5g_stage4.py
python -m pytest -q
python run_phase5g_stage4.py preflight
```

The add-on includes **22 new tests**; if the previous Windows total of **316** is unchanged, the expected full project count is **338 tests**. The preflight should end with:

```text
PHASE5G_STAGE4_PREFLIGHT_OK source_files=15 frozen_models=3 semantic_sha256=e1255cd17f984ebd501565f31550a7179fcc849b678b9324e88b5bff0f555a7c final_seeds_accessed=False
```

Preflight verifies the complete frozen Stage 3 file manifest, all 15 Stage 3-locked source hashes, the immutable protocol, selected model hashes and checkpoint identities **before any topology generation**. If it fails, do not edit the existing source or frozen models to get around the lock.

## 3. Development-only eight-cell smoke

```cmd
python run_phase5g_stage4.py smoke --out outputs\phase5g\stage4_dev_smoke
```

Expected final lines:

```text
PHASE5G_STAGE4_DEV_COMPLETE cohorts=8 rows=48 role=dev final_data_accessed=false
PHASE5G_STAGE4_DEV_SMOKE_OK rows=48 final_seeds_accessed=False
```

This executes **all 8 prespecified cell types × all 6 policies** on 60-node *development* network 501, traffic stream 74000. It checks full independent protected-feasibility replay, all 6 policies, K2 fixed six-slot neural action padding, both network families, bandwidth and holding-duration stress, input hashes, trajectory congestion and one atomic saved cohort per cell. Result files live in `outputs\phase5g\stage4_dev_smoke\cohorts\`; `development_summary.csv` is descriptive engineering evidence only.

If interrupted, rerun with the **same output path and `--resume`**; completed whole cohorts are verified and preserved:

```cmd
python run_phase5g_stage4.py smoke --out outputs\phase5g\stage4_dev_smoke --resume
```

## 4. Replay the selected Stage 2 models against their **existing validation** trace

```cmd
python run_phase5g_stage4.py validation-replay --out outputs\phase5g\stage4_validation_replay_dev
```

Expected:

```text
PHASE5G_STAGE4_VALIDATION_REPLAY_OK policies=2 requests=2000 matched_full_decisions=True final_seeds_accessed=False
```

This is a **500-node validation-regression** check using the **already-used** Stage 2 validation topology 701 and traffic stream 76000. It compares all 2,000 saved action/ledger fingerprints for each selected GNN/MLP to the unmodified Stage 3 frozen validation traces. It performs no training, model selection or final evaluation. The source and test environment make this computation more demanding than the 60-node smoke; do not infer performance from these regression results.

## 5. Representative 500-node **development** scaling test

```cmd
python run_phase5g_stage4.py scale-dev --out outputs\phase5g\stage4_scale_dev
```

Expected:

```text
PHASE5G_STAGE4_SCALE_DEV_OK rows=6 final_seeds_accessed=False
```

This runs **all six policies on a single 500-node H_BASE development cohort** (topology 501, stream 74000) and records separate candidate-generation, policy and oracle execution costs. This is a feasibility/resource test, *not* statistical or held-out evidence. Keep its result archive even if it is computationally demanding. If interrupted, use `--resume` without changing the source or models.

**Do not run the full 2,880-policy final study yet.** The Stage 4 final command is intentionally guarded by an external independent audit release; this package does **not** contain that release.

## 6. Upload evidence for independent Stage 4 implementation audit

When the checks pass, run:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5g\stage4_dev_smoke','outputs\phase5g\stage4_validation_replay_dev','outputs\phase5g\stage4_scale_dev' -DestinationPath 'Phase5G_Stage4_Development_Evidence.zip' -Force"
```

Upload `Phase5G_Stage4_Development_Evidence.zip` together with the test counts and preflight output. Retain the source handoff ZIP, Stage 3 freeze archive and original complete Stage 2 evidence on your computer.

## 7. After independent audit

If the independent review finds no unresolved implementation/feasibility/analysis problems, a **separate, exact-hash audit release** can unlock Stage 4 final execution using the unchanged source and frozen models. Only then run the reserved 20 topology seeds × 3 streams × 8 cells × 6 policies. The final analyst will compute **20 topology-cluster** paired intervals, **Holm** for two co-primary and eight secondary tests, and descriptive stress/K2 interactions. No interim tuning, model replacement, pruning or restarts with a changed configuration.

## Scientific limits

The experiment is synthetic end-to-end link-disjoint working/backup protection with reserved capacity under single physical-link faults. It does **not** prove compliance with MPLS-TP MSRP or represent actual Irish Rail topology or restoration times. Supervisor acceptance of this approximation is still pending. The scenario-aware minimax baseline is a one-decision exact enumerator of its declared current-state objective, **not** a global future-aware MILP.
