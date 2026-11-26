# Phase 5G — Stage 3 Start-to-Finish Windows Guide

**Purpose:** Freeze the two Stage 2 selected policies, preserve the original Phase 5F Enhanced PPO checkpoint, lock the next evaluation grid/statistical protocol, and prepare evidence for a new independent Stage 4 evaluation. **No RL training or held-out simulation is performed here.**

## Files to download

Download the `Phase5G_Stage3_Model_Freeze_and_Protocol_AddOn.zip` linked in the accompanying chat. Extract the ZIP **into the existing project root** containing `run_phase5g_stage2.py`, `phase5g_stage2`, `stage5_frozen`, and `outputs`. The ZIP only adds `run_phase5g_stage3.py`, `stage3_reference/*`, `tests/test_phase5g_stage3.py`, and documentation. Do not copy previous archived `phase5g_stage2` source snapshots over the installed implementation.

## Check the folder layout

The project must already contain:

```text
MPLS-AI\
  run_phase5g_stage3.py
  PHASE5G_STAGE3_PROTOCOL_LOCK.md
  stage3_reference\original___init__.py
  stage3_reference\original_workflow.py
  stage5_frozen\C_enhanced_seed27.pt
  stage5_frozen\D_full_hybrid_seed7.pt
  outputs\phase5g\stage2_train\gnn_seed27\...
  outputs\phase5g\stage2_train\mlp_seed17\...
  outputs\phase5g\stage2_selection_PROVENANCE_AUDITED\selection_manifest.json
  outputs\phase5g\stage2_selection_PROVENANCE_AUDITED\source_equivalence_report.json
```

If the original Stage 2 folders live elsewhere, pass `--runs-root` and `--selection-root` explicitly. Do not rename or modify training provenance to satisfy a check.

## Step 1 — Run the new tests

Open Windows **Command Prompt** and run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5g_stage3.py
```

Expected: `14 passed`.

Then use your existing full suite if convenient:

```cmd
python -m pytest -q
```

The previous reported total was 302, and this add-on adds 14 tests, so **316 is the expected count if no other tests changed**. Do not assume the count if you have added other tests. The Stage 3 standalone tests are the specific required gate; report any unrelated test failures separately.

## Step 2 — Inspect and lock the protocol before final data access

Read `PHASE5G_STAGE3_PROTOCOL_LOCK.md` in full. The plan uses **six policies** across **eight 500-node cells** (`hierarchical`/`ladder`; base, 1.5× bandwidth, shortened holding time; K2 sensitivity). There are **20 reserved topology seeds × 3 reserved streams × 8 cells = 480 matched cohorts**, producing **2,880 policy simulations**.

Two confirmatory hypotheses in the hierarchical/base K6 dynamic load4 cell: **GNN vs matched MLP** and **GNN vs scenario-aware minimax**. Holm adjustment covers these two primary comparisons. The document defines eight Holm-adjusted secondary comparisons; stress and K2 sensitivity are exploratory. Each confidence interval uses **20 independent topology-level paired averages** rather than treating 60 traffic streams as independent.

Do **not** access any 801–820/77000–77002 final cases. Stage 3 does not have any functionality to generate these seeds. Do **not** retrain or select models, and do not change Stage 2 or Phase 5F source files. Model-protection abstraction supervisor approval remains outstanding; record any decision separately and retain the scientific limitation even if approval is obtained.

## Step 3 — Run read-only six-run preflight

```cmd
python run_phase5g_stage3.py preflight
```

Expected output ends with:

```text
PHASE5G_STAGE3_PREFLIGHT_OK six_runs=6 models=2 prior_C_verified=True heldout_accessed=False
```

This inspects all six run provenance files and best checkpoint byte fingerprints, independently recalculates the selected validation scores, verifies the docstring-only source exception, checks two selected model parameter counts/finite tensors, and confirms the immutable Phase 5F C/D checkpoints. **No training or new network simulation.** If the preflight fails, stop and retain all files; do not bypass the check.

## Step 4 — Freeze checkpoints and protocol (once)

```cmd
python run_phase5g_stage3.py freeze --out outputs\phase5g\stage3_FROZEN
```

Expected:

```text
PHASE5G_STAGE3_VERIFY_OK models=gnn,mlp,C_frozen prior_D_archived=True final_seeds_not_accessed=True
PHASE5G_STAGE3_FREEZE_OK selected_gnn=seed27_ep48 selected_mlp=seed17_ep24 heldout_accessed=False
```

The freeze directory is deliberately **refused if it already exists**. Never delete or overwrite a successfully frozen package just to rerun this command. It copies selected GNN/MLP weights and validation trace files, the prior frozen C/D checkpoint backups, six original training-provenance JSONs, the provenance-aware selection record and exact Stage 3 protocol. A freeze manifest lists SHA-256 fingerprints of every included file and 15 original executable-source files.

## Step 5 — Independently verify the freeze

```cmd
python run_phase5g_stage3.py verify --root outputs\phase5g\stage3_FROZEN
```

Expected:

```text
PHASE5G_STAGE3_VERIFY_OK models=gnn,mlp,C_frozen prior_D_archived=True final_seeds_not_accessed=True
```

The verifier fails if a model, source file, plan or copy of the manifest-governed files has changed. It does not generate final scenarios.

## Step 6 — Export the Stage 3 evidence only

In Command Prompt:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5g\stage3_FROZEN' -DestinationPath 'Phase5G_Stage3_Freeze_Evidence.zip' -Force"
```

Upload `Phase5G_Stage3_Freeze_Evidence.zip` and the test/preflight/verify console results. **Do not upload new held-out results at this stage**, and do not rerun the six Stage 2 experiments.

## After the freeze

Stage 4 will implement a once-only evaluator with deterministic loading and checkpoint-source fingerprints. It must first pass engineering tests on *development-role* seeds, with no attempt to use the reserved final network topology seeds 801–820 or demand streams 77000–77002. The Stage 4 final experiment is authorized only after auditing this Stage 3 freeze and the Stage 4 implementation.

**Research limitation:** all conclusions remain about a synthetic, end-to-end single-link-failure protected-pair surrogate, **not MSRP** or a measured operational Irish Rail network.
