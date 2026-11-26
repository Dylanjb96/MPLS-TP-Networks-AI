# Phase 5G — Stage 1 installation, Windows CMD and outputs

## Purpose

This package is a **separate Phase 5G development foundation**, not full-scale GNN training and not a confirmatory experiment. It does **not** replace any `phase5f*`, `run_phase5f*`, Stage 4 checkpoints or Stage 5 evidence files. Stage 5 final seeds are forbidden, and Phase 5G's new reserved seeds cannot be used with this CLI.

## 1 — Backup, install, preserve freezes

Back up your existing project and evidence. Extract `Phase5G_Stage1_GNN_Optimization_Robustness_AddOn.zip` **into**:

`C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI`

It adds `phase5g/`, `run_phase5g.py`, `tests/test_phase5g.py`, and `.md` guidance. It does **not** include frozen model weights; the existing `stage5_frozen/` directory from your Phase 5F Stage 5 installation must remain installed. Do **not** change your Phase 5F/Stage 5 sources; the CLI invokes the frozen C model's source-hash check.

Open Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python --version
```

Your previously verified environment was Python 3.11.9, NetworkX 3.5, NumPy 2.2.6 and PyTorch 2.7.1+cu118. Do not upgrade libraries for this stage.

## 2 — Tests

```cmd
python -m pytest -q tests/test_phase5g.py
python -m pytest -q
```

Expected from the development package: **22 new Phase 5G tests passed**. If the rest of your project remains unchanged, **about 285 total tests** (263 previous +22). If there are any failures, stop and save the full traceback; do not proceed to model training.

## 3 — Freeze and topology preflight

```cmd
python run_phase5g.py preflight
```

Expected:

- `PHASE5G_PREFLIGHT_FAMILY_OK family=hierarchical ...`
- `PHASE5G_PREFLIGHT_FAMILY_OK family=ladder ...`
- `PHASE5G_PREFLIGHT_OK frozen_C_sha256=e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`

A frozen source-model mismatch is a safety failure; investigate it rather than bypassing the check.

## 4 — Initial GNN learning and capacity-safety smoke

```cmd
python run_phase5g.py gnn-smoke --out outputs\phase5g\gnn_smoke_dev --episodes 2 --agent-seed 7 --load 1.6 --mode dynamic
```

Expected `PHASE5G_GNN_SMOKE_OK`. This generates:

- `gnn_training_history.csv` — only two tiny PPO development episodes.
- `gnn_validation.csv` — one independent validation topology comparison with greedy, footprint, scenario-minimax and GNN.
- `gnn_engineering_smoke.pt` — **NOT a selected or final model**.
- `gnn_smoke_manifest.json` — source hashes and checkpoint fingerprint.

The GNN may underperform every baseline. This is expected in a smoke test and is not grounds for tuning the held-out data.

## 5 — New topology and online optimization development pilot

```cmd
python run_phase5g.py pilot --out outputs\phase5g\optimization_topology_dev --sizes 60 --families hierarchical ladder --topology-seeds 501 502 --stream-seeds 74000 --load 2.5 --mode dynamic --stress base --k 6 --policies greedy footprint scenario_minimax C_frozen
```

Expected `PHASE5G_DEV_PILOT_OK policy_rows=16`. This evaluates the original frozen Enhanced PPO C, greedy, footprint and exact single-arrival scenario-minimax on identical sequences *within each synthetic network* and verifies every resulting trace through the separate single-fault oracle.

**Only after the first 60-node pilot succeeds**, an optional scale pilot (no need to run yet):

```cmd
python run_phase5g.py pilot --out outputs\phase5g\optimization_150_dev --sizes 150 --families hierarchical ladder --topology-seeds 501 --stream-seeds 74000 --load 2.5 --mode dynamic --stress base --k 6 --policies greedy footprint scenario_minimax C_frozen
```

Both pilot runs use **development seeds** and must not be interpreted as final Phase 5G confirmation. Each output folder is created once. If already present, do not delete/reuse it; keep it and share the evidence.

## 6 — Files to send for audit

Send your CMD outputs for steps 2 and 3, plus:

`outputs\phase5g\gnn_smoke_dev\` (four files), and
`outputs\phase5g\optimization_topology_dev\` (two files).

ZIP for upload (run from your project folder):

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5g\gnn_smoke_dev','outputs\phase5g\optimization_topology_dev' -DestinationPath 'Phase5G_Stage1_Development_Evidence.zip' -Force"
```

No Phase 5G final evaluation script is included. Do **not** run Phase 5F's old held-out seeds again to compare GNN/optimizer; those results remain frozen. The next deliverable, after audit, is a full matched-budget GNN-versus-MLP training framework and then an independent test protocol.
