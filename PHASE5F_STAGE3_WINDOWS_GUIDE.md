# Phase 5F — Stage 3: Windows Installation, Tests and Hybrid PPO Smoke

**Artifact status:** development-only engineering add-on. It contains the first trainable **candidate-aware greedy-guided PPO** policy on the reproducible Stage 2 synthetic transport simulator. It is **NOT** the complete full-scale Stage 4 training, frozen checkpoint selection, or Stage 5 held-out evaluation. It cannot establish that RL outperforms greedy or capacity-footprint optimization.

## 1. Prerequisites

Your previously reported Windows environment: Python 3.11.9, NetworkX 3.5, NumPy 2.2.6, PyTorch 2.7.1+cu118. These are already appropriate. Do **not** change package versions unless tests show a compatibility problem. Keep your original Phase 3–5E and Phase 5F Stage 1–2 files and experimental output unchanged.

Extract `Phase5F_Stage3_Candidate_Aware_PPO_AddOn.zip` into the project root:

```
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI
```

It adds *only* `phase5f_stage3/`, `run_phase5f_stage3.py`, `tests/test_phase5f_stage3.py`, and these Stage 3 documents. It requires existing `phase5f/` and `phase5f_stage2/` modules. There is no overwrite of Stage 2 software.

From **Windows Command Prompt** (not PowerShell):

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5f_stage3.py
python -m pytest -q
```

The new module provides **13 tests**. With your previously reported 211 existing passing tests, the expected combined test count is **224 passed** if the project remains otherwise unchanged. A different count is not inherently a failure; investigate all actual test failures.

## 2. Stage 3 preflight — first

```cmd
python run_phase5f_stage3.py preflight
```

Expected: a line beginning with `STAGE3_PREFLIGHT_OK` for a structurally generated 60-node network and six link-disjoint protected candidates with multiple currently feasible routes. The topology SHA is deterministic and should match the Stage 3 implementation. Stop and send the complete error output if it fails.

## 3. Run the development-only training smoke

Only after the tests and preflight succeed:

```cmd
python run_phase5f_stage3.py smoke --out outputs\phase5f\stage3_smoke_dev --nodes 60 --episodes 4 --train-topologies 201 202 --train-streams 71000 71001 --val-topologies 301 --val-streams 72000 --load 2.5 --traffic static dynamic --k 6 --headroom 10 --max-working-paths 40 --agent-seed 7 --learning-rate 0.0003 --bc-coef 0.01 --gate-threshold 0.1
```

This uses training topology seeds **201/202**, training request-stream seeds **71000/71001**, validation topology **301**, and validation stream **72000**. It generates four training episodes on declared development streams and independently evaluates learned and conventional policies on validation streams. The output folder must *not already exist*; the script intentionally refuses to overwrite prior results.

**Development checks:** (i) candidate-score shape and feasible masking, (ii) stochastic PPO update and optional greedy imitation regularization, (iii) dynamic resource expiry, (iv) no voluntary rejection, (v) same-stream greedy/footprint/hybrid comparison, (vi) exact capacity and single-link-failure-scenario ledger recomputation, and (vii) action-level logging.

The script writes:

| Output | Contents |
|---|---|
| `training_history.csv` | Four on-policy training episodes, admission, overrides and PPO updates |
| `validation_summary.csv` | Same-stream greedy, footprint and deterministic hybrid results |
| `validation_hybrid_decisions.csv` | Per-request feasibility, greedy decision, actual action and override flag |
| `smoke_model.pt` | **Development smoke checkpoint only**; not selected final model |
| `stage3_smoke_manifest.json` | Dataset roles, fingerprints, software, source SHA, checkpoint SHA and limitations |

**Upload these four small CSV/JSON files:** `training_history.csv`, `validation_summary.csv`, `validation_hybrid_decisions.csv`, `stage3_smoke_manifest.json`. Do not upload the `.pt` checkpoint unless a binary inspection is specifically needed. Include the test/preflight terminal output in your message.

## 4. Interpreting results

A four-episode smoke test only establishes that model training, feasible action selection, reproducibility metadata and reporting execute. **Positive override counts are not evidence of better admission.** Negative validation results are not reasons to open or tune on held-out data. No final checkpoint is selected in Stage 3.

The learner uses a **shared candidate-scoring neural MLP**, not a graph neural network. It is topology-aware through request, candidate path, residual capacity, route length, network-size and single-failure-scenario features. A GNN may be investigated as an explicit representation ablation later.

PPO updates use rewards obtained after a full simulated episode. The policy never sees future request details at decision time. A request with at least one feasible route must be admitted, and no infeasible candidate may be selected.

## 5. Frozen partitions and protections

- **Stage 2 development evidence:** topology seeds 101–103; stream seeds 70000–70002, already inspected.
- **Stage 3/4 training:** topology seeds 201–220, stream seeds 71000–71007.
- **Stage 3/4 validation:** topology seeds 301–310, stream seeds 72000–72002.
- **Stage 5 final reserved:** topology seeds **401–420**, stream seeds **73000–73002**; *never touched* in this add-on.
- Predeclared development loads: **1.6, 2.5 and 4.0 requests per node**. These supersede the original planning values 1.6/3.2/4.8 as a documented **development-stage change**, not a post-heldout adaptation.

All networks and railway service types are **synthetic research abstractions**. The fault model is an end-to-end link-disjoint working/backup pair under a modeled single physical-link failure; it is **not MSRP** or a demonstration of switching latency, safety or actual Irish Rail network performance. Supervisor approval remains outstanding.

## 6. What happens next

When these four files pass independent review, the next deliverable is **Stage 4 full training and validation**, with multi-seed runs, frozen configuration selection, and separate held-out controls. **Do not run a 500-node final evaluation from this Stage 3 smoke.**
