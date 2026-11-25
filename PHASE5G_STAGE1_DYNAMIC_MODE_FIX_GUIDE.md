# Phase 5G Stage 1 — Forward-only dynamic traffic flag patch (V1.1)

## Why this patch exists

The Stage 1 GNN simulator was inferring whether an episode was dynamic from the **individual request expiry time**:

```
dynamic = demand.expiry <= len(demands)
```

This is not valid. A legitimate dynamic request may expire after the final arrival and was therefore incorrectly labelled as static in the state encoder. The Stage 3 global feature vector contains a `dynamic` flag at index 5 (zero-based). On the 60-node, load 1.6 training cohort, 57/96 dynamic requests have expiry beyond the arrival horizon.

This patch passes the actual cohort traffic mode explicitly into GNN observation encoding, adds a regression test, and increments the Phase 5G development version to `PHASE5G_DEVELOPMENT_FOUNDATION_V1_1`.

This is a **prospective engineering correction made before Phase 5G full model training**. It must not be used to re-interpret the previously uploaded Stage 1 smoke checkpoint. The original V1 evidence should be retained as an immutable development record. The simulator ledger, protection model, candidate generator, frozen Phase 5F models, and Phase 5F held-out results are not altered.

## Installation (Windows CMD)

1. Keep a copy of `outputs\phase5g\gnn_smoke_dev` and `outputs\phase5g\optimization_topology_dev` unchanged.
2. Extract this ZIP **into the MPLS-AI project root**, allowing overwrite of **only these three files**:
   - `phase5g/__init__.py`
   - `phase5g/workflow.py`
   - `tests/test_phase5g.py`
3. Run:

```
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5g.py
python -m pytest -q
python run_phase5g.py preflight
```

The patched Phase 5G module should report **23 tests passed**. With the previous total unchanged, the overall project should have approximately **286 tests passed**. If you see failures or a different total, report the actual output rather than assuming it is safe to continue.

This patch was checked independently on Python 3.13.5 / NetworkX 3.6.1 / NumPy 2.3.5 / PyTorch 2.10 CPU: 23 Phase 5G tests, 100 combined Phase 5F+5G add-on tests, development-only preflight, and fresh one-episode static/dynamic GNN smoke checks all passed. Your Windows versions differ; repeat the tests locally.

**No Stage 5 or new Phase 5G reserved final seeds should be run.** The next work is Phase 5G Stage 2 matched-budget model training, not a repeat of this development experiment.
