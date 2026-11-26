# Phase 5F — Stage 1 Synthetic Network/Traffic/Routing Opportunity Probe

## Install safely

1. Back up the completed Phase 5E project and frozen outputs first. Do **not** replace or edit `phase3`, `phase4`, `phase5`, `phase5b`, `phase5c`, `phase5d`, `phase5e` or their models.
2. Extract the `Phase5F_Stage1_A_B_C_Prototype.zip` archive into your project root. It only adds the new `phase5f/`, `run_phase5f.py`, `tests/test_phase5f.py`, and Phase5F documentation. The Stage 1 modules are standalone and **do not** change or use the previous Phase 3 protection simulator.
3. In Windows Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5f.py
python -m pytest -q
```

The original Phase 5E local suite previously reported 186 passes. This add-on has **13 additional tests**; if the earlier suite is identical, the new total should be **199**. If local tests differ, report the actual count rather than assume this number.

## Preflight: topology sizes and sample working/backup candidates

```cmd
python run_phase5f.py preflight --sizes 60 150 250 500 --seed 101
```

This validates physically distinct node/edge counts, absence of single-link bridges, unique physical links and availability of at least one sample pair. **A sample showing six candidates does not prove six feasible choices on loaded networks.**

## Pilot A × B × C on 60 synthetic nodes

```cmd
python run_phase5f.py probe --out outputs\phase5f\stage1_60_dev --sizes 60 --topology-seeds 101 102 103 --stream-seeds 70000 70001 70002 --requests-per-node 1.6 --k 2 6 --traffic static dynamic --policies greedy random_feasible footprint --headroom 10 --max-working-paths 40
```

Development-only pilot on multiple topologies and request streams. This will create 3 × 3 × 2 × 2 × 3 = **108 run records** and 9 request inventories (one inventory per topology/request pair).

## Profile larger synthetic systems before full 500-node work

```cmd
python run_phase5f.py probe --out outputs\phase5f\stage1_150_dev --sizes 150 --topology-seeds 101 --stream-seeds 70000 --requests-per-node 1.6 --k 2 6 --traffic static dynamic --policies greedy footprint
python run_phase5f.py probe --out outputs\phase5f\stage1_250_dev --sizes 250 --topology-seeds 101 --stream-seeds 70000 --requests-per-node 1.6 --k 2 6 --traffic static dynamic --policies greedy footprint
python run_phase5f.py probe --out outputs\phase5f\stage1_500_dev --sizes 500 --topology-seeds 101 --stream-seeds 70000 --requests-per-node 1.6 --k 2 6 --traffic static dynamic --policies greedy footprint
```

**All three are development-only.** These are deliberate *small-scale* pilots and do not create test results. 500 nodes is more expensive because path-pair generation and all single-edge-failure capacity scenarios scale with size.

## Later load sensitivity — development ONLY, separate folders

For instance, if 60-node and 500-node admissions are almost always successful, diagnose candidate richness and contention before choosing a primary offered-load stratum:

```cmd
python run_phase5f.py probe --out outputs\phase5f\stage1_60_load32_dev --sizes 60 --topology-seeds 101 --stream-seeds 70000 --requests-per-node 3.2 --k 2 6 --traffic static dynamic --policies greedy footprint
```

Repeat on 500 nodes **only after measuring runtime**. Pre-specified additional load level 4.8 is allowed for development. Do not manipulate load based on test results or to maximize the observed RL advantage.

## Files to upload first

Please upload the ZIPs or the following **three files from each Stage 1 output folder**: `probe_manifest.json`, `probe_runs.csv`, `candidate_inventory.csv`. Also provide your local pytest/preflight console results. Upload the 60-node pilot first; we can examine route diversity and software correctness before a heavier 500-node sweep.

## What the data actually mean

- `candidate_inventory.csv`: number of unique *structurally valid link-disjoint working/backup pairs* per request; not how many are simultaneously capacity-feasible.
- `probe_runs.csv`: matched seed/topology/policy/traffic results: admitted, priority-weighted admission, departures, actual discretionary choices and basic elapsed time. Candidate generation occurs once per request stream and its timing is reported separately; measured runtime here is a local code benchmark, not network reconfiguration or restoration time.
- `probe_manifest.json`: role, cohort stream hashes, topology hashes, source hashes and experimental assumptions.

**The Phase 5F protection abstraction is different from Phase 5E's.** No Phase 5E model weights are reusable *without retraining and new feature/action adaptation*. The prototype makes no claim about real Irish Rail MPLS-TP configuration, performance, safety or IEEE/RFC MSRP compliance.
