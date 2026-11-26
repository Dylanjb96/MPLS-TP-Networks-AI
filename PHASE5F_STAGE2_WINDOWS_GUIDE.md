# Phase 5F — Stage 2, prospective national-scale simulator validation

**Status:** Development-only, **not** a Phase 5F trained learner and **not** final held-out evaluation. The previous Stage 1 outputs remain frozen. The synthetic graph is not the true Irish Rail topology; link-disjoint end-to-end fallback is **not** an MPLS-TP Shared Ring Protection or MSRP conformance implementation.

## 1. Installation and version discipline

The user reported: **Python 3.11.9; NetworkX 3.5; NumPy 2.2.6; PyTorch 2.7.1+cu118**. These are the Windows development versions. Earlier independent Stage 1 replay ran on a different environment (Python 3.13.5, NetworkX 3.6.1, NumPy 2.3.5) and observed nine differing footprint-policy rows; **a library-version cause is plausible but not proven**.

1. Back up your completed project and `outputs\phase5f\stage1_*` folders.
2. Extract `Phase5F_Stage2_Validation_AddOn.zip` into the project root. It adds `phase5f_stage2/`, `run_phase5f_stage2.py`, `tests/test_phase5f_stage2.py` and this guide. **It does not overwrite existing Phase 3–5F Stage 1 source.**
3. No changes to your installed package versions are needed for Stage 2 if the versions you reported remain the same. Do **not** downgrade Torch or NumPy merely to retry Stage 1.

In **Command Prompt**:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5f_stage2.py
python -m pytest -q
```

The add-on includes **12 new tests**. The earlier Stage 1 Windows project had 199 passing tests, so the expected combined count is **211** if nothing else has changed.

## 2. Cross-environment fingerprint gate — first run

```cmd
python run_phase5f_stage2.py preflight --sizes 60 150 250 500 --seeds 101 102 103 --fingerprint
```

Record the **entire terminal output**, especially the final `CROSS_MACHINE_FINGERPRINT_SHA256` and the three structural hashes for each network size. Our independent Linux reference, using Python 3.13.5 / NetworkX 3.6.1 / NumPy 2.3.5, is:

```text
CROSS_MACHINE_FINGERPRINT_SHA256=a5ab116ce0441b02f49a1bdf493c7f67082241a55cf901444970f712ab78aa79
```

A matching fingerprint is a positive cross-environment check for **these specific test networks and one candidate sample per network**; it is **not** a universal replay guarantee. A mismatch means **stop before substantial probes**, upload the output for investigation; **do not modify earlier evidence**.

Expected reference structural hashes for 60 nodes: topology 101=`bf8d14b7e90f1e68a5241ad9559c34a62ca6e432856a8cf5503d5aa7417246bc`; 102=`b137c47544931f1e1fa79d7d6c428444404ee639e9e5509b2d7973db861e3584`; 103=`fddbdfa52a659b1bc27a26227f6dc5e74c726ee180ef903cd9f0644d66a3e940`.

The full physical graph SHA includes routing-relevant link lengths, capacities, physical IDs and connectivity. The structural digest includes physical connectivity/capacity/layers but deliberately omits link lengths and IDs.

## 3. First controlled Stage 2 developmental probe: 60 nodes

**Only after the hash check succeeds:**

```cmd
python run_phase5f_stage2.py probe --out outputs\phase5f\stage2_60_dev --sizes 60 --topology-seeds 101 102 103 --stream-seeds 70000 70001 --loads 1.6 2.5 4.0 --k 2 6 --traffic static dynamic --policies greedy footprint --headroom 10 --max-working-paths 40
```

This produces 3 topologies × 2 request streams × 3 predeclared loads × 2 traffic modes × 2 candidate limits × 2 policies = **144 algorithm-cell records**. Networks differ in **physical edges**, not merely link distances. All arms are matched on demand streams within the relevant network/stream/load cohort. The three loads are development calibrations, not selected based on RL held-out performance.

These files are produced:

- `stage2_probe_runs.csv`: per-policy admission, weighted admission, override and **trajectory-dependent feasible-candidate counts 0..6**; separates candidate generation, policy-simulation and combined computational times.
- `stage2_candidate_inventory.csv`: per-request generated candidate IDs and counts (structural validity, **not** loaded-network feasibility).
- `stage2_topology_diversity.csv`: edge-set overlap and Jaccard scores between topology seeds.
- `stage2_manifest.json`: complete source, software, physical topology, demand and ordered-candidate SHA fingerprints; protection-model limits.

The script refuses to overwrite an existing output directory.

## 4. Staged scaling pilots, after 60-node review

150/250 nodes with two development loads:

```cmd
python run_phase5f_stage2.py probe --out outputs\phase5f\stage2_150_250_dev --sizes 150 250 --topology-seeds 101 --stream-seeds 70000 --loads 1.6 2.5 --k 2 6 --traffic static dynamic --policies greedy footprint
```

500 nodes with three predeclared loads, after profiling runtime and memory:

```cmd
python run_phase5f_stage2.py probe --out outputs\phase5f\stage2_500_dev --sizes 500 --topology-seeds 101 --stream-seeds 70000 --loads 1.6 2.5 4.0 --k 2 6 --traffic static dynamic --policies greedy footprint
```

The 500-node runs can be compute-intensive; first do the 60-node experiment and review its outputs. For genuine *final* topology generalization, reserve **fresh, unused** topology and demand seeds; no held-out streams are included in these commands.

## 5. Files to upload first

**First** share the exact `pytest` and cross-environment preflight output. Once the preflight fingerprint matches, run the 60-node probe and upload its **four** output files (listed above), preferably as a ZIP preserving names. We'll independently verify physical network diversity, canonical candidate SHA hashes, feasibility distributions, policy contrasts and numerical consistency before 500-node training.

## Research integrity notes

- Phase 5F Stage 2 uses a **new seeded graph generator and integer-weight candidate ordering**, deliberately different from Stage 1. Stage 1 evidence and Stage 2 evidence must not be interchanged as the same simulator version.
- Footprint tie-breaking uses exact rational costs and route path tuples, not the candidate-list index. This mitigates the identified ambiguous tie case. Dependency/version differences are **not assumed resolved** without the fingerprint checks above.
- There is no trained RL agent in this add-on. Do not claim hybrid improvement, national real-world fidelity, MSRP conformance, restoration timing or railway safety from these probes.
- Supervisor sign-off on the revised end-to-end link-disjoint single-failure surrogate remains outstanding.
