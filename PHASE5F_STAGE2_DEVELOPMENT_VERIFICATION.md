# Phase 5F Stage 2: development verification (independent Linux container)

Date: 23 November 2026. All results here are **development-only** and from an environment **different** from the user's Windows PC.

## Original Stage 1 provenance and observed discrepancy

The user reported Python 3.11.9, NetworkX 3.5, NumPy 2.2.6 and PyTorch 2.7.1+cu118. Independent Stage 1 replays occurred with Python 3.13.5, NetworkX 3.6.1, NumPy 2.3.5. Nine of 108 footprint-heuristic 60-node records had reproducibility differences; the cause was **not** conclusively diagnosed. No earlier data, source, or results have been modified.

## Stage 2 source and checks

Stage 2 is a **new additive module**, not an overwrite of Stage 1. Main corrections:

1. Regional gateway positions, local order, crosslinks and backbone bypasses vary with topology seed; structural and full graph SHA checks cover connectivity and routing-relevant attributes.
2. Routing path weights are deterministic integers, with physical-edge salt, rather than floating-point addition; backup shortest-path tie handling uses a stable sorted-neighbour heap. Exact rational footprint scoring with route-tuple tie-break replaces the original candidate-index tie.
3. The candidate inventory is SHA-fingerprinted including **ordered working/backup node paths**, not merely a count of candidates.
4. Per-seed feasibility bins 0–6 measure available **capacity-feasible** alternatives as reservations evolve. An independent reconstruction of all single-link-failure scenario loads is tested for small networks and for the final active ledger in each probe.
5. Environment metadata, network hashes, topology overlap, execution times and optional RSS are output in the Stage 2 manifest/CSVs. Stage 2 uses predeclared development seeds and load levels only.

**Automated results in this container:** 12/12 new tests; 25/25 combined Stage 1+2 tests. The user's remaining earlier tests are not mounted in this minimal Stage 2 development directory; their prior Windows suite showed 199 passes.

Preflight sizes 60, 150, 250, 500 each validated for topology seeds 101, 102 and 103 (12 networks); all 12 generated six sample link-disjoint pairs. **Different structural hashes for all three seeds at every network size**. The Linux `CROSS_MACHINE_FINGERPRINT_SHA256` is:

`a5ab116ce0441b02f49a1bdf493c7f67082241a55cf901444970f712ab78aa79`

**Small integration check:** 60 nodes × topology seeds 101, 102 and 103 × one stream 70000 × load 1.6 × static/dynamic × k 2/6 × greedy/footprint = 24 development policy-cell records, 288 candidate inventories. Execution passed and all final scenario ledgers independently recomputed.

**National-scale integration check:** 500 nodes, seed 101, stream 70000, 1.6 requests/node, static traffic, k=6: generated 800 candidate inventories and completed both greedy and footprint controls. One observed development run (not a statistical result): greedy admitted 673/800; footprint admitted 681/800. Feasible-choice bins 0–6 were populated, supporting the richer-choice diagnostic. These are Stage 2 simulator conditions, **not** directly comparable to previously published Stage 1 conditions because network and candidate-generation semantics changed.

## Remaining validation before training RL

- **Cross-machine fingerprint match on Windows**. The presented hash is from this Linux environment, not from the user's Windows machine.
- At least two independent current software environments should replay *identical Stage 2 cohort outputs*, including ordered candidate SHA. The candidate generator still relies on NetworkX's shortest-simple-path enumeration, so integer weights and canonical tie-breaks reduce but **do not eliminate every theoretical cross-version variation**.
- Higher-load and multi-topology national-scale probes, including capacity-feasible candidate distributions and timing/memory profiles.
- Supervisor review of the Stage 2 surrogate: single-failure end-to-end link-disjoint pair reservation is **not** true MPLS-TP MSRP and says nothing about operational railway safety/restoration latency.
- No RL training, model selection or held-out results yet exist for Stage 2.
