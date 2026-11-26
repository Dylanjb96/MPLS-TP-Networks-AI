# Phase 5F — Irish-Rail-Scale Synthetic MPLS-TP Provisioning: Integrated A+B+C Study

**Status (23 November 2026):** Prospective research design and **Stage 1 topology/traffic/protection feasibility prototype**. This is **not** a trained or evaluated Phase 5F hybrid and is **not** an empirical improvement claim. Prior Phases 5–5E and all of their frozen held-out results remain unchanged.

## 1. Source-based motivation and crucial accuracy boundary

Iarnród Éireann's **2023 Annual Report**, p. 18, says its operational-telecommunications replacement programme required renewal of *circa 500 nodes* using MPLS-TP. The **2019 MPLS-TP procurement notice** describes more than 2,000 km of existing fibre and operational applications including signalling, voice, level-crossing and video. Sources:

- 2023 Annual Report: https://www.cie.ie/getContentAsset/b0915028-f920-4857-bdb8-c9f3a71ec476/7bb26847-c361-42bb-98b0-48e612da32bf/2023-Iarnrod-Eireann-Report-Eng.pdf?language=en-ie
- 2019 procurement: https://irl.eu-supply.com/ctm/Supplier/PublicPurchase/146725/1/0

**These sources do not disclose a current live MPLS-TP node inventory, exact geographical/node layout, link capacities, gateway assignments, traffic intensities, or network availability.** Every parameter beyond the approximate national-scale target and application categories is an explicitly synthetic research assumption. Never call the model a digital twin, an authentic Irish Rail topology, a railway safety analysis, or a deployed MSRP implementation.

## 2. Unified research question

> Does a capacity- and protection-feasible greedy-guided deep RL policy achieve higher **per-sequence priority-weighted admission** than strong conventional heuristics when a synthetic railway transport network changes along three dimensions: **A** national scale, **B** genuinely richer protected candidate sets, and **C** service holding times and departures?

Subquestions: Do alternative routes offer measurable admission headroom? Which A/B/C treatments and interactions account for any *hybrid-minus-greedy* gain? How does provisioning time and memory grow? Does the proposed learner ever differ from greedy, unlike Phase 5E? Does the result generalize to unseen synthetic topologies and request streams?

Null or negative results are valid. No result is to be selected by the held-out test metrics.

## 3. The crucial change from Phase 5E

The current Stage 1 prototype implements an **end-to-end, edge-disjoint working/backup path pair** for each demand. All normal and individual physical-link-failure scenarios are checked under symmetric directional capacity reservations. A failed working link causes that service to use its precomputed backup. The backup may not share a **physical edge** with the working path. This is the `END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP` research abstraction.

**This is not the earlier Phase 3 `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP` model.** The earlier model switches ring-local affected segments to their opposite arc with fixed gateways; this prototype switches the whole service to a precomputed backup. Even in a 60-node scenario, Phase 5F results CANNOT be directly compared with Phase 5E as if only the routing algorithm changed. Within Phase 5F, all policies in a cell must share the **same new** independently audited feasibility rule.

A real network may have correlated conduit, node, supply, fibre-route or disaster failures not represented by single-edge failures. Such hazards are neither modelled nor ruled out.

## 4. The A × B × C factorial design

| Factor | Low condition (0) | High condition (1) | Controlled meaning |
|---|---|---|---|
| **A: network scale** | 60 synthetic nodes | 500 synthetic nodes | Identical topology **generation family**, not old Phase 5E topology versus a new family |
| **B: routing choice** | First **2** protected candidates | First **6** protected candidates | Same generator, same ordering; low candidate set is a strict prefix of high |
| **C: holding time** | No departures inside horizon | Seeded synthetic lifetimes with departures before arrivals | Same offered arrivals, priorities, bandwidths and endpoints; only release rule changes |

**Eight pre-specified primary conditions:** `60/K2/static`, `60/K6/static`, `60/K2/dynamic`, `60/K6/dynamic`, `500/K2/static`, `500/K6/static`, `500/K2/dynamic`, `500/K6/dynamic`.

150 and 250 nodes are **development scalability checkpoints only** (unless separately preregistered before fresh evaluation), not silently pooled into the primary 2×2×2 effects.

A differs in node count, traffic volume and graph layout from different draws of one generator. Report scale effects as **synthetic system-level sensitivity**, not a clean causal estimate isolating node count under otherwise identical traffic graphs. The **hybrid-minus-greedy paired difference within each cell** is the main policy-performance estimate, not raw cross-size admission.

### Topology assumptions, Stage 1

Seeded hierarchical regional rings, a cyclic backbone, independent backbone bypass links and occasional interregional links. There are 60/150/250/500 nodes, fixed positive bandwidth ceilings and 10% headroom. No required match to real Irish geographical routes, physical km per corridor, installed capacity or actual gateway count. A structural no-bridge test is mandatory. Physical links are counted once even at a gateway shared by rings.

### Protected candidate assumptions, Stage 1

K shortest **simple working routes** are explored up to a bounded deterministic search budget; for each, a disjoint backup is computed after deleting the working physical edges. Results are unique ordered working/backup pairs. For B=2, use the *first two of exactly the same B=6 inventory*. Record actual available candidates and actual feasible choices; never infer action richness merely from the configured K.

### Synthetic traffic assumptions, Stage 1

Service labels: signalling control, operational voice, level-crossing control, OCS control, CCTV/video and passenger information. Synthetic priority weights: 5 for the first three classes, 3 for OCS, 2 for CCTV, 1 for passenger information. Synthetic bandwidth ranges and class frequencies are stored in source code; **none is operator-measured**. Endpoint selection mixes within-region, backbone-facing and cross-region demands. In dynamic mode duration is sampled from `[0.15, 0.35, 0.60, 1.00, 1.50] × sequence_length` in synthetic event-time steps. Before a request at time `t`, release established LSPs with expiry ≤ `t`. Static mode suppresses release until after the horizon. This dynamic model does not include interruption, teardown signalling, reconfiguration cost, or measured busy-hour load.

### Offered-load design

Start with **1.6 requests/node** in Stage 1 development, but *do not lock it as the final primary offered load yet*. On development-only data, inspect greedy rejection and resource utilization at **predefined** load multipliers `[1.6, 3.2, 4.8]` requests/node. Before any RL final test, lock either (i) a fixed selected primary load with a written objective rule, or (ii) all three as distinct load strata. A sensible selection gate is a nondegenerate policy problem for both A levels; if this cannot be achieved with one load level, keep load as an explicit factor and report the limitation. Never calibrate to make RL win, and never tune load against test outcomes.

## 5. Protection feasibility and technical verification

The stage-1 capacity ledger is indexed by **normal + each single-link failure scenario**, and by **two directed arcs per physical link**. For every admitted demand, the active path is working unless the failed link belongs to its working route, in which case the end-to-end backup is used. All admitted demands share each scenario's capacity budget, **but loads from different exclusive failure scenarios are not summed**.

An accepted request must satisfy each scenario and each directed arc with headroom. Rejection is compulsory if no supplied candidate passes the check. Accepted requests are immutable until their synthetic expiry; no preemption. Release subtracts the old route's identical scenario reservations. Each run independently recomputes its final reservation table; small unit tests also use a separately written explicit fault loop.

**Development gate:** All topology bridge checks, backup-disjointness, normal/failure capacities, release bookkeeping, request identity parity, same-prefix candidate inventories, reproducible provenance and no held-out leakage must pass before training.

## 6. Algorithm development — one integrated GG-RL framework

The existing Stage 1 ZIP does **NOT** yet implement RL. After opportunity and scalability gates pass, implement a topology-size-invariant, action-masked greedy-guided learner:

1. Generate and independently audit feasible protected candidate pairs for the current request.
2. Greedy selects candidate 0 among feasible ranked candidates; residual action 0 means follow greedy, residual actions 1..K−1 select a different *currently feasible* candidate.
3. Encode current request, candidate bandwidth footprint, per-link residual/scenario capacity, normal/backup hops, utilization, service priority and active reservations. No future demand labels are visible to the policy.
4. Begin with a **size-invariant candidate scoring network** using aggregated current-only link/path statistics and PPO. A topology-message-passing graph encoder is a prospective **representation ablation**, evaluated only if the simpler implementation's development gates show that candidate features lose meaningful topology information. A GNN must not be added after examining final held-out results.
5. Train on complete, sequential synthetic demand episodes. Admission is rewarded by positive service priority (no arbitrary reward advantage from choosing an infeasible path). Document PPO clipping, entropy, learning rate, discount, seeds, total environment interactions and regularization. Use explicit exploration on development streams; Phase 5E's overly strong greedy imitation may be a concern, but do **not** claim a new network overcomes this without evidence.
6. During inference the model may only choose masked feasible candidates. All commits pass the independent scenario-capacity check. A failure or invalid action is reported, not silently allowed.
7. **Do not infer algorithm quality from override count alone.** Record override rate and compare observed episode reward against greedy with paired samples.

### Pre-registered explanatory controls

Mandatory within every cell: **first-feasible greedy**, **random feasible**, **minimum working+backup inverse-capacity footprint**, learned **GG-RL**, plus **RL-only (no greedy guidance)** and **GG-RL without enhanced resilience/path features** as design ablations. CSPF/MILP may be included on smaller sizes if independently adapted to the *same* formulation; do not call an omitted 500-node MILP beaten merely because it timed out.

If the representation ablation (simple vs GNN) is adopted, hold the candidate generator, data budget, feasibility oracle and held-out streams fixed. State who receives which features; compare runtime and model size too.

## 7. Pre-reserved seeds and independent statistical units

**Stage 1 development only (implemented guard):** topology seeds 100–199; request-stream seeds 70000–70099. These may be freely used for diagnostics, never as final tests.

**Proposed frozen cohorts for subsequent stages** (not yet executed):

| Role | Synthetic topology seeds | Stream seeds per topology | Distinct topology × demand pairs |
|---|---|---|---:|
| Training | 201–220 (20) | 71000–71007 (8) | 160 |
| Validation and model selection | 301–310 (10) | 72000–72002 (3) | 30 |
| **Final held-out** | **401–420 (20)** | **73000–73002 (3)** | **60** |

These cohort values must be confirmed unused before training. The same topology seed/stream-seed pair should be used for greedy, controls and GG-RL within a condition; **different node counts naturally produce different concrete graphs and endpoint instances**. Training may visit a pair repeatedly, but evaluation must use no training/validation topology or request seed.

Unit of independent inference should respect clustering: the three streams sharing a topology seed are **not 60 mutually independent topology replicates**. Recommend averaging paired per-stream differences within each of 20 held-out topology seeds; infer primary mean improvement across 20 independent topology draws, with paired t interval (df=19) and topology-cluster bootstrap as a sensitivity. The eight A/B/C cells are repeated conditions: preserve correlations when comparing main effects and interactions. Do not pool per-request admission events as independent observations.

## 8. Locked outcomes and statistical analysis plan

**Primary:** mean across independent held-out topology clusters of the **paired hybrid minus greedy per-stream priority-weighted admission fraction**, in percentage points, within the **500-node/K6/dynamic** target cell. This is the named confirmatory comparison. Secondary cells and factorial interactions are explanatory; calculate A, B and C marginal contrasts of hybrid improvement and their two-/three-factor interactions, with family-wise adjustment for designated secondary contrasts.

Secondary: ordinary admission, number accepted, per-service-class weighted admission, forced rejections, two-feasible and ≥3-feasible shares, route override fraction, capacity/failure violations (should be zero), final state fingerprints, peak scenario utilization, admission/runtime latency excluding and including candidate generation, training time, peak memory, source/seed hashes. Report **per decision** inference separately from restoration time (which is not modelled).

Acceptance gates (development-only):
- **Choice richness:** ≥10% of offered requests must have >2 distinct *structurally valid* candidate pairs under B=6 on at least one 500-node development topology; verify the *feasible* >2 share separately. This is a planning gate, not a hypothesis test.
- **Meaningful resource competition:** At least one load condition must have nonzero forced rejection in both static and dynamic primary candidate settings; otherwise primary-load selection remains open and models may learn trivial all-admit behaviour.
- **Correctness:** zero invalid admitted plans and zero scenario violations in all reproducible local tests; fixed-seed deterministic replay; release balances rechecked against explicit scenario recompute.
- **Scale:** candidate generation must be profiled separately from online decision/verification; if runtime is not practical at 500 nodes, reduce *development algorithm budgets or coding cost with documented, identical changes to all methods*, not silently remove difficult requests.

No test-driven model, gate, candidate-generation or reward adjustments are permitted after the 73000-series data are opened. If development shows no override signal or poor action diversity, record that as a finding; any redesign is a named, new preregistered method with newly sealed final evaluation data.

## 9. Already-frozen Phase 5E and interpretability

Phase 5E's 500? **No: its frozen primary system was 60 nodes**, two structurally generated candidates, static 100-request sequences under the distinct *ring-segment scenario surrogate*. Its GG-RPPO policy followed greedy in 2,131 of 2,131 feasible-choice opportunities across 60 test sequences, with mean weighted admission 40.966% and about 45% higher reported selection time. Preserve this as the baseline motivating Phase 5F, never replace those reported results with Phase 5F conditions or claim a direct protocol-to-protocol win.

## 10. Stage gates and deliverables

- **5F.0 (done here):** written protocol; standalone first-cut topology/traffic/protected-pair generators; fast scenario ledger; deterministic development probe; tests; manifest.
- **5F.1 (next):** run development probes, inventory candidate counts and admission pressure for sizes 60/150/250/500 and K2/K6, static/dynamic. Verify all outputs and profile performance. Calibrate/lock offered load *before* training.
- **5F.2:** freeze topology/traffic/protection definitions, seed partitions and baseline algorithms; add full-time trace logging, independent feasibility oracle and benchmarks.
- **5F.3:** implement and test unified GG-RL, the controls and the fixed development ablations.
- **5F.4:** train across topology/demand seeds, freeze checkpoint by validation-only selection.
- **5F.5:** one-shot untouched held-out evaluation of the 8-cell design with topology-cluster inference.
- **5F.6:** integrate into the dissertation, clearly separate old 60-node findings from new model results and limitations.

**Important implementation status:** Stage 1 code is a *working feasibility/discovery prototype*. It does not train RL, run a full primary study, use real Irish Rail confidential data, prove MSRP compliance, or demonstrate railway operational safety.
