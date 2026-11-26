# Phase 5G — Graph Learning, Online Optimization, and Robustness

**Status:** prospective extension, **development foundation**. This file records research commitments **before any Phase 5G confirmatory (801–820 / 77000–77002) evaluations**. The finalized confirmatory statistical plan will be version-locked after development and *before* the Phase 5G final evaluator can be built or used. Phase 5F Stage 5 results and selected checkpoints are frozen and must not be reinterpreted as a new independent test.

## Motivation, prior evidence and discipline

Phase 5F Stage 5 previously compared a frozen Enhanced PPO (C) with greedy, footprint and a greedy-guided hybrid (D) in a 500-node synthetic scenario-based environment. The C advantage over greedy was small. That result is prior evidence that motivated **this new study**, not a Phase 5G independent replication. Neither selected C nor D may be retrained or reselected in response to Phase 5G test data. All further model choices use **new training and validation partitions only**.

**Scientific objectives:**

1. Test whether a stronger, transparent **scenario-aware online optimization objective** changes the comparison with the frozen Enhanced PPO.
2. Test whether the frozen PPO and any newly selected GNN policy transfer to an **independently designed corridor-ladder network family**, as well as bandwidth and holding-time shocks. Avoid claiming real-network transfer.
3. Test whether **physical-graph message passing** provides benefit over an appropriately controlled, retrained non-GNN candidate MLP. If time and resources permit, compare a similarly budgeted discrete DDQN policy, prioritizing DDQN over discrete SAC to limit extra implementations.

All comparisons must hold the candidate generator, online information set, headroom, selected scenarios, offered demands and capacity oracle constant **within each evaluated cohort**. Different network families cannot generally share identical endpoints; match policies *within* family, rather than claiming individual service-by-service cross-family pairing.

## Network, reliability and traffic boundary

This remains **synthetic**; not Irish Rail's physical layout, actual demand, deployed protection configuration, measured failover time or safety evidence. The reference simulator reserves working/backup **end-to-end edge-disjoint pairs** under a **single-physical-link-failure scenario surrogate**. It **does not implement IEEE/IETF MPLS-TP MSRP**. Formal supervisor signoff is needed on the abstraction, or it must be reported explicitly as a surrogate, before any stronger operational claim.

Network families:

- `hierarchical`: existing frozen Phase 5F Stage 2 structurally seeded regional rings/backbone.
- `ladder`: newly implemented dual corridor-rail ladder with rungs and redundant local corridors. Has no bridges, but it is not a validated railway engineering topology and represents only **one** additional synthetic family.

Stress arms (same simulated requests across policies inside each arm):

- `base`: existing synthetic demand generation.
- `bandwidth_150`: deterministic 1.5× demand bandwidth, rounded up; other fields unchanged.
- `short_hold`: dynamic expiry durations halved (floor, minimum 1); other fields unchanged.

Primary environment family: 500 nodes, K=6, dynamic, load 4.0. Sensitivity: K=2, static traffic, and 1.6/2.5 loads if compute budgets permit. Headroom 10%; maximum searched working paths 40. **Do not choose extra arms in response to apparent confirmatory results.**

## Data roles (no overlap with prior Stage 5)

| Role | Topology seeds | Traffic-stream seeds | Rule |
|---|---:|---:|---|
| Development / scaling | 501–520 | 74000–74019 | Debugging; outcomes not confirmatory |
| GNN/MLP/DDQN training | 601–630 | 75000–75019 | Model updates only |
| GNN/MLP/DDQN validation | 701–710 | 76000–76009 | Hyperparameters and checkpoints only |
| **Reserved NEW Phase 5G final** | **801–820** | **77000–77002** | **Do not access until model freeze and final protocol** |
| Prior Phase 5F final | 401–420 | 73000–73002 | Frozen, NEVER reused for tuning |

Phase 5G foundation code **cannot** generate its reserved final role through the CLI; no final runner is bundled. All runs must produce network/request/candidate hashes and source hashes. Store immutable C checkpoint SHA-256 `e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9` and D checkpoint SHA-256 `e55b7b551b09bffe036d5bb75a471d87bb75568b48fa64aac483f947454e8e44`.

## Controls and optimization meaning

The initial stronger conventional baseline is **`scenario_minimax`**: for each arrival, enumerate *all currently feasible* candidates (at most six), calculate their **prospective global maximum single-fault scenario utilization**, break ties by incremental squared scenario-utilization energy, then capacity footprint and stable path order. This is an **exact enumeration for the defined one-arrival objective**, NOT global multi-request optimization, NOT a MILP, and **not guaranteed to outperform greedy or footprint** on eventual admission. No information about future requests is available. It uses the *same* ScenarioLedger and is externally audited by an independently implemented `SingleFaultOracle` for the accepted trajectory. A more computationally bounded online MILP can be explored in development only with an explicit fair same-information-set design; it is **not included in Stage 5G foundation**.

## New GNN architecture and comparisons

The Stage 5G graph encoder carries, at the *current decision only*, node features (backbone/degree/incident scenario state/source/destination), physical-link features (length, headroom-adjusted capacity, normal and worst-case load), canonical incidence lists, and per-candidate **working and backup edge masks**. Two shared neighbor-message-passing layers embed the physical graph; each feasible candidate pools its own working and backup links, combines them with the existing 17 candidate and eight global features, and receives a masked action score. It is **a real graph message-passing network**, unlike the Stage 5F fixed-size feature scorer. No PyTorch Geometric requirement. Discretionary actions alone are sampled by PPO, and the chosen route is enforced by the existing feasibility ledger plus independent trace oracle. The initial smoke trains this architecture for correctness only; it is *not* a full model or confirmation.

Before confirmatory testing, implement and freeze **matched-budget GNN and non-GNN MLP training** on the new 601–630 and 701–710 splits, at least 3 independent agent initialization seeds each (e.g. 7, 17, 27). Match PPO steps, episodes, candidate generator, observations available, and validation scenarios. Any advantage of newly trained GNN relative to original frozen C may also reflect training data or budget; causal attribution of graph message-passing needs the matched-budget MLP ablation. The frozen C must remain a separate untouched policy.

Optional DDQN: discretized masked six-action network, same observation families/budgets and seed partitions, with no extra final tuning. If infeasible to build and validate properly, omit and document as future work rather than dilute main evidence.

## Planned statistical analysis (must be finalized before final seed access)

- Pair algorithms **within identical topology×stream×family×stress×K×traffic×load cohorts**; compare mean *per-cohort priority-weighted admission*, not pooled request-level p-values.
- 20 independent topology seeds with three within-topology streams; compute each topology's mean paired policy difference first (**n=20 clusters**, not n=60 independent).
- Report two-sided cluster-level paired 95% t confidence intervals and p-values, absolute percentage-point improvements, admitted services, intervention rate, scenario feasibility, and per-decision/policy walltime; optionally preregister a cluster bootstrap sensitivity analysis.
- Candidate co-primary hypotheses for **hierarchical / base / 500 / K6 / dynamic / load4.0**: newly validated GNN vs frozen C, and frozen C vs scenario-minimax. Correct these two co-primary p-values (Holm). Secondary: GNN vs matched-budget MLP; frozen C vs scenario-minimax on ladder; GNN vs C on ladder; degradation under bandwidth/holding shocks. Pre-specify multiplicity correction for each family and all family interactions before running the final.
- Distinguish **out-of-family transfer** (GNN trained and selected using hierarchical networks only, then tested on unseen ladder graphs) from within-family topology-seed generalization; do not inadvertently validate GNN on ladder outcomes.
- Report **non-improvement** and uncertainty as rigorously as positive outcomes. Do not revise checkpoints, baselines, stress arms or success thresholds after first contact with 801–820/77000–77002.

## Stage boundaries

- **5G-1** (this add-on): implement topology/stress family, online scenario-minimax, real message-passing GNN PPO engineering smoke, tests, immutable C verification, development pilot. **Done locally; Windows verification pending.**
- **5G-2**: implement budget-matched GNN vs non-GNN PPO training with proper model selection, optional DDQN; audit training evidence.
- **5G-3**: formalize final estimands, policies, resource budget, statistical corrections and frozen checkpoint/source hashes. Document supervisor protection-model decision. Write once-only independent final evaluator **without running final seeds**.
- **5G-4**: execute NEW 801–820 / 77000–77002 test once; independently audit; then commence Phase 6.
