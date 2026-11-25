# Phase 5F Stage 5 — Locked final held-out evaluation protocol

**PRE-EXECUTION PROTOCOL.** The reserved Stage 5 topology and traffic seeds must not be evaluated by the development team before installing and accepting this protocol. This file is distributed **before** any final evaluation. Running `preflight`, `devcheck`, or the 21 new tests **does not generate any held-out network**. The actual held-out evaluation requires the separate explicit `final --confirm-final` command.

## 1. Research question and frozen comparisons

**Primary test:** On previously unseen **synthetic 500-node networks**, at **K=6** feasible end-to-end working/backup candidate limit, **dynamic service holding times**, **4.0 offered service requests per node**, determine whether the **preselected C_enhanced PPO** delivers a higher mean priority-weighted admission ratio than **greedy routing**.

Precommitted primary estimand: for each of 20 held-out **topology seeds**, average the matched priority-weighted admission difference over three held-out demand stream seeds. The final effect is the equally weighted mean of these 20 topology-level paired differences, expressed in percentage points. **The 20 network topologies, not 60 streams nor thousands of LSPs, are the inferential clusters.** A two-sided paired one-sample t interval (95%, df=19) and two-sided p-value are the primary inferential summary. A fixed-seed 10,000-replicate topology-cluster bootstrap interval is a sensitivity analysis. A primary positive point estimate is not automatically evidence of statistical superiority.

**Three secondary primary-cell comparisons**, analyzed as a Holm-corrected family of two-sided tests: (i) selected C vs footprint, (ii) nominated D_full_hybrid vs greedy, (iii) C vs D_full_hybrid. Greedy, footprint, random feasible, selected C and nominated D are **all** included in every matched cell. Random feasible is a descriptive control, not the main comparator.

## 2. Freeze of the two trained checkpoints

**Selected validation winner:** `C_enhanced`, agent seed 27, training episode 60, SHA-256:

`e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`

**Preserved proposed full hybrid:** `D_full_hybrid`, agent seed 7, episode 84, SHA-256:

`e55b7b551b09bffe036d5bb75a471d87bb75568b48fa64aac483f947454e8e44`

The bundled `stage5_frozen/selection_manifest.json` is the exact official Stage 4 selection artifact. All Stage 1/2/3/4 source hashes and both checkpoint file hashes must match before either model may be loaded. The models use the same exact frozen candidate-shared actor with 17 candidate features, eight global features and six masked action slots. **C** applies enhanced resource features, no greedy-default input, no behavior-cloning guidance and no evaluation gate. **D** applies the enhanced resource features and the combined guidance bundle with a frozen deterministic 0.1 logit-margin greedy fallback gate. No training, hyperparameter changes, selection, calibration, or learned threshold adjustments occur in Stage 5.

## 3. Full factorial, fixed-load sensitivity analysis

Factor **A (network scale):** 60 vs 500 synthetic nodes (the **same Stage 2 generator** throughout).

Factor **B (route candidates):** first 2 vs first 6 of the *same ordered candidate list*.

Factor **C (service holding):** static vs dynamic departure semantics, with the **same arrival request attributes** and topology; only expiry/holding differs.

Evaluate all **eight** A×B×C cells at fixed **4.0 offered requests/node**. The primary national-scale cell is included once. For each cell, use **all** 20 final topology seeds × **all** three final stream seeds × all **five** frozen policies. Thus:

- 8 cells × 20 topologies × 3 streams = **480 distinct matched cell/cohort evaluations**.
- 480 cohorts × 5 policies = **2,400 full sequential policy episode evaluations**.
- 20 independent topology clusters, each with 3 streams, for any within-cell estimate.

All seven classical 2³ factorial effects are computed on the **C minus greedy admission gain** using paired per-topology differences and a fixed factorial sign convention. The seven two-sided topology-cluster p-values receive a separate Holm correction and are **secondary/exploratory**; report effect magnitude and uncertainty rather than cherry-picking the most favorable cell. The 60-node studies are genuine synthetic-model ablations, **not direct comparisons to old Phase 5E's different simulator**.

The Stage 2 development loads 1.6 and 2.5 are **not** part of the Stage 5 held-out grid. The fixed load 4.0 was designated as the main high-contention setting **before Stage 5 held-out access**. The prior lighter-load development investigations remain documented and must not be relabeled as final test data. No final-load search is permitted after observing final results.

## 4. Data partitions

| Partition | Physical network topology seeds | Traffic-stream seeds |
|---|---|---|
| Stage 4 training | 201–220 | 71000–71007 |
| Stage 4 validation | 301–304 (selection subset) | 72000–72001 |
| **Stage 5 FINAL HELDOUT** | **401–420** | **73000, 73001, 73002** |

**No final cohort is used for model selection.** `preflight` examines only source files and frozen model evidence. `devcheck` tests only the already-used development cohort `60/301/72000/2.5/dynamic/K6`. Automated tests use development scenarios or fabricated numerical records, not any graph from held-out seed ranges.

## 5. Simulator, fairness and safety/research scope

All policies encounter identical candidate inventory, offered requests, and network capacity/headroom on a given cell/topology/stream. Each has its own independent scenario reservation ledger and its own evolving admitted-service state. The source algorithm's `run_episode` forcibly admits whenever at least one feasible route exists (it does not include a voluntary reject action). A scenario-capacity checker independent of the production ledger replays each decision trace, checks all admitted actions and all rejections against a separate direct failure-scenario accounting implementation, and periodically verifies the complete feasible candidate mask.

The simulator is the **synthetic `END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP`**. It does **not** implement MPLS-TP MSRP, shared-ring rerouting, MPLS-TP standards compliance, real railway switching/restoration times, safety certification, or the actual Iarnród Éireann network/topology/traffic. Supervisor approval of using this abstraction is still outstanding. Runtime measures are **computer simulation wall times**; report them separately from route-generation time and never as real railway switching times.

## 6. Integrity, output and one-time execution

The package ships a **single immutable final plan**. The final CLI has **no flags** to choose different final seeds, switch algorithms, adjust network load or change the test size. It requires an explicit `--confirm-final`. Before opening any reserved seed, it records a locked source/checkpoint/parameter/runtime hash. At the end of each matched cell, one compressed, atomic evidence record is written with five matched policy results, source/stream/candidate hashes, independently replayed failure-scenario oracle digests, and full decision traces for **both RL models on the primary cell**.

An interrupted run may use `--resume` **only with exactly the same source, checkpoints, runtime, input plan and output directory**. Completed cohorts are read and validated rather than recalculated. Existing final outputs or successfully finalized analyses cannot be silently overwritten. Do not manually delete, edit, or rerun unfavorable cohorts.

Only after all 480 matched cohorts are present and consistent does Stage 5 compute:

- `heldout_runs.csv`: all 2,400 policy episodes with matched hashes and runtime measures.
- `heldout_cell_descriptives.csv`: the 40 cell×policy descriptive summaries.
- `heldout_primary_inference.csv`: the one primary and three multiplicity-adjusted secondary comparisons.
- `heldout_factorial_effects.csv`: seven exploratory interaction/main effects, Holm adjusted.
- `heldout_topology_paired_deltas.csv`: all per-topology paired differences for cross-checking.
- `stage5_final_manifest.json`: full provenance and SHA-256 for each exported results table.

The `verify` CLI independently reads these files in **read-only mode** after finalization and recomputes the primary estimate; no new graph or demand is generated.

## 7. Interpretation constraints

Do not retune or rerun Stage 4 after observing Stage 5 results. Report unsuccessful or nonsignificant results transparently. Apparent validation improvement (+0.7889 pp for C vs greedy) is a selection score and should **not** be presented as a held-out result. Do not assert RL superiority until inferential uncertainty is reviewed. All 20 topology realizations are generated within the same synthetic family, so generalization remains **within that family**, not to an independent actual deployed railway network. Stage 5 completes the *comparative synthetic study*, not operational railway qualification.
