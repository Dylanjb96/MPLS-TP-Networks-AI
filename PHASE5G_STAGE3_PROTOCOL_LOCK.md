# Phase 5G — Stage 3 preregistered checkpoint and final evaluation protocol

**Protocol ID:** `PHASE5G_STAGE3_FROZEN_CONFIRMATORY_PLAN_V1`  
**Status:** PRE-FINAL; must be frozen with original checkpoint bytes **before** running the reserved Phase 5G test seeds. Stage 3 does **not** evaluate held-out networks. Stage 4 will implement the separate final runner using this exact protocol. Any change thereafter is explicitly labeled an amendment; results from an accessed test set cannot be represented as fresh confirmation after model tuning.

## Scientific boundary and models

The research compares resilience-aware protected path-pair admission in a **synthetic**, end-to-end link-disjoint working/backup, single-physical-link-fault scenario-reservation surrogate. It is not standards-compliant MPLS-TP MSRP, not an actual Irish Rail network, not restoration timing evidence, and not a railway safety demonstration. Supervisor sign-off on the abstraction remains **pending** (`protection_model_approved=false`); this does not change any synthetic simulation measurements but constrains interpretation.

Six fixed policies, identically offered requests and candidate inventories within every cohort:

1. `gnn_PPO`: Stage 2 selected GNN seed **27**, episode **48**, SHA-256 `7db8b92ab9bc1648a9224eea727efe035848543fe514c7d7c14be3ee584658c9`; score **0.8873967834723134** on Stage 2 *validation only*.
2. `mlp_PPO`: matched-budget Stage 2 selected MLP seed **17**, episode **24**, SHA-256 `942c8edd516ad28b4012733bfd8504af01fde50f47444be7d15c8e02995e3363`; score **0.8717528121563379** on *validation only*.
3. `C_frozen`: unchanged Phase 5F Enhanced PPO C seed 27, SHA-256 `e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`.
4. `scenario_minimax`: online exact enumeration of up to K feasible route pairs for *current-arrival maximum single-fault scenario utilization* objective with declared stable tie-breaks. **Not** a globally optimal sequential MILP and no future-demand information.
5. `footprint`: established capacity-footprint online heuristic.
6. `greedy`: established first-feasible online baseline.

Both newly learned policies use masked, deterministic argmax at test time and no additional greedy gate / behavior-cloning adjustment. Never retrain, retune, restart a training job, change preprocessing, choose a new checkpoint, or drop an unfavourable policy after inspecting final results. The historical docstring-only source-drift exception for Stage 2 is preserved verbatim in the Stage 2 selection record; Stage 3 refuses unrelated source changes.

## Partitions and roles

| Role | Topology seeds | Traffic streams | Access |
|---|---|---|---|
| Phase 5G development | 501–520 | 74000–74019 | Already used for development only |
| Stage 2 training | 601–630 | 75000–75019 | Training only |
| Stage 2 validation | 701–703 (predeclared used subset of 701–710) | 76000–76001 (subset of 76000–76009) | Selection only |
| **New, untouched Phase 5G final** | **801–820** | **77000–77002** | **Stage 4 once after freeze** |
| Prior Phase 5F final | 401–420 | 73000–73002 | Never reuse for model selection |

No network/traffic from reserved Phase 5G final is used by Stage 3 preflight, freeze, tests or verification. A provenance statement is not proof of all unsubmitted local activity; the final runner must implement enforced seed separation and immutable output integrity.

## Fixed final design

All final cells use **500 nodes**, priority-weighted admission as primary score, **load 4.0 requests/node**, dynamic traffic, **10% capacity headroom**, max explored working paths **40**, same scenario model, identical cohorts across all policies; candidate generation occurs before choosing the route, no future requests are exposed. Every cell uses all **20 independently generated topology seeds × 3 demand streams per topology = 60 matched cohorts**. Two network families: `hierarchical` (training family; unseen seeds) and `ladder` (withheld out-of-family; Stage 2 trained/selected on hierarchical only).

**Eight cells locked for final evaluation:**

| Cell | Network family | Stress | K | Role |
|---|---|---|---:|---|
| H_BASE | hierarchical | base | 6 | Primary confirmatory cell |
| L_BASE | ladder | base | 6 | Out-of-family transfer |
| H_BW | hierarchical | bandwidth_150 | 6 | Bandwidth-shock robustness |
| L_BW | ladder | bandwidth_150 | 6 | Bandwidth-shock transfer |
| H_HOLD | hierarchical | short_hold | 6 | Service-duration robustness |
| L_HOLD | ladder | short_hold | 6 | Service-duration transfer |
| H_K2 | hierarchical | base | 2 | Candidate-diversity sensitivity |
| L_K2 | ladder | base | 2 | Candidate-diversity transfer |

Total **8 cells × 20 topologies × 3 streams = 480 matched cohorts**, each with six policies, **2,880 full policy simulations**. All cells must be completed for an unqualified complete-study claim; if resources prevent completion, report explicitly which prespecified cells were missing and why, with no results-dependent subset selection. No opportunistic load-change / static-traffic arms will be added to this confirmatory evaluation; any such work is separate exploratory analysis on fresh data.

## Fixed statistical analysis

- Unit of independent replication: **20 network topologies**, not 60 demand streams or individual service arrivals. Pair policies in each topology×stream×cell. Average each topology's three stream-level policy admission-ratio differences (100×[policy – comparator] = percentage points), and analyse the 20 resulting paired differences using a **two-sided one-sample t-based 95% CI** with 19 df and p-value (two-sided). Topology cluster mean differences are the primary estimand; report individual requests and admitted counts only descriptively, not as independent statistical replicates. Report effect size in percentage points, cluster standard deviation, CI, p, positive/negative/tie clusters. For 0 variance and zero mean, use p=1, CI=[0,0]; 0 variance and nonzero mean require special transparent treatment rather than a misleading finite t statistic.
- **Two co-primary hypotheses** in `H_BASE`: `GNN – MLP` (graph representation vs matched neural architecture) and `GNN – scenario_minimax` (vs strongest planned online optimizer). Compare two-sided null differences of zero. Adjust the two p-values with **Holm family-wise α=0.05**; do not make a superiority claim from selection-validation results or from raw unadjusted p-values.
- **Eight predeclared secondary hypotheses** tested together with a separate Holm adjustment across eight: `L_BASE GNN–MLP`, `L_BASE GNN–scenario_minimax`, `H_BASE C_frozen–scenario_minimax`, `L_BASE C_frozen–scenario_minimax`, `H_BASE GNN–C_frozen`, `L_BASE GNN–C_frozen`, `H_BASE GNN–greedy`, `L_BASE GNN–greedy`. Clearly distinguish secondary family from the two co-primary hypotheses; no unadjusted secondary p-value is called confirmatory.
- Stress contrasts and K2 sensitivity are **descriptive/exploratory**: report paired differences and 95% cluster intervals for each policy contrast; include stress-minus-base / K2-minus-K6 cluster-level differences and uncertainty without treating their unadjusted p-values as confirmed discovery. Include uncertainty and negative results. Report deviations for any requested additional testing as *post hoc*.
- Additional diagnostics for **every** policy and cell: unweighted request admissions, priority-weighted admissions, number of discretionary routing choices, overrides versus greedy (for RL policies), candidate diversity / feasible counts, departures, oracle-protection-feasibility violations, maximum single-scenario utilization, candidate-generation wall time separately from **policy** wall time, per-decision latency where available. Hardware/timing qualification required. If a violation occurs, preserve all evidence and flag cohort; do not silently omit it or report it as valid success.
- Paired evaluation implementation must verify exact **network/request/candidate digests** among policies for each cohort, source fingerprints, actual six model/optimizer identities, the full 480-cohort grid, and **no duplicate/missing results**. All model checkpoint hashes must match Stage 3 manifest before any final network is generated. Final outputs are append-only / resumable; once-only evaluation is not additional tuning. Rerunning for technical recovery is allowed only with identical input manifests and without changing performance parameters or data selection.

## Confirmatory access policy and stage boundaries

**Stage 3:** verify all 6 Stage 2 run identities and provenance exceptions; freeze Stage 2 selected GNN+MLP and older frozen C, protocol, and evaluation-source SHA fingerprints. Do **not** generate final networks, calculate final policy performance, train DDQN or run Stage 4 code. Stage 3 does not certify performance on unfamiliar networks.

**Stage 4:** implement, unit-test and smoke-test final evaluator **on development seeds only**; independently review that evaluator and the statistical plan against this locked protocol. Only then authorize a single evaluation on 801–820 / 77000–77002. The Stage 4 source addition must not change earlier code hashes. Preserve all original Stage 5F records unchanged.

**Optional DDQN** was not trained under Stage 2; it is **excluded from the locked six-policy final evaluation** and can only be reported as separately preregistered additional work on a fresh independent final data partition. This prevents post-selection addition of competitors to the original confirmatory family.
