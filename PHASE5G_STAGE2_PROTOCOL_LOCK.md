# Phase 5G Stage 2 — Matched-budget graph-vs-tabular PPO protocol

**Status:** Development-stage neural training and validation; **not** confirmatory held-out evaluation. This document is a prospective Stage 2 training plan, created before any Phase 5G final results were examined. No Phase 5G final data are accessed by this package. Completed Phase 5F Stage 5 results remain frozen and are not part of model training or selection.

## Primary scientific question

Can explicit physical-graph message passing improve priority-weighted protected LSP admission over a similarly trained, **approximately parameter-matched**, non-GNN PPO, when both see identical synthetic offered request sequences, protected candidates, failure-scenario reservations and validation cases?

## Architecture contrast

| Arm | Architecture | Hidden width | Learnable parameters |
|---|---|---:|---:|
| MLP-PPO | CandidateActorCritic on the original 17 candidate + 8 global causal resource features | 101 | **15,758** |
| GNN-PPO | Two physical-graph message-passing layers + candidate working/backup pooling + the same 17+8 features | 32 | **15,778** |

Difference **20 parameters (0.127%)**. This is a stronger control than equal hidden widths. The GNN does have **additional graph inputs**; this is the intended representation intervention. It still does not isolate every potential architecture difference, because computational cost and inductive biases differ.

Both arms use the same trainable actor-critic PPO objective, mask invalid candidate slots, optimize complete-episode remaining weighted-admission reward, and never inspect future requests during decisions.

### Fixed matched training settings

- **6 runs:** MLP and GNN × agent seeds **7, 17, 27**.
- **48 training episodes per run**. Every 24-episode block covers all combinations of 4 network sizes × 3 loads × 2 traffic modes exactly once. Within matched agent seeds the GNN and MLP use identical episode cohorts and schedules.
- Network sizes **60, 150, 250, 500**; loads **1.6, 2.5, 4.0** requests/node; static and dynamic traffic; hierarchical synthetic topology **only** for training and model selection.
- Training topologies **601–630** and traffic seeds **75000–75019**. The schedule cycles through these predetermined partitions; there is no adaptive case selection.
- Validation topologies **701, 702, 703** and streams **76000, 76001**, under **500 nodes / 6 candidate pairs / dynamic / load 4.0**. These 6 paired validation cohorts are repeated at episodes 12, 24, 36 and 48, and used for model selection only.
- For each training episode the policy processes **every request** with stochastic on-policy actions. At most **48 uniformly selected discretionary action samples** per episode update the policy; the subsampling is deterministic from seed+episode and applies to both arms. Training interaction count and gradient-update sample count are recorded separately. Some congested trajectories may yield fewer eligible actions.
- PPO: Adam learning rate **0.0003**, two epochs/update, mini-batch size **8**, clip **0.2**, gamma **1.0**, entropy weight **0.01**, value weight **0.5**, gradient norm max **0.5**. No behavior cloning guidance, no greedy logit gate. Both arms are evaluated deterministically and masked to feasible candidates.
- Single-link-failure scenario oracle, **10% capacity headroom**, **K=6**, maximum explored working paths **40**. Candidates generated before routing policy selection, no request suffix given to either policy.
- Validation compares both learned models (on identical cohorts) with **greedy, footprint, scenario-minimax, and the frozen Phase 5F Enhanced PPO C**. Each recorded validation choice is checked by the independent single-fault trace validator. The inherited **MSRP noncompliance** limitation remains.
- Best checkpoint **within each run**: highest mean priority-weighted admission across the six hierarchical validation cohorts; ties prefer the earliest checkpoint (strict improvement only).
- Final Stage 2 architecture selection: mean of three per-run best validation values; ties prefer simpler MLP. Within each arm, highest individual seed result, then smallest seed. Both best-per-arm models are preserved regardless of winner.

### Generalization safeguards

- **Corridor-ladder family is excluded from Stage 2 training and validation**. Phase 5G Stage 1 ladder development pilots do not select these models. The dedicated ladder robustness evaluation will use a separately locked Stage 3/4 protocol.
- Phase 5G **final** seeds **801–820** and **77000–77002** remain inaccessible. Phase 5F held-out **401–420** and **73000–73002** are likewise forbidden.
- No additional source retuning, model selection, or operating-condition choice using either final holdout is permitted.
- The previous frozen Stage 5 checkpoint hash must remain `e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`.

### Interpretation and limitations

- This is a computationally bounded **fixed-budget** comparison, not proof that either model has converged. Training duration may be inadequate for either architecture; report learning curves and inference cost.
- Interactions are **matched by offered arrivals and seed-matched schedules**. The policies induce different reservation states, so the numbers of discretionary actions and actual PPO update samples may differ; report both. Graph message passing has a higher per-action computational cost, which must be included in any practical recommendation.
- Stage 2 validation scores are **not confirmatory evidence**. A separate locked generalization study on fresh seed partitions is needed before comparative performance claims.
- All environments are **synthetic**, not the physical Irish Rail communications infrastructure; the protection model is `END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP`, not standards-compliant MSRP or restoration-time engineering.
- Approval by the academic supervisor for the protection abstraction is outstanding.
- DDQN is a **separate optional RL comparator**. If included, it requires its own prospective budget protocol and stage-specific training implementation **before Phase 5G final evaluation**. The present package intentionally concentrates on the stronger GNN-versus-parameter-matched-MLP question.
