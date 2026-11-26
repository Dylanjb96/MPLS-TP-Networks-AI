# Phase 5F Stage 4 — Research protocol, frozen before full training

**Status:** Development/training/validation only. Full-scale experiments are to be run on the user's Windows machine; the included engineering smoke and miniature 12-run selector are *not* evidence of comparative advantage. Stage 5 held-out data must remain unread.

## Research objective

Test whether a candidate-aware PPO provisioning policy trained across genuinely distinct synthetic protected-transport networks can learn sequential choices that yield greater **mean per-cohort priority-weighted admission** than greedy and capacity-footprint heuristics. The national-scale primary condition is 500 synthetic nodes, six end-to-end disjoint route pairs, dynamic holding times, and 4.0 offered requests/node. Network and traffic parameters are **synthetic assumptions**, not Irish Rail engineering data.

The physical-failure capacity model is `END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP` and `protection_model_approved=false`. The observed decision times are simulator implementation timings, not protection switching or railway restoration times. Supervisor approval of the abstraction is outstanding.

## Frozen factors

**Training programme:** 96 maximum PPO episodes per run, across network sizes **60, 150, 250, 500**, traffic load values **1.6, 2.5, 4.0** requests/node and both **static and dynamic** holding conditions. A shuffled 24-combination cycle ensures exposure to all size × load × traffic combinations within each complete 24-episode block. At each episode use a training-only physical-network seed and a training-only demand seed. PPO learns from entire sequential service request episodes; no voluntary refusal is permitted when a feasible protected pair exists. Priorities contribute to cumulative admitted reward; no future demands are exposed in the online features.

**Model:** the exact frozen Stage 3 six-slot candidate-shared actor and value network, hidden size 64, candidate features 17, global features 8, masked categorical action selection, update learning rate 0.0003, undiscounted episodic reward (gamma = 1.0), PPO epochs 2, minibatch 64, entropy coefficient 0.01. *Greedy guidance* denotes the **joint treatment** of the default-choice indicator, a behaviour-cloning regularizer (weight 0.01) and a deterministic evaluation-only margin gate (0.1). The individual causal contributions of these three constituents are **not** isolated by the primary four-arm design.

### Four-arm, three-initialization factorial

| Arm name | Guidance bundle | Enhanced congestion/capacity features | Agent initializations |
|---|---|---|---|
| `A_standard` | No | No | 7, 17, 27 |
| `B_guided` | Yes | No | 7, 17, 27 |
| `C_enhanced` | No | Yes | 7, 17, 27 |
| `D_full_hybrid` | Yes | Yes | 7, 17, 27 |

All arms use the **same network actor architecture, candidate generation, action masking, and scenario-feasibility ledger**. Feature ablation zeroes the same fixed feature channels both during on-policy action sampling and PPO gradient updates. `A/C` exclude the greedy reference indicator, behaviour-cloning penalty, and confidence gate; `B/D` include the entire guidance bundle. All policies still must accept a request whenever a feasible route exists.

### Training, validation and test partition

| Role | Physical topology seeds | Demand-stream seeds |
|---|---|---|
| **Training** | 201–220 | 71000–71007 |
| **Validation pool** | 301–310 | 72000–72002 |
| **Frozen Stage 4 model-selection subset** | **301–304** | **72000–72001** |
| **Stage 5 untouched final evaluation** | **401–420** | **73000–73002** |

No Stage 4 command exposes a final-test role, nor accepts seeds outside the exact development ranges. The final test seeds must not be supplied even as a development or cache argument. The old Phase 5C/E heldout seeds are not used for training or tuning.

### Checkpointing and selection

Every 12 episodes (and at the last episode), evaluate the policy deterministically on all **8 predeclared validation cohorts**, each with 500 nodes, K=6, dynamic holding, load 4.0 and 10% headroom. Use exactly the same candidate inventory and request stream to evaluate greedy, footprint and the current learned policy independently. Early stop after five consecutive non-improving validations; the maximum remains 96 episodes. **Within a run**, save the earliest checkpoint achieving the maximum mean per-cohort priority-weighted validation admission. **Between configurations**, choose the highest mean of the three individual best-checkpoint validation scores. Break exact ties by the order A, B, C, D. **Within the winning configuration**, choose the highest-scoring initialization, then lowest numeric agent seed.

The `D_full_hybrid` ablation-best checkpoint is also explicitly preserved even when another arm wins the selection; it is not substituted for the selected winner. In Stage 5, a separately preregistered experiment will need to compare the locked selected policy and the nominated full hybrid with greedy, footprint and random feasible under matched network/traffic conditions.

The validation set is shared across arms and used repeatedly for checkpoint selection: selection scores are *not* unbiased evidence of population superiority. Stage 5 holds the independent evaluation.

### Main inference and interpretation boundaries

- The true independent units include **distinct network topology realizations** and demand streams. Multiple streams on the same topology induce clustering; Stage 5 must account for this (e.g. topology-level paired estimates and hierarchical uncertainty); do not treat each offered LSP as independent.
- Report weighted and ordinary admission, override rate, discretionary choices, infeasible-action attempts, compute time and sensitivity to network scale, candidate count, traffic dynamics and offered load.
- Avoid claiming a larger network alone caused gains when candidate generation or traffic model also changed. A/B/C ablations within a frozen cell are required to identify contributions.
- No exact rollout policy-improvement theorem is claimed for the trained approximate PPO model. The learned policy may underperform greedy or footprint.
- The generator is *inspired by publicly reported national communications scale*, not validated as Irish Rail's live topology or operating traffic. No MSRP, switching time, safety or service assurance claims.

### Reproducibility and audit gates

1. Do not modify frozen Stage 1/2/3 source packages. Verify Stage 3 source hashes with preflight.
2. Run Stage 4 tests and the full local tests; investigate failures before training.
3. Run a new-output-directory miniature 60-node Stage 4 integration test.
4. Freeze this protocol and use the standard 12-run batch without altering its options between arms.
5. Each complete run saves the training plan, history, periodic validation, selection checkpoints, checkpoint hash, source hashes, topology/request/candidate hashes, software versions and no-heldout flags.
6. Review all 12 run manifests, and run strict validation-only selection once.
7. **Do not run Stage 5 final test yet.** Independently verify the 12 training outputs and selection before designing/running Stage 5 evaluation.

**Load calibration amendment:** Original Stage 1 research planning mentioned 1.6/3.2/4.8 requests/node; Stage 2 development-only calibration used 1.6/2.5/4.0. The latter set is frozen for this Stage 4 training protocol *before* the untouched Stage 5 evaluation.
