# Phase 5C — Frozen research protocol draft (supervisor review required)

**Version:** `PHASE5C_DISCRETE_RL_V1`  
**Research evolution:** Post-Phase5B, motivated after examining held-out outcomes on seeds 31000–31029 and 41000–41029. This adaptive extension must be described transparently; do not label it as original Phase 1 preregistration.

## Objective, hypothesis and models

**Question:** Can learning-based protected LSP provisioning improve priority-weighted and ordinary sequential admission compared with greedy, CSPF-style, online MILP V2, random-feasible and earlier DDQN/SAC policies, within the frozen finite candidate set?

**Primary endpoint:** mean seed-level priority-weighted admission. **Secondary endpoints:** ordinary admission; paired admission deltas; selection time; training compute and simulator interactions; accepted reservation-state differences; scenario utilization. No claim of RFC 8227 protection switching or railway safety certification.

**Algorithms:** masked PPO, discrete offline BCQ, CQL, AWAC (offline stage or explicitly separate hybrid extension), and IQL. Previous DDQN, discrete SAC and conventional algorithms remain fixed.

## Data separation and baselines

- Synthetic topology: 60 nodes, dual gateway, topology seed 42.
- Headroom: 10%; requests per seed: 100; protected candidates: at most 2.
- Train demand: 10000–10079; validation: 20000–20009; fresh final test: 51000–51029.
- Previously inspected test seeds 31000–31029, 41000–41029 and exploratory Phase4 seeds 2027–2041 are prohibited for prospective Phase5C selection.
- All policies use the identical Phase3 candidate generator, Phase3 capacity-feasibility checker and independent empty simulator ledger for a paired demand seed.
- No future-demand oracle; forced rejection only if no candidate is feasible.

## Offline dataset

Build a **single**, versioned dataset on training streams only: each seed yields one greedy episode and one random-feasible episode, 16,000 transitions at the primary configuration. No nontraining demand seeds in logs, no new simulator interaction during offline gradient updates. Store rows, legality mask, action, reward, next rows, next mask, terminal flag, behavior ID, demand-seed ID and stream hashes. Verify dataset SHA-256. This is simulated logged experience, not real infrastructure traffic.

## Planned hyperparameter grid and confirmation

All other parameters remain the defaults documented in the guide. Train **three random initializations 7/17/27 per configuration**. Configurations per family:

| Family | Hyperparameter | Values |
|---|---|---|
| PPO | learning rate | 0.0003, 0.0001 |
| BCQ | behavior probability threshold | 0.3, 0.1 |
| CQL | conservative coefficient | 1.0, 0.5 |
| AWAC | advantage temperature | 1.0, 0.5 |
| IQL | expectile | 0.7, 0.9 |

Select the best configuration by **mean** validation priority-weighted admission over all three seeds, with ordinary admission as secondary; within selected configuration use the checkpoint selected on validation priority-weighted admission. Fixed training limits: 160 PPO episodes; 3,000 offline gradient steps; validation every 10 PPO episodes or 300 offline steps; patience 8 validation checks. An optional AWAC hybrid extension may finetune the validation-selected offline configuration for 80 online TRAIN episodes using 64 gradient steps per episode and 50:50 offline/new online transition mixes, with fixed epsilon 0.1. This hybrid model is an explicitly different compute-budget treatment.

## Inference, audit and statistics

Evaluate all validation-selected checkpoints exactly once on new held-out test seeds 51000–51029. Record request hashes, checkpoint SHA-256, source hashes, all decisions, capacity audit outcomes, final-state digests and timing. Compute seed-level paired differences relative to greedy and direct paired contrasts with DDQN/SAC. Present estimates, uncertainty, and sample sizes; discuss multiplicity and limited topology generality. Performance on the final test must not motivate retraining/tuning on those same seeds.

## Scope and validity

Model is `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`: ring-local alternative segments for simulated single physical-link failures. Not an RFC 8227 MSRP signaling, protocol or timing implementation. Not an industrial railway network or safety assurance. A two-candidate restricted route-selection problem is not unrestricted graph routing. Supervisor approval of modeling scope remains outstanding.

**Interpretation policy:** algorithm performance is an empirical question; there are no preset expected improvements and negative findings must be retained. Cross-topology transfer/generalization requires a distinct, newly specified evaluation design.


## Development implementation validation

Development evidence: 13 new tests passed, 157 complete project tests passed; end-to-end 12-node integration smoke and 60-node micro-compatibility smoke passed. These demonstrate functionality only, not comparative effectiveness or protection-model validation.
