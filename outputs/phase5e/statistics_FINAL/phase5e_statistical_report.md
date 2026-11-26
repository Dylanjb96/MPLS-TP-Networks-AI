# Phase 5E: Greedy-Guided Residual Reinforcement Learning — statistical report

- Evaluation role: **primary**.
- Matched demand sequences: **60**; offered requests per sequence: **100**.
- Main inferential unit: demand seed, NOT individual requests.
- All comparisons share candidate generation, fixed topology, forced-only rejection and the same single-link scenario surrogate.
- Nominal CIs: two-sided 95% Student-t intervals across demand seeds.
- P-values: paired, two-sided t-tests; Holm family-wise adjustment across all alternatives to greedy.
- Zero-width confidence intervals indicate zero *observed* sample variance, not population equivalence.
- Time reflects measured selection/provisioning (not restoration or protection switching).
- Previously inspected test streams were never used for the hybrid selection.
- Do not modify selected checkpoints or thresholds after examining this evaluation.

## Admission results

| Algorithm | Admitted / offered | Weighted admission | Ordinary admission | Selection ms/request |
|---|---:|---:|---:|---:|
| greedy | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 33.46 |
| cspf | 2172/6000 | 41.015% [39.259%, 42.772%] | 36.200% | 41.48 |
| milp_v2 | 2172/6000 | 41.015% [39.259%, 42.772%] | 36.200% | 65.46 |
| random_feasible | 2165/6000 | 40.950% [39.207%, 42.693%] | 36.083% | 50.10 |
| random_override | 2165/6000 | 41.100% [39.366%, 42.835%] | 36.083% | 49.97 |
| capacity_override | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 45.59 |
| hybrid_gated | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 48.56 |
| hybrid_ungated | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 48.79 |
| ppo | 2153/6000 | 40.733% [38.966%, 42.501%] | 35.883% | 49.20 |
| bcq | 2153/6000 | 40.846% [39.089%, 42.603%] | 35.883% | 52.11 |
| cql | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 45.96 |
| awac | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 45.98 |
| iql | 2170/6000 | 40.966% [39.197%, 42.735%] | 36.167% | 45.92 |
| original_ddqn | 2146/6000 | 40.830% [39.092%, 42.568%] | 35.767% | 52.74 |
| discrete_sac | 2158/6000 | 40.924% [39.260%, 42.587%] | 35.967% | 51.41 |

## Paired changes vs greedy

| Algorithm | Weighted difference pp (95% CI) | Holm-adjusted p | Positive / negative / zero seeds |
|---|---:|---:|---:|
| cspf | +0.049 [-0.062, +0.160] | 1 | 1 / 1 / 58 |
| milp_v2 | +0.049 [-0.062, +0.160] | 1 | 1 / 1 / 58 |
| random_feasible | -0.016 [-0.518, +0.486] | 1 | 7 / 11 / 42 |
| random_override | +0.134 [-0.460, +0.727] | 1 | 6 / 16 / 38 |
| capacity_override | +0.000 [+0.000, +0.000] | 1 | 0 / 0 / 60 |
| hybrid_gated | +0.000 [+0.000, +0.000] | 1 | 0 / 0 / 60 |
| hybrid_ungated | +0.000 [+0.000, +0.000] | 1 | 0 / 0 / 60 |
| ppo | -0.233 [-0.460, -0.006] | 0.6214 | 3 / 16 / 41 |
| bcq | -0.120 [-0.693, +0.452] | 1 | 9 / 26 / 25 |
| cql | +0.000 [+0.000, +0.000] | 1 | 0 / 0 / 60 |
| awac | +0.000 [+0.000, +0.000] | 1 | 0 / 0 / 60 |
| iql | +0.000 [+0.000, +0.000] | 1 | 0 / 0 / 60 |
| original_ddqn | -0.136 [-0.661, +0.389] | 1 | 13 / 31 / 16 |
| discrete_sac | -0.043 [-0.530, +0.444] | 1 | 7 / 17 / 36 |

## Interpretation boundaries

- A hybrid improvement is demonstrated only if measured in its pre-specified condition against same-condition greedy, with uncertainty assessed.
- Enhanced-state and candidate-count ablations must not be conflated with changes to learning architecture.
- Primary 60-node, k=2 inference cannot establish results across railway topologies.
- No full MPLS-TP MSRP protocol, restoration-time, safety or railway deployment claim.
- Negative and null results are scientifically informative; do not select variants based on test metrics.
