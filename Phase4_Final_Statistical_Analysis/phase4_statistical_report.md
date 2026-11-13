# Phase 4 — Final seed-level statistical analysis

## Scope

- Input files scanned: **11**.
- Original rows: **174**; unique matched policy runs: **141**; duplicate exports removed: **33**.
- Unique paired seed/condition groups: **47**.
- All three algorithms were present for every retained condition, and their request hashes matched within each paired seed.
- No repeated-export timings are treated as independent observations.
- Main inference unit: **demand seed / whole sequential request stream**, not the 25–100 individual requests.
- Topology seed is fixed at 42 in the illustrated benchmark: these CIs do **not** include uncertainty across different network topologies.

## Primary tables

### 60-node offered-load sensitivity, 10% headroom

| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |
|---|---|---:|---|---:|---:|
| 25 requests | greedy | 5 | 74.40% [61.93%, 86.87%] | 80.36% | 39.62 |
| 25 requests | cspf | 5 | 74.40% [61.93%, 86.87%] | 80.36% | 48.98 |
| 25 requests | milp | 5 | 74.40% [61.93%, 86.87%] | 80.36% | 56.29 |
| 50 requests | greedy | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 55.54 |
| 50 requests | cspf | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 70.90 |
| 50 requests | milp | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 81.75 |
| 100 requests | greedy | 15 | 35.53% [31.95%, 39.12%] | 39.64% | 54.08 |
| 100 requests | cspf | 15 | 35.53% [31.95%, 39.12%] | 39.64% | 65.22 |
| 100 requests | milp | 15 | 35.53% [31.95%, 39.12%] | 39.64% | 84.73 |

### 60-node headroom sensitivity, 50 requests

| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |
|---|---|---:|---|---:|---:|
| 0% headroom | greedy | 5 | 57.60% [50.94%, 64.26%] | 64.28% | 46.83 |
| 0% headroom | cspf | 5 | 57.60% [50.94%, 64.26%] | 64.28% | 57.85 |
| 0% headroom | milp | 5 | 57.60% [50.94%, 64.26%] | 64.28% | 77.77 |
| 10% headroom | greedy | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 55.54 |
| 10% headroom | cspf | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 70.90 |
| 10% headroom | milp | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 81.75 |
| 20% headroom | greedy | 5 | 52.80% [45.65%, 59.95%] | 60.17% | 46.61 |
| 20% headroom | cspf | 5 | 52.80% [45.65%, 59.95%] | 60.17% | 67.35 |
| 20% headroom | milp | 5 | 52.40% [44.67%, 60.13%] | 59.72% | 76.00 |

### Network-size sensitivity, 50 requests, 10% headroom

| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |
|---|---|---:|---|---:|---:|
| 60 nodes | greedy | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 55.54 |
| 60 nodes | cspf | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 70.90 |
| 60 nodes | milp | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 81.75 |
| 100 nodes | greedy | 5 | 63.20% [48.34%, 78.06%] | 68.95% | 136.01 |
| 100 nodes | cspf | 5 | 63.20% [48.34%, 78.06%] | 68.95% | 195.48 |
| 100 nodes | milp | 5 | 63.20% [48.34%, 78.06%] | 68.95% | 231.69 |
| 150 nodes | greedy | 5 | 66.80% [50.09%, 83.51%] | 73.39% | 216.72 |
| 150 nodes | cspf | 5 | 66.80% [50.09%, 83.51%] | 73.39% | 310.47 |
| 150 nodes | milp | 5 | 66.40% [48.98%, 83.82%] | 72.93% | 354.76 |

## Paired comparisons: algorithm minus greedy

| Network | Requests | Headroom | Policy | Seeds | Admission difference (95% t-CI), pp | Positive / negative seeds |
|---:|---:|---:|---|---:|---|---:|
| 60 | 50 | 0% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 0% | milp | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 25 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 25 | 10% | milp | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 10% | milp | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 100 | 10% | cspf | 15 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 100 | 10% | milp | 15 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 20% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 20% | milp | 5 | -0.40 [-1.51, +0.71] | 0 / 1 |
| 100 | 50 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 100 | 50 | 10% | milp | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 150 | 50 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 150 | 50 | 10% | milp | 5 | -0.40 [-1.51, +0.71] | 0 / 1 |

## Interpretation rules and limitations

1. **Paired comparisons are essential.** Compare each CSPF-style/MILP run to greedy using exactly the same topology, headroom, request count, demand seed and request SHA-256.
2. **Zero observed paired differences are not proof of statistical equivalence.** If every observed difference is zero, the generated t interval is [0,0] mechanically; this is a descriptive property of the sample and not a valid equivalence test.
3. **For fewer than two seeds no confidence interval is reported.** The one-seed 12-node and early 100-node development results are smoke tests, not inferential evidence.
4. **Small n / sparse differences** make t intervals approximate. Do not interpret wide intervals as confirming superiority or infer equivalence from a nonsignificant difference.
5. **Only one topology seed** was used per network size. The comparisons describe a fixed synthetic configuration and do not generalize over all possible physical architectures.
6. **Fixed requests across network sizes** do not represent equal load relative to network capacity. Interpret network-size trends as sensitivity under fixed offered-request counts, not isolated causal effects of scale.
7. **Runtime** reflects this implementation, computer and workload. It is selection time, not deployment switching time or end-to-end protection recovery time.
8. **Request prefix validation** between 25-, 50- and 100-request experiments requires the saved per-run requests.csv files. The aggregate phase4_runs.csv hashes alone cannot prove prefix identity.
9. **Decision-divergence case studies** (60 nodes, headroom 20%, seed 2028; 150 nodes, headroom 10%, seed 2029) require per-request decisions.csv and accepted_protection_plans.json to identify the first divergent routing choice.
10. **Protection scope** remains RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP: no true MSRP protocol signalling, node failure, simultaneous failures or restoration timing.

## Next actions before Phase 5

Archive the source code, configurations, all baseline run outputs, test logs and this analysis. Ask the supervisor to approve the protection-model scope and two-candidate action-space limitation. Define separate DDQN training, validation and held-out test seeds. On the held-out test seeds, re-run all classical baselines alongside the final locked DDQN policy with exactly the same requests and feasibility validator.
