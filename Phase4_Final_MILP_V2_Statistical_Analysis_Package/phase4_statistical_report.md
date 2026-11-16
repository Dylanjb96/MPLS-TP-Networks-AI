# Phase 4 — Final MILP V2 seed-level statistical analysis

## Scope

- Input files scanned: **5**.
- Original rows: **135**; unique matched policy runs: **135**; duplicate exports removed: **0**.
- Unique paired seed/condition groups: **45**.
- All three algorithms were present for every retained condition, and their request hashes matched within each paired seed.
- No repeated-export timings are treated as independent observations.
- Primary algorithms: **greedy**, **CSPF-style**, **MILP V2**. Original MILP V1 is historical and excluded from the primary comparison.
- Main inference unit: **demand seed / whole sequential request stream**, not the 25–100 individual requests.
- 45 matched conditions across five experiment families; reuse of seeds across conditions means these are **not 45 independent network realizations**.
- Topology seed is fixed at 42 in the illustrated benchmark: these CIs do **not** include uncertainty across different network topologies.

## Executive findings

- **All observed admission differences:** 90/90 paired policy comparisons were zero; priority-weighted differences were also zero in 90/90 comparisons.
- **Heavy load (60 nodes, 100 requests, 10% headroom, 15 seeds):** 533/1,500 requests admitted per policy; mean admission 35.53% with 95% seed-level t-CI [31.95%, 39.12%].
- **Heavy-load mean selection time:** greedy 50.61 ms, CSPF-style 61.90 ms, MILP V2 103.42 ms per request.
- **Different final reservation-state digests:** CSPF differed from greedy in 16/45 groups; MILP V2 differed from greedy in 16/45 groups. Equal admission is not identical routing.
- **Interpretation:** MILP V2 does not show an admission advantage over greedy in the tested conditions; this does not prove general equivalence, nor does it establish behaviour of real railway/MPLS-TP networks.

## Primary tables

### 60-node offered-load sensitivity, 10% headroom

| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |
|---|---|---:|---|---:|---:|
| 25 requests | greedy | 5 | 74.40% [61.93%, 86.87%] | 80.36% | 36.44 |
| 25 requests | cspf | 5 | 74.40% [61.93%, 86.87%] | 80.36% | 52.18 |
| 25 requests | milp_v2 | 5 | 74.40% [61.93%, 86.87%] | 80.36% | 120.88 |
| 50 requests | greedy | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 44.38 |
| 50 requests | cspf | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 62.06 |
| 50 requests | milp_v2 | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 101.88 |
| 100 requests | greedy | 15 | 35.53% [31.95%, 39.12%] | 39.64% | 50.61 |
| 100 requests | cspf | 15 | 35.53% [31.95%, 39.12%] | 39.64% | 61.90 |
| 100 requests | milp_v2 | 15 | 35.53% [31.95%, 39.12%] | 39.64% | 103.42 |

### 60-node headroom sensitivity, 50 requests

| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |
|---|---|---:|---|---:|---:|
| 0% headroom | greedy | 5 | 57.60% [50.94%, 64.26%] | 64.28% | 50.31 |
| 0% headroom | cspf | 5 | 57.60% [50.94%, 64.26%] | 64.28% | 71.63 |
| 0% headroom | milp_v2 | 5 | 57.60% [50.94%, 64.26%] | 64.28% | 141.76 |
| 10% headroom | greedy | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 44.38 |
| 10% headroom | cspf | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 62.06 |
| 10% headroom | milp_v2 | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 101.88 |
| 20% headroom | greedy | 5 | 52.80% [45.65%, 59.95%] | 60.17% | 46.04 |
| 20% headroom | cspf | 5 | 52.80% [45.65%, 59.95%] | 60.17% | 58.69 |
| 20% headroom | milp_v2 | 5 | 52.80% [45.65%, 59.95%] | 60.17% | 101.15 |

### Network-size sensitivity, 50 requests, 10% headroom

| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |
|---|---|---:|---|---:|---:|
| 60 nodes | greedy | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 44.38 |
| 60 nodes | cspf | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 62.06 |
| 60 nodes | milp_v2 | 5 | 52.40% [43.04%, 61.76%] | 58.00% | 101.88 |
| 100 nodes | greedy | 5 | 63.20% [48.34%, 78.06%] | 68.95% | 103.54 |
| 100 nodes | cspf | 5 | 63.20% [48.34%, 78.06%] | 68.95% | 147.59 |
| 100 nodes | milp_v2 | 5 | 63.20% [48.34%, 78.06%] | 68.95% | 256.61 |
| 150 nodes | greedy | 5 | 66.80% [50.09%, 83.51%] | 73.39% | 212.69 |
| 150 nodes | cspf | 5 | 66.80% [50.09%, 83.51%] | 73.39% | 307.54 |
| 150 nodes | milp_v2 | 5 | 66.80% [50.09%, 83.51%] | 73.39% | 577.13 |

## Paired comparisons: algorithm minus greedy

| Network | Requests | Headroom | Policy | Seeds | Admission difference (95% t-CI), pp | Positive / negative seeds |
|---:|---:|---:|---|---:|---|---:|
| 60 | 50 | 0% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 0% | milp_v2 | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 25 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 25 | 10% | milp_v2 | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 10% | milp_v2 | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 100 | 10% | cspf | 15 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 100 | 10% | milp_v2 | 15 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 20% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 60 | 50 | 20% | milp_v2 | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 100 | 50 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 100 | 50 | 10% | milp_v2 | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 150 | 50 | 10% | cspf | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |
| 150 | 50 | 10% | milp_v2 | 5 | +0.00 [+0.00, +0.00] | 0 / 0 |

## Interpretation rules and limitations

1. **Paired comparisons are essential.** Compare each CSPF-style/MILP V2 run to greedy using exactly the same topology, headroom, request count, demand seed and request SHA-256.
2. **Zero observed paired differences are not proof of statistical equivalence.** If every observed difference is zero, the generated t interval is [0,0] mechanically; this is a descriptive property of the sample and not a valid equivalence test.
3. **For fewer than two seeds no confidence interval is reported.** The included primary comparisons contain five or fifteen demand seeds; earlier one-seed development runs are not part of this report.
4. **Small n / sparse differences** make t intervals approximate. Do not interpret wide intervals as confirming superiority or infer equivalence from a nonsignificant difference.
5. **Only one topology seed** was used per network size. The comparisons describe a fixed synthetic configuration and do not generalize over all possible physical architectures.
6. **Fixed requests across network sizes** do not represent equal load relative to network capacity. Interpret network-size trends as sensitivity under fixed offered-request counts, not isolated causal effects of scale.
7. **Runtime** reflects this implementation, computer and workload. It is selection time, not deployment switching time or end-to-end protection recovery time.
8. **Request prefix validation** between 25-, 50- and 100-request experiments was previously checked on the archived per-run requests.csv files for seeds 2027–2031. These V2 aggregate CSVs independently establish matching hashes only *within each paired condition*.
9. **Historical MILP V1** showed two one-request admission deficits in investigated scenarios, linked to earlier route choices and sensitive to tiny objective tie-breaking. Preserve V1 results as a separately versioned implementation-sensitivity study. Do not combine V1 and V2 inferential datasets.
10. **Protection scope** remains RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP: no true MSRP protocol signalling, node failure, simultaneous failures or restoration timing.
11. **No evidence of an admission gain** from MILP V2 in these conditions is not proof of general statistical equivalence. Identical pairs can coexist with different final reservation states.

## Next actions before Phase 5

Archive the source code, configurations, all baseline run outputs, test logs and this analysis. Ask the supervisor to approve the protection-model scope and two-candidate action-space limitation. Define separate DDQN training, validation and held-out test seeds. On the held-out test seeds, re-run all classical baselines alongside the final locked DDQN policy with exactly the same requests and feasibility validator.
