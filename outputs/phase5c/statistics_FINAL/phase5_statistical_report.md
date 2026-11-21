# Phase 5C — New held-out on-policy and offline deep RL comparison

- Matched held-out demand sequences: **30**.
- Main inferential unit: **whole demand seed**, not individual requests.
- Comparisons: identical requests (SHA-256 verified) and topology configuration per seed.
- Confidence intervals: two-sided **95% Student t** intervals over seed-level results.
- This does **not** include uncertainty across topology seeds or railway infrastructure.
- Zero-width intervals from zero sample variance **do not establish equivalence**.
- Timings include policy selection; DDQN additionally includes feature construction / masking.
- Underlying protection model is the Phase-3 single-link scenario surrogate, **not MSRP**.

## Held-out admission

| Policy | Admitted / Offered | Mean admission (95% CI) | Weighted admission | Selection ms/request |
|---|---:|---:|---:|---:|
| awac | 1102/3000 | 36.73% [33.96%, 39.51%] | 41.83% | 45.58 |
| bcq | 1089/3000 | 36.30% [33.58%, 39.02%] | 41.49% | 51.68 |
| cql | 1102/3000 | 36.73% [33.96%, 39.51%] | 41.83% | 45.59 |
| cspf | 1102/3000 | 36.73% [33.96%, 39.51%] | 41.83% | 41.00 |
| discrete_sac | 1092/3000 | 36.40% [33.65%, 39.15%] | 41.58% | 50.94 |
| greedy | 1102/3000 | 36.73% [33.96%, 39.51%] | 41.83% | 32.64 |
| iql | 1102/3000 | 36.73% [33.96%, 39.51%] | 41.83% | 45.50 |
| milp_v2 | 1102/3000 | 36.73% [33.96%, 39.51%] | 41.83% | 64.99 |
| original_ddqn | 1075/3000 | 35.83% [33.15%, 38.52%] | 41.07% | 52.32 |
| ppo | 1095/3000 | 36.50% [33.73%, 39.27%] | 41.63% | 48.85 |
| random_feasible | 1100/3000 | 36.67% [33.82%, 39.52%] | 41.86% | 49.78 |

## Paired differences (algorithm minus greedy)

| Algorithm | Admission difference (95% CI), percentage points | Positive / Negative / Zero seeds |
|---|---:|---:|
| awac | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| bcq | -0.43 [-0.78, -0.08] | 3 / 13 / 14 |
| cql | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| cspf | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| discrete_sac | -0.33 [-0.56, -0.11] | 1 / 10 / 19 |
| iql | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| milp_v2 | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| original_ddqn | -0.90 [-1.26, -0.54] | 0 / 18 / 12 |
| ppo | -0.23 [-0.42, -0.05] | 1 / 8 / 21 |
| random_feasible | -0.07 [-0.42, +0.29] | 3 / 8 / 19 |

## Reporting checklist

1. Disclose checkpoint selection by **validation only**; do not retune after observing held-out results.
2. Report both weighted and unweighted admissions, not merely cumulative RL reward.
3. Report learning curves, three random training initializations if possible, and compute budgets.
4. Compare baseline versions **greedy / CSPF-style / MILP V2** from the same test seeds.
5. State that inference-only latency is not restoration/switching time.
6. Report finite two-candidate limitation, fixed topology seed, forced-only rejection, and no MSRP signaling.
7. Do not claim statistical equivalence from zero observed differences.

Phase5C uses NEW held-out demand seeds; previous Phase5/5B test seeds excluded.
Offline methods use a fixed training-only transition dataset; masked PPO collects on-policy simulation experience.
