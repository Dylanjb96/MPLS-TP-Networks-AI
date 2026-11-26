# Phase 5B — New held-out comparison: tuned DDQN, discrete SAC, original DDQN, and baselines

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
| cspf | 1075/3000 | 35.83% [33.45%, 38.22%] | 40.77% | 40.37 |
| discrete_sac | 1067/3000 | 35.57% [33.21%, 37.93%] | 40.64% | 50.77 |
| greedy | 1075/3000 | 35.83% [33.45%, 38.22%] | 40.77% | 32.41 |
| milp_v2 | 1070/3000 | 35.67% [33.31%, 38.03%] | 40.58% | 64.10 |
| original_ddqn | 1039/3000 | 34.63% [32.32%, 36.95%] | 39.78% | 51.93 |
| random_feasible | 1060/3000 | 35.33% [32.90%, 37.76%] | 40.38% | 49.31 |
| tuned_ddqn | 1039/3000 | 34.63% [32.32%, 36.95%] | 39.78% | 52.25 |

## Paired differences (algorithm minus greedy)

| Algorithm | Admission difference (95% CI), percentage points | Positive / Negative / Zero seeds |
|---|---:|---:|
| cspf | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| discrete_sac | -0.27 [-0.43, -0.10] | 0 / 8 / 22 |
| milp_v2 | -0.17 [-0.45, +0.11] | 0 / 2 / 28 |
| original_ddqn | -1.20 [-1.71, -0.69] | 0 / 19 / 11 |
| random_feasible | -0.50 [-0.88, -0.12] | 0 / 8 / 22 |
| tuned_ddqn | -1.20 [-1.71, -0.69] | 0 / 19 / 11 |

## Reporting checklist

1. Disclose checkpoint selection by **validation only**; do not retune after observing held-out results.
2. Report both weighted and unweighted admissions, not merely cumulative RL reward.
3. Report learning curves, three random training initializations if possible, and compute budgets.
4. Compare baseline versions **greedy / CSPF-style / MILP V2** from the same test seeds.
5. State that inference-only latency is not restoration/switching time.
6. Report finite two-candidate limitation, fixed topology seed, forced-only rejection, and no MSRP signaling.
7. Do not claim statistical equivalence from zero observed differences.

Phase5B is a post-Phase5 extension evaluated on NEW seeds; historical 31000..31029 excluded.
