# Phase 5 — Held-out masked DDQN versus conventional baselines

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
| cspf | 1051/3000 | 35.03% [32.82%, 37.24%] | 40.03% | 63.68 |
| ddqn | 1020/3000 | 34.00% [31.76%, 36.24%] | 39.22% | 81.83 |
| greedy | 1051/3000 | 35.03% [32.82%, 37.24%] | 40.03% | 51.27 |
| milp_v2 | 1051/3000 | 35.03% [32.82%, 37.24%] | 40.03% | 98.33 |
| random_feasible | 1034/3000 | 34.47% [32.19%, 36.74%] | 39.49% | 76.22 |

## Paired differences (algorithm minus greedy)

| Algorithm | Admission difference (95% CI), percentage points | Positive / Negative / Zero seeds |
|---|---:|---:|
| cspf | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| ddqn | -1.03 [-1.55, -0.52] | 2 / 16 / 12 |
| milp_v2 | +0.00 [+0.00, +0.00] | 0 / 0 / 30 |
| random_feasible | -0.57 [-1.10, -0.03] | 1 / 10 / 19 |

## Reporting checklist

1. Disclose checkpoint selection by **validation only**; do not retune after observing held-out results.
2. Report both weighted and unweighted admissions, not merely cumulative RL reward.
3. Report learning curves, three random training initializations if possible, and compute budgets.
4. Compare baseline versions **greedy / CSPF-style / MILP V2** from the same test seeds.
5. State that inference-only latency is not restoration/switching time.
6. Report finite two-candidate limitation, fixed topology seed, forced-only rejection, and no MSRP signaling.
7. Do not claim statistical equivalence from zero observed differences.
