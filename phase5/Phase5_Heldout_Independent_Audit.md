# Phase 5 — Independent audit of held-out DDQN evaluation

**Audit date:** 16 November 2026  
**Source files:** `phase5_runs.csv`, `phase5_paired_deltas.csv`, `phase5_provenance.json`, `condition_summary(2).csv`, `paired_summary(2).csv`, `phase5_statistical_report.md`.

## Executive finding

The held-out experiment completed **150 policy runs**: 30 matched demand seeds (31000–31029), 100 service requests per seed, and five policies. **DDQN did not outperform the conventional baselines** in this evaluation. It admitted **1,020/3,000 (34.00%)** services versus **1,051/3,000 (35.03%)** for greedy, CSPF-style and MILP V2. The paired deficit is **−1.03 percentage points**, with a two-sided 95% Student-t confidence interval **[−1.55, −0.52]**. DDQN's **mean priority-weighted admission** was **39.22%**, versus **40.03%** for the three conventional baselines; the paired difference was **−0.81 percentage points**, with a 95% interval **[−1.43, −0.19]**. These intervals describe variation across the 30 synthetic demand seeds on the *fixed* 60-node topology, not uncertainty across different railway networks.

The selected checkpoint is recorded as agent seed **7**, training episode **20**, chosen by validation only. The held-out checkpoint hash is `83e3b892033f377fc184f4b5d42a84971ec3830d39b08a64a7bc81d73a188acc`. The actual `.pt` binary was **not supplied for independent hashing**.

## Held-out metrics

| Policy | Admitted / 3,000 | Mean admission | Mean weighted admission | Mean policy selection (ms/request) | Final-state digests matching greedy, of 30 |
|---|---:|---:|---:|---:|---:|
| Greedy | 1,051 | 35.03% | 40.03% | 51.27 | 30 |
| CSPF-style | 1,051 | 35.03% | 40.03% | 63.68 | 7 |
| MILP V2 | 1,051 | 35.03% | 40.03% | 98.33 | 5 |
| Random feasible | 1,034 | 34.47% | 39.49% | 76.22 | 0 |
| **DDQN** | **1,020** | **34.00%** | **39.22%** | **81.83** | **0** |

All rates are seed-level means across 30 seeds. Because each seed offers 100 requests, mean ordinary admission also equals admitted divided by 3,000. Weighted admission is the arithmetic mean of each seed's weighted-admission fraction, rather than a pooled-weight fraction.

## Paired contrasts against greedy

| Policy | Ordinary admission difference (percentage points), 95% CI | Weighted admission difference (percentage points), 95% CI | Better / worse / tied demand seeds |
|---|---|---|---|
| CSPF-style | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 / 0 / 30 |
| MILP V2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 / 0 / 30 |
| Random feasible | −0.57 [−1.10, −0.03] | −0.54 [−1.10, +0.03] | 1 / 10 / 19 |
| **DDQN** | **−1.03 [−1.55, −0.52]** | **−0.81 [−1.43, −0.19]** | **2 / 16 / 12** |

DDQN admitted 31 fewer services than greedy in the aggregate. Its ordinary admission deficit and weighted admission deficit both have nominal seed-level 95% confidence intervals excluding zero. This does **not** demonstrate superiority of greedy in general, nor does it adjust for multiple comparisons. The all-zero observed paired differences for CSPF-style and MILP V2 **do not prove algorithmic or statistical equivalence**.

DDQN's ordinary admission was **0.47 percentage points** below random feasible on the same 30 seeds, with a *post-hoc* paired 95% confidence interval approximately **[−0.96, +0.03]** percentage points; this interval includes zero. Its weighted admission was also numerically lower than random feasible, but no reliable advantage or deficit relative to random is established by these two summaries.

## Computational interpretation

The mean recorded DDQN selection time was **81.83 ms/request** versus **51.27 ms/request** for greedy (**+30.56 ms**, about **60% more**). DDQN remained faster than the MILP V2 benchmark's **98.33 ms/request**. DDQN's mean **inference-only** figure was approximately **0.362 ms/request**; the much larger full selection time includes feature construction, feasibility/masking work and other policy-selection overhead. Shared candidate generation is excluded from policy decision timing. These are implementation-specific times, **not MPLS-TP protection restoration or signaling times**.

## Integrity checks independently recalculated from run records

- Exactly **150 run rows**: five named policies × 30 unique demand seeds, without duplicate `(policy, demand_seed)` keys. Paired file has **120 rows**: four policies × 30 seeds against greedy.
- For each seed, all five run rows have the **same request SHA-256**, and it agrees with the reported provenance hash for that seed. The 30 seeds match **31000–31029**.
- Network configuration agrees across all rows: **60 nodes**, topology seed **42**, dual gateways as recorded in provenance, **10%** capacity headroom, **2** maximum policy candidates and **100** requests per seed.
- Every run reconciles `admitted + rejected = requests` and `normal_capacity_rejections + postfault_capacity_rejections + no_plan_rejections = rejected`.
- Every individual admission fraction matches its admitted count. The recorded `selection_seconds` matches `avg_selection_ms` to within the expected five-millionths-of-a-millisecond rounding tolerance.
- Every record reports **zero** MILP solver fallback and non-optimal counts. For all 30 DDQN records, `forced_rejections == rejected`; there were no logged voluntary rejection actions.
- All **120** paired differences reproduce the corresponding differences from the run records, for admitted counts, ordinary admission, weighted admission and policy-selection time.
- All reported condition means, sample standard deviations, total admitted/offered counts and 95% Student-t confidence intervals reproduce from the **150** run records; all paired means, sample SDs, confidence intervals and positive/negative/tied-seed counts likewise reproduce.
- Final-state digest equality against greedy: CSPF-style **7/30**, MILP V2 **5/30**, random feasible **0/30**, DDQN **0/30**. Identical admission counts do not imply identical route selection or reservation state.

**Outcome:** No substantive numerical inconsistency was found in the supplied aggregate benchmark exports. Reported duration fields have small decimal rounding differences only.

## Interpretation and limitations

This is a negative learning result: the validation-selected DDQN policy did not improve protected admission and was measurably worse than greedy under the evaluated synthetic workload. The fact that CSPF-style and MILP V2 matched greedy's admissions while often taking different final routes supports the narrower conclusion that admission is insensitive to some decisions, not that the algorithms are equivalent. More complex optimization or learning does not automatically yield higher sequential admission in the available two-candidate space.

The experiment uses **one fixed topology seed**, a **restricted two-candidate action space**, forced-only rejection, and a synthetic single-physical-link failure protection approximation. The recorded model is `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`. It is **not** an RFC 8227 shared-ring protection implementation and cannot establish live railway resilience, signaling operation, switching time or safety certification. Supervisor approval remains pending.

This audit **did not** independently execute the simulator, reconstruct the accepted protection paths, inspect individual `decisions.csv` or `accepted_protection_plans.json` outputs, inspect neural-network weights, or independently re-hash the `.pt` checkpoint. The provenance's no-lookahead and test-seed-isolation flags were recorded, not proved by source-code review in this audit.

## Recommended next steps

1. **Freeze** agent seed 7 / episode 20, the code and configuration hashes, all 150 held-out run results and the held-out seed range. Do not train, select or tune against seeds 31000–31029 after observing these outcomes.
2. Generate the already-supported **learning curve**, **held-out admission** and **selection time** figures without changing the experiment.
3. For explanatory *post-hoc* analysis, inspect `decisions.csv` and accepted protection plan artifacts from a small subset of seeds (for example, 31006 and 31012 where DDQN lost four admissions). Mark this as descriptive diagnosis rather than model selection; preserve exact request/route provenance.
4. Write the Phase 5 results faithfully, including all three training-initialization validation scores, the selection criterion, the worse held-out DDQN outcome, random-feasible control and the restricted protection abstraction. Preserve reviewer/supervisor approval as pending.
5. If proposing a **new** DDQN algorithm after this result, label it a new version and choose a **new, untouched** final test seed family. Do not reuse the examined 31000–31029 seeds as an independent test after tuning.
