# Phase 5B — Independent audit of the new held-out comparison

**Audit date:** 19 November 2026  
**Inputs:** `phase5_runs(1).csv`, `phase5_paired_deltas(1).csv`, `phase5_provenance(1).json`, `condition_summary(3).csv`, `paired_summary(3).csv`, `phase5_statistical_report(1).md`.  
**Scope:** Independent CSV arithmetic, paired seed analysis, and provenance consistency. This is **not** a full replay of the simulation, full route-path validation, or independent inspection of model checkpoint bytes.

## 1. Experimental design and integrity

- **210 complete algorithm runs**: 7 policy labels × 30 held-out demand seeds (41000–41029), with 100 offered requests per policy–seed. No duplicate policy–seed pairs.
- All 7 policy labels are present for every seed, and within each seed all policies have the **same recorded request SHA-256** and configuration. The full request contents were not independently regenerated.
- The provenance lists exactly seeds 41000–41029, the 60-node dual-gateway topology with topology seed 42, 10% headroom, a maximum of two candidate arrangements, and the same Phase 3 surrogate.
- All admission/rejection totals reconcile for all 210 runs, including reported normal-capacity, post-failure-capacity and no-plan rejection categories. No out-of-range peak utilizations were found.
- There are **zero recorded MILP solver fallbacks and zero recorded non-optimal solver outcomes**. This is a check of logged outcomes, not an independent optimization proof.
- All 180 per-seed paired CSV rows reconcile against the raw runs. Independently calculated mean, sample standard deviation, two-sided 95% Student-t intervals and totals agree with both supplied summary files to their reported precision.
- Historical Phase 5 test seeds (31000–31029) are separate from the new held-out seeds. Checkpoint-selection manifest and provenance report validation-only model selection, but checkpoints were not uploaded for byte-level verification.

## 2. Held-out results

Each policy received 3,000 requests. Mean admission rate is the mean of the 30 per-seed admission rates. Weighted admission is a mean of per-seed weighted ratios. Times are arithmetic means of per-seed policy-selection time per request, **not** network failover latency.

| Policy | Admitted / 3,000 | Admission | Weighted admission | Selection ms/request |
|---|---:|---:|---:|---:|
| Greedy | 1,075 | 35.83% | 40.77% | 32.41 |
| CSPF-style | 1,075 | 35.83% | 40.77% | 40.37 |
| MILP V2 | 1,070 | 35.67% | 40.58% | 64.10 |
| Random feasible | 1,060 | 35.33% | 40.38% | 49.31 |
| Original DDQN | 1,039 | 34.63% | 39.78% | 51.93 |
| Tuned DDQN | 1,039 | 34.63% | 39.78% | 52.25 |
| Discrete SAC | 1,067 | 35.57% | 40.64% | 50.77 |

## 3. Paired effects with 95% confidence intervals

All effects below are **percentage-point** differences, computed on 30 whole demand seeds. Intervals are **unadjusted** two-sided Student-t intervals; simultaneous comparisons or multiplicity-adjusted claims would require further analysis.

| Contrast (first minus second) | Admission difference (95% CI) | Weighted admission difference (95% CI) |
|---|---:|---:|
| Discrete SAC − Greedy | **−0.267 [−0.435, −0.099]** | −0.127 [−0.367, +0.113] |
| Discrete SAC − Original DDQN | **+0.933 [+0.444, +1.423]** | **+0.864 [+0.207, +1.520]** |
| MILP V2 − Greedy | −0.167 [−0.445, +0.112] | −0.193 [−0.562, +0.176] |
| Original DDQN − Greedy | **−1.200 [−1.713, −0.687]** | **−0.991 [−1.613, −0.368]** |
| Random feasible − Greedy | **−0.500 [−0.877, −0.123]** | **−0.386 [−0.716, −0.055]** |

The principal Phase 5B result is that **Discrete SAC improves upon the original DDQN on this fresh held-out dataset**, both in ordinary admission (28 more requests) and mean priority-weighted admission (+0.864 pp). SAC **still admits eight fewer services than greedy**. SAC's weighted difference versus greedy has a 95% CI including zero: this is **not** evidence that SAC and greedy are equivalent, nor proof SAC is superior. On the predefined weighted endpoint, superiority over greedy was *not demonstrated*. The nominal unadjusted ordinary-admission CI for SAC versus greedy is entirely negative.

## 4. Identical DDQN checkpoints, not two independently improved models

The recorded SHA-256 digests of `original_ddqn` and `tuned_ddqn` checkpoints are identical: `83e3b892033f377fc184f4b5d42a84971ec3830d39b08a64a7bc81d73a188acc`. Across all 30 matched seeds, their admitted counts, weighted admission ratios, total rewards and final reservation-state digests are also identical. Their recorded runtime differs, consistent with measurement noise or runtime effects. Treat these as **two evaluation labels of the same checkpoint**, not independent model evidence. No DDQN tuning gain was demonstrated.

## 5. Follow-up worth investigating: MILP V2 versus greedy

In this new held-out set, MILP V2 admitted **1,070** services versus greedy's **1,075**. Differences arose in only **two** seeds:

| Seed | Greedy admitted | MILP V2 admitted | Gap |
|---|---:|---:|---:|
| 41026 | 34 | 33 | −1 |
| 41027 | 41 | 37 | **−4** |

These new results do **not** imply that the lexicographic tie-breaking correction failed: online min–max utilization and greedy may select different feasible paths with different downstream consequences. They do show that Phase 4's 45 matched conditions with equal admission should **not** be generalized into universal equality. A first-divergence audit of `decisions.csv` and `accepted_protection_plans.json` for both policies in seeds 41026 and 41027 would be useful before claiming why these gaps occurred. The seed-level 95% paired t interval for MILP V2 − greedy admission includes zero.

## 6. Timing interpretation

Discrete SAC averaged **50.77 ms** policy selection versus greedy **32.41 ms**, CSPF-style **40.37 ms**, and MILP V2 **64.10 ms**. SAC is approximately **56.7% slower than greedy** and **20.8% faster than MILP V2** in these logged selection measurements. These measures include more than neural-network inference and omit shared candidate generation from the policy timer. They are **not** physical MPLS-TP protection-switching times.

## 7. Research limitations and next actions

1. **Freeze the current Phase 5B models and report these new held-out results without retuning on seeds 41000–41029.** The historical Phase 5 test results on seeds 31000–31029 remain a separate prior experiment, not interchangeable observations from a prospectively unified trial.
2. If investigating MILP V2's new admission gaps, conduct a **diagnostic first-divergence audit** using stored request/decision/path data. Do not revise V2 using held-out results and then present a rerun on the same seeds as untouched confirmation.
3. Include three training initializations and the limited hyperparameter search in the dissertation. The main Phase 5B finding is improved performance of discrete SAC relative to DDQN, **not** superiority over greedy.
4. The fixed topology seed 42, finite two-candidate action space, forced-only rejection, and Phase 3 single-physical-link scenario surrogate constrain interpretation. The simulator is `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`; these results do not demonstrate RFC 8227 MSRP signaling, switching latency, simultaneous failures, railway safety or generalization across independent network topologies. Supervisor approval is recorded as pending.

**Audit conclusion:** Six files are internally consistent. Discrete SAC recovers much of DDQN's admission deficit while remaining slightly below greedy on unweighted admission; weighted superiority over greedy has not been demonstrated. No claim of conventional algorithm equivalence is supported.
