# Phase 5D — Expanded Counterfactual Audit (development-only)

**Evidence inspected:** `counterfactual_branches(1).csv`, `counterfactual_summary(1).json`, and `diagnostic_manifest(3).json`, compared with the earlier Phase 5D manifest and inspected Phase 5D add-on source. This is a consistency and methodological audit; it is **not** an independent simulator replay.

## Results

| Outcome of one alternative-route override followed by greedy continuation | Branch count | Share |
| --- | ---: | ---: |
| Alternative increases suffix priority-weighted reward | 0 | 0.0% |
| Alternative produces the same suffix reward | 197 | 98.5% |
| Alternative reduces suffix reward | 3 | 1.5% |
| **Total** | **200** | **100.0%** |

The mean alternative-minus-greedy *cumulative normalized reward* is **−0.005 per branch** (sum **−1.0**). This is **not an admission-rate difference in percentage points**. All three negative cases also lose one admission; the other 197 cases have no admission-count difference. There are no positive admission-count differences.

### Three adverse overrides

| Demand seed | Request | Zero-based index | Weighted reward difference | Admitted difference | Continuation steps |
| --- | --- | ---: | ---: | ---: | ---: |
| 21005 | R0002 | 1 | -0.6 | -1 | 99 |
| 21009 | R0009 | 8 | -0.2 | -1 | 92 |
| 21009 | R0016 | 15 | -0.2 | -1 | 85 |

All **200 alternative branches** have a different terminal reservation-state digest from their matched greedy branch, despite 197 matching admission/reward outcomes. Changing reservations alone is not evidence of better future admission.

## Experimental scope

- Ten development seeds **21000–21009**, 100 requests per seed; **20 deterministically spaced eligible decision points** per seed along each greedy trajectory, not a random sample of independent trials.
- Each decision point has exactly two feasible protected candidates on the greedy prefix state. Candidate 0 is selected for the baseline branch; candidate 1 for the alternative branch.
- The initial reservation state and subsequent request sequence are matched, with **first-feasible greedy decisions** for the rest of the original 100-request sequence (the continuation ends at the sequence boundary). The supplied CSV has **1–100 rollout steps** depending on the branch's position.
- The rollout looks at *realized future requests*, information **unavailable to a deployable policy**. These records cannot be used as online oracle features.
- The experiment tests one override followed by greedy, **not** a learned, repeatedly overriding hybrid; results neither bound every possible hybrid policy nor demonstrate impossibility of future improvements.
- The 200 branches are clustered in only ten request seeds and share historical prefixes. Do not treat them as 200 independent samples for confidence intervals or significance tests.
- Fixed topology seed 42, 60 nodes, dual gateway, headroom 10%, two candidates; source protection model is **`RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`**, with supervisor protection-model approval pending.

## Integrity checks

- PASS: Exactly 200 unique seed/request branch records
- PASS: 20 points in each of ten seeds
- PASS: No missing values
- PASS: Different candidate choices in each branch
- PASS: Identical starting state is recorded as one snapshot per branch
- PASS: 200 distinct branch-final state outcomes (compared within pairs)
- PASS: Weighted reward differences arithmetically reconcile
- PASS: Admission differences arithmetically reconcile
- PASS: Continuation covers the remaining requests
- PASS: Summary outcome counts and mean agree
- PASS: Core source hashes match previous Phase 5D counterfactual manifest
- PASS: Topology matches previous Phase 5D counterfactual manifest
- PASS: Role is development only; no model training

**Passed: 13/13 checks.** Source hashes and metadata are self-reported provenance; uploaded simulator output and logs were checked for consistency, but the code was not rerun against every request here.

## Research conclusion and Phase 5E gate

This expanded diagnostic provides **no positive one-step override training examples** in the tested greedy-prefix states. A residual model trained solely on these labels cannot learn to recognize positive overrides; a conservative policy should default to greedy when its predicted advantage is not sufficiently supported. Do not oversell a gain that has not been observed.

A defensible Phase 5E study can compare **(1) frozen greedy, (2) simple greedy+PPO/SAC/BCQ gates, (3) a non-learning override control, and (4) a learned residual policy**, without assuming that the hybrid wins. Predefine development-only threshold choices and freeze them before the fresh test seeds **61000–61059**, after confirming those seeds were unused.

If the scientific objective requires discovering positive override opportunities, investigate **training-only/development-only** sensitivity cases (diverse loads, horizons, requests or enlarged candidate sets) with a separate experiment definition, matched conventional baselines, and explicit disclosure that these are new experimental conditions—not direct improvements under the original two-candidate formulation. Do not modify or retune the already published Phase 5C held-out results.
