# Phase 5D — Independent diagnostic file audit

**Status:** The three planned Phase 5D diagnostic workloads are represented. All 12 uploaded files were read directly. This is an independent consistency and interpretation audit of the recorded data; it is **not** a byte-for-byte replay of the underlying simulator.

## Scope and provenance

- Behaviour: 10 development-validation demand seeds 20000–20009 × 100 demands; 1,000 decisions per algorithm. 8 policies, including greedy and seven pretrained RL methods. 8,000 trajectory rows and 7,000 frozen greedy-state comparison rows.
- Counterfactual: 10 development seeds 21000–21009, four stratified two-feasible choice points per seed, up to 50 subsequent arrivals. 40 paired branches; known future used **only in retrospective replay**.
- MILP V2: historical already-inspected seed 41027, 100 arrivals; not a fresh held-out or a candidate for hybrid tuning.
- All source/config metadata correspond to the 60-node, dual-gateway, topology-seed-42, 10%-headroom, two-candidate experiment. The protection model is `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`, not an RFC 8227 MSRP implementation; protection approval remains false.
- All five Phase 5C frozen-model SHA-256 strings in the behaviour manifest match the earlier `selection_manifest.json`. Actual checkpoint bytes were not provided to this audit.

## Frozen-model behaviour on identical greedy states

| Policy | Agreement in 347 two-feasible states | Alternative route choices | Candidate-1 proportion | Per-seed final digest matches greedy |
|---|---:|---:|---:|---:|
| ppo | 280/347 (80.69%) | 67 | 19.31% | 0/10 |
| bcq | 208/347 (59.94%) | 139 | 40.06% | 0/10 |
| cql | 347/347 (100.00%) | 0 | 0.00% | 10/10 |
| awac | 347/347 (100.00%) | 0 | 0.00% | 10/10 |
| iql | 347/347 (100.00%) | 0 | 0.00% | 10/10 |
| original_ddqn | 149/347 (42.94%) | 198 | 57.06% | 0/10 |
| discrete_sac | 170/347 (48.99%) | 177 | 51.01% | 0/10 |

**Decisive finding:** CQL, AWAC and IQL chose exactly the greedy candidate on every one of the 347 shared two-feasible states. On their separate complete trajectories, all 1,000 recorded actions, admission flags, candidate identifiers, before-state digests, and after-state digests also match greedy exactly. This establishes behavioural identity for the tested validation trajectories, **not** policy identity for every possible network state.

For the other models, choosing a different route often changes the subsequent reservation state. Their own-trajectory two-feasible sample sizes differ from the common greedy-state sample because routing decisions alter future feasibility.

## Retrospective counterfactual analysis — an essential limitation

- Branch points: 40; positive alternative future weighted reward: 0; negative: 0; ties: 40.
- All 40 alternative branches also tied greedy in suffix admitted-count; yet all 40 paired branch final-state digests differ.
- The branches therefore changed recorded reservation states without changing admitted services or the priority-weighted admission reward in these sampled suffixes. This indicates **zero observed supervised advantage signal in the currently sampled 40 examples**; it does *not* establish that better actions are impossible elsewhere.
- The 50-step horizon, only four stratified branches per seed, and the greedy continuation policy all limit the scope of the negative result. The chosen branch positions include several late requests whose remaining suffix is shorter than 50 arrivals.

**Action gate before residual RL:** expand the retrospective probe on the same development seeds to a full remaining 100-request suffix and more early/mid/late two-feasible branches. Do **not** train a supervised override predictor on 40 labels that are all zero.

Suggested Windows Command Prompt command (from project root; output folder must be new):

```cmd
python run_phase5d.py counterfactual --config config/primary_60_dual.json --out outputs\phase5d\counterfactual_dev_21000_fullsuffix20 --requests 100 --headroom 10 --max-candidates 2 --seed-start 21000 --seed-count 10 --max-branches 20 --horizon 100
```

If the broader experiment finds nonzero advantages, use **training-only** demand seeds (10000–10079) to construct hybrid training targets and reserve other development sequences for validation. If it remains near-zero, report the limited headroom candidly and consider a separately specified capacity/load sensitivity experiment instead of tuning on old held-out outcomes.

## Historical MILP V2 divergence: seed 41027

- Greedy admitted 41/100 (weighted 45.6081%). MILP V2 admitted 37/100 (weighted 40.2027%).
- First candidate difference: `R0009` (zero-indexed request 8), greedy candidate 0 versus MILP candidate 1; both requests admitted. First admission difference: `R0020` (zero-indexed 19), greedy admitted while MILP rejected.
- Nine total divergence rows: one earlier route-choice difference and eight later admission-outcome differences. Of the latter, MILP lost six admissions gained by greedy and gained two services greedy rejected: net −4.
- Priority units lost net = 16 out of 296 offered, corresponding to −5.4054 percentage points priority-weighted admission. These figures reconcile with the trace and summary.
- The MILP statuses are 37 `OPTIMAL_ONLINE_MINIMAX_LEXICOGRAPHIC_V2` and 63 `INFEASIBLE` for its requests. Its per-arrival minimax objective is not offline maximization of whole-horizon admissions.
- **Causal caution:** the trace identifies the first divergence and later consequences, but does not isolate whether changing only `R0009` would prevent every later admission loss.

## Independent consistency checks

**23 / 23 checks passed.**

- PASS — All trajectory keys are unique.
- PASS — All same-state keys are unique.
- PASS — All 80 trajectory (seed,policy) groups contain 100 records.
- PASS — All 70 same-state (seed,policy) groups contain 100 records.
- PASS — Greedy reference-state digests match trajectory file.
- PASS — No voluntary rejection in trajectory file.
- PASS — Two-feasible flag matches trajectory counts.
- PASS — No voluntary rejection in same-state file.
- PASS — Two-feasible flag matches same-state counts.
- PASS — Per-seed admission totals reconcile.
- PASS — Per-seed rewards reconcile.
- PASS — Per-seed final digests reconcile.
- PASS — All trajectory digests form sequential state chains.
- PASS — Both MILP/greedy traces form sequential state chains.
- PASS — MILP and greedy traces each contain 100 requests.
- PASS — Frozen checkpoint SHA matches Phase 5C selection: ppo.
- PASS — Frozen checkpoint SHA matches Phase 5C selection: bcq.
- PASS — Frozen checkpoint SHA matches Phase 5C selection: cql.
- PASS — Frozen checkpoint SHA matches Phase 5C selection: awac.
- PASS — Frozen checkpoint SHA matches Phase 5C selection: iql.
- PASS — Counterfactual rows total 40.
- PASS — Counterfactual reward differences reconcile.
- PASS — Counterfactual admission differences reconcile.

The checks establish file-to-file consistency, no recorded voluntary rejections in decision logs, checkpoint reference alignment, accounting, and sequential reservation-state digest continuity. They do **not** independently recalculate Phase 3 capacity feasibility from complete plan objects, verify the actual checkpoint bytes, or establish network safety.

## Decision for Phase 5E

1. **Finished:** the Phase 5D core diagnostic runs, and CQL/AWAC/IQL have now been shown to exactly imitate greedy on the observed development trajectories.
2. **Before investing in residual RL:** execute and audit the broader counterfactual sensitivity run. The present 40/40 ties provide no nonzero labels for learning an advantage predictor.
3. **Then specify Phase 5E:** compare frozen-model selective overrides, random overrides, simple capacity-based heuristics, and (only if training labels exhibit meaningful variation) a learned residual override model, with identical feasible masks and strictly no voluntary rejection.
4. **Freeze before testing:** no tuning on Phase 5, 5B, 5C held-out demand seeds; reserve proposed 61000–61059 as a new, uninspected Phase 5E final test. Scope and sample size need declaration before running it.

**Overall finding:** Phase 5D explains why several offline RL policies tied greedy, but it has not yet demonstrated a profitable greedy override. A hybrid is a testable hypothesis, not an assured improvement.
