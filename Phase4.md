# Phase 4 — Conventional protected-LSP provisioning comparison

**Project:** Resilience-Aware LSP Provisioning in Interconnected Railway MPLS-TP Networks: A Comparative Evaluation of Conventional Optimization and Deep Reinforcement Learning.

**Model carried forward unchanged:** `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`.

## Scientific scope and key limitation

Phase 4 **does not change Phase 3 physical topology, ring-segment plan generator or independent admission validator**. It compares online policies that choose one plan for each incoming service request; already accepted services cannot be moved, and future requests are unknown. All policies use the same synthetic request streams, candidate ordering, headroom, bidirectional capacity semantics and every-single-physical-link-failure feasibility check. This is a **simulation-only, deterministic ring-segment protection surrogate**, not an RFC 8227 MSRP implementation, safety assurance, recovery-speed model, ring-tunnel reservation implementation or actual railway NMS model.

For the measured primary 60-node topology and five Phase 3 demand seeds, **each of 250 requests produced exactly TWO policy-valid protected candidates**, so increasing `max_candidates` beyond 2 has no effect unless the model is changed. That limits scope for conventional algorithms and DDQN; a negative comparison can be scientifically useful.

## Install without overwriting your Phase 3 project

1. Make a backup of the working `MPLS-AI` directory (97 tests passed before installing Phase 4).
2. Extract the **contents** of the Phase 4 add-on ZIP directly into `MPLS-AI`, so `MPLS-AI/phase4/cli.py` exists. Do **not** extract a nested project directory.
3. From Windows Command Prompt opened in your project folder:

```cmd
python -m pip install -r requirements_phase4.txt
python -m pytest -q tests/test_phase4.py
python -m pytest -q
```

The matching packaged Phase 3 baseline with the 30 boundary tests plus Phase 4 tests passed **122 tests** in the development environment (97 existing + 25 new), but your own local result is the authoritative installation check. If you use a virtual environment, activate it before running commands.

## Three classical policies (precisely defined)

| Policy | Definition | Can it look ahead? | Uses Phase 3 validator? |
|---|---|---|---|
| `greedy` | Pick the **first robust-feasible** plan in Phase-3 default shortest-working-path ordering. Exactly reproduces Phase-3 first-feasible selection. | No | Yes |
| `cspf` | **CSPF-style candidate-constrained** policy: among robust-feasible Phase-3 plans, prefer fewest normal working hops, then least projected **normal-state** peak directed utilization. It is not unrestricted graph CSPF. | No | Yes |
| `milp` | One-request **robust online min–max MILP**: select one of the same enumerated plans minimizing worst utilization over normal + all one-link-failure states, respecting headroom and existing reservations. Uses HiGHS through `scipy.optimize.milp`. | No | Yes, again after solver |

**No claims of a global admission optimum.** The online MILP optimizes route choice only for the current service, conditional on already committed choices. It cannot optimize future admission because future demand is not shown. With two candidates, exhaustive enumeration would be a perfectly adequate exact alternative for this *single-step* problem; MILP is retained as a transparent optimization baseline for Phase 5, not as proof that MILP is necessary.

### Mathematical model for MILP

Let `S={normal} ∪ {each physical-link failure}`. Let `A` denote directed physical arcs (both directions of every undirected physical link), and `J` the fixed Phase-3 candidate set for the current request `i`.

- `L(s,a)`: already committed Mb/s on arc `a` in scenario `s`.
- `D(j,s,a)`: current request's additional Mb/s on arc `a` if candidate `j` is chosen in `s` (symmetrical forward/reverse assumption).
- `C(a)`: usable capacity of directed arc `a` after configurable headroom.
- `x_j ∈ {0,1}`: choose candidate `j`.
- `z ∈ [0,1]`: upper bound on worst directed-arc utilization over all scenarios.

```
minimize            z + ε × normalized_working_hops(x)       (ε = 1e-7 tie-breaker)
subject to          ∑_{j∈J} x_j = 1
                    L(s,a) + ∑_{j∈J} D(j,s,a) x_j ≤ C(a) × z   ∀ s∈S, a∈A
                    x_j ∈ {0,1},   0 ≤ z ≤ 1
```

A failed physical link's arc is never used by a structurally valid candidate in that failure state. An infeasible model means **no candidate** fits the given robust scenario budgets; no service is admitted. Existing services are not reoptimized. If the solver times out or returns a nonoptimal status, a greedy fallback is explicitly labelled, and output fields count fallbacks; never silently report such a result as a MILP optimum.

## Commands: start → pilot → complete experiment

### 1. Small debugging pilot (12 nodes, 3 algorithms)

```cmd
python -m phase4.cli --config config/debug_12_dual.json --requests 10 --seeds 2027 --headrooms 10 --out outputs\phase4\debug_12
```

### 2. Primary 60-node paired pilot (5 demand seeds, 50 requests each)

```cmd
python run_phase4_benchmark.py --config config/primary_60_dual.json --requests 50 --seeds 2027 2028 2029 2030 2031 --headrooms 10 --max-candidates 2 --out outputs\phase4\pilot_60
```

This performs **15 runs**, i.e. 5 paired demand sequences × 3 algorithms. Each algorithm receives its own **fresh** reservation state. Use a new `--out` path for each repetition, as existing output directories are deliberately not overwritten.

### 3. Headroom sensitivity (0%, 10%, 20%)

```cmd
python run_phase4_benchmark.py --config config/primary_60_dual.json --requests 50 --seeds 2027 2028 2029 2030 2031 --headrooms 0 10 20 --max-candidates 2 --out outputs\phase4\headroom_60
```

This runs 45 policy sequences. It provides within-seed comparisons; the small seed count is a **pilot**, not a basis for claims of statistical superiority. For main dissertation evaluation, plan at least 20 independent demand seeds per condition and include load variation and topology variation.

### 4. Optional larger-network smoke checks

```cmd
python run_phase4_benchmark.py --config config/scale_100_dual.json --requests 15 --seeds 2027 --headrooms 10 --out outputs\phase4\scale_100
```

Test 150 nodes only after the 100-node run; larger fault enumerations and MILP solve times increase workload. These are **size checks**, not isolated scalability measures unless demand intensity and topology differences are controlled.

## Output inventory (generated in `--out`)

- `phase4_runs.csv`: one row per policy × seed × headroom; accepted count, weighted admissions, decisions time, reasons, state and request hashes.
- `phase4_summary.csv`: **across-seed** mean and sample standard deviation for each policy/headroom.
- `phase4_paired_deltas.csv`: CSPF and MILP differences relative to greedy, *within the same seed/headroom*.
- `phase4_provenance.json`: frozen model, software versions, paired workload hashes, scope and fairness checklist.
- `runs/h<HEADROOM>_d<SEED>_<POLICY>/`: each run's requests, decisions, accepted preplanned paths, category analysis, summary and SHA-256 manifest.

The `selection_ms` metric measures policy selection and validation only, while `candidate_generation_ms` measures Phase-3 candidate construction separately. Times are environment-dependent and should not be treated as comparative scientific timing without warmups, repeat measurements and machine-specific disclosure.

## Pilot result included in supplied guide

Development environment, frozen 60-node topology seed 42, 10% headroom, 5 demand seeds 2027–2031, 50 requests each:

| Policy | Admission rate mean | Weighted admission mean | Mean decision time/request (development machine) |
|---|---:|---:|---:|
| Greedy | 52.4% | 58.0% | 24.4 ms |
| CSPF-style | 52.4% | 58.0% | 32.0 ms |
| MILP online | 52.4% | 58.0% | 46.4 ms |

The policy methods **sometimes selected different plans but admitted exactly the same count** on these five test sequences. A zero observed difference in five seeds does **not** prove equivalence and does not establish any algorithm as best. Rerun on your system; timings will vary.

## Integrity and acceptance gate

1. Confirm the 122-test local suite passes; no regressions against the prior 97.
2. Confirm greedy Phase-4 output exactly reproduces Phase-3 first-feasible admissions when topology, seed, request count, headroom and candidate limit are identical.
3. Confirm identical `request_sha256` for paired policies and equal candidate budget (`max_candidates=2`).
4. Confirm MILP `fallback_count=0` for runs used to claim online MILP results; otherwise separate/fix the failed runs.
5. Verify each accepted plan against Phase 3 `audit/admit` and file manifests (automatic in tests / solver commit checks).
6. Compare per-request choices and class/category outcomes, not just aggregate admissions.
7. Report confidence intervals and paired differences across independent demand seeds, not pseudo-replicated individual requests.
8. Freeze Phase-3 model and obtain supervisor approval of its limited surrogate protection semantics. Do not revise candidate-generation policy mid-comparison.
9. Decide whether the **two-action setting** supports the intended DDQN comparison or needs a separately versioned expanded model. If expanded, repeat all previous validation for every algorithm.

### Study stages

**Phase 4a:** algorithms and validation (provided here) → **4b:** five-seed matched pilot (provided) → **4c:** multi-headroom, offered-load and topology-sensitivity assessment → **4d:** final statistical report and freeze benchmark protocol → **Phase 5:** DDQN using exactly the same candidates, observation timing and independent checker.

## Standards context and documentation

- RFC 8227, MPLS-TP Shared-Ring Protection, is a materially richer ring-tunnel/RPS protocol than this surrogate: https://www.rfc-editor.org/info/rfc8227/
- SciPy documentation for the mixed-integer optimizer: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
- The results do not model packet loss, protection switching latency, OAM/RPS events, real railway NMS provisioning or physical equipment behaviour.