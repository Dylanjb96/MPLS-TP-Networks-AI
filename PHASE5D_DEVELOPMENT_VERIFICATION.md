# Phase 5D development verification and preflight

**Scope:** additive `phase5d` package installed beside the user's existing Phase 3–5C code. No existing source file, trained model or evaluation result was modified. No new neural-network training was conducted.

## Uploaded source review

The uploaded `/mnt/data/MPLS_AI_Phase5D_Source.zip` contained config, Phase 3/4/5/5B/5C source and 157 existing tests but omitted the existing `railway_tp` and `enhancements` modules and historical Phase 2 output fixtures. The earlier trusted Phase 5 *full working project* was used **only as a compatibility fixture in the local development checkout** for running those tests and demonstrations; those older modules are **not bundled or substituted into the add-on**. The user should install the add-on into their already-working existing local project.

SHA-256 checks of the corresponding files in the uploaded archive against the supplied Phase 5C evaluation provenance: **12/12 matched**, including Phase 3 `model.py`, Phase 4 `cli.py` and `policies.py`, Phase 5 `agent.py`, `environment.py`, `experiment.py`, `statistics.py`, and Phase 5C `common.py`, `dataset.py`, `offline_agents.py`, `ppo.py`, `workflow.py`. This verifies the extension targets the same key source files used by the supplied evaluation records. The archived `railway_tp` files could not be compared against an uploaded current copy.

## Test results

- New Phase 5D regression/unit tests: **10 passed**.
- Full merged development checkout tests: **167 passed**, consisting of 157 preexisting and 10 new tests.
- 12-node baseline-only behaviour CLI smoke: **passed**.
- 12-node isolated counterfactual CLI smoke: **passed**.
- 12-node explicit-retrospective MILP tracing CLI smoke: **passed**.
- 60-node baseline-only behaviour micro-test: **passed**.
- 60-node counterfactual micro-test (20 requests): **passed**.
- 60-node, 100-request development counterfactual with bounded two positions, 20-request suffix: **passed**. Both sampled positions had zero observed difference. This is a software check, not a hybrid result.
- All-seven-model checkpoint-loading/forward-pass integration: **passed with synthetically initialized 12-node checkpoints**. Frozen original Phase 5/5B/5C `.pt` binaries were not included in the source ZIP; the actual final-model diagnostic must run on the user's computer.

## Historical MILP V2 diagnosis preflight

The 60-node full-horizon run of already-inspected seed **41027**, marked `RETROSPECTIVE_ALREADY_INSPECTED_NOT_FOR_TUNING`, reproduced the reported **greedy 41 admissions versus MILP V2 37 admissions**. Weighted ratios were 0.45608108 and 0.40202703, respectively. The first *candidate-choice* divergence occurred at zero-based request index **8**, request `R0009`: greedy selected feasible candidate 0, while MILP V2 selected feasible candidate 1. The first *admission outcome* divergence occurred at index **19**, request `R0020`: greedy admitted the request, whereas MILP V2 rejected it for modeled capacity infeasibility. There were **9** differing sequential choices across the 100 requests, and MILP V2 recorded **37 optimal single-request solves and 63 infeasible instances**, with no logged solver fallbacks in this replay.

These observations support a plausible resource-allocation trajectory explanation for the 4-admission difference. They **do not prove** that the first different decision individually caused all four downstream losses. A controlled isolated branch experiment would be needed for that causal attribution. The reproduction uses the matching Phase 3–5C source plus earlier Phase 5 `railway_tp` development fixture, so the user should rerun the trace against their actual local Phase 2 package.

## Scientific and practical boundaries

- Training data and previous held-out results remain untouched. Phase 5E proposed 61000–61059 test seeds are prohibited by diagnostic seed guards.
- Phase 5D's counterfactuals use known future development requests in **retrospective branch simulation only**; a Phase 5E hybrid cannot inspect such future requests when making a real decision.
- Same-state agreement is distinct from end-state digest equality; no equivalence or superiority is inferred from identical results.
- The surrogate is `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`. No compliant operational protection or railway safety inference is drawn.
- A full 100-request lookahead with eight branches per seed can be computationally expensive; the main recommended diagnostic uses four branch points and 50 requests of lookahead, with longer horizons as sensitivity analyses.
