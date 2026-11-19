# Phase 5C — Development Verification Evidence

**Version**: `PHASE5C_DISCRETE_RL_V1`  
**Date**: 19 November 2026  
**Baseline source**: earlier matching integrated Phase 5 project plus Phase 5B add-on; preserves Phase 3 and MILP V2.

## Software tests

- New Phase 5C pytest module: `tests/test_phase5c.py` — **13 passed**.
- Whole extracted Phase 2–5B + Phase 5C suite: **157 passed in 17.12 seconds** (one development run).
- New tests cover masked distributions, illegal-action exclusion, all four offline agents' gradient updates and checkpoint round-trips, BCQ supported-action masking, forbidden data splits, dataset hash tamper detection, PPO training/checkpoints, offline training/checkpoints, model-selection protocol, held-out seed block and AWAC offline-to-online finetuning.

## End-to-end development smoke: 12 nodes

- Synthetic 12-node dual-gateway network; tiny 4-request episodes; 2 training and 2 validation seeds; 3 artificial agent initializations per algorithm.
- Logged dataset constructed with greedy and random-feasible trajectories.
- **15 training runs** exercised PPO, BCQ, CQL, AWAC and IQL (three seeds each).
- Validation-only selection produced a manifest with one checkpoint per algorithm, including hashes.
- **18 algorithm-run evaluations** completed (2 smoke demand seeds × 9 algorithms), comprising greedy, CSPF-style, MILP V2, random-feasible and the five new RL methods.
- Statistics and figures completed: `phase5_runs.csv`, paired comparisons, seed-level statistical report and three PNG charts.
- **Status:** full execution and output wiring confirmed.

## 60-node compatibility micro-test

- 60-node dual-gateway topology, topology seed 42, one 6-request evaluation stream.
- Built a tiny offline dataset from two training seeds; trained PPO and BCQ on training-only streams; validated and loaded checkpoints.
- Evaluated **6 policies**: greedy, CSPF-style, MILP V2, random-feasible, PPO and BCQ.
- **Status:** integration and same topology path-generator/Phase3 commit mechanism worked for this micro-test.

## Delimitations of the checks

These are **software smoke tests only**: tiny work requests offer little congestion and are not a scientifically meaningful performance comparison. We did **not** run the planned 80-seed, 16,000-transition offline dataset or the prospective 60-node multi-initialization full training and new 30-seed held-out evaluation. No claim is made that PPO, BCQ, CQL, AWAC, or IQL improves admissions or reproduces every detail of the original algorithm papers. The underlying protection model is `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`, not operational RFC 8227 MSRP.

## Reproduce

Install the Phase 5C add-on into the original project without overwriting existing phases, then execute:

```cmd
python -m pytest -q tests/test_phase5c.py
python -m pytest -q
```

Read `README_PHASE5C_START_TO_FINISH.md` for the full 60-node training, validation selection, held-out evaluation, statistical analysis and dissertation-report sequence. Document any differences in Python, NumPy, SciPy, PyTorch, CPU/GPU and operating system when reporting timings.
