# Phase 5 — Independent development validation record

**Environment:** development container, Phase 4 V2 baseline overlaid on the archived Phase 4 full project; Python/PyTorch installed locally. **This record validates software wiring only, NOT research performance.**

## Automated tests

- Original Phases 2–4 code from the archived full working project, with the MILP V2 add-on installed.
- New `tests/test_phase5.py`: **7 passed**.
- Full suite including original Phase 4 V2 tests: **138 passed**.

## 12-node end-to-end smoke experiment

- Trained for 4 episodes of 10 requests (demand seeds 10000–10002), two validation seeds, replay started at 8 experiences.
- Confirmed **33 gradient updates**; `best_model.pt`, `last_model.pt`, `training_history.csv` and provenance emitted.
- Checkpoint automatically selected on validation (best episode 2).
- Two distinct *smoke-only* held-out streams, five policies each: 10 policy runs; policy outputs, SHA hashes, paired CSV and summary generated.
- All five methods happened to admit all 10 services in both 12-node smoke streams. **This is not a learning benefit and not suitable for final inference.**
- Generated three charts (learning, held-out admission and selection cost).

## 60-node end-to-end smoke experiment

- 4 training episodes × 25 requests; 2 validation streams; 2 new test streams.
- All five policies ran: greedy / CSPF-style / MILP V2 / random-feasible / DDQN.
- Test seed 30000: all policies admitted 22/25; test seed 30001: all policies admitted 13/25.
- This is an **undertrained integration pilot**, NOT a dissertation finding or evidence of algorithm equivalence.

## Boundaries and next steps

- Full train/validation/held-out Phase 5 experiments **have not been completed** by the development tests.
- No model-performance claims on the 100-request 30-seed held-out workload are made.
- Human supervisor approval of `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP` remains pending.
- The only acceptable final findings are those produced by the user's full executed held-out protocol.
