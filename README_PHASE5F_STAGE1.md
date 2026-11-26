# Phase 5F A+B+C integrated synthetic national-scale study — Stage 1

This ZIP **only adds** a new `phase5f` package and accompanying tests/docs. It does **not** overwrite any earlier phases and does not yet train a reinforcement learning model.

**Run:** `python -m pytest -q tests/test_phase5f.py` followed by `python run_phase5f.py preflight --sizes 60 150 250 500`.

For commands and upload instructions read `PHASE5F_STAGE1_START_GUIDE.md`.

See `PHASE5F_RESEARCH_PROTOCOL.md` for the 2×2×2 factorial, held-out reservation, source-supported national-scale motivation, changes from previous protection model and statistical plan.

**Critical:** synthetic reference scale of approximately 500 nodes; not Irish Rail's confidential topology; scenario-based disjoint end-to-end paths; not MPLS-TP MSRP or safety validation.
