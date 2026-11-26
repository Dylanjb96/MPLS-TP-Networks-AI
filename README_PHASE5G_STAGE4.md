# Phase 5G Stage 4 — Development evaluator add-on

**Only install on top of the verified Phase 5G Stage 3 freeze.** The package adds files; it must not overwrite your models or earlier sources.

- `phase5g_stage4/protocol.py`: precise eight-cell frozen design, seed roles, comparisons and locked stress semantics.
- `phase5g_stage4/evaluation.py`: hash-verified model loading, all six policies, K2 feature padding, independent exact peak scenario oracle, atomic and resumable matched-cohort evidence, Stage 2 validation replay, audit-release gate.
- `phase5g_stage4/statistics.py`: mandatory complete 2,880-row paired grid, t-based topology-cluster intervals, two-test primary Holm family, eight-test secondary Holm family, exploratory stress comparisons.
- `run_phase5g_stage4.py`: `preflight`, `smoke`, `validation-replay`, `scale-dev`, and **audit-release gated** `final`/`report-final`.
- `tests/test_phase5g_stage4.py`: 22 development-only tests; no held-out graph generation.
- `PHASE5G_STAGE4_START_TO_FINISH_WINDOWS.md`: complete Windows runbook.
- `PHASE5G_STAGE4_ENGINEERING_AUDIT.md`: source and implementation verification, explicit untested gates.

**Important:** The official Phase 5G final evaluation is **not authorised** by this development-only add-on. Do not create `stage4_AUDIT_RELEASE.json` yourself. Run all development tests, smoke, validation replay and 500-node scale cohort; upload their evidence for an independent implementation audit. Only after audit should a separate release be issued for the reserved held-out seeds.
