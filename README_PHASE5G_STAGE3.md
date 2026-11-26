# Phase 5G Stage 3 — pre-heldout freeze package

**No final-evaluation command is included.** The Stage 3 script never generates topologies, traffic, or demand streams.

New files only:

- `run_phase5g_stage3.py` — read-only preflight, one-time freeze, and integrity verification.
- `stage3_reference/original___init__.py` and `stage3_reference/original_workflow.py` — read-only reference copies used to verify the **previously independently audited exact docstring-only Stage 2 source exception**. Do **not** place them over active project source files.
- `tests/test_phase5g_stage3.py` — 14 offline tests.
- `PHASE5G_STAGE3_PROTOCOL_LOCK.md` — eight-cell prespecified confirmatory design and topology-clustered inference.
- `PHASE5G_STAGE3_WINDOWS_GUIDE.md` — Windows start-to-finish instructions.

**Do not retrain** Stage 2 models, change/checkpoint-select after the final seeds are used, reuse Phase 5F Stage 5 seeds, or run Phase 5G final held-out data before implementing and reviewing Stage 4. Preserve your previous 13 MB full six-run archive; the new freeze is a curated snapshot, not a substitute for it.
