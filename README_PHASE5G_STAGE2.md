# Phase 5G Stage 2 — Matched GNN/MLP neural training

**Install as an add-on on top of Phase 5G Stage 1 V1.1**. Do not replace any Phase 5F or Phase 5G Stage 1 code.

- `phase5g_stage2/workflow.py`: paired GNN versus near-parameter-matched MLP training, safe candidate cache, independent validation oracle, episode-level resumability, immutable best checkpoints, validation-only model selector.
- `run_phase5g_stage2.py`: CLI (preflight, quickcheck, train, status, select). **No held-out command.**
- `Phase5G_Stage2_6_Run_Windows.cmd`: complete 6-run batch at fixed 48 episodes per run.
- `tests/test_phase5g_stage2.py` : 16 tests.
- `PHASE5G_STAGE2_PROTOCOL_LOCK.md`: design and guardrails.
- `PHASE5G_STAGE2_START_TO_FINISH_WINDOWS_GUIDE.md`: exact CMD steps.
- `PHASE5G_STAGE2_DEVELOPMENT_VERIFICATION.md`: local verification and limitations.

Neural models are both pure PyTorch and use CPU for comparable deterministic execution. A graph network can have higher computational overhead even with matched parameter count.

Start with `python -m pytest -q tests/test_phase5g_stage2.py` and then `python run_phase5g_stage2.py preflight`.
