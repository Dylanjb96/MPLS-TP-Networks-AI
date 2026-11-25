# Phase 5G Stage 1 — Development verification, not final evidence

Completed in the authoring environment, with the frozen Phase 5F Stage 5 checkpoint present and unmodified:

- `python run_phase5g.py preflight`: PASS for both hierarchical and corridor-ladder families, both with six feasible candidate pairs; frozen C checkpoint SHA-256 confirmed as `e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`.
- `python -m pytest -q tests/test_phase5g.py`: **22 passed**.
- `python -m pytest -q tests/test_phase5f.py tests/test_phase5f_stage2.py tests/test_phase5f_stage3.py tests/test_phase5f_stage4.py tests/test_phase5f_stage5.py tests/test_phase5g.py`: **99 passed**.
- `python run_phase5g.py gnn-smoke --out <development output> --episodes 2`: PASS, trained real PyTorch message-passing network and used independent single-fault trace verification on all four validation policies.
- `python run_phase5g.py pilot --out <development output> --sizes 60 --families hierarchical ladder --topology-seeds 501 --stream-seeds 74000 --load 1.6 ...`: PASS, 8 policy rows.
- `python run_phase5g.py pilot --out <development output> --sizes 150 --families hierarchical ladder --topology-seeds 501 --stream-seeds 74000 --load 2.5 ...`: PASS, 8 policy rows.

**Remaining limitations / checks:** The user's Windows environment must pass its own tests and preflight. No full-scale GNN training, no matched-budget MLP ablation, no DDQN implementation, no Phase 5G confirmatory evaluation and no GNN superiority claim. The new network families and operating shocks require scientific review and the protection abstraction requires supervisor review. Future final statistics and model selection must be locked independently from Stage 5F held-out evidence.
