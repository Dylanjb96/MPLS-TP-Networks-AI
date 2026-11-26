# Phase 5G Stage 2 — Independent implementation verification

This document records *development* checks performed in the package-authoring environment. Full six-run training and Phase 5G confirmatory final evaluation **have not** been executed.

## Verified behaviours

- Corrected Phase 5G V1.1 was overlaid on the previously independently replayed source tree; existing frozen Phase 5F model data and source hashes were not changed.
- New Stage 2 preflight succeeded for two neural architectures; preserved Phase 5F frozen model SHA-256 `e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`.
- **GNN 15,778 parameters; MLP 15,758** (20-parameter / 0.127% difference), using exactly the same 17+8 causal state features for both with additional graph tensors for GNN.
- **16 new regression tests passed**, including held-out seed rejection, topology/parameter lock, mask checks, graph and MLP PPO update, bounded/disjoint action sampling, candidate-cache corruption detection, training completion, strict overwrite refusal, controlled simulated interruption/resumption, and selection-completeness rejection.
- **39 Phase 5G Stage 1+2 tests passed** with the corrected V1.1 source.
- GNN two-episode, 60-node development quickcheck completed; training journal, independent oracle-verified validation, best checkpoint and provenance saved.
- A separate **complete miniature six-run matrix** (2 architectures × 3 model seeds × 1 episode) was executed and the architecture selector successfully finished. This is exclusively a functional check using a trivial 60-node low-load validation cell; its scores have no substantive research interpretation.
- All generated development tests and the small selector used only permitted training/validation/development seed partitions. Phase 5G final reserved seeds and Phase 5F Stage 5 seeds were not opened.

## Required Windows checks before real six-run training

The author's environment used Python/PyTorch/NetworkX versions different from the user's Windows installation. The user must run `python -m pytest -q tests/test_phase5g_stage2.py`, `python -m pytest -q`, `python run_phase5g_stage2.py preflight` and the provided `quickcheck`. Expected new tests: **16**; expected full project (if unchanged): **302**.

## Research limitations and methodological guardrails

- Full GNN and MLP model training is not yet done. A positive or negative GNN conclusion is not warranted.
- GNN and MLP have approximately equal parameter counts, but different graph inputs and computational cost. This is a *matched neural-capacity and interaction-budget comparison*, not proof of an isolated graph-messaging causal effect under every other architecture choice.
- Training is fixed at 48 episodes/seed; as a finite exploratory budget, it need not ensure convergence. Count actual sample updates and inference costs in later analyses.
- The preplanned validation score uses only 6 hierarchical-topology cases per checkpoint, with correlated streams within each topology. These cases select checkpoints only; no statistical superiority is inferred from them.
- The later robustness/generalization evaluation on the corridor-ladder network requires its own prospectively locked protocol and fresh data; *none is implemented by this package.*
- Additional DDQN is intentionally deferred to an explicitly budgeted optional follow-on before any Phase 5G final test.
- IEEE/MPLS-TP MSRP compliance and actual railway safety/recovery behaviour are **not claimed**.
