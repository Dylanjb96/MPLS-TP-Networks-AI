# Phase 5G Stage 1 — Runnable development add-on

This additive package contains:

- `phase5g/protocol.py`: strictly nonoverlapping new development/training/validation partitions; new confirmatory seeds reserved and inaccessible.
- `phase5g/topology.py`: a structurally distinct bridge-free corridor-ladder synthetic topology family alongside the existing hierarchical family.
- `phase5g/optimization.py`: exact one-arrival scenario-minimax optimization over feasible candidate paths, without future information.
- `phase5g/gnn.py`: *true edge-message-passing* GraphCandidateActorCritic with per-candidate working/backup edge pooling and infeasible-action masking, plus an on-policy clipped PPO update.
- `phase5g/workflow.py`: independent single-fault oracle-checked development pilots and a very short GNN smoke experiment.
- `run_phase5g.py`: `preflight`, `gnn-smoke`, `pilot`; deliberately no `final` command.
- `tests/test_phase5g.py`: 22 unit/integration tests including masked actions, graph-node permutation equivariance, online scenario optimization, topology diversity, oracle checks and forbidden-seed safeguards.

Development verification: new 22 tests passed; combined Phase 5F Stage 1–5 + Phase 5G local suite **99 tests passed**. Two-episode GNN smoke and two-family optimization pilots at 60 and 150 nodes executed successfully in the authoring environment. These are **not final experimental findings**, not full GNN training and not proof of a GNN advantage.

Review `PHASE5G_RESEARCH_PROTOCOL_DRAFT.md` before using the tool and follow `PHASE5G_STAGE1_START_GUIDE_WINDOWS.md` for exact commands.

**Limitations:** Synthetic research networks, single-link-failure end-to-end protection surrogate (not MSRP), no actual Irish Rail topology, no operational restoration-time or safety claims. The stronger optimization method is **one-step scenario-aware exact candidate enumeration, not a MILP or a full-sequence clairvoyant optimizer**. The optional DDQN comparator and full multi-seed GNN training/final evaluator are **subsequent deliverables**.
