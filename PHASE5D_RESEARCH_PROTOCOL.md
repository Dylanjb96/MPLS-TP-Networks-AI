# Phase 5D research protocol (diagnostic only)

**Version:** `PHASE5D_FROZEN_POLICY_DIAGNOSTICS_V1`.

## Objectives

1. Explain the observed Phase 5C equivalence of greedy, CQL, AWAC and IQL on all 30 held-out 51000–51029 streams. Since equal outcomes and final state digests do not prove identical action sequences, test *identical-state agreement* explicitly.
2. Distinguish on-policy trajectory differences from identical-state policy differences for PPO, Discrete SAC, BCQ, DDQN, CQL, AWAC and IQL.
3. Explain the previously observed Phase 5B seed-41027 MILP V2 admission loss as a **retrospective** descriptive trace.
4. Measure the occurrence and sign of **realized** counterfactual benefits from choosing a non-greedy candidate at a greedy-prefix two-feasible state; no claim of knowing expected future benefit or guaranteeing policy improvement.

## Frozen inputs

- Primary network: 60 nodes, topology seed 42, dual gateways, 10% headroom, at most two Phase-3 policy-valid candidate protected routes, 100 sequential requests per stream.
- Phase 3 `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP` is the independent feasibility authority; forced-only rejection is unchanged.
- Phase 5C selected checkpoint hashes from `selection_manifest.json`, plus the previously frozen DDQN/SAC binaries.
- Phase 5D code cannot train/checkpoint/update a network. Evaluated policies use deterministic `greedy=True` inference as in Phase 5C held-out evaluation.
- Development only: 10000–10079, 20000–20009 and prospective 21000–21029. Historical test 31000, 41000, 51000 ranges may be described retrospectively only, never used for new model selection.
- Proposed Phase 5E test 61000–61059 is explicitly forbidden in all Phase 5D modes.

## Tests and records

- **A. Common-state policy agreement:** Hold simulator at each state along a greedy demand stream, compute all model choices without committing them, compare only two-feasible states for primary behavioural similarity.
- **B. Own-trajectory policy comparison:** Replay each model's actual sequential allocations independently, recording the current state's greedy choice, committed candidate, acceptance, and ledger digest. Report admitted and weighted admission separately. Differences in later actions can reflect different preceding network states.
- **C. Retrospective counterfactual branches:** At selected two-feasible greedy-prefix states, clone the entire simulation; commit first/second candidate separately, then apply the same future demand suffix under greedy. Positive difference in realized priority-weighted reward denotes an *opportunity on that realized suffix*, not an online-observable oracle advantage.
- **D. MILP trace:** Starting from identical zero-reservation states for historical seed 41027, independently replay greedy and MILP V2. Identify first choice divergence, first admission divergence, solver status, and all later outcomes.

## No hypothesis laundering

Phase 5D is **post-hoc diagnosis** of previously examined results. It does not provide new independent superiority evidence, new confidence levels for prior Phase 5C claims, a Phase 5E test result or fresh training labels from the held-out test sets. The prospective Phase 5E hybrid is an additional experiment requiring a separately frozen protocol, non-learning controls, and a genuinely untouched evaluation stream. It is not defensible to repeatedly adjust hybrid rules on held-out data until a positive result appears.

## Completion criteria

All new Phase 5D tests pass; outputs include request hashes, checkpoint hashes and source hashes; zero invalid committed actions; first MILP divergence reproduced or discrepancy openly reported; development counterfactuals produce independent ledger digests; and an evidence-based decision is made about whether a selective greedy override has sufficient opportunity to justify new Phase 5E model training.

## Scope

No MSRP/RFC 8227 signaling, node/simultaneous faults, switching times, live telemetry, or operational safety claims. The work addresses this fixed synthetic simulator, finite candidate action space and synthetic traffic distribution only.
