# Phase 5F Stage 1 — Local Development Verification (23 November 2026)

**Scope:** standalone source prototype only; not a final held-out test or RL result.

- Source reviewed: the user's current `MPLS_AI_Phase5D_Source.zip` and the supplied `MPLS_TP_Phase5E_Hybrid_Greedy_Residual_RL_AddOn.zip` to understand existing simulator boundaries. No existing source files altered or packaged for replacement.
- Public basis: Iarnród Éireann 2023 Annual Report pp.18 and its 2019 procurement documents (see protocol).
- `python -m pytest -q tests/test_phase5f.py` → **13 passed** in the development environment.
- `python run_phase5f.py preflight --sizes 60 150 250 500 --seed 101` → four synthetic connected, bridge-free networks successfully built; sample route counts:

| Nodes | Links | Regions | Sample disjoint pairs |
|---:|---:|---:|---:|
| 60 | 73 | 5 | 6 |
| 150 | 182 | 12 | 6 |
| 250 | 301 | 21 | 6 |
| 500 | 594 | 42 | 6 |

- Stage 1 generation at 500 nodes, seed 101, demand seed 70000 and 1.6 offered requests per node (800 synthetic demands): approximately 14.4 seconds to prepare up to 6 candidate pairs for all requests in this development container. This is an **uncontrolled illustrative run**, not a hardware-portable performance claim or the claimed cost of full ML experiments.
- 500-node *single development stream*, same topological seed, same request identities for static/dynamic:

| Condition | Greedy weighted admission | Footprint heuristic weighted admission |
|---|---:|---:|
| K2/static | 81.01% | 81.40% |
| K6/static | 87.92% | 89.56% |
| K2/dynamic | 94.30% | 94.66% |
| K6/dynamic | 97.08% | 98.18% |

**These are strictly exploratory single-stream software smoke-test values** and must never be used as Phase 5F final results, a p-value, or proof RL improves upon greedy. Note the change from Phase 5E's protection model, routing graph and synthetic demand rules.

- Additional integrity checks: deterministic topology, common-random-numbers parity of static/dynamic arrivals, candidate-prefix invariance, genuine pair link-disjointness, normal and per-link-failure directed capacity conservation, dynamic releases, explicit independent fault-path recomputation oracle, and reserved-test-seed guard.

**Limitations:** Real deployment reconstruction is not attempted, new protection abstraction is not MSRP, the >2 feasible-choice share remains to be established on development populations, 60-node offered load appears weak (often near all-admit), and the full RL model/training/held-out experimental pipeline is **not yet implemented**. 500-node training performance and model efficacy are not established.
