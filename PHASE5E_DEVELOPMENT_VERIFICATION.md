# Phase 5E development verification and engineering audit

**Development date:** 2026-11-22. **Scope:** local implementation and testing only, not a completed full-scale hybrid RL experiment.

## Source integration and provenance

- Used the exact `MPLS_AI_Phase5D_Source.zip` uploaded in the current conversation and the previously supplied `MPLS_TP_Phase5D_Diagnostics_AddOn.zip`.
- No change to user-supplied Phase 3/4/5/5C code or earlier Phase 5D sources; all twelve Phase 3–5C/5D core SHA256 digests computed locally matched the Phase 5D `diagnostic_manifest(3).json` values.
- The user's source-only archive **omits `railway_tp` and Phase 2 enhancements**, so the local test workspace incorporated those support modules and historical Phase 2 test fixtures from the older `MPLS_TP_Phase5_Full_Working_Project.zip`. Their byte hashes **do not match the currently uploaded Phase 5D provenance**. Those older support modules/fixtures are **not included** in the distributed Phase 5E add-on. Run the new `preflight` command on the user's actual project to confirm compatibility; do not overwrite the user's existing Phase 2 modules.
- No checkpoint `.pt` files were supplied for final hybrid or original Phase 5C evaluation. No real hybrid performance gains can yet be reported.

## Implemented files

- `phase5e/policy.py`: native-to-residual masks and action mapping; nonfuture congestion features and conservative gating; non-learning controls.
- `phase5e/agent.py`: on-policy residual PPO, action masking, greedy warm start/imitation, checkpoint save/load with safe PyTorch weight loading.
- `phase5e/workflow.py`: read-only compatibility checks, development opportunity probes, six-run training machinery, validation-only checkpoint selection, guarded evaluation, matched paired results, Holm-adjusted statistics, plots, reproducibility checks and source hashes.
- `phase5e/cli.py`, `run_phase5e.py`, `tests/test_phase5e.py`.
- Pre-registration protocol, Windows start-to-finish guide and lightweight requirement list.

## Tests actually performed

- `python -m pytest -q tests/test_phase5e.py` — **19 passed** (new tests).
- `python -m pytest -q` — **186 passed** (167 previous tests + 19 new), after temporarily restoring earlier Phase 2 fixture outputs into the local test workspace. The package itself does not contain those fixture outputs; the user's existing project should already have them.
- Tests cover: sealed test seed ranges, cohort disjointness, k=3 residual mapping including first feasible candidate not at index 0 and forced rejection, invalid mask refusal, policy-confidence gating, enhanced feature finite values, Phase-3 feasibility integration, safe checkpoint load, source preflight, historical greedy equivalence, k=4 candidate inventory rather than assumed extra candidates, matching request hashes, complete small-scale train–select–development-evaluate–statistics–plot pipeline and non-overwrite handling.
- A separate **60-node development-only probe**, 100 requests on seed **23000**, k=4, completed. Greedy admitted 40/100, random override 40/100, conservative capacity override 40/100. This is a *smoke check*, NOT the final research result or statistical evidence of equivalence.
- Direct candidate generation inspection on locally reconstructed 60-node topology for **1,000 requests across seeds 23000–23009** at k=8 returned **two structurally valid candidates in each instance**. This cannot be asserted for the user's exact current Phase 2 source until they replicate `probe` locally.
- A short 60-node **training smoke** completed (10 requests per episode, 2 train seeds, 2 validation seeds, 2 PPO episodes). Results: no meaningful congestion challenge; it is evidence the wiring works, not evidence of hybrid improvement.
- A 12-node integrated fixture completed two small training runs with different network initialization seeds, selected a checkpoint, evaluated eight policy labels over two development seeds, generated a statistical report and three plots. None used reserved 61000 or 62000 test seeds.

## Integrity constraints and known limitations

- The prior Phase 5C/DDQN/SAC checkpoint binaries were not available in this workspace. The optional seven legacy ML baselines require those original local files and matching metadata on the user's PC.
- The test suite **does not prove** a trained 60-node hybrid can outperform greedy. Full three-seed training for each configuration, validation-only selection, and 60-seed final held-out evaluation are user-executed future steps.
- Local PyTorch was CPU-only, and the exact GPU setup on the user's Windows computer was not reproduced. Source references use installed PyTorch `weights_only=True` to load checkpoints.
- A higher candidate cap did not produce extra choices in the locally reconstructed primary 60-node topology. For any claimed richer decision space, independently validate candidate generation and comparably rerun all conventional policies.
- The three prior downside cases in Phase 5D's 200 single-override analysis are not evidence of universal greedy optimality. Full-episode residual training remains a hypothesis, not an asserted improvement.
- All results remain restricted to `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP` with `protection_model_approved=false`; no MSRP signalling or restoration-time implementation.

**Summary:** implementation built and local end-to-end checks passed; the prospective full-scale Phase 5E experimental outcome is not yet known.
