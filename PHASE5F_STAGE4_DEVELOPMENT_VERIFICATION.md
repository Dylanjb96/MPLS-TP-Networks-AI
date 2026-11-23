# Phase 5F Stage 4 — Implementation and Development Verification

**Scope:** This report establishes software engineering readiness for the user's *development-only* multi-topology PPO training. It is **not** an assertion that the twelve full training experiments, independent held-out comparison, or dissertation hypotheses have already been completed.

## Implementation supplied

- `phase5f_stage4/ablation.py`: fixed 2×2 guidance × enhanced-feature factorial, with equal-shape masked features for acting and training.
- `phase5f_stage4/workflow.py`: training over the frozen Stage 2/3 synthetic protected-route simulator, 96-episode maximum, 24-variant curriculum, deterministic cohort generation, three baselines per validation cell, content-validated gzip/JSON candidate cache, PPO parameter updates, audit logs, checkpoint hashes, and safe episode-level resume.
- `phase5f_stage4/selection.py`: strict 12-complete-run selection by grouped mean weighted validation admission, checkpoint integrity and cohort identity checks; preserves both the selected factor configuration and the nominated full hybrid.
- `run_phase5f_stage4.py`: preflight, protocol, train, select. **No held-out evaluation function.**
- `Phase5F_Stage4_12_Run_Windows.cmd`: controlled 12-run sequential Windows batch; resumes incomplete runs; never overwrites completed evidence.
- `tests/test_phase5f_stage4.py`: seed separation, route feasibility, feature ablations, curriculum, source/cache integrity, training and checkpoint persistence, resume fidelity and strict selection prerequisites.

## Checks actually executed in the assistant's environment

| Test / check | Observed result |
|---|---|
| New Stage 4 regression suite | **18 passed** |
| Combined Stage 1–4 regression tests (installed add-on tests, not user's full project) | **56 passed** |
| Stage 4 neural preflight | `STAGE4_PREFLIGHT_OK`; six feasible actions; a valid choice in each of four arms |
| Stage 3 provenance preflight using earlier uploaded smoke manifest | Passed, all matching existing Stage 2/3 source hashes |
| Small 60-node training/validation engineering run | **2 PPO episodes** completed; validation and model saving worked; zero feasibility violations |
| Interrupted training and resume | Passed; RNG/optimizer/model progress restored; unit test checks exact resulting model weights and CSV history against uninterrupted run |
| Miniature all-12-run selection test | All 4 configurations × 3 initializations completed on **one short 60-node development condition**, and strict validation-only selection succeeded |
| Source change/overwrite guard | Covered by regression tests; incomplete selection rejected |
| Access to reserved Stage 5 seeds | No usage; Stage 4 seed-role validator rejects reserved topology/stream seeds |

**Important:** The miniature 12-run engineering test used only one episode per model, one training topology, and one 60-node validation cohort. Its selected configuration and scores **must not be cited** as the proposed method's research performance and must never be confused with the formal 12-run training protocol.

## Verification not performed here

1. The assistant has **not** run the full 12 × up-to-96-episode, multi-topology training programme. Its large-scale outputs are to be generated on the user's Windows computer.
2. The user's complete local **224-test** project suite was not available in this reconstructed local environment. If unchanged, 18 new tests should give **242 total** on the user's machine. User's actual pytest output is authoritative.
3. Cross-machine exact checkpoint byte equality has not been established; SHA-256 fingerprints permit local artifact-integrity audits.
4. No final held-out topology/traffic seeds have been opened or evaluated. Training and validation findings cannot establish RL superiority.
5. The national-scale synthetic network is **not** a known Irish Rail deployed MPLS-TP network. The single-link-failure end-to-end link-disjoint scenario surrogate is **not** standards-compliant MSRP and has not been approved by the supervisor.

## Frozen development specification

The Stage 4 default full experiments use network sizes 60/150/250/500; load 1.6/2.5/4.0 requests/node; static and dynamic holding; K=6; 10% headroom; seed partition training topology 201–220 / streams 71000–71007, validation topology 301–304 / streams 72000–72001; Stage 5 heldout topology 401–420 / streams 73000–73002 never consulted; the primary model-selection validation cell is 500 nodes / 6 candidates / dynamic traffic / load 4.0. Three model seeds (7/17/27) are required in each of four configurations.

Each full run freezes its `training_plan.json` on creation and produces `training_history.csv`, `validation_summary.csv`, `validation_checkpoints.csv`, `training_provenance.json`, `best_model.pt`, and `latest_state.pt`. The selection command produces `selection_trials.csv`, `selection_configurations.csv`, and `selection_manifest.json` only after verifying all twelve completed trials.
