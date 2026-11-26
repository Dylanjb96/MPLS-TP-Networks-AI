# Phase 5E — Prospectively specified hybrid Greedy–RL research protocol

**Status:** Implementation and local development checks completed; full-scale training, model selection, new held-out evaluation and dissertation conclusions remain to be performed by the research team. **Do not describe this protocol as a completed experiment.**

## Research question

Can a **Greedy-Guided Residual PPO (GG-RPPO)** policy improve *priority-weighted protected LSP admission* over the first-feasible greedy baseline on previously unseen demand streams, without violating the unchanged Phase-3 capacity and single-link failure-scenario constraints? Does improved instantaneous state information, rather than merely adding a more complex learner, explain any improvement?

The outcome may be negative or null. Earlier Phase 5C showed no RL method exceeding greedy on the 51000–51029 held-out dataset. Phase 5D's 200 single-override counterfactuals on 21000–21009 yielded **0 improvements, 197 ties, and 3 deteriorations**. These observations motivate caution, not a presumption of hybrid benefit.

## Fixed primary experimental condition

- Topology: `config/primary_60_dual.json`, `nodes=60`, topology seed `42`, dual gateway; one fixed network, *not* 60 topology replicates.
- Protection model: `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`. The simulator checks normal state and each physical single-link failure scenario; **not an MPLS-TP Shared Ring Protection implementation** and does not measure failover/restoration time.
- 100 sequential synthetic service requests per demand seed; 10% reserved headroom; maximum 2 pre-generated structurally valid protected LSP candidates. Selection is online: **no future arrivals are available**.
- Rejection is compulsory **only** when no candidate passes the unchanged Phase-3 audit. No voluntary rejection and no modification of previously admitted LSPs.
- Baselines: first-feasible greedy, CSPF-style, online MILP V2, random feasible, plus frozen Phase 5C PPO/BCQ/CQL/AWAC/IQL and original DDQN/SAC when checkpoints are provided.
- Hybrid non-learning controls: random candidate override; selective override to a candidate with strictly lower projected single-fault peak utilization. Frozen learned-policy controls: residual PPO with and without conservative probability-margin gate.
- Prior Phase 5/5B/5C/5D results are historical context only, never training/test-selection data for GG-RPPO.

## Agent and algorithm

**A single unified residual algorithm with feature ablation** (not seven new standalone algorithms):

1. Call the unmodified `OnlineEpisode.observe()` to receive original 26 features and Phase-3 mask for each structurally valid candidate.
2. Determine the **first feasible candidate** as greedy's action. Residual action `0` means obey greedy. Residual action `1...k-1` means choose the remaining feasible candidates, in original order. When none is feasible, action `0` maps to the original forced rejection (`k`).
3. Optional *enhanced* features add an explicit greedy-default indicator, differences from greedy in prospective utilization/hops/scenario slack, and projected fault-scenario arc statistics. **All features depend only on the current request, committed reservation ledger, and precomputed current candidate paths**. No test labels, future requests or Phase 5D retrospective rollout results enter observations.
4. An action-masked categorical PPO actor and separate state-value network train **on policy** from complete request streams. Use the same priority reward as prior phases: `priority_weight/5` per admitted request; rejected requests get zero. Use `gamma=1.0` (finite sequence undiscounted primary objective) with GAE(0.95), clipped PPO updates, entropy regularisation.
5. Greedy-only warm-start and a greedy action-0 imitation loss at discretionary decisions regularise actor training. These do not imply safe improvement guarantees; they can inhibit useful overrides.
6. Deterministic evaluation uses action `0` unless the best alternative's categorical probability exceeds greedy's categorical probability by more than the **pre-fixed gate margin**. This is a policy preference margin, **not a calibrated probability of future admission improvement**. The ungated ablation chooses actor argmax directly.
7. All choices are passed once to the original `OnlineEpisode.step()`, which calls `ProtectionSimulator.admit()`/`audit()` and refuses invalid choices. Invalid residual actions raise errors; masks are not overridden by the agent.

### Pre-registered ablation configurations

**Configuration A: base** — original 26 features + explicit greedy indicator (27 total); hidden=64; lr=0.0003; greedy imitation weight=0.1; warmstart steps=50; gate margin=0.15; PPO epochs=4; gamma=1.0; 160 maximum episodes; batch=64.

**Configuration B: enhanced** — all base settings but includes the 7 extra current-state congestion features (34 total).

Three distinct agent initialization seeds: **7, 17, 27**. Thus **6 full primary training runs**. If a configuration is changed, record the revised pre-registration before training; never choose a new configuration based on held-out outcomes.

### Selection rules (no test involvement)

- Training streams: demand seeds **11000–11079** (80), repeated over episodes; greedy warm-start only on the first four of these.
- Validation streams: **22000–22019** (20); `validate_every=10`, patience=8 recorded non-improvements; a maximum of 160 training episodes.
- Per run: best checkpoint by mean priority-weighted *validation* admission; ordinary validation admission tie break.
- Per configuration: mean of three independent best checkpoint validation weighted scores; mean ordinary admission tie break; deterministic configuration-key tie break. Within the winning configuration choose best checkpoint, lowest agent seed on tie.
- Validation requests (sha256), training cohorts, source hashes and checkpoint checksums are recorded/verified. `selection_manifest.json` is the frozen selection artifact. The optional ungated result is an ablation **not** used to re-select checkpoints.
- Do not retrain/reselect/tune after seeing 61000-series data. Do not retry the final held-out experiment with alternative hyperparameters to search for a win.

### Probe and independent decision-space limitation

Development-only `probe` on **23000–23009** counts actual generated candidate sets, discretionary decision points, and paired performance of two non-learning overrides. It can be run at `k=2` and `k=4` *before* primary selection, without accessing final test streams.

**Important confirmed development observation:** on a locally reconstructed 60-node seed-42 topology using an older archived Phase 2 support package, direct inspection of 1,000 requests across development seeds 23000–23009 with `max_candidates=8` returned **exactly two structurally valid candidates for all 1,000 requests**. The user's current `railway_tp` package was absent from the source-only upload, so this particular inventory must be replicated on that current project. Thus increasing `max_candidates` from 2 to 4 does **not by itself** create a richer action space on this tested topology; report actual candidate inventories rather than claim otherwise. A new candidate generator or topology is a new experimental treatment requiring independent structural/feasibility verification, a new preregistration, and same-condition greedy/CSPF/MILP baselines. This add-on **does not modify candidate generation**.

### Locked testing and comparison

- Primary untouched test streams: **61000–61059**, 60 independent **demand seeds**, 100 requests each. Only open after validation-only selection manifest is frozen.
- Development-only `--role development` explicitly rejects all historical or future held-out seed ranges.
- Optional `k>2` sensitivity: separately label `--role sensitivity` and use previously reserved **62000–62059**, only after its own k-specific model selection. If the candidate inventory shows only 2 candidates, this is **not** an expanded-choice test and should not be used to claim an expanded action space.
- All methods use exactly the same request sequence and original Phase-3 feasibility oracle per demand seed. No topology change between algorithms within a condition.
- Primary effect: paired difference in priority-weighted admission (hybrid gated minus greedy), mean across the 60 demand seeds. Also record ordinary admission, admit counts, per-seed wins/losses/ties, final reservation-state digests and selection latency. Descriptive 95% t intervals over **whole demand seeds**; two-sided paired t-tests with Holm adjustment across the full family of alternative-vs-greedy comparisons. Unadjusted 95% CIs remain nominal; zero-width CIs are **not equivalence proofs**.
- No oracle planning using actual future test requests. Candidate generation is precomputed *once per demand seed* and its time is reported separately from per-request selection time. Selection latency includes observations, action choice and Phase-3 admission for the hybrid.
- Sensitivity (if any) must not be pooled with fixed-k primary estimates; tests of additional load/topology scenarios require independent cohort/condition definitions and matching baselines.

## Integrity and limitations

- The user's source ZIP did not contain `railway_tp` or Phase 2 support files. In local development these were supplied from an earlier full-project archive **only for test execution**; neither is distributed/overwritten in the add-on. The exact current Phase 3/4/5/5C/5D source hashes match previous uploaded Phase 5D provenance. The local Phase 2 support files have different byte hashes than the user's Phase 5D provenance; real provenance for Phase 5E must be generated by running on the user's unchanged project.
- This is a bounded, fixed topology, finite-candidate simulator with no dynamic LSP teardown/preemption, restoration timing, production protocol signalling or deployment validation. No safety compliance is implied.
- Equal observed outcomes across a finite collection of demand seeds do not establish general policy equivalence. CQL/AWAC/IQL replication of greedy in Phase 5D describes **observed states**, not every possible state.
- Changes to the candidate generator, restoration model, topology or traffic model require separate experimental labels and independent approval. `protection_model_approved` remains false.

## Evidence archive

Preserve all original Phase 5/5B/5C/5D archives; six Phase 5E training directories; `selection_trials.csv`; `selection_manifest.json`; model/checkpoint hashes; 60-seed run and paired CSVs; statistical report and figures; exact source revision; Python/NumPy/PyTorch/SciPy versions; and optional accepted-protection plans.
