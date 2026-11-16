# PHASE 5 — From start to finish: Masked Double Deep Q-Network (DDQN)

**Dissertation:** Resilience-Aware LSP Provisioning in Interconnected Railway MPLS-TP Networks: A Comparative Evaluation of Conventional Optimization and Deep Reinforcement Learning.

**Status:** This package supplies a complete **implementable and testable workflow**. It does **not** contain a trained 60-node research model or claim that DDQN has outperformed the Phase 4 baselines. Train and run your own held-out evaluation before stating results.

## 0. Research scope and protection-model boundary (mandatory)

This work reuses the exact Phase 3 model `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP` (preplanned ring-local opposite-arc substitution under one physical-link failure, fixed gateway transitions, independent directed capacities). It is **not RFC 8227 MSRP**, does not implement protection signalling or switching times, and establishes no railway safety claim. Supervisor scope approval remains pending. **Do not silently extend protection rules or add unvalidated routes**.

Phase 4 greedy/CSPF-style/MILP V2 are frozen baselines. The online MILP is per-request min-max utilization, not a clairvoyant offline optimum. All baselines and DDQN receive the same sorted, capped, policy-valid candidates and the same Phase 3 feasibility oracle. No actor sees future demand requests.

## 1. Install into the EXISTING project

Back up your `MPLS-AI` repository, its `phase4/policies.py`, and all existing `outputs/phase4` before making changes. Extract this add-on **at the root** of the working project, **not** in a new `phase5V2` or parallel folder. It adds a `phase5/` package, scripts, and tests; it does **not overwrite** `phase3/` or `phase4/`.

In Windows Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python --version
python -m pip install -r requirements.txt
python -m pip install -r requirements_phase4.txt
python -m pip install -r requirements_phase5.txt
python -c "import torch; print('PyTorch',torch.__version__); import phase4.policies as p; print('MILP V2 available:', 'milp_v2' in p.ALL_POLICIES)"
```

PyTorch installation may require selecting **Windows / Pip / Python / CPU** on the official PyTorch installation page if `pip` resolution fails. Python 3.11 is suitable. There is no requirement for an NVIDIA GPU.

### Check automated tests

```cmd
python -m pytest -q tests/test_phase5.py
python -m pytest -q
```

The package adds **7** Phase 5 tests. On the project version that previously passed **131**, the complete suite should report **138 passed**. Tests do not guarantee learning success or real protocol correctness.

## 2. Design: a fair sequential provisioning agent

**Current request** -> same Phase 3 policy-valid protected plans -> independent Phase 3 normal/fault capacity audit -> safety mask -> DDQN chooses a feasible plan -> `ProtectionSimulator.admit` commits atomically -> next request.

- Action `0..K-1`: one of up to `K=2` current-request candidates **only if Phase 3 says feasible**.
- Action `K`: **forced reject**, allowed only when no candidate is feasible (or none is generated). **No voluntary reject**; this preserves comparability to the non-preemptive Phase 4 baselines.
- Reward: **priority weight / 5** for accepted requests; 0 for forced rejection. This optimizes a discounted priority-weighted sequential objective (default gamma 0.99), not directly a guarantee of maximizing terminal admitted count.
- State: fixed-length features per candidate, including current request bandwidth, class priority, source/destination, service category, progress, number of accepted services, current peak scenario utilization, current feasible choices, working/fault path lengths, and prospective utilization. **The feature vector is only a partial observation of the full reservation state. Do not call the agent fully Markov.**
- Network: 2-hidden-layer, action-shared feedforward Q network (64 units each). The same small network scores each candidate; the invalid choices are masked to `-inf` before action selection.
- DDQN: online network selects `argmax` next legal action, a separately updated target network evaluates that action. Experience replay, Huber loss, target sync, gradient clipping, and epsilon-greedy training are implemented. The final **greedy-in-Q** actor uses epsilon 0.
- Feasibility oracle: all accepted services are *still* independently committed by Phase 3 audit/admit; the neural network never bypasses it.
- Safety mask caveat: the mask itself requires scenario-capacity checks. Its cost is measured inside DDQN decision time. This is an **oracle-assisted online selection** policy, not a completely autonomous network controller.

For complete details of the chosen implementation, see `PHASE5_RESEARCH_DECISIONS.json`, `phase5/environment.py`, `phase5/agent.py`.

## 3. Confirm end-to-end wiring on a small network (not a scientific experiment)

```cmd
python run_phase5_train.py --config config/debug_12_dual.json --out outputs\phase5\smoke_train_12 --requests 10 --train-seed-count 3 --val-seed-count 2 --episodes 4 --validate-every 2 --patience 10 --batch-size 8 --learning-starts 8 --train-every 1
python run_phase5_evaluate.py --checkpoint outputs\phase5\smoke_train_12\best_model.pt --config config/debug_12_dual.json --out outputs\phase5\smoke_eval_12 --requests 10 --test-seed-start 30000 --test-seed-count 2 --policies greedy cspf milp_v2 random_feasible ddqn
python run_phase5_statistics.py --eval-dir outputs\phase5\smoke_eval_12 --out outputs\phase5\smoke_stats_12
```

Do **not** include these smoke runs in the dissertation's main statistical inference. Keep separate output directories (the scripts refuse to overwrite an existing directory).

## 4. Freeze the experiment protocol **before** the held-out test

Primary experiment: 60-node dual gateway, physical topology seed **42**, candidate cap **2**, headroom **10%**, **100** service requests per episode.

| Partition | Demand seeds | Usage |
|:--|:--|:--|
| Phase 4 historical | 2027–2041 | Previously analysed; **not untouched test data** |
| Phase 5 training | 10000–10079 | Replay and gradient updates |
| Phase 5 validation | 20000–20009 | Checkpoint/hyperparameter selection |
| Phase 5 held-out test | 31000–31029 | **One-shot final reporting only** |

The smoke-only seeds 30000–30001 were used during software development and are excluded from the official untouched 31000–31029 test range. Seed split overlap is rejected in code. The topology is fixed, so generalization to new physical architectures is *not* established. Same 100-request stream per seed is reused for every evaluation policy. `phase5_runs.csv` includes SHA-256 hashes to check matching.

**Before running the final held-out test:** document the chosen learning rate, target update, discount, exploration decay, training count, random initialization, model checkpoint, and selection rule. Do not adjust them after inspecting test admission results.

## 5. Main 60-node training

```cmd
python run_phase5_train.py --config config/primary_60_dual.json --out outputs\phase5\train_60_seed7 --requests 100 --headroom 10 --max-candidates 2 --train-seed-start 10000 --train-seed-count 80 --val-seed-start 20000 --val-seed-count 10 --episodes 160 --validate-every 10 --patience 8 --agent-seed 7
```

Output directory: `training_history.csv`, `training_provenance.json`, `best_model.pt` and `last_model.pt`. **Use `best_model.pt`, selected by mean validation priority-weighted admission, then unweighted admission, not the last model.** The best model is chosen without using test seeds.

For a stronger study, repeat training with different independent network initializations **17** and **27**, with identical train/validation demand seeds and new output paths, then compare *validation* performance to choose a final model. Record variation among initializations; do not select based on test outcomes.

You may pilot on **20 training episodes** to check throughput and learning correctness, but a smoke or pilot run is not a convergence demonstration. Keep pilot and main runs separate.

## 6. One-shot, paired, blind held-out test

After locking the model and protocol, run:

```cmd
python run_phase5_evaluate.py --checkpoint outputs\phase5\train_60_seed7\best_model.pt --config config/primary_60_dual.json --out outputs\phase5\test_60_FINAL --requests 100 --headroom 10 --max-candidates 2 --test-seed-start 31000 --test-seed-count 30 --policies greedy cspf milp_v2 random_feasible ddqn --save-plans
```

If your validation-selected checkpoint is from a different training initialization, **change the checkpoint path**. Each policy receives an independent fresh simulation state. There are 30 paired test streams × 5 policies = **150 policy runs**. `random_feasible` is a diagnostic policy: chooses uniformly among verified feasible candidates (fixed per-stream random seed), and is not part of the Phase 4 optimized baseline trio.

**Timing:** candidate generation is precomputed once per demand stream, outside timed selection for all policies. DDQN selection time includes safe-masking feature computation and network inference, **not just neural inference**; `avg_inference_only_ms` is also reported. Do not confuse either with actual protection-switching latency.

Outputs:

- `phase5_runs.csv`: policy/seed admission, weighted rate, rejection categories, timing, request hashes, state digests.
- `phase5_paired_deltas.csv`: per-seed algorithm-minus-greedy admission and weighted-admission differences.
- `phase5_provenance.json`: fixed protocol, checkpoint hash, train/validation metadata, versions, seeds, limitations.
- `runs/seed_<seed>_<policy>/requests.csv`, `decisions.csv`, `summary.json`, `manifest.json` and, if `--save-plans`, `accepted_protection_plans.json`.

Do not rerun the final test to choose a different model. If a critical implementation bug is found, explicitly archive/version the old test as invalid, fix code, and create **new untouched seeds** for a revised test.

## 7. Final statistical analysis

```cmd
python run_phase5_statistics.py --eval-dir outputs\phase5\test_60_FINAL --out outputs\phase5\test_60_statistics
```

Uses the **whole demand seed** as the statistical replicate. Creates `condition_summary.csv`, `paired_summary.csv`, and `phase5_statistical_report.md`. Reports mean and seed-level SD, 95% Student t intervals, paired admission/priority-weighted differences, and selection timing.

For stable academic conclusions, discuss uncertainty and the practical significance of admission differences; **zero sample differences do not prove equivalence**. Compare DDQN with all three locked Phase 4 baselines and with `random_feasible`; do not compare with old MILP V1 as the primary baseline. More repeated seeds/topology realizations would improve external validity, but must be **preplanned** and classified accordingly.

## 8. Draw the Phase 5 figures

```cmd
python run_phase5_plots.py --training outputs\phase5\train_60_seed7\training_history.csv --statistics-dir outputs\phase5\test_60_statistics --out outputs\phase5\test_60_figures
```

Generates: `phase5_learning_curve.png`, `phase5_heldout_admission.png`, `phase5_selection_time.png`. Do not interpret high training reward as evidence of test improvement.

## 9. Optional robustness studies, only after the main protocol is frozen

1. **Training initialization robustness:** train with agent seeds 7, 17, 27, select using validation only; report training-result variance.
2. **Load:** a *separately trained* 25- or 50-request model if studying different horizons; don't silently compare a 100-request-trained checkpoint on a new horizon. Use `--allow-shift` only for **labelled out-of-distribution transfer**, not for the primary fairness comparison.
3. **Headroom:** 0%, 20% as explicitly labelled transfer tests with `--allow-shift`; preferably independently trained/validated policies for each condition if optimizing performance.
4. **Topology:** 100- and 150-node transfer or independently trained DDQN with **new topology seeds** for structural generalization. A 60-to-150 model evaluation may be valuable but cannot, alone, establish architectural generalization.
5. **Reward ablation:** test gamma=1.0 vs 0.99 (validation selection only); report both raw and weighted admission.
6. **Category balancing:** previous Phase 3 sampled workload overrepresented some endpoint categories; repeat category-balanced experiments as separate pre-registered conditions if investigating service-class equity.
7. **Action-space limitation:** primary 60-node workload has only two generated policy-valid candidates; this limits the possible advantage of DDQN. A different candidate generator is **a new, separately versioned protection experiment** and cannot simply replace Phase 3.

## 10. Final Phase 5 dissertation chapter outline

- **5.1 Objective and hypothesis.** Can current-state learning improve future protected admission relative to frozen conventional policies?
- **5.2 Formal environment.** Sequential requests; constrained action set, Phase 3 oracle, protected capacities, explicit partial observability.
- **5.3 DDQN model.** Neural-network architecture, replay, target network, Double Q target, masking, exploration and reward.
- **5.4 Data splits and training protocol.** Synthetic request generation, three separated seed ranges, hyperparameter and checkpoint selection without test leakage.
- **5.5 Verification.** Automated test log, invalid-action rejection and deterministic replay, accepted-plan hashes/feasibility audit.
- **5.6 Training.** Learning and validation curves; sensitivity to initialization/hyperparameters.
- **5.7 Held-out comparison.** Greedy/CSPF/MILP V2/random feasible vs DDQN, weighted and raw admission, seed-level uncertainty and computational cost.
- **5.8 Counterexamples / case studies.** Identify requests and earlier route choices underlying differences, without assuming DDQN must win.
- **5.9 Limitations.** Partial state, fixed topology, synthetic traffic, two candidate plans, no real deployment/safety or RFC 8227 conformance, no restoration timing.
- **5.10 Conclusion.** State whether learning offered a **replicated, meaningful** benefit, or whether greedy was sufficient under this surrogate.

## 11. Final acceptance checklist

- [ ] Fresh backup of Phase 3 / Phase 4 source and results made.
- [ ] `pytest -q tests/test_phase5.py` passes.
- [ ] `pytest -q` passes with previous 131 tests retained.
- [ ] Small-network train/eval/report completes, with positive gradient-step count.
- [ ] Main training and **validation-only** checkpoint selection complete.
- [ ] Test-seed range fixed before running and never used for hyperparameter tuning.
- [ ] All baseline and DDQN test request hashes match per seed; no missing runs/fallbacks unreported.
- [ ] All accepted test routes independently audited by Phase 3; archive manifests if generated.
- [ ] Seed-level statistical analysis and learning curves finished.
- [ ] Conclusions do not overstate the two-candidate action space or topology generalization.
- [ ] Supervisor explicitly approves the protection abstraction / scope before academic sign-off.

## 12. Suggested references for your methodology chapter

- van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-Learning*. Proceedings of AAAI, 30(1). DOI: `10.1609/aaai.v30i1.10295`.
- RFC 8227 (2017). *MPLS-TP Shared-Ring Protection (MSRP) Mechanism for Ring Topology* — **reference standard, NOT implemented by this simulator**.
- PyTorch official getting started: `https://pytorch.org/get-started/locally/`.

### Integrity and publication

This is a research prototype with no train/test result precomputed beyond smoke validation. Any future score must come from the user's executed experiments. Preserve source and exact output hashes. Running the Phase 5 code does not resolve the supervisor's pending approval of the Phase 3 model.
