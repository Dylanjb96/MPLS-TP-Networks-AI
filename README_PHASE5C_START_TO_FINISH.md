# Phase 5C — End-to-End Implementation and Experiment Guide

**Project:** Resilience-Aware LSP Provisioning in Interconnected Railway MPLS-TP Networks  
**Extension:** Masked PPO + discrete BCQ, CQL, AWAC and IQL  
**Code version:** `PHASE5C_DISCRETE_RL_V1`  
**Model:** `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`  
**Status:** Software implemented and development-tested; full 60-node training, held-out evaluation, and dissertation claims **still require your local runs**.  
**Scope:** Single-link-failure scenario approximation, not RFC 8227 standards-compliant MPLS-TP Shared-Ring Protection; no railway safety certification.

---

## 0. Why these algorithms

The existing problem is a **discrete choice among at most two Phase-3-generated protected LSP candidate arrangements**; the third action is a **forced reject if no feasible candidate exists**. A candidate must satisfy the independent Phase-3 audit before selection. No method can bypass this oracle, create an unconstrained route, voluntarily reject feasible work, or see future demand.

| Algorithm | Mode | Implemented formulation | Train-time environment access |
|---|---|---|---|
| Masked PPO | Online, on-policy | Categorical masked actor, learned state value, GAE and clipped PPO ratio | Yes — training seeds only |
| Discrete BCQ | Offline | Behavior-cloned categorical action support, masked Q-learning, behavior-likelihood threshold | No — updates read the fixed offline dataset only |
| Discrete CQL | Offline | Masked double-Q-style target selection with CQL(H) log-sum-exp penalty on legal actions | No |
| Discrete AWAC (offline stage) | Offline | Discrete actor weighted by estimated logged-action advantage and Bellman critic | No |
| AWAC hybrid (optional) | Offline → online | Warm-start the offline AWAC model, add training-only on-policy simulation transitions, update using mixed replay | Yes — training seeds only |
| Discrete IQL | Offline | Expectile value regression, twin logged-action Q regressions, advantage-weighted categorical behavior cloning | No |

These are **explicit discrete research adaptations**, not reproductions of third-party libraries. In particular, an offline-only AWAC run does not constitute the original offline-to-online AWAC training protocol; use `finetune-awac` to evaluate the hybrid version. For the main five-method comparison, select the AWAC variant **by validation only**.

**Existing comparator algorithms** remain greedy, CSPF-style, MILP V2, random-feasible, original DDQN and Phase 5B discrete SAC. The final result should not treat original/tuned DDQN as independent if their checkpoint hashes are equal.

## 1. Install without overwriting earlier phases

Existing Windows project:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
```

1. Back up the project and all Phase 4, Phase 5 and Phase 5B experiment directories.
2. Extract `MPLS_TP_Phase5C_Five_Algorithms_AddOn.zip` **into the existing project root**. It adds `phase5c/`, `run_phase5c.py`, `tests/test_phase5c.py`, this guide and the protocol. **It does not replace** Phase 3, Phase 4, Phase 5, or Phase 5B source files.
3. Confirm that `phase5b/` already exists (this add-on reuses the Phase 5B SAC loader and Phase 5 runner). If it is missing, install the earlier Phase 5B add-on first.
4. Dependencies:

```cmd
python -m pip install -r requirements_phase5c.txt
python -m pytest -q tests/test_phase5c.py
python -m pytest -q
```

The add-on includes **13 new tests**; in the matched development environment, **157 tests passed** in the full project. Your machine must independently pass tests before the full research experiment.

Check available commands:

```cmd
python run_phase5c.py --help
python run_phase5c.py train-ppo --help
python run_phase5c.py train-offline --help
python run_phase5c.py finetune-awac --help
```

### Project addition

```text
MPLS-AI/
  phase5c/
    __init__.py
    common.py
    dataset.py
    ppo.py
    offline_agents.py
    workflow.py
    selection.py
    plots.py
    cli.py
  run_phase5c.py
  tests/test_phase5c.py
  requirements_phase5c.txt
  README_PHASE5C_START_TO_FINISH.md
  PHASE5C_RESEARCH_PROTOCOL.md
```

## 2. Use an appropriate experimental design

The **fixed primary topology** is the same 60-node dual-gateway topology, topology seed 42, 10% headroom, two maximum candidates and 100 requests per demand seed.

| Data role | Seeds | Rule |
|---|---|---|
| Online PPO training and offline behavior-data collection | **10000–10079** | Permitted for all training updates |
| Validation/checkpoint/hyperparameter selection | **20000–20009** | Only these data choose winners |
| Previously examined Phase 5 test | 31000–31029 | Never retune using these |
| Previously examined Phase 5B test | 41000–41029 | Never retune using these |
| **Phase 5C new final held-out test** | **51000–51029** | Evaluate **once**, after freezing all decisions |
| Phase 4 exploratory seeds | 2027–2041 | Previously examined, not a new final test |

**Primary outcome**: mean priority-weighted protected-service admission. Also report ordinary admission, seed-level paired differences with uncertainty, runtime, training budget, final-state digests, action feasibility, and the effect of restricted candidate choice.

The offline logged data and PPO on-policy data have **different experience-collection mechanisms and computational budgets**. Report dataset transitions, PPO episodes, gradient steps and wall time **separately**. Do not call the methods equal-sample-budget comparisons unless the budgets are explicitly matched.

## 3. Run small 12-node software checks first

These are **not research results**. They only verify packaging and execution on your machine. Use dedicated output folders so no smoke result gets included in full model selection.

```cmd
python run_phase5c.py dataset --config config/debug_12_dual.json --out outputs\phase5c\smoke\offline_data --requests 4 --seed-start 10000 --seed-count 2

python run_phase5c.py train-ppo --config config/debug_12_dual.json --out outputs\phase5c\smoke\ppo_seed7 --requests 4 --train-seed-count 2 --val-seed-count 2 --episodes 2 --validate-every 1 --agent-seed 7

python run_phase5c.py train-offline --algorithm bcq --config config/debug_12_dual.json --dataset outputs\phase5c\smoke\offline_data --out outputs\phase5c\smoke\bcq_seed7 --requests 4 --val-seed-count 2 --updates 4 --validate-every 2 --agent-seed 7
```

Repeat the offline smoke command with `cql`, `awac`, and `iql` and distinct output directories. Check that each produces `best_model.pt`, `training_history.csv`, `training_provenance.json` and `last_model.pt`. An independent end-to-end 12-node development demonstration exercised all five algorithms, algorithm selection, 9-policy evaluation, statistics and three PNG figures. This **does not imply good model performance** in a nontrivial workload.

## 4. Create the frozen offline training dataset

Run **once** for all offline algorithms. The dataset records **two independent simulated trajectories for each of 80 training seeds**: one greedy, one random-feasible. All transitions preserve state rows, action mask, action, priority-weighted reward, next-state rows, next mask and terminal flag. With 100 requests × 80 seeds × 2 policies, the planned count is **16,000 transitions**.

```cmd
python run_phase5c.py dataset --config config/primary_60_dual.json --out outputs\phase5c\offline_data_60 --requests 100 --headroom 10 --max-candidates 2 --seed-start 10000 --seed-count 80
```

Outputs:

```text
outputs/phase5c/offline_data_60/
  offline_transitions.npz
  offline_dataset_manifest.json
```

The loader verifies SHA-256 integrity. **Do not rewrite the dataset** between algorithm runs. The dataset is collected from simulation, not observational railway logs; report that clearly. It is not a demonstration of safe online exploration in real infrastructure.

## 5. Main on-policy PPO training (three agent initializations)

Start with an agreed initial PPO configuration; use validation metrics to compare any planned alternatives. To execute three reproducible runs from **Command Prompt** (use `%S` interactively; use `%%S` in a `.bat` file):

```cmd
for %S in (7 17 27) do python run_phase5c.py train-ppo --config config/primary_60_dual.json --out outputs\phase5c\train\ppo_lr3e4_seed%S --requests 100 --headroom 10 --max-candidates 2 --train-seed-start 10000 --train-seed-count 80 --val-seed-start 20000 --val-seed-count 10 --episodes 160 --validate-every 10 --patience 8 --agent-seed %S --learning-rate 0.0003 --ppo-clip 0.2 --ppo-epochs 4 --entropy-coef 0.01
```

For a **small, controlled second configuration**, repeat into `ppo_lr1e4_seed%S` with `--learning-rate 0.0001`. Do not silently alter the workload or test seeds.

PPO trains from simulator interactions and uses masks at action sampling, scoring and policy-gradient updates. The PPO ratio is clipped to the configured range; generalized advantage estimation is calculated over complete 100-request episodes. The best checkpoint is chosen by validation **priority-weighted admission**, with ordinary admission as a tie-breaker.

## 6. Offline BCQ, CQL, AWAC and IQL training

These models learn from **exactly the same frozen offline dataset**. They must not collect new simulator transitions during optimization; simulation is permitted **only for validation evaluation** of the frozen or intermediate policy, without gradient updates.

Use the following interactive Windows CMD commands (each covers seeds 7, 17 and 27):

**BCQ**, behavior threshold 0.3:

```cmd
for %S in (7 17 27) do python run_phase5c.py train-offline --algorithm bcq --dataset outputs\phase5c\offline_data_60 --config config/primary_60_dual.json --out outputs\phase5c\train\bcq_t03_seed%S --requests 100 --headroom 10 --max-candidates 2 --val-seed-start 20000 --val-seed-count 10 --updates 3000 --validate-every 300 --patience 8 --agent-seed %S --behavior-threshold 0.3
```

**CQL**, conservative penalty coefficient 1.0:

```cmd
for %S in (7 17 27) do python run_phase5c.py train-offline --algorithm cql --dataset outputs\phase5c\offline_data_60 --config config/primary_60_dual.json --out outputs\phase5c\train\cql_a10_seed%S --requests 100 --headroom 10 --max-candidates 2 --updates 3000 --validate-every 300 --agent-seed %S --conservative-alpha 1.0
```

**AWAC**, advantage temperature 1.0 (offline stage):

```cmd
for %S in (7 17 27) do python run_phase5c.py train-offline --algorithm awac --dataset outputs\phase5c\offline_data_60 --config config/primary_60_dual.json --out outputs\phase5c\train\awac_b10_seed%S --requests 100 --headroom 10 --max-candidates 2 --updates 3000 --validate-every 300 --agent-seed %S --awac-lambda 1.0
```

**IQL**, expectile 0.7:

```cmd
for %S in (7 17 27) do python run_phase5c.py train-offline --algorithm iql --dataset outputs\phase5c\offline_data_60 --config config/primary_60_dual.json --out outputs\phase5c\train\iql_e07_seed%S --requests 100 --headroom 10 --max-candidates 2 --updates 3000 --validate-every 300 --agent-seed %S --expectile 0.7
```

For the planned **second hyperparameter configurations**, run a second three-seed batch for each method in a new output directory:

| Method | First config | Second config | Additional flag for second config |
|---|---|---|---|
| PPO | LR 0.0003 | LR 0.0001 | `--learning-rate 0.0001` |
| BCQ | threshold 0.3 | threshold 0.1 | `--behavior-threshold 0.1` |
| CQL | alpha 1.0 | alpha 0.5 | `--conservative-alpha 0.5` |
| AWAC | lambda 1.0 | lambda 0.5 | `--awac-lambda 0.5` |
| IQL | expectile 0.7 | expectile 0.9 | `--expectile 0.9` |

Name the new directories uniquely, e.g. `bcq_t01_seed%S`, `cql_a05_seed%S`, `awac_b05_seed%S`, `iql_e09_seed%S`. With two configs × three seeds × five methods, this pre-specified small grid comprises **30 base training runs**. These are *starting configurations*, not a claim of globally optimal hyperparameters. After the study is frozen, avoid expanding the grid based on test performance.

### Optional: the full offline-to-online AWAC extension

If you want to compare the offline AWAC stage with its hybrid online refinement, **before touching the Phase 5C final test**, finetune each offline AWAC model on **training seeds only**:

```cmd
for %S in (7 17 27) do python run_phase5c.py finetune-awac --config config/primary_60_dual.json --offline-checkpoint outputs\phase5c\train\awac_b10_seed%S\best_model.pt --dataset outputs\phase5c\offline_data_60 --out outputs\phase5c\train\awac_hybrid_b10_seed%S --requests 100 --headroom 10 --max-candidates 2 --train-seed-start 10000 --train-seed-count 80 --val-seed-start 20000 --val-seed-count 10 --episodes 80 --updates-per-episode 64 --validate-every 10 --patience 8 --exploration-epsilon 0.1 --online-fraction 0.5
```

This stage adds freshly simulated training-seed transitions and mixes them 50/50 by mini-batch with the frozen offline dataset. The above example uses the `awac_b10` configuration; if the `b05` configuration won validation, use its corresponding offline checkpoints instead. The selected AWAC result must disclose whether it was **offline only** or **offline-to-online**. This hybrid receives additional simulator interactions and should not be described as an equal-interaction comparison to the offline-only methods.

## 7. Freeze validation-selected models (before final test)

Use the built-in aggregation command. It expects **at least three different agent seeds per hyperparameter configuration for every algorithm** and uses mean validation priority-weighted admission across initializations to choose the configuration. It then chooses the best validation checkpoint within the winning configuration.

```cmd
python run_phase5c.py select --runs-root outputs\phase5c\train --out outputs\phase5c\selection_FINAL --min-seeds 3
```

Outputs:

```text
outputs/phase5c/selection_FINAL/
  selection_trials.csv
  selection_manifest.json
```

Inspect `selection_manifest.json`. It records exactly **five selected model checkpoints** and their SHA-256 checksums. It should contain `ppo`, `bcq`, `cql`, `awac`, `iql`, with the AWAC regimen explicitly labelled. Do not include smoke-test folders in `--runs-root`. The selection code rejects mixed network configurations and overlapping/changed seed cohorts.

**Once this step is complete, freeze the selected configuration and source code**. The new final test seeds must not be used for any further model selection, debugging motivated by outcomes, or tuning.

## 8. Final held-out comparison: 51000–51029 (once)

This example includes the prior frozen DDQN and SAC checkpoints as well as all new five methods and four non-learning baselines, for a total of **11 algorithm labels**, **30 demand seeds** and **330 algorithm-run records**.

```cmd
python run_phase5c.py evaluate --selection outputs\phase5c\selection_FINAL\selection_manifest.json --config config/primary_60_dual.json --out outputs\phase5c\heldout_51000_FINAL --requests 100 --headroom 10 --max-candidates 2 --test-seed-start 51000 --test-seed-count 30 --original-ddqn outputs\phase5\train_60_seed7\best_model.pt --discrete-sac outputs\phase5b\tuning_v1\confirm\discrete_sac_cfg2_seed7\best_model.pt --save-plans
```

If you omit `--original-ddqn` or `--discrete-sac`, there will be fewer comparator labels (9 for the five new methods + four non-learning baselines). The `--save-plans` flag saves substantial route files; keep it for independent protection-path audits. All algorithms receive **identical generated request streams per demand seed**, empty initial reservation state, fixed candidate generator and the same Phase-3 feasibility authority.

Expected directory contents:

```text
heldout_51000_FINAL/
  phase5_runs.csv
  phase5_paired_deltas.csv
  phase5_provenance.json
  runs/
    seed_51000_greedy/
      requests.csv
      decisions.csv
      summary.json
      accepted_protection_plans.json    (when --save-plans)
      manifest.json
    ...
```

The evaluation code rejects reuse of the already examined 31000/41000 test seeds and refuses to overwrite existing output directories. It also verifies that checkpoint architecture, feature schema and simulator settings match the evaluation configuration.

## 9. Statistical analysis and figures

```cmd
python run_phase5c.py statistics --eval-dir outputs\phase5c\heldout_51000_FINAL --out outputs\phase5c\statistics_FINAL

python run_phase5c.py plots --stats outputs\phase5c\statistics_FINAL --out outputs\phase5c\figures_FINAL
```

Key files:

```text
statistics_FINAL/
  condition_summary.csv
  paired_summary.csv
  phase5_statistical_report.md
figures_FINAL/
  phase5c_admission.png
  phase5c_weighted_admission.png
  phase5c_selection_time.png
```

**Report the full paired comparisons, not just the best algorithm.** Demand seed (whole 100-request sequence) is the experimental replicate. The Student-t intervals reflect variability across demand sequences for one topology only. With 11 methods and multiple endpoints, consider adjusted multiplicity or frame p-values as exploratory. Zero-width differences do **not** prove equivalence. The neural network's policy-selection time includes feature generation and masks but does **not** represent MPLS-TP switching latency or restoration times.

## 10. Required audits before declaring Phase 5C complete

1. Confirm 157-project-test baseline on the matching installed project and all new algorithm tests. Re-run after any source edit.
2. Audit action masks: no inadmissible selected candidate, no voluntary reject, no Phase3 audit/admit bypass.
3. Confirm `offline_transitions.npz` and its SHA-256 manifest are unchanged for BCQ/CQL/AWAC/IQL.
4. Confirm all validation-selected checkpoint hashes in `selection_manifest.json` match the loaded evaluation checkpoints.
5. Confirm the new 51000–51029 test seeds are absent from all training and validation provenance.
6. Verify each run has 100 offered requests, admission + rejection reconciles, and requests hashes match across policies for each seed.
7. Verify paired confidence intervals and metrics independently from `phase5_runs.csv` (not only the summary report).
8. Inspect cases with differences via `decisions.csv` and accepted plan files, without retroactively retuning algorithms.
9. Report offline dataset coverage, forced rejects, actual total trained steps and the limitations of a fixed network seed and max-two-candidate space.
10. Preserve the full previous Phase5 and Phase5B reports separately as historical evidence; the research programme expanded after observing those results.

## 11. Dissertation Phase 5C report structure

- **5C.1 Research motivation:** why existing DDQN/SAC results prompted a **post hoc extension**, with disclosure of earlier held-out performance.
- **5C.2 Stable provisioning simulator:** Phase-3 single-link failure scenario surrogate, no MSRP compliance claims.
- **5C.3 MDP:** request stream, observation rows and masks, two candidate actions plus forced rejection, priority-weighted reward, sequential terminal state.
- **5C.4 Online algorithm:** masked PPO / clipped objective / generalized advantage estimation.
- **5C.5 Offline dataset:** deterministic greedy + random-feasible data collection on simulation training seeds, SHA-256 and behavior support.
- **5C.6 Offline algorithms:** discrete BCQ threshold, CQL penalty, AWAC advantage-weighted actor (and optional hybrid fine-tuning), IQL expectile value fit.
- **5C.7 Training and selection:** shared vs unequal simulation budgets, three initializations, two-point prospective validation grids.
- **5C.8 Held-out study:** new paired test seeds, all policies and compute timings.
- **5C.9 Statistics:** seed-level paired effects and uncertainty, adjusted comparisons as needed.
- **5C.10 Negative results/limitations:** constrained action space, partial state, topology generalization not evaluated, no standards-conformant protection signaling.
- **5C.11 Implications:** what learning methods can and cannot add given this strictly defined provisioning environment.

## 12. Research references (algorithm families)

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, https://arxiv.org/abs/1707.06347
- Fujimoto et al. (2019), *Benchmarking Batch Deep Reinforcement Learning Algorithms* (discrete BCQ), https://arxiv.org/abs/1910.01708
- Kumar et al. (2020), *Conservative Q-Learning for Offline Reinforcement Learning*, https://arxiv.org/abs/2006.04779
- Nair et al. (2020), *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*, https://arxiv.org/abs/2006.09359
- Kostrikov et al. (2021/2022), *Offline Reinforcement Learning with Implicit Q-Learning*, https://arxiv.org/abs/2110.06169

**Implementation disclaimer:** these citations motivate the mathematical families. This package is a compact Python/PyTorch discrete-action research adaptation. Its small tests are software correctness checks, not independent academic verification of exact paper reproducibility.
