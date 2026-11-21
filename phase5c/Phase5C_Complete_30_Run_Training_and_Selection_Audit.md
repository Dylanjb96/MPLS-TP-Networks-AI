# Phase 5C — Independent audit of 30 training runs and validation-based selection preview

**Scope:** Two pre-specified hyperparameter configurations × three initialization seeds (7, 17, 27) × five discrete RL algorithms. The audit uses all 30 uploaded `training_history.csv` / `training_provenance.json` pairs. This is a *validation-only* audit, not an independent held-out evaluation. The actual model checkpoint files were not provided; proposed selections below are predictions of the packaged selector from the recorded provenance, not verified checkpoint-hash selections.

## Executive conclusion

- **All 30 planned base training runs are present and internally consistent** with their corresponding history files. All 10 algorithm–configuration groups have exactly three distinct initialization seeds. No missing runs or inconsistent summary metrics were found.
- **The expected selector winners are:** PPO lr=0.0003/seed 27; BCQ threshold=0.1/seed 7; CQL conservative α=1.0/seed 7; AWAC λ=1.0/seed 7; IQL expectile=0.9/seed 7. The CQL/AWAC/IQL configuration choices are *arbitrary deterministic tie-breaks*, not measured improvements.
- **A model-behaviour diagnostic is advisable before the final holdout:** 23/30 best checkpoints share the exact same pair of validation outcomes (39.896416203% weighted and 34.8% ordinary admission). All 18 CQL/AWAC/IQL selected checkpoints share those outcomes, across both parameter configurations and all seeds. This does not establish identical actions or indicate a confirmed software bug.
- The next authoritative step is to run the packaged `select` command locally, which checks existence and SHA-256 hashes of actual `.pt` checkpoint binaries. **Do not access Phase 5C held-out seeds 51000–51029 until the manifest and provenance have been frozen.**

## All hyperparameter configurations — three-seed validation comparison

Primary selection metric: mean *priority-weighted* admission at the **best validation-selected checkpoint from each agent initialization**, not the best individual score across configurations. All scores are percentages. Ordinary admission is the secondary metric.

| Algorithm | Configuration | Seed 7 weighted | Seed 17 weighted | Seed 27 weighted | Mean weighted | Mean ordinary admission | Selection |
|---|---|---:|---:|---:|---:|---:|---|
| PPO | Learning rate = 0.0003 | 39.8964% | 40.0142% | 40.0445% | **39.9850%** | 34.7000% | **Chosen** |
| PPO | Learning rate = 0.0001 | 39.8964% | 38.5503% | 40.2672% | **39.5713%** | 34.4667% | Not chosen |
| BCQ | Behaviour threshold = 0.3 | 39.8964% | 39.8964% | 40.0300% | **39.9409%** | 34.7333% | Not chosen |
| BCQ | Behaviour threshold = 0.1 | 40.3433% | 39.9852% | 39.8964% | **40.0750%** | 34.6000% | **Chosen** |
| CQL | Conservative penalty α = 1.0 | 39.8964% | 39.8964% | 39.8964% | **39.8964%** | 34.8000% | **Chosen** |
| CQL | Conservative penalty α = 0.5 | 39.8964% | 39.8964% | 39.8964% | **39.8964%** | 34.8000% | Not chosen |
| AWAC | Advantage temperature λ = 1.0 | 39.8964% | 39.8964% | 39.8964% | **39.8964%** | 34.8000% | **Chosen** |
| AWAC | Advantage temperature λ = 0.5 | 39.8964% | 39.8964% | 39.8964% | **39.8964%** | 34.8000% | Not chosen |
| IQL | Expectile τ = 0.7 | 39.8964% | 39.8964% | 39.8964% | **39.8964%** | 34.8000% | Not chosen |
| IQL | Expectile τ = 0.9 | 39.8964% | 39.8964% | 39.8964% | **39.8964%** | 34.8000% | **Chosen** |

**Details:** PPO lr 0.0001 generated the highest *individual* PPO score (seed 27 = 40.2672%), but its seed-17 result (38.5503%) pulled its three-seed mean below lr 0.0003. The validation selection rule therefore chooses lr 0.0003. BCQ threshold 0.1 has the best mean weighted score of its two configurations (40.0750%), while its ordinary admission mean (34.60%) is lower than for threshold 0.3 (34.73%). This difference in ranking is expected because weighted admission is the declared objective. None of these validation comparisons establishes held-out superiority.

## Validation-only expected selected checkpoints

| Algorithm | Selected configuration | Agent seed | Best training checkpoint | Checkpoint weighted validation | Checkpoint ordinary admission |
|---|---|---:|---|---:|---:|
| PPO | Learning rate = 0.0003 | 27 | Episode 90 | 40.0445% | 34.60% |
| BCQ | Behaviour threshold = 0.1 | 7 | Gradient step 300 | 40.3433% | 34.40% |
| CQL | Conservative penalty α = 1.0 | 7 | Gradient step 300 | 39.8964% | 34.80% |
| AWAC | Advantage temperature λ = 1.0 | 7 | Gradient step 300 | 39.8964% | 34.80% |
| IQL | Expectile τ = 0.9 | 7 | Gradient step 300 | 39.8964% | 34.80% |

The expected checkpoint paths, as specified by the commands in the existing Phase 5C guide, are:

- **PPO:** `outputs\phase5c\train\ppo_lr3e4_seed27\best_model.pt`
- **BCQ:** `outputs\phase5c\train\bcq_t01_seed7\best_model.pt`
- **CQL:** `outputs\phase5c\train\cql_a10_seed7\best_model.pt`
- **AWAC:** `outputs\phase5c\train\awac_b10_seed7\best_model.pt`
- **IQL:** `outputs\phase5c\train\iql_e09_seed7\best_model.pt`

For CQL, AWAC, and IQL, both configurations have identical mean weighted and ordinary admission; the packaged `phase5c/selection.py` breaks those ties lexicographically using its serialized configuration key, selecting α=1.0, λ=1.0 and τ=0.9, respectively. All three seeds then tie within each chosen configuration, and the selector chooses seed 7. This tie-breaking should be disclosed; it is not evidence of a physically meaningful hyperparameter optimum.

## Internal consistency checks

| Audit check | Finding |
|---|---|
| All planned histories and provenance files | 30 matched pairs; 10 distinct configurations, three agent seeds per config |
| History ↔ provenance scores | Best checkpoint recorded validation weighted/unweighted values agree within CSV rounding |
| Checkpoint markers | Recorded best episode/gradient step matches the final positive best-checkpoint history marker |
| Training completion | Final history row matches the recorded completed episode/gradient-step count |
| Numerical validity | No nonfinite losses; validation scores in [0,1] |
| Training/validation cohorts | Identical across all 30 runs; training demands 10000–10079, validation demands 20000–20009 |
| Demand stream hashes | 80 PPO train stream hashes equal the uploaded offline dataset manifest and original Phase 5 train hashes; all 10 validation stream hashes equal original Phase 5 |
| Topology & capacity | All 60-node dual gateway, topology seed 42, headroom 10%, max 2 candidates, 100 requests/seed |
| Offline dataset identity | All 24 offline runs report the same NPZ SHA-256 and manifest SHA-256; uploaded dataset manifest file itself independently hashes to 831dbccb...c433 |
| Shared original simulator source | All 30 records list the same Phase 3/4/5 source file SHA-256 mappings |
| Artifact scope | Checkpoint binaries and the 16,000-transition NPZ were not supplied; their contents and SHA-256 cannot be independently verified here |

### Repeated validation scores: interpretation and diagnostic

**23 out of 30 best checkpoints** share the exact recorded pair weighted=0.39896416203 and unweighted=0.348. **17 of the 30 entire histories** show only one rounded distinct weighted validation value at *every* validation checkpoint; these are all CQL/AWAC/IQL histories. The remaining CQL run (α=0.5, seed 7) has two distinct validation scores but selects the same common best score.

Possible explanations include convergence to similar deterministic candidate rules, the limited two-candidate action space, domination by forced feasibility constraints, or a policy-construction/observation limitation. **Metrics alone cannot distinguish those explanations.** Before interpreting these as learning successes, compare the *selected actions and protected reservation plans on validation-only demand sequences*, and inspect checkpoint inference on two-feasible-choice states. Such validation-only diagnosis is not a reason to re-open old or new test sets.

## Exact next commands (Windows Command Prompt)

Make sure the `outputs\phase5c\train` directory contains only the intended 30 Phase 5C base training runs, each with `training_provenance.json` and `best_model.pt`, and no smoke-test or hybrid AWAC runs. Run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python run_phase5c.py select --runs-root outputs\phase5c\train --out outputs\phase5c\selection_FINAL --min-seeds 3
```

Then inspect `outputs\phase5c\selection_FINAL\selection_trials.csv` (expect 10 configurations, 3 initializations each) and `outputs\phase5c\selection_FINAL\selection_manifest.json` (expect 5 winners). Check these against the selection preview in the companion `Phase5C_Provisional_Selection_Preview.json`. The actual selector will compute checkpoint SHA-256 values and make the result authoritative, unlike the preview.

**Do not run `evaluate --test-seed-start 51000 ...` yet.** After checking the manifest, freeze model binaries, hashes, code version and the predeclared comparison protocol. The eventual held-out test should be a single pass over 51000–51029; prior tested seeds 31000–31029 and 41000–41029 have been inspected and must not be reused as new holdout data.

## Research scope and limitations

This is a fixed synthetic topology and a fixed two-candidate, forced-only rejection environment; the 16,000 offline transitions come from greedy and random-feasible logged behaviour, not operational railway records. PPO learns online with simulated interaction while BCQ/CQL/AWAC/IQL learn offline; their training experiences and computational budgets should be reported separately. AWAC in these 30 runs is **offline-only**; the optional AWAC hybrid training stage was not included. Ten repeatedly used validation demand seeds are a potential model-selection overfitting source. Three training initializations are not thirty independent demand tests. The protection model remains `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`; no demonstrated RFC 8227 MSRP compliance, restoration latency or railway operational safety; supervisor approval remains pending.

## Traceable records (all 30)

| Algorithm | Configuration | Agent seed | Best point | Weighted validation | Ordinary admission | History filename | Provenance filename |
|---|---:|---:|---:|---:|---:|---|---|
| PPO | 0.0003 | 7 | 10 | 39.8964% | 34.80% | `training_history(20261120-111129).csv` | `training_provenance(20261120-111130).json` |
| PPO | 0.0003 | 17 | 160 | 40.0142% | 34.70% | `training_history(20261120-111144).csv` | `training_provenance(20261120-111145).json` |
| PPO | 0.0003 | 27 | 90 | 40.0445% | 34.60% | `training_history(20261120-111158).csv` | `training_provenance(20261120-111158).json` |
| PPO | 0.0001 | 7 | 10 | 39.8964% | 34.80% | `training_history(20261121-154910).csv` | `training_provenance(20261121-154910).json` |
| PPO | 0.0001 | 17 | 60 | 38.5503% | 33.60% | `training_history(20261121-154919).csv` | `training_provenance(20261121-154919).json` |
| PPO | 0.0001 | 27 | 10 | 40.2672% | 35.00% | `training_history(20261121-154927).csv` | `training_provenance(20261121-154927).json` |
| BCQ | 0.3 | 7 | 600 | 39.8964% | 34.80% | `training_history(20261120-110823).csv` | `training_provenance(20261120-110823).json` |
| BCQ | 0.3 | 17 | 600 | 39.8964% | 34.80% | `training_history(20261120-110837).csv` | `training_provenance(20261120-110838).json` |
| BCQ | 0.3 | 27 | 2700 | 40.0300% | 34.60% | `training_history(20261120-110856).csv` | `training_provenance(20261120-110857).json` |
| BCQ | 0.1 | 7 | 300 | 40.3433% | 34.40% | `training_history(20261121-153919).csv` | `training_provenance(20261121-153919).json` |
| BCQ | 0.1 | 17 | 900 | 39.9852% | 34.60% | `training_history(20261121-153929).csv` | `training_provenance(20261121-153929).json` |
| BCQ | 0.1 | 27 | 2100 | 39.8964% | 34.80% | `training_history(20261121-153939).csv` | `training_provenance(20261121-153939).json` |
| CQL | 1.0 | 7 | 300 | 39.8964% | 34.80% | `training_history(20261120-110916).csv` | `training_provenance(20261120-110916).json` |
| CQL | 1.0 | 17 | 300 | 39.8964% | 34.80% | `training_history(20261120-110933).csv` | `training_provenance(20261120-110933).json` |
| CQL | 1.0 | 27 | 300 | 39.8964% | 34.80% | `training_history(20261120-110951).csv` | `training_provenance(20261120-110951).json` |
| CQL | 0.5 | 7 | 300 | 39.8964% | 34.80% | `training_history(20261121-154003).csv` | `training_provenance(20261121-154003).json` |
| CQL | 0.5 | 17 | 300 | 39.8964% | 34.80% | `training_history(20261121-154013).csv` | `training_provenance(20261121-154013).json` |
| CQL | 0.5 | 27 | 300 | 39.8964% | 34.80% | `training_history(20261121-154023).csv` | `training_provenance(20261121-154023).json` |
| AWAC | 1.0 | 7 | 300 | 39.8964% | 34.80% | `training_history(9).csv` | `training_provenance(9).json` |
| AWAC | 1.0 | 17 | 300 | 39.8964% | 34.80% | `training_history(10).csv` | `training_provenance(10).json` |
| AWAC | 1.0 | 27 | 300 | 39.8964% | 34.80% | `training_history(20261120-110812).csv` | `training_provenance(20261120-110812).json` |
| AWAC | 0.5 | 7 | 300 | 39.8964% | 34.80% | `training_history(20261121-153846).csv` | `training_provenance(20261121-153846).json` |
| AWAC | 0.5 | 17 | 300 | 39.8964% | 34.80% | `training_history(20261121-153855).csv` | `training_provenance(20261121-153855).json` |
| AWAC | 0.5 | 27 | 300 | 39.8964% | 34.80% | `training_history(20261121-153903).csv` | `training_provenance(20261121-153904).json` |
| IQL | 0.7 | 7 | 300 | 39.8964% | 34.80% | `training_history(20261120-111049).csv` | `training_provenance(20261120-111049).json` |
| IQL | 0.7 | 17 | 300 | 39.8964% | 34.80% | `training_history(20261120-111103).csv` | `training_provenance(20261120-111103).json` |
| IQL | 0.7 | 27 | 300 | 39.8964% | 34.80% | `training_history(20261120-111116).csv` | `training_provenance(20261120-111116).json` |
| IQL | 0.9 | 7 | 300 | 39.8964% | 34.80% | `training_history(20261121-154826).csv` | `training_provenance(20261121-154826).json` |
| IQL | 0.9 | 17 | 300 | 39.8964% | 34.80% | `training_history(20261121-154837).csv` | `training_provenance(20261121-154837).json` |
| IQL | 0.9 | 27 | 300 | 39.8964% | 34.80% | `training_history(20261121-154847).csv` | `training_provenance(20261121-154847).json` |

**Audit method:** Compared each JSON provenance with the corresponding history CSV, matched timestamp labels within ±1 second where the browser had appended a slightly different filename timestamp, independently computed ten group means and replayed the selector's config/seed tie-break rules by reading the distributed `phase5c/selection.py`. Source hashes and request-stream hashes were checked against one another and against the previously provided original Phase 5 provenance and offline dataset manifest. No access to actual `.pt` neural-network binaries or `.npz` transitions was available.
