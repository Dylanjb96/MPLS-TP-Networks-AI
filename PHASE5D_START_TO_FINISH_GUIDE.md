# Phase 5D — Start-to-finish execution guide

**Purpose:** Diagnose the already-frozen Phase 5 / 5B / 5C policies, investigate Greedy versus MILP V2 divergence, and measure *retrospective* two-route opportunity. **Phase 5D does not train, tune, alter, or replace any policy.** It is the evidence-gathering stage before a *separately preregistered* Phase 5E hybrid Greedy–RL investigation.

## 0. Source verification and scientific restrictions

The uploaded `MPLS_AI_Phase5D_Source.zip` contained `phase3`, `phase4`, `phase5`, `phase5b`, `phase5c`, configuration files and existing tests. The **12 files** whose SHA-256 signatures were recorded in the Phase 5C evaluation provenance matched the corresponding uploaded Phase 3–5C files exactly. The uploaded archive did **not** contain `railway_tp` (the earlier Phase 2 package) or `enhancements`; those already exist in your working local project if your original 157 tests pass. Do not overwrite them with older versions.

The framework is a **single-physical-link scenario surrogate**, `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`. It is not operational MSRP, standards validation, service restoration timing, or railway safety certification.

## 1. Install

Back up the existing project. **Extract only the contents of `MPLS_TP_Phase5D_Diagnostics_AddOn.zip` into the existing MPLS-AI project root**, alongside `run_phase5c.py`, not into a nested new project. It adds:

- `phase5d/__init__.py`, `phase5d/analysis.py`, `phase5d/cli.py`
- `run_phase5d.py`
- `tests/test_phase5d.py`
- This guide and the research protocol.

It does **not** overwrite existing `phase3`, `phase4`, `phase5`, `phase5b`, `phase5c`, `railway_tp`, trained checkpoints or results. It uses existing Phase 5C requirements.

Open Windows **Command Prompt**, not PowerShell, and run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5d.py
python -m pytest -q
```

Expected with your unmodified previous 157-test project: **10** new tests, **167 total**. If the installed project has since added tests, its total may differ.

## 2. Small 12-node software checks

Each output directory must not already exist; Phase 5D refuses to overwrite results.

```cmd
python run_phase5d.py behavior --config config/debug_12_dual.json --out outputs\phase5d\smoke_12_behavior --requests 8 --seed-start 20000 --seed-count 2 --baseline-only
```

```cmd
python run_phase5d.py counterfactual --config config/debug_12_dual.json --out outputs\phase5d\smoke_12_counterfactual --requests 20 --seed-start 21000 --seed-count 2 --max-branches 2 --horizon 20
```

```cmd
python run_phase5d.py milp-trace --config config/debug_12_dual.json --out outputs\phase5d\smoke_12_milp --requests 10 --seed 41027 --retrospective
```

These small runs verify code paths and data isolation; **do not use their outcomes as 60-node research results**.

## 3. Main seven-policy behaviour comparison — development validation seeds

The diagnostic *loads the five selected Phase 5C checkpoints and DDQN/SAC* with the same feature schema, network configuration and request horizon as their saved metadata. The five Phase 5C checkpoint SHA-256 hashes must exactly match `selection_manifest.json`. No model weights are updated.

```cmd
python run_phase5d.py behavior --config config/primary_60_dual.json --out outputs\phase5d\behavior_validation_20000 --requests 100 --headroom 10 --max-candidates 2 --seed-start 20000 --seed-count 10 --selection outputs\phase5c\selection_FINAL\selection_manifest.json --original-ddqn outputs\phase5\train_60_seed7\best_model.pt --discrete-sac outputs\phase5b\tuning_v1\confirm\discrete_sac_cfg2_seed7\best_model.pt
```

Produces:

| File | Role |
|---|---|
| `same_state_choices.csv` | Each RL policy evaluated on the *same greedy-trajectory* observation, without changing that baseline state. Primary policy imitation diagnostic. |
| `trajectory_choices.csv` | Each policy separately commits its own choices, with prefix and post-state digests, and a greedy choice for *that policy's current state*. |
| `per_seed_summary.csv` | Each policy's admission, priority-weighted admission, first sequential action/admission divergence, and final-state equality. |
| `policy_summary.csv` | Cross-seed comparison: two-feasible action rates, agreement, final-state digest counts, and admission. |
| `diagnostic_manifest.json` | Config, stream hashes, checkpoint hashes, source hashes, experiment role. |

Interpretation: **same-state agreement** controls for reservation-state differences; **own-trajectory agreement** explains what a deployed policy actually did after its earlier decisions affected the ledger. Report both. Forced rejects and one-feasible decisions are *not* evidence of discretionary policy similarity. Different checkpoint hashes by themselves do not establish different decisions. Actor probabilities are policy preferences, **not calibrated probabilities of improved future admission**. Q-values are model estimates, not proven causal advantages. The BCQ behaviour-support mask is recorded where applicable.

## 4. New-development counterfactual greedy-versus-alternative choice diagnostic

This experiment does **not** invoke future information in any learned model. It *retrospectively* starts from greedy-prefix states where both options are feasible and branches into two **independent copies** of the exact same reservation state. Branch A takes greedy's first feasible candidate; Branch B takes the alternative. Subsequent known requests and the greedy continuation policy are identical in both branches. This is a **known-future realized-difference analysis**, not a deployment mechanism or expected future performance guarantee.

```cmd
python run_phase5d.py counterfactual --config config/primary_60_dual.json --out outputs\phase5d\counterfactual_dev_21000 --requests 100 --headroom 10 --max-candidates 2 --seed-start 21000 --seed-count 10 --max-branches 4 --horizon 50
```

The main run uses a bounded **50-request lookahead** at four sampled choice points per development seed for practical runtime. A full 100-request suffix is a useful *secondary sensitivity check* once the main run is reviewed. For each seed, branch positions are selected deterministically across the full greedy trajectory's two-feasible positions, not just the earliest ones. The output `counterfactual_branches.csv` includes the initial ledger digest, both branch returns, their difference, admission counts and final-state digests. Positive `alternative_minus_greedy_weighted_reward` identifies an opportunity **on that particular realized synthetic suffix**, not proof a learned router can predict it.

Outputs: `counterfactual_branches.csv`, `counterfactual_summary.json`, `diagnostic_manifest.json`.

The selected `21000–21029` development seeds are **not** the Phase 5E final held-out seeds. If possible, later repeat the diagnostic on 21010–21029, but never consult 61000–61059 during model development.

## 5. Historical MILP V2 versus greedy divergence trace

Seed **41027** was already examined in Phase 5B. This is a **retrospective explanation**, not a new independent test, and its results must not be used for selecting Phase 5E hyperparameters. The `--retrospective` flag is mandatory.

```cmd
python run_phase5d.py milp-trace --config config/primary_60_dual.json --out outputs\phase5d\milp_41027_retrospective --requests 100 --headroom 10 --max-candidates 2 --seed 41027 --retrospective
```

Outputs: `milp_greedy_trace.csv`, `divergence_points.csv`, `trace_summary.json`, `diagnostic_manifest.json`. Examine **first candidate-choice divergence**, **first admission divergence**, final accepted counts, and solver status counts. MILP V2 minimizes *current-request* worst-scenario utilization; it does not optimize admission over the entire 100-request horizon. Later differences cannot automatically be attributed exclusively to the first divergence without an isolated branch experiment.

## 6. Stop-and-review before Phase 5E

Upload a ZIP containing:

- `outputs\phase5d\behavior_validation_20000\` (all five files)
- `outputs\phase5d\counterfactual_dev_21000\` (three files)
- `outputs\phase5d\milp_41027_retrospective\` (four files)

Exclude large unrelated prior outputs and all `.pt` binaries if you prefer: the diagnostic output contains checkpoint hashes, so the existing frozen models can be identified.

Review the number of true choices; fraction of policy deviations from greedy *on identical states*; whether alternative choices produce positive, negative or zero *realized* suffix benefits; whether model preferences predict the differences; and the exact MILP divergence sequence. If a large majority of alternatives tie or harm, a greedy-plus-RL hybrid may have very little headroom in this two-candidate formulation. This is a valid negative research outcome.

After that, write a separate Phase 5E protocol, evaluate frozen-model threshold hybrids and simple non-learning controls on **development seeds only**, and train a residual-improvement agent only if the diagnostic suggests an exploitable signal. Freeze all methods before the **proposed, currently untouched 61000–61059 Phase 5E held-out evaluation**. A hybrid must never see the actual future demand sequence when selecting a route.

## Troubleshooting

- `ModuleNotFoundError: railway_tp`: your local project lacks the preexisting Phase 2 package; Phase 5D does not install it. Check that you are in your existing project root.
- `checkpoint not found`: selection-manifest paths are relative to the current working directory. Run from the root that contains `outputs`.
- `SHA-256 mismatch`: **do not replace the frozen checkpoint**; verify you extracted the correct selection manifest and have not retrained or modified the file.
- `checkpoint metadata mismatch`: the 12-node smoke check is **baseline only**. Do not use a 60-node checkpoint on a 12-node simulator.
- `Refusing to overwrite`: choose a *new* output directory for rerun, preserving previous output files.
- `development seeds ... only`: intended guard. Previously examined final-test streams require explicit retrospective mode, and Phase 5E 61000–61059 seeds remain blocked even in retrospective mode.
