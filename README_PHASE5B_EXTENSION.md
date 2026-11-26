# Phase 5B — Prospective RL extension: validation tuning + masked discrete SAC

## Scientific status and research integrity

This is an **exploratory, explicitly post-Phase-5 extension**. The previously inspected
held-out data, demand seeds **31000–31029**, were used to report that DDQN
admitted 1020/3000 versus 1051/3000 for greedy, CSPF-style and MILP V2.
Those 30 seeds **must not be used for choosing new models or settings**.
The add-on rejects those seeds (also historical 2027–2041 and 30000–30001)
in training and test scheduling. Preserve all old Phases 3–5 code and outputs.

For the extension use training **10000–10079**, validation **20000–20009**, and a
prospectively chosen, NEW test **41000–41029**. Declare test settings before
running it; use that set once for final reporting. If you later change models
based on 41000 results, label the change exploratory and preselect a further
new test dataset. Do not mix independent claims across topologies.

All policies use the identical `OnlineEpisode`, common Phase-3
`ProtectionSimulator.audit/admit` feasibility rule, same precomputed policy-valid
candidate sets, **two candidate arrangements maximum**, fixed gateway topology,
single-link-failure scenario surrogate and **forced-only rejection**. Neither
agent can see future service requests. This is **not RFC 8227 MSRP**; there is
no switching-protocol implementation or railway safety certification.
Supervisor scope approval remains pending.

## Why *discrete* SAC, not conventional continuous SAC?

The action is an index of **one of up to two feasible protected paths**. It is
not a continuous control signal. SAC for continuous actions would therefore
introduce an artificial action mapping. The supplied implementation is
**categorical / discrete SAC**, with twin Q critics, a categorical actor,
exact expectation over discrete actions, fixed entropy weight `alpha`,
Polyak target updates and independent replay. Invalid actions receive zero
selection probability. Deterministic evaluation chooses the highest-probability
feasible action. This is a pedagogical/research implementation, **not a claim
of published algorithm equivalence or empirical superiority**.

Reference: Christodoulou (2019), *Soft Actor-Critic for Discrete Action Settings*,
https://arxiv.org/abs/1910.07207 ; Huang and Ontañón (2022), *A Closer Look
at Invalid Action Masking in Policy Gradient Algorithms*,
https://doi.org/10.32473/flairs.v35i.130584.

An additional masked PPO experiment is a possible next branch; avoid adding
multiple algorithms before interpreting these two learning approaches.

## Installation (Windows)

1. Back up your existing `MPLS-AI` project and all experimental data.
2. Extract this ZIP **into the root of the existing MPLS-AI folder**. It adds
   `phase5b/`, `run_phase5b.py` and `tests/test_phase5b.py`.
   It does NOT replace `phase3/`, `phase4/`, `phase5/`, or `tests/test_phase5.py`.
3. From Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5b.py
python -m pytest -q
```

On the matching project 6 new tests and 144 total tests pass. Your local
environment should verify independently. Use `python -m pip install -r
requirements_phase5.txt` if your PyTorch environment is not already installed.

## 1. Tiny software smoke test (NOT dissertation results)

```cmd
python run_phase5b.py tune --config config/debug_12_dual.json --out outputs\phase5b\smoke_tune --requests 5 --train-seed-count 2 --val-seed-count 2 --max-configs 1 --confirm-top 1 --screen-episodes 2 --confirm-episodes 2 --validate-every 1 --patience 5 --agent-seeds 7 --algorithms ddqn discrete_sac
```

This checks the wiring, not meaningful learning. The output should include
`tuning_trials.csv`, `tuning_selection.json`, and separate `best_model.pt` files.
Do not reuse a pre-existing output directory; the code deliberately refuses to
overwrite results.

## 2. Check whether the choice set offers meaningful control

Before spending CPU on multiple networks, count how often the 60-node greedy
trajectory actually offers **two or more feasible candidate arrangements**.
On states with zero or one feasible candidate, no RL algorithm has meaningful
choice under the fixed action mask, so its potential impact is limited.

```cmd
python run_phase5b.py diagnose --config config/primary_60_dual.json --out outputs\phase5b\choice_diagnostic --requests 100 --headroom 10 --max-candidates 2 --seed-start 20000 --seed-count 10
```

Read `choice_opportunities_summary.json` and `choice_opportunities.csv`.
This counts opportunities under GREEDY's historical reservation trajectory;
other policies can subsequently see different feasible sets. It does not
explain whether either candidate improves future admission.

## 3. Controlled tuning on 60 nodes (exploratory)

```cmd
python run_phase5b.py tune --config config/primary_60_dual.json --out outputs\phase5b\tuning_v1 --requests 100 --headroom 10 --max-candidates 2 --train-seed-start 10000 --train-seed-count 80 --val-seed-start 20000 --val-seed-count 10 --max-configs 3 --confirm-top 1 --screen-episodes 80 --confirm-episodes 160 --validate-every 10 --patience 8 --agent-seeds 7 17 27 --algorithms ddqn discrete_sac
```

**Fixed screening settings**: DDQN learning rates 0.0003/0.0001, discount
0.99/0.95 and exploration decays 4000/8000 transitions (three combinations
in the command above). SAC uses entropy weights alpha **0.05, 0.1, 0.2**,
learning rate 0.0003, discount 0.99 and target tau 0.01.

For each model family, run three screening configurations with one agent
initialization; confirm the best configuration by training seeds 7, 17 and 27.
All screening and confirmation runs use the same 10 validation streams.
The chosen configuration is the one with the highest **mean validation
priority-weighted admission across confirmation initializations**. The
selected checkpoint within the configuration is the top validation performer.
Ordinary admission breaks a tie. This is deliberately **not** test-set selection.

The full command involves **12 training runs** (3 screen + 3 confirm per
algorithm), some up to 160 episodes, and can be computationally expensive.
Record total environment interactions, gradient updates, wall time and
validation variability; validation may overfit after repeated use.

The `tuning_selection.json` file lists the selected checkpoints for each
algorithm. **Do not evaluate the new test set until the choice is frozen.**

## 4. New, untouched test experiment (once, after selection)

Replace `<TUNED_DDQN_CHECKPOINT>` and `<DISCRETE_SAC_CHECKPOINT>` with the actual
`selected_checkpoint` paths from `outputs\phase5b\tuning_v1\tuning_selection.json`.
Pass the ORIGINAL agent seed 7 checkpoint exactly as saved, not a re-trained
model. Do not copy checkpoint bytes into the extension directory.

```cmd
python run_phase5b.py evaluate --config config/primary_60_dual.json --out outputs\phase5b\new_heldout_41000 --requests 100 --headroom 10 --max-candidates 2 --test-seed-start 41000 --test-seed-count 30 --original-ddqn outputs\phase5\train_60_seed7\best_model.pt --tuned-ddqn "<TUNED_DDQN_CHECKPOINT>" --discrete-sac "<DISCRETE_SAC_CHECKPOINT>" --save-plans
```

7 policies are evaluated: greedy, CSPF-style, MILP V2, random-feasible,
original DDQN, tuned DDQN, discrete SAC. Every policy receives identical
new request streams. The new test contains 30 seeds x 100 requests per policy;
the 30 whole-stream seed-level replicates are paired, and all acceptances are
committed by the Phase-3 simulator. The experiments must **not** reuse
31000–31029 even to 'check improvements'.

Outputs include `phase5_runs.csv`, `phase5_paired_deltas.csv`,
`phase5_provenance.json`, and per-seed decisions, state digest, manifests,
request lists and optional accepted route/fault plans.

## 5. Seed-level statistics and limitations

```cmd
python run_phase5b.py statistics --eval-dir outputs\phase5b\new_heldout_41000 --out outputs\phase5b\new_heldout_statistics
```

Outputs: `condition_summary.csv`, `paired_summary.csv`, and
`phase5_statistical_report.md` (generic Phase 5 statistics calculator reused
with adjusted Phase 5B heading).

**Primary endpoint**: mean priority-weighted admission per stream. Secondary:
ordinary admission, per-request selection time, fallback counts, final state
digest, path hops, and failure-state feasibility. Use paired seed differences
with 95% intervals, not per-request pseudoreplication. A narrow or zero-width
confidence interval is not proof of algorithm equivalence. Timing is policy
selection, **not protection switching time**.

**Important implementation limitation**: this Phase 5B add-on deliberately
preserves original Phase 5 environment and reward. This environment exposes a
*partial summary* of reservation state, not a fully Markov observation. Only up
to two fixed policy-valid candidate arrangements are offered. Training SAC
cannot create new routes or override eligibility. Accordingly, improvement is
not guaranteed; equal or worse admission is an informative outcome.

## 6. What to upload for review

After validation tuning: `tuning_trials.csv` and `tuning_selection.json`, plus
one `training_provenance.json` and `training_history.csv` for the selected run
of each algorithm. Keep checkpoint files locally for evaluation.

After the new held-out evaluation: `phase5_runs.csv`,
`phase5_paired_deltas.csv`, `phase5_provenance.json`,
`condition_summary.csv`, `paired_summary.csv`, `phase5_statistical_report.md`.

## Software engineering notes

- The add-on imports the existing Phase 5 implementations without replacing
  your Phase 3/4/5 source code.
- Phase 5B's SAC is a fixed-temperature, masked categorical actor-critic with
  twin Q critics; temperature optimization and prioritized replay are NOT
  implemented. Label it as such.
- Output directories are immutable once created, and checkpoints record
  topology configuration, learning seed, validation score and provenance.
- The Phase 5B evaluation reuses the Phase 5 per-policy simulation routine and
  relabels neural-network runs accurately in `summary.json`, `decisions.csv`
  and SHA-256 manifests.
- The 12-node smoke test checks integration, NOT empirical training quality.
