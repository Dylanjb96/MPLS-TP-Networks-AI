# Phase 5B — Independent validation-tuning audit

**Status:** All 12 tuning-trial records reconciled; all six supplied confirmation training histories and provenance files passed structural checks. This is a validation-data audit, **not** a held-out performance evaluation.

## Design and provenance

- 60-node dual-gateway synthetic topology, topology seed 42, 10% bandwidth headroom, 100 requests per episode, at most two protected candidates.
- Training requests: demand seeds 10000–10079. Validation requests: 20000–20009. The user-supplied Phase 5B selection manifest declares historical test seeds 31000–31029 unused.
- Both algorithms were screened using one initialization (seed 7), three configurations each; only the top screened configuration per algorithm was confirmed using seeds 7, 17 and 27.
- All six confirmation provenance files match the original Phase 5 hashes for the training and validation request streams and the shared Phase 3/4/5 source files.
- The selected checkpoints are paths in the manifest; their binary contents/existence have **not** been independently checked because checkpoint files were not supplied.

## Screening: best validation scores under initialization seed 7

| Algorithm | Config | Hyperparameters | Priority-weighted admission | Ordinary admission | Best episode |
|---|---:|---|---:|---:|---:|
| ddqn | 0 | `{"epsilon_decay_steps": 4000, "gamma": 0.99, "learning_rate": 0.0003}` | 40.5282% | 34.50% | 20 |
| ddqn | 1 | `{"epsilon_decay_steps": 8000, "gamma": 0.99, "learning_rate": 0.0001}` | 39.7862% | 34.10% | 40 |
| ddqn | 2 | `{"epsilon_decay_steps": 8000, "gamma": 0.95, "learning_rate": 0.0003}` | 39.8608% | 34.10% | 20 |
| discrete_sac | 0 | `{"alpha": 0.05, "gamma": 0.99, "learning_rate": 0.0003, "target_tau": 0.01}` | 40.0915% | 34.50% | 30 |
| discrete_sac | 1 | `{"alpha": 0.1, "gamma": 0.99, "learning_rate": 0.0003, "target_tau": 0.01}` | 39.9896% | 34.40% | 60 |
| discrete_sac | 2 | `{"alpha": 0.2, "gamma": 0.99, "learning_rate": 0.0003, "target_tau": 0.01}` | 40.0935% | 34.50% | 70 |

## Confirmation: best validation checkpoint for each initialization

| Algorithm | Agent seed | Best episode | Completed episodes | Weighted admission | Ordinary admission | Gradient updates |
|---|---:|---:|---:|---:|---:|---:|
| ddqn | 7 | 20 | 100 | 40.5282% | 34.50% | 2469 |
| ddqn | 17 | 50 | 130 | 40.2069% | 34.80% | 3219 |
| ddqn | 27 | 130 | 160 | 39.5929% | 34.40% | 3969 |
| discrete_sac | 7 | 70 | 150 | 40.0935% | 34.50% | 3719 |
| discrete_sac | 17 | 30 | 110 | 39.7562% | 34.60% | 2719 |
| discrete_sac | 27 | 80 | 160 | 40.0925% | 34.30% | 3969 |

## Chosen checkpoints

**ddqn:** config 0, initialization seed 7; mean confirmed weighted validation admission **40.1093%** and mean ordinary admission **34.5667%**.
Selected checkpoint: `outputs\phase5b\tuning_v1\confirm\ddqn_cfg0_seed7\best_model.pt`.

**discrete_sac:** config 2, initialization seed 7; mean confirmed weighted validation admission **39.9807%** and mean ordinary admission **34.4667%**.
Selected checkpoint: `outputs\phase5b\tuning_v1\confirm\discrete_sac_cfg2_seed7\best_model.pt`.

## Important interpretation and limitations

1. **No DDQN tuning gain demonstrated.** Selected DDQN config 0 is precisely the original hyperparameter configuration (learning rate 0.0003, gamma 0.99, epsilon decay 4000) and seed 7. The confirmed seed-7 `training_history.csv` and `training_provenance.json` are **byte-identical** to the previously supplied original Phase 5 files. The original and “tuned” labels should not be treated as two independently optimized DDQN algorithms.
2. **SAC screening margin is tiny.** Alpha 0.20 edged alpha 0.05 in the single-initialization screen by about **0.0021 percentage points** weighted admission (40.0935% versus 40.0915%). This does not demonstrate that alpha 0.20 is robustly superior.
3. **DDQN vs SAC on validation:** selected DDQN configuration averaged **40.1093%** weighted admission across confirmed seeds; SAC averaged **39.9807%**. The difference is about **0.1286 percentage points** in DDQN’s favour. It is not a held-out effect estimate or a demonstration of statistical significance.
4. **Only one configuration per family received multi-seed confirmation.** Selection across configurations was performed first using seed 7; therefore the manifest’s stated “config by average weighted over seeds” applies only among confirmation finalists (one each), not to the entire hyperparameter grid.
5. **Repeated use of the 10 validation demand seeds can overfit the selection procedure.** Do not retune in response to final results or present the Phase 5 original test seeds as untouched.
6. **No standards or railway safety claim.** The capacity/failure model remains `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP`, not RFC 8227 MSRP, and modelling approval is pending.

## Next experiment: prospectively specified new held-out evaluation

Freeze both selected checkpoints and run a once-only paired comparison on demand seeds **41000–41029**, 100 requests per seed, topology seed 42, 10% headroom, k=2. Include greedy, CSPF-style, MILP V2, random feasible, original DDQN, confirmed DDQN and confirmed discrete SAC. Retain two DDQN labels for provenance if desired, but explicitly flag potential duplication of the original seed-7 configuration.

```cmd
python run_phase5b.py evaluate --config config/primary_60_dual.json --out outputs\phase5b\new_heldout_41000 --requests 100 --headroom 10 --max-candidates 2 --test-seed-start 41000 --test-seed-count 30 --original-ddqn outputs\phase5\train_60_seed7\best_model.pt --tuned-ddqn outputs\phase5b\tuning_v1\confirm\ddqn_cfg0_seed7\best_model.pt --discrete-sac outputs\phase5b\tuning_v1\confirm\discrete_sac_cfg2_seed7\best_model.pt --save-plans
python run_phase5b.py statistics --eval-dir outputs\phase5b\new_heldout_41000 --out outputs\phase5b\new_heldout_statistics
```

## File integrity checksums for uploaded selection records

- `tuning_trials.csv`: SHA-256 `df74f4cc6d4a62e70cc036f84dc2fdfe69ac32f9018c685a507158bb825e2a54`
- `tuning_selection.json`: SHA-256 `ef36f4b88ff841e66705d00507319bf764d601b5fa20358c55d9268551cd7779`

**Audit scope:** This report checked the uploaded tuning artifacts and the provided Phase 5B command implementation. It did not rerun network training, verify binary checkpoint weights, or evaluate on new test requests.
