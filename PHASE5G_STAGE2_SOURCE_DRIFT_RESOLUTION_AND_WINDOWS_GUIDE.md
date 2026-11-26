# Phase 5G Stage 2 — source drift resolution and provenance-aware selection

**Status (26 November 2026):** The uploaded `Phase5G_Stage2_Source_Drift_Evidence.zip` was examined independently. The six training runs used three **byte-distinct** source snapshots, but the only differences in the two affected files are conversions of Python docstrings to comments. The executable abstract syntax trees are **identical when docstring statements are excluded**. This strongly supports preserving all six trained checkpoints and running a **strict, explicitly documented provenance-aware selection**, rather than retraining or rewriting training metadata.

## Independently established facts

| Source file | Original SHA-256 prefix | Later SHA-256 prefix | Actual difference | Executable AST after stripping docstrings |
|---|---|---|---|---|
| `phase5g_stage2/__init__.py` | `8c31570c5588` | `aaa0c17a9616` | module docstring converted to comments | Same |
| `phase5g_stage2/workflow.py` | `d3d2568e0a55` | `5c6ca89a1459` | module and three function/class docstrings converted to comments | Same |

The Python `__doc__` attributes differ, but none of the changed sections contains a changed routing, feature, loss, optimizer, training, validation, checkpoint, or model-selection statement. No `__doc__` access was found in the changed files. Importantly, the protected-candidate cache keys depend on **six upstream generator files**, not on the two affected Stage 2 source files. The audit package also provides Git snapshots containing the exact source versions.

The submitted provenance establishes the following three training snapshots:

| Train snapshot | Training runs |
|---|---|
| Original `__init__`, original `workflow` | `mlp_seed7` |
| Original `__init__`, later `workflow` | `mlp_seed17`, `mlp_seed27` |
| Later `__init__`, later `workflow` | `gnn_seed7`, `gnn_seed17`, `gnn_seed27` |

Every `training_provenance.json` declares 48 episodes, the same full non-architecture training settings and frozen Phase 5F C fingerprint. The uploaded source-drift evidence **does not include model checkpoints, validation CSVs, latest states or traces**, so their validity **cannot yet be independently certified from this archive**.

## Provisional validation-only statistics

*The following scores are copied from the six submitted provenance records and await source-aware verification against the actual checkpoint and validation files.*

| Architecture | Seed 7 | Seed 17 | Seed 27 | Three-seed mean |
|---|---:|---:|---:|---:|
| MLP-PPO | 83.2439% | 87.1753% | 84.6970% | **85.0387%** |
| GNN-PPO | 87.7200% | 81.5694% | 88.7397% | **86.0097%** |

By the **predeclared** highest-three-seed-mean rule, the provisional winner is GNN-PPO; the highest scoring individual GNN is seed 27 and the highest scoring MLP is seed 17. This is a **validation/model-selection observation, not an unseen held-out result or a statistically established GNN advantage**. The different outcomes across three seeds merit attention.

## Download and installation

Download the companion ZIP `Phase5G_Stage2_Provenance_Aware_Selection_Recovery.zip`. Extract it **into your existing MPLS-AI project root** (`C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI`). It includes:

- `select_phase5g_stage2_provenance_aware.py`: new, standalone selector. **Does not modify the existing `phase5g_stage2` modules, previous source files, checkpoints, training results or provenance.**
- `phase5g_stage2_source_reference/`: four **read-only** exact original/later source snapshots. These are for independent verification only; never copy them over the installed implementation.
- This guide.

In **Windows Command Prompt** run:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python select_phase5g_stage2_provenance_aware.py --selftest
python -m pytest -q tests/test_phase5g_stage2.py
python run_phase5g_stage2.py preflight
```

Expected source self-test: `PHASE5G_STAGE2_SOURCE_EQUIVALENCE_SELFTEST_OK affected_files=2 known_runs=6`.

Then use a **new output directory** (do not use the old failed selection directory if it exists):

```cmd
python select_phase5g_stage2_provenance_aware.py --runs-root outputs\phase5g\stage2_train --out outputs\phase5g\stage2_selection_PROVENANCE_AUDITED
```

When the saved files satisfy every integrity check, output ends with:

```text
SOURCE_AWARE_RUN_OK run=mlp_seed7 ...
SOURCE_AWARE_RUN_OK run=mlp_seed17 ...
SOURCE_AWARE_RUN_OK run=mlp_seed27 ...
SOURCE_AWARE_RUN_OK run=gnn_seed7 ...
SOURCE_AWARE_RUN_OK run=gnn_seed17 ...
SOURCE_AWARE_RUN_OK run=gnn_seed27 ...
PHASE5G_STAGE2_SOURCE_AWARE_SELECTION_OK winner=gnn mean=0.86009702 ...
```

The new selector checks all six run identities, known exact source snapshots, the **other thirteen source hashes byte-for-byte**, frozen C checkpoint identity, complete training histories and reconstructed schedules, latest-state journals, selected checkpoint and trace hashes, **checkpoint-embedded original source signatures**, validation checkpoint selection, complete matched baseline/policy validation rows, and deterministic selection rules. It refuses to overwrite an existing output folder. The resulting `selection_manifest.json` clearly labels this as a **provenance-aware** selection and retains the individual source signatures. The original, unchanged strict selector will continue to reject the mixed byte-level fingerprints; **do not patch it to blindly skip source checks**.

**Important:** If this new selector reports `SOURCE_AWARE_SELECTION_BLOCKED`, stop and share the exact error; do not make manual changes to training outputs or rerun training to suppress it.

## Send the evidence back for the final Stage 2 audit

When the command succeeds, package the six training folders and the new selection records, **without including the potentially large cohort cache**:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5g\stage2_train','outputs\phase5g\stage2_selection_PROVENANCE_AUDITED' -DestinationPath 'Phase5G_Stage2_Training_And_Selection_Evidence.zip' -Force"
```

Upload `Phase5G_Stage2_Training_And_Selection_Evidence.zip`. An independent audit can then read **your actual checkpoints and validation records** and confirm the selector's result before any new Phase 5G held-out evaluation is unlocked.

## Boundaries

- **No new RL training** is authorized by this source audit; the six historical experiments are preserved.
- The new selector reads only Stage 2 **training/validation evidence**, not held-out Phase 5G seeds `801–820` / streams `77000–77002`.
- Do not use prior Phase 5F Stage 5 held-out data to tune any Phase 5G models.
- The physical-link-disjoint single-fault protection model remains a **synthetic surrogate, not standards-compliant MSRP**, and supervisor approval of the abstraction remains pending.

## Developer verification of this recovery implementation

The standalone selector was syntax-checked, the two exact version pairs passed the normalized-AST hash-locked proof, and a **synthetic six-run fixture** (48 episode rows and matched validation structures per run) successfully exercised all six branches and produced `winner=gnn mean=0.86009702`. A deliberately tampered source fingerprint was correctly rejected **before** creating any output directory. This is a software engineering test, **not an independent replay of the user's real six full training runs**.
