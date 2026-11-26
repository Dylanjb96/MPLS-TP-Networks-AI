# Phase 5F — Stage 4: independent audit of the twelve-run training evidence

**Audit date:** 24 November 2026. **Evidence:** user-uploaded `Phase5F_Stage4_Training_Evidence.zip` (12 run folders, 84 files). **Scope:** complete training/validation *record consistency* audit, safe checkpoint-metadata inspection, SHA-256 verification and execution of the frozen Stage 4 validation-only selector on reconstructed evidence. **Not performed:** independent re-simulation of 1,020 training episodes or all validation rollouts, independent physical-capacity oracle recheck, statistical testing on previously untouched held-out data, or assessment of actual railway MPLS-TP MSRP compliance.

## Executive outcome

**PASS — all twelve Stage 4 training runs supplied complete, internally consistent evidence.** An independent Python audit passed **855/855 structural, numerical, provenance and checkpoint checks**. All **12** frozen training schedules were independently reconstructed and matched their recorded schedule hashes and episode records. All **13** claimed frozen source files match the supplied Stage 1–4 package hashes. All **12** selected checkpoint files passed SHA-256, metadata and finite-parameter checks, loaded using `torch.load(..., weights_only=True)`.

The **actual frozen Stage 4 selection implementation** was executed using reconstructed copies of all twelve run folders. It returned:

`STAGE4_SELECTION_OK config=C_enhanced seed=27 score=0.886764`

This is a **validation-only model-selection outcome, not a held-out demonstration** of superiority. The proposed full hybrid `D_full_hybrid` did not win the four-arm comparison. Preserve its nominated checkpoint separately for final ablation evaluation.

## Experimental completeness and checks

| Check | Independent audit |
|---|---:|
| Planned factorial arms | 4/4 |
| Initializations per arm | 3/3 (7, 17, 27) |
| Complete evidence folders | 12/12 |
| Files expected / supplied | 84/84 |
| Completed training episodes | 1,020 in aggregate; 7/12 runs stopped early per predeclared rule |
| Offered training requests across episodes | 661,271 |
| PPO training actions | 427,934 |
| Periodic validation evaluations (each with eight matched cohorts) | 85 |
| Validation CSV rows | 2,040 (85 × 8 × 3 policies) |
| Source-package matches | 13/13 source hashes, consistent across runs |
| Checkpoint-file SHA-256 checks | 12/12 |
| Checkpoint metadata and finite tensors | 12/12 |
| Independent schedule reconstruction | 12/12 |
| Reported feasibility violations | 0 (as logged; not a new independent oracle check) |
| Software logged | Windows Python 3.11.9, NetworkX 3.5, NumPy 2.2.6, PyTorch 2.7.1+cu118 |
| Training / validation seed partitions | Matched frozen protocol |
| Reserved final seeds in training/validation rows | None detected |
| Manifest no-heldout flags | 12/12 false for `final_test_seeds_accessed` |

Training uses sizes 60/150/250/500, loads 1.6/2.5/4.0, both traffic modes and six end-to-end protected candidates. Model selection uses the predeclared **500-node, dynamic, load-4.0, K=6** condition, on **4 validation topologies × 2 streams = 8 matched cohorts**. All 12 runs had matching network/request/candidate hashes and matching baseline results for their shared validation cohorts.

The manifest flags and absence of final-seed references in the files are evidence of correct partition handling; they cannot prove what was or was not opened elsewhere on the user's computer.

## Validation selection by model configuration

**These are means of each initialization's *selected best validation checkpoint*, in percentages; consequently they are exposed to model-selection optimism.** They are not final generalization estimates.

| Arm | Guidance bundle | Enhanced features | Mean best validation weighted admission | Difference to greedy | Difference to footprint |
|---|---|---|---:|---:|---:|
| `A_standard` | No | No | 86.1955% | −1.6920 pp | −1.7597 pp |
| `B_guided` | Yes | No | 87.8875% | +0.0000 pp | −0.0677 pp |
| **`C_enhanced`** | No | Yes | **88.2717%** | **+0.3842 pp** | **+0.3165 pp** |
| `D_full_hybrid` | Yes | Yes | 88.0000% | +0.1125 pp | +0.0448 pp |
| Greedy comparator | — | — | 87.8875% | — | −0.0677 pp |
| Footprint comparator | — | — | 87.9552% | +0.0677 pp | — |

**Predeclared group winner: `C_enhanced`**, followed by `D_full_hybrid`, `B_guided`, then `A_standard`. `B_guided` exhibited precisely zero overrides in its selected checkpoints for all three seeds; its eight validation cohorts therefore matched greedy exactly. In the `D_full_hybrid` arm, two of the three selected checkpoints likewise made no overrides. The guidance treatment combines a greedy-indicator feature, behaviour-cloning penalty and evaluation confidence gate, so this factorial does **not** distinguish which constituent suppressed interventions.

## Individually nominated checkpoint models

| Role | Checkpoint | Selected episode | Mean weighted admission | Greedy | Footprint | Δ greedy | Δ footprint | Override events / discretionary decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall selected model** | **`C_enhanced_seed27`** | **60** | **88.6764%** | 87.8875% | 87.9552% | **+0.7889 pp** | **+0.7212 pp** | 966 / 11,854 (8.15%) |
| Proposed full-hybrid comparison | `D_full_hybrid_seed7` | 84 | 88.2250% | 87.8875% | 87.9552% | +0.3375 pp | +0.2699 pp | 24 / 11,781 (0.20%) |

**Individual model SHA-256 fingerprints:**

- Selected `C_enhanced_seed27/best_model.pt`: `e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9`
- Nominated full hybrid `D_full_hybrid_seed7/best_model.pt`: `e55b7b551b09bffe036d5bb75a471d87bb75568b48fa64aac483f947454e8e44`

For the selected `C_enhanced_seed27` model, the paired validation difference against greedy was positive in **six of eight** topology/stream cohorts and negative in **two**. Its **four topology-level mean differences** were positive, but these same cohorts were repeatedly consulted for checkpoint and configuration selection, so this is **not independent evidence** of improvement.

## Methods used in the independent audit

1. Opened the user-uploaded ZIP directly and verified the full twelve-folder, seven-file-per-run structure.
2. Parsed all training plans, provenance manifests and training/validation CSVs; independently checked indices, arithmetic, policy matching, weighted-admission aggregates, action counts, override bounds, reported zero violations, early stopping and highest-score-earliest-episode rules.
3. Independently reconstructed the randomized 96-episode episode schedule for each arm/initialization; compared schedule SHA-256 and the full completed training prefix to the user data.
4. Recomputed exact source digests from the original supplied Stage 1, 2, 3 and 4 code ZIPs and checked all thirteen manifests' source entries.
5. Recomputed all twelve selected checkpoint-file SHA-256 values and safely opened checkpoint metadata/weights with `weights_only=True`; checked architecture dimensions, configuration, selected episode, score, source/cohort hashes and tensor finiteness.
6. Rebuilt an isolated copy of the unmodified Stage 4 selection implementation and the twelve training folders, and ran `python run_phase5f_stage4.py select ...` successfully. No final-test network or stream seeds were generated or evaluated.

**Important limitation:** The audit is a consistency and selection-reproducibility assessment, not a rerun of every on-policy training interaction. Claims such as 'zero capacity-feasibility violations' refer to recorded simulator fields rather than an independently recomputed protection feasibility oracle. Additionally, the research system uses a synthetic national-scale network and an **end-to-end link-disjoint, single-link-failure capacity surrogate**, not Irish Rail's real network, standards-compliant MSRP, switching times or railway safety assurance.

## Next action — freeze the official selection on Windows

Keep all twelve training outputs and all model checkpoints unchanged. In Windows Command Prompt at the project root, run **exactly once**:

```cmd
python run_phase5f_stage4.py select --runs-root outputs\phase5f\stage4_train --out outputs\phase5f\stage4_selection_FINAL
```

Expected: `STAGE4_SELECTION_OK config=C_enhanced seed=27 score=0.886764`.

Upload the resulting `selection_trials.csv`, `selection_configurations.csv`, and `selection_manifest.json`, plus the Stage 4 pytest/preflight terminal output if available. Do not rerun or retune any of the twelve training experiments. Do not access reserved final topology seeds 401–420 or traffic seeds 73000–73002 until the official selection manifest has been independently checked and the separate Stage 5 protocol is frozen.

**Stage 4 status:** All supplied training evidence and the local selector reproduction **PASS**. The user's **official Windows selection manifest remains to be produced and audited** before Stage 5.
