# Phase 5G — Stage 4: independent forensic audit of final held-out evaluation

**Audit verdict: PASS for the submitted full-grid, source/checkpoint/protocol provenance, per-request trace consistency and independently recalculated inferential statistics.** The **sampled independent protection-oracle replay** is reported separately below. This verdict does not imply the GNN is the winner; the scientific results show a meaningful limitation on cross-topology transfer.

**Evidence analysed:** `Phase5G_Stage4_Final_Reports.zip` plus the eight corresponding `H_BASE`, `L_BASE`, `H_BW`, `L_BW`, `H_HOLD`, `L_HOLD`, `H_K2`, and `L_K2` cohort archives. Source/freeze identities independently checked against the earlier submitted Stage 3 frozen archive, Stage 4 original add-on/source handoff, and external independent-development audit release. All are synthetic 500-node networks, not actual railway maps.

## 1. Independent audit scope

| Requirement | Independent finding |
|---|---:|
| Eight preregistered cells, each with 20 topologies × three streams | **8/8 complete; 480/480 cohorts** |
| Six policies per matched cohort | **2,880/2,880 policy runs** |
| Request-level traces inspected | **5,760,000 decisions** (2,000 × 2,880) |
| Archived cohort-body SHA-256 integrity | **480/480 valid** |
| Duplicate, missing or extra cell/topology/stream/policy combinations | **0** |
| Independently repeated programmatic checks | **156,977 checks; 0 failures** |
| Request-level admissions, priority-weighted ratios, override counts, feasible/structural counts, departures | **Reconstructed and matched for all 2,880 policy records** |
| Within-cohort network/request/candidate digests | **All six policies identical per cohort** |
| K=2 candidate action bounds | **Validated in all K=2 traces** |
| Baseline versus stressed/K2 generator identity consistency | **Verified using all supplied input digests** |
| Frozen model SHA-256 and previous Stage 3 protocol/release hashes | **All match the supplied frozen source and release** |
| Logged single-fault protection violations | **0** (as recorded; full independent oracle replay is sampled) |
| Independent reconstruction of test statistics and Holm adjustments | **All two co-primary, eight secondary, and 12 exploratory contrasts matched** |
| Changed model weights or retraining during final analysis | **None present in the submitted locked manifests; cannot certify activity outside supplied files** |

Each cohort file contains an integrity-sealed JSON body with all six policy results and all 12,000 request-level decisions. The independent auditor recalculated this body SHA-256 from the raw canonical contents, matched its six rows byte-semantically against the reporting CSV, and independently reconstructed reported priority-weighted admission from every request weight and admitted bit. The 480 separate input fingerprints and 20 topology seed identities were checked independently of the official reporting code. In addition, regenerating all 20 network seeds confirmed **20 different graph adjacencies** within each of the hierarchical and corridor-ladder families; this is still a controlled synthetic generator, not sampling across real railway infrastructures.

**Independent oracle scope:** The additional attestation is based on reconstructing the exact graph, request stream and candidate inventory for topology 801 / traffic 77000 on **all eight cells**, then replaying each of the six policies through the earlier separate Phase 5F single-link-failure oracle. It checks every accepted path, forced rejection, capacity feasibility, active/departure counters and terminal oracle state digest for **48 complete policy trajectories / 96,000 individual requests**. The remaining 472 cohorts have extensive trace/provenance checks, but they have **not** each been rerun from independently regenerated candidate inventories. Therefore zero independently detected violations cannot be extrapolated to a guarantee about unseen error modes across all 2,880 runs.

## 2. Final priorities — independent, held-out weighted admission rates

Each cell has **60 matched cohorts** (20 independent topology seeds, three streams per topology), with **120,000 offered requests per policy**. Values are mean of 60 within-cohort priority-weighted admission ratios (percent).

| Cell | GNN-PPO | MLP-PPO | Previous PPO C | Online minimax | Footprint | Greedy |
|---|---:|---:|---:|---:|---:|---:|
| H_BASE | 89.978% | 87.933% | 88.356% | 90.292% | 87.921% | 87.929% |
| L_BASE | 87.392% | 89.276% | 89.119% | 89.684% | 88.054% | 89.299% |
| H_BW | 82.008% | 80.166% | 80.338% | 82.616% | 79.922% | 80.153% |
| L_BW | 79.187% | 81.013% | 80.802% | 81.433% | 79.778% | 81.000% |
| H_HOLD | 96.047% | 94.722% | 94.875% | 96.305% | 94.732% | 94.712% |
| L_HOLD | 94.287% | 95.649% | 95.434% | 95.976% | 95.118% | 95.646% |
| H_K2 | 84.633% | 83.218% | 83.627% | 84.964% | 83.085% | 83.218% |
| L_K2 | 87.226% | 88.706% | 88.203% | 88.898% | 87.517% | 88.700% |

**The online scenario-aware minimax heuristic has the highest mean admission in all eight conditions.** It is a current-request, candidate-enumeration online optimizer, **not** a globally optimal sequential MILP or future-informed oracle.

In the primary hierarchical/base condition, GNN admits **101,708** requests out of 120,000 versus **102,385** for minimax. In the ladder/base transfer condition, these are **98,523** versus **102,289**. Admission totals are descriptive; hypothesis testing is performed on topology-level paired priority-weighted fractions.

## 3. Two preregistered co-primary inferential tests

The independent inference uses **20 topology clusters**. Within each topology, it averages the three matched stream-level differences. The reported intervals are *nominal two-sided 95% Student-t intervals*, while p-values receive **Holm family-wise correction over the two primary hypotheses**. Positive effects indicate the first policy has higher priority-weighted admission. *pp = percentage points*.

| H_BASE comparison (first − second) | Effect (pp) | 95% CI (pp) | Holm-adjusted p | Positive / negative topology clusters |
|---|---:|---:|---:|---:|
| gnn_PPO − mlp_PPO | +2.0454 | [+1.8306, +2.2601] | 6.7721e-14 | 20 / 0 |
| gnn_PPO − scenario_minimax | -0.3141 | [-0.5966, -0.0316] | 0.031153 | 5 / 15 |

**Interpretation:** GNN-PPO improves upon the matched neural MLP-PPO on the in-family test set (**+2.0454 pp**, 20/20 clusters positive). However, **the online minimax is statistically better than GNN-PPO**, since GNN minus minimax is **−0.3141 pp** (Holm-adjusted p = **0.03115**, 15/20 topology clusters negative). It would be scientifically incorrect to claim that GNN wins the overall algorithm comparison.

## 4. Preregistered secondary tests: out-of-family generalization

The eight secondary comparisons form a **separate Holm correction family**. All eight effects reproduced exactly from the submitted cohort rows; direction and adjusted p-values were verified independently.

| Cell | Contrast | Effect (pp) | 95% CI (pp) | Holm-adjusted p |
|---|---|---:|---:|---:|
| L_BASE | gnn_PPO − mlp_PPO | -1.8848 | [-2.1794, -1.5902] | 1.5916e-10 |
| L_BASE | gnn_PPO − scenario_minimax | -2.2929 | [-2.5324, -2.0534] | 2.4611e-13 |
| H_BASE | C_frozen − scenario_minimax | -1.9362 | [-2.2395, -1.6328] | 1.5916e-10 |
| L_BASE | C_frozen − scenario_minimax | -0.5654 | [-0.7653, -0.3656] | 1.0588e-05 |
| H_BASE | gnn_PPO − C_frozen | +1.6220 | [+1.3501, +1.8940] | 2.6468e-10 |
| L_BASE | gnn_PPO − C_frozen | -1.7275 | [-1.9841, -1.4708] | 9.8819e-11 |
| H_BASE | gnn_PPO − greedy | +2.0493 | [+1.8094, +2.2892] | 1.697e-12 |
| L_BASE | gnn_PPO − greedy | -1.9077 | [-2.1966, -1.6189] | 1.1457e-10 |

**The important reversal:** The hierarchical-test GNN advantage over MLP (+2.0454 pp) becomes a **−1.8848 pp disadvantage on unseen corridor-ladder networks**, and it also falls **−2.2929 pp below minimax** on ladder. For both ladder comparisons, 20/20 topology clusters favour the competitor. GNN also loses to greedy and the previous Enhanced PPO on ladder despite beating both on hierarchical; this is a strong *topology-family transfer limitation* for the frozen trained policy, not evidence that GNN-based approaches generally cannot transfer.

## 5. Robustness and candidate diversity

The bandwidth stress is `ceil(1.5 × bandwidth)` with unchanged physical capacity, and `short_hold` replaces each holding time by `max(1,floor(original_duration/2))`. The K=2 condition truncates the protected pair inventory and pads the neural encoder only. These operations are locked by the Stage 4 semantic signature and independently checked through input SHA-256 relationships.

The **12 stress-minus-base topology-level difference-in-differences** are explicitly **exploratory** (no multiplicity adjustment). The especially relevant differences in GNN–MLP margin are:

| Robustness contrast | Change in GNN–MLP margin (pp) | 95% nominal CI (pp) |
|---|---:|---:|
| H_BW_minus_H_BASE | -0.203 | [-0.673, +0.267] |
| H_HOLD_minus_H_BASE | -0.721 | [-0.948, -0.493] |
| H_K2_minus_H_BASE | -0.630 | [-0.985, -0.275] |
| L_BW_minus_L_BASE | +0.059 | [-0.313, +0.430] |
| L_HOLD_minus_L_BASE | +0.523 | [+0.190, +0.856] |
| L_K2_minus_L_BASE | +0.405 | [+0.040, +0.771] |

The short-hold and K2 changes reduce GNN relative advantage within hierarchical networks; shorter holds and K2 partially lessen the negative ladder transfer difference, but do **not** reverse the ladder/base finding. These exploratory effects are not independently family-wise-confirmatory.

## 6. Route decisions and computational overhead

The **MLP trained policy behaves almost exactly like greedy** despite being trained with a matched interaction/update budget: across the eight conditions it overrides greedy only **159** times in **726,083** discretionary choices, and **347 of 480** entire MLP decision traces match greedy exactly. The GNN overrides greedy **331,968** times in **721,774** discretionary choices. The neural comparison is parameter and interaction matched, but a largely greedy MLP is an important limitation when attributing gains specifically to graph message passing. The two architectures also have different graph information and inductive biases.

| Base family | GNN mean policy time / 2,000 requests | MLP | Online minimax | Greedy | GNN/minimax time ratio |
|---|---:|---:|---:|---:|---:|
| Hierarchical | 20.066 s | 10.528 s | 10.931 s | 3.554 s | 1.84× |
| Ladder | 20.520 s | 12.593 s | 12.298 s | 3.977 s | 1.67× |

These are **Windows simulation policy elapsed times**, not measured network protection switching/recovery delays. Candidate-generation time is shared and recorded separately; no inference should be drawn about operational railway latency.

## 7. Integrity restrictions and research interpretation

1. **The final test is finished, not a development set.** Preserve all nine uploaded ZIPs, frozen checkpoints, code, and report files. Do not retrain, reselect, tune, remove negative cohorts, or rerun with changed conditions on the already accessed final seeds. Any post hoc analysis should be explicitly labeled exploratory, and new confirmatory claims would require a separately reserved dataset.
2. **No operational protection claim.** The model is a synthetic end-to-end working/backup link-disjoint pair under a single-physical-link-fault scenario-reservation surrogate, **not IEEE/RFC MPLS-TP MSRP** and **not a real Irish Rail physical network**. It does not evaluate recovery time, packet loss, or operational failover.
3. **Sampling scope of independent oracle audit.** All archived per-request traces are checksummed and their aggregated values independently reconstructed; an earlier separate physical-fault oracle checks a fixed preselected subset of eight out of 480 cohorts across all six policies. Complete final-world rerunning and external empirical validation are not claimed.
4. **Statistical generalization boundary.** The 20 topology seeds create distinct weighted graph adjacencies within each synthetic family, but they remain realizations of one generator per family, not draws from all possible railway infrastructure designs. Report the stated inference as conditional on these synthetic topology-generating families and traffic generator.
5. **Supervisor sign-off remains outstanding.** Separate academic review is still needed for claiming the surrogate is appropriate to the dissertation’s intended MPLS-TP protection scope. Source code proof, statistical significance and synthetic results do not replace standards or supervisor verification.

## 8. Suggested Phase 6 dissertation conclusion (evidence-aligned)

The frozen graph message-passing PPO showed a positive **within-family** held-out gain against matched non-graph PPO and simpler greedy/previous-PPO controls. However, the separately frozen **online scenario-aware minimax** achieved slightly higher priority-weighted admission on hierarchical networks, with a Holm-adjusted paired result favouring the optimizer. Moreover, graph-PPO performance transferred poorly to an independently generated corridor-ladder family, significantly underperforming matched MLP, prior PPO, greedy and minimax on the ladder/base condition. Across all eight operating conditions the strongest mean-admission policy was scenario-aware minimax. Accordingly, the evidence supports the **conditional value of graph representations for the training-family distribution** and shows **limited cross-family generalization** of the selected GNN policy, rather than a universal deep-RL superiority claim.

## Evidence deliverables

- `Phase5G_Stage4_Final_Independent_Audit_Results.json`: independent check counts, all comparisons and source archive SHA-256 fingerprints.
- `Phase5G_Stage4_Final_Independent_Policy_Summary.csv`: 48 verified cell×policy means, admission counts, times, overrides.
- `Phase5G_Stage4_Final_Independent_Statistical_Comparisons.csv`: independently reconstructed primary, Holm-adjusted secondary and unadjusted exploratory contrasts.
- `Phase5G_Stage4_Final_Oracle_Sample_Attestation.json`: independently regenerated scenario fingerprints and full independent single-link-failure replay for 48 policy runs, one predetermined cohort per condition.
- `audit_phase5g_stage4_final_independent.py` and `audit_phase5g_stage4_oracle_sample.py`: replayable audit code. The scripts need the original uploaded input ZIPs and the source bundle present at their stated filesystem paths.
