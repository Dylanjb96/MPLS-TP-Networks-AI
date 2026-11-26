# Phase 5F Stage 5 — Independent Held-Out Evidence Audit

**Audit status:** COMPLETE — supplied Stage 5 evidence is internally consistent; the reported primary improvement of Enhanced PPO is reproduced from the frozen outcome records.  
**Evidence reviewed:** `Phase5F_Stage5_Final_Evidence.zip`, SHA-256 `b98fa23506874e5e7d0405d0c206a231d0db6d3b6cdf902f488c238a53eb6b1f`  
**Research boundary:** a **synthetic** railway-inspired network and an **end-to-end link-disjoint single-link-failure protection surrogate**, **not** IEEE/MPLS-TP Shared Ring Protection (MSRP), an operational safety evaluation, or the actual Iarnród Éireann network.

## 1. Conclusion

The **preselected Enhanced PPO (configuration C, agent seed 27, episode 60)** achieved a **small but statistically supported improvement in priority-weighted LSP admission** on the **predeclared 500-node, K=6, dynamic traffic, load-4.0 primary test**:

- **Enhanced PPO C:** **88.0952%** weighted admission; **99,349 / 120,000** requests admitted (82.7908% unweighted).
- **Greedy:** **87.7692%** weighted admission; **98,897 / 120,000** admitted (82.4142% unweighted).
- **Paired effect:** **+0.3260 percentage points (pp)**, computed as the mean of **20 topology-level paired differences**, each averaged over the **three common traffic streams**.
- **95% Student-t confidence interval:** **[+0.1221, +0.5299] pp**; **two-sided p = 0.003391** (19 degrees of freedom).
- **10,000-resample topology-cluster bootstrap 95% interval:** **[+0.1469, +0.5184] pp**.
- C had positive mean differences for **16/20 topology clusters** (4 negative), and improved upon greedy in **42/60 individual matched topology–stream cases**.
- The **452 additional admitted requests** across the 60 primary scenarios are an *unweighted count*, complementary to (but not interchangeable with) the weighted percentage-point improvement.

**Crucial qualification:** the originally proposed **Full Hybrid PPO (configuration D, seed 7)** did **not** show an improvement over greedy on the primary held-out experiment. The supported result belongs to **Enhanced PPO without explicit greedy guidance**, not to the full greedy-guided hybrid.

## 2. Preregistered primary and secondary contrasts

All effects are **topology-clustered mean paired differences in weighted admission**, in **percentage points**, at **500 nodes / K=6 / dynamic traffic / load 4.0**.

| Predeclared contrast | Effect (pp) | Student-t 95% CI (pp) | Two-sided p | Holm-adjusted p (secondary only) | Reading |
|---|---:|---:|---:|---:|---|
| **Primary: C − Greedy** | **+0.3260** | **[+0.1221, +0.5299]** | **0.003391** | N/A | Positive primary result |
| Secondary: C − Footprint | +0.2718 | [+0.0569, +0.4868] | 0.015915 | **0.031831** | Positive after adjustment |
| Secondary: D − Greedy | −0.0126 | [−0.1572, +0.1320] | 0.857016 | 0.857016 | No supported improvement |
| Secondary: C − D | +0.3386 | [+0.1201, +0.5571] | 0.004273 | **0.012818** | C beats D after adjustment |

The primary test was specified separately; the **three secondary comparisons** were adjusted together by **Holm's step-down method**. The experimental unit for the reported confidence intervals and p-values is the **topology seed (n=20)**, *not* 60 mutually independent demand streams or 120,000 independent requests.

### Five-policy primary-cell descriptives

| Policy | Weighted admission | Ordinary admission | Admitted / 120,000 |
|---|---:|---:|---:|
| Greedy | 87.7692% | 82.4142% | 98,897 |
| Capacity-footprint | 87.8234% | 82.4633% | 98,956 |
| Random feasible | 84.5036% | 78.5792% | 94,295 |
| **C — Enhanced PPO** | **88.0952%** | **82.7908%** | **99,349** |
| D — Full Hybrid PPO | 87.7566% | 82.4017% | 98,882 |

## 3. C versus greedy across all eight predeclared factorial cells

All 480 matched cohorts comprise **20 topologies × 3 traffic streams × 8 cells**. Each cohort contains **five policy outcomes**, resulting in **2,400 policy evaluations**. Each cell below averages 60 topology–stream runs and is **descriptive**, not an independently confirmatory test.

| Nodes | Candidates | Traffic | Greedy weighted | Footprint weighted | C weighted | D weighted | C − Greedy (pp) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 60 | 2 | Static | 74.7779% | 74.3341% | 74.8138% | 74.7779% | +0.0359 |
| 60 | 2 | Dynamic | 92.2594% | 92.2479% | 92.1215% | 92.2594% | −0.1379 |
| 60 | 6 | Static | 77.3423% | 74.8734% | 77.5403% | 77.5136% | +0.1980 |
| 60 | 6 | Dynamic | 93.4405% | 92.7053% | 93.8253% | 93.4784% | +0.3847 |
| 500 | 2 | Static | 59.1966% | 59.5851% | 59.2679% | 59.2158% | +0.0713 |
| 500 | 2 | Dynamic | 83.0399% | 83.1749% | 83.0054% | 83.0009% | −0.0344 |
| **500** | **6** | **Dynamic** | **87.7692%** | **87.8234%** | **88.0952%** | **87.7566%** | **+0.3260** |
| 500 | 6 | Static | 62.4465% | 62.5494% | 62.9638% | 62.4976% | +0.5173 |

The selected C model's mean weighted admission exceeded greedy in **6/8 cells** and was worse in **2/8**; these are not eight statistically independent confirmatory demonstrations.

## 4. Factorial contrast: what A, B and C actually explain

The preregistered 2×2×2 contrasts concern the **C-minus-greedy advantage**, rather than absolute admission of either algorithm. All seven were tested on topology-level contrasts and their p-values were jointly Holm-adjusted.

| Contrast (C minus greedy advantage) | Effect (pp) | Holm-adjusted p | Interpretation |
|---|---:|---:|---|
| A: 500 versus 60 nodes | +0.0999 | 1.0000 | No adjusted evidence |
| **B: six versus two candidates** | **+0.3728** | **0.000172** | **Supported increase in C's relative advantage** |
| C: dynamic versus static traffic | −0.0710 | 1.0000 | No adjusted evidence |
| A×B | +0.0304 | 1.0000 | No adjusted evidence |
| A×C | −0.0775 | 1.0000 | No adjusted evidence |
| B×C | +0.0688 | 1.0000 | No adjusted evidence |
| A×B×C | −0.1115 | 0.3187 | No adjusted evidence |

Across the eight cells, C's average advantage was approximately **−0.0163 pp at K=2** versus **+0.3565 pp at K=6**. Hence the difference is **+0.3728 pp**. This does **not** establish that K=6 produces *only* +0.3728 pp in absolute admission; it quantifies the **relative C-versus-greedy gain** when increasing candidate count. A causal attribution to a particular neural feature or to network scale alone would overstate what this factorial result identifies.

## 5. Why the chosen model differs from the proposed hybrid

At the primary cell:

- **C:** **7,958 overrides** across **88,143 discretionary decisions** (**9.03%** override rate).
- **D:** **202 overrides** across **87,541 discretionary decisions** (**0.23%** override rate).
- C's and D's policy trajectories differ, so their counts of discretionary decisions are **not identical denominators**.
- C gained 452 admissions compared with greedy, while D admitted **15 fewer**. Do **not** treat the number of overrides as the number of improved individual decisions: routing has downstream effects on future requests.

Both primary RL models have a full 2,000-request decision trace per matched cohort (**60 cohorts × 2 RL models = 120 traces = 240,000 individual records**). Their admission, weighted-priority admission, feasibility counts, cumulative decisions and overrides were checked independently against the cohort summaries; **no discrepancies** were identified. All rejected requests in the recorded RL primary traces had zero currently feasible protected candidates. This is an audit of the **recorded behavior**, not a full independent reconstruction of physical-link capacity from original request and path data.

### Computation cost

Average measured **simulation-phase** time per primary cohort was about **3.47 s for greedy**, **3.80 s for footprint**, **10.64 s for C**, and **10.57 s for D**. C therefore required approximately **3.07× the greedy simulation-phase time** in this Windows run. Candidate generation was measured separately; this ratio **must not** be described as the full end-to-end ratio or as physical restoration/switching latency.

## 6. Evidence integrity: independent calculations

The received 20 MB ZIP contains **487 files**: seven top-level report/lock files and **480 compressed scenario-cohort files**.

**A total of 123,776 independent machine-checked assertions** were evaluated, with **zero failures**. Checks covered:

- **Complete factorial grid:** 2 sizes × 2 route limits × 2 traffic modes × 20 held-out topologies × 3 held-out streams × 5 policies = **2,400 unique records**; no missing/duplicate cells.
- **Matched comparisons:** five policies in each cohort share identical network, demand-stream and candidate-inventory fingerprints; the request fingerprint also matches across K=2 and K=6 within the same size/mode/topology/stream.
- **Topology diversity:** all **20 full network fingerprints are distinct** for each size (60 and 500), consistent with distinct generated routing instances.
- **Arithmetic and bounds:** admitted + rejected = requests; ordinary admission recomputed; feasible candidate totals and discretionary/override counts bounded; active_end = admitted − departures; 0 reported feasibility violations; recorded peak scenario utilization ≤ 1.
- **Compressed cohort provenance:** all **480** JSON-gz cohorts match **all 2,400 report rows** field by field, including the per-policy final scenario-oracle digest and accepted/departed counts.
- **Evidence hashing:** all five report-file SHA-256 hashes match the finalized manifest; the lock identity and both selected checkpoint digest declarations agree with the frozen plan.
- **Companion frozen package:** the six Stage 5 source-file hashes and both actual frozen PyTorch checkpoint SHA-256 hashes in the previously supplied Stage 5 add-on independently match the plan. The Stage 4 source-file hashes are declarations in the held-out archive; full local source reproduction is not being claimed here.
- **Statistics:** independently recomputed all **40 policy×cell descriptive rows**, four primary/secondary t-statistics/95% intervals/bootstrap intervals and Holm corrections, seven factorial effects and Holm corrections, and all **480 topology-level paired-delta rows**.
- **Primary RL traces:** verified **240,000 request-level records** across the 120 primary RL traces; they reconcile to outcomes and weight-based admission, with no invalid recorded actions or unforced rejection.

The original Stage 5 statistical CSV export failed after running the cohorts, and the provided reporting-only recovery was used. The **recovered outputs are now complete and internally consistent**. A preserved pre-recovery copy of all cohort binaries was not supplied, so a byte-for-byte before/after comparison of the cohort files is not independently possible; it is **not necessary to rerun the held-out experiment** for this report.

The archive declares that no training or validation seeds were mixed into the final grid, and its topology/stream indices comply with the reserved partitions. These are evidence-based checks, **not a claim to have independently monitored the user's computer**.

## 7. Statistical and engineering limitations

1. **Small absolute effect:** +0.3260 pp weighted admission is statistically supported for the prespecified synthetic primary condition, but its operational practical significance remains unknown. Present both effect size and computational overhead.
2. **Synthetic modeling:** No real Irish Rail topology/capacities, physical-fault observations, operational signalling workload, protection-switching latency, or MSRP standards compliance has been established. The surrogate was marked **not yet supervisor-approved**.
3. **Correlation and scope:** the 60 primary streams reuse **three stream seeds across 20 topologies**. Topology-clustered inference is preferable to treating 60 runs as independent; uncertainty about completely new traffic distributions is not fully captured by only three held-out stream realizations.
4. **No full simulator replay:** this audit verifies supplied cohort records, weighted priority decisions, both frozen checkpoint hashes in the previously distributed package and aggregate statistics. It does **not** independently rerun all 480 final scenarios, regenerate all underlying paths, or rederive the protection oracle from raw per-link reservations. Reported `feasibility_violations=0` is supported by consistent records, not an independent hardware or standards-compliance proof.
5. **Selection matters:** C was selected using development validation; the final holdout was only used for the above test. Do not change the PPO architecture, intervention gate, threshold, seed ranges or chosen primary cell after reading these outcomes; avoid treating post-hoc selection among ablation cells as confirmatory evidence.
6. **Not directly comparable with old Phase 5E:** Phase 5F changed topology generation, route candidates, traffic model and protection abstraction. Differences relative to Phase 5E cannot be attributed exclusively to the algorithm.

## 8. Freeze and dissertation integration

**Recommended research disposition:** mark Phase 5F Stage 5 as **complete and frozen**. Preserve the original ZIP and SHA-256, the two selected checkpoints, the locked plan, all 480 cohort files, reports and the Stage 4 validation-selection record. **Do not retrain/reselect based on the held-out results, and do not rerun final seeds to search for better results.**

In the dissertation, make the distinction explicit:

> Under a predeclared, 500-node synthetic end-to-end link-disjoint single-link-failure provisioning surrogate, validation-selected Enhanced PPO achieved a statistically supported but modest increase in topology-clustered priority-weighted admission over greedy (+0.3260 percentage points; 95% t confidence interval +0.1221 to +0.5299; p=0.00339). The proposed greedy-guided full hybrid did not improve on greedy (−0.0126 percentage points; p=0.857). The result concerns this synthetic surrogate, not a deployed railway or standards-compliant MSRP implementation.

**Next stage:** Phase 5F Stage 6 — results chapter, methodological qualification, limitations, computational cost analysis, and final dissertation integration. No further held-out tuning is scientifically appropriate.
