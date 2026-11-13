# Phase 4 — MILP Version 2 (lexicographic selection) add-on

## Purpose and compatibility

This incremental patch is for the *existing* MPLS-AI Phase 4 project. It introduces a **new** algorithm label `milp_v2` and preserves the original algorithm label `milp` (Version 1). It does **not** replace the Phase 3 protection model, the greedy policy, CSPF-style policy, the benchmark request generator, or your archived Phase 4 results.

V1 selected a candidate by minimizing `z + 1e-7 * normalized_working_hops`. V2 follows a mathematically lexicographic objective:

1. **Primary:** minimize `z`, the greatest directed-arc utilization across normal operation and every modeled isolated physical-link failure scenario, subject to the *same* working/failure capacity constraints and previously committed services.
2. **Secondary:** among feasible candidates attaining the exact minimum `z`, choose the fewest normal working-path hops.
3. **Tertiary:** break any remaining tie using the lexicographically smallest `candidate_id`.

V2 solves the primary optimization with the original SciPy MILP formulation, **without** the `1e-7` coefficient. Because only one candidate is selected from a finite generated pool (two per request in your Phase 3 sensitivity experiment), the secondary problem is solved *exactly* by checking every feasible candidate with integer bandwidth/usable-capacity ratios represented as Python `Fraction`s. This finite enumeration is mathematically equivalent to a second discrete optimization over candidate identities, but is **not** a second MILP solver call. It removes ambiguous tiny-weight numerical tie-breaking.

All chosen paths are still independently checked and atomically committed by `phase3.model.ProtectionSimulator.admit`. The model remains `RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP` and does **not** emulate RFC 8227 MSRP signaling or restoration timing.

## Install on Windows (from Command Prompt)

1. Back up your working project folder, including existing `outputs/phase4` results.
2. Open Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
copy phase4\policies.py phase4\policies_before_v2.py
copy phase4\cli.py phase4\cli_before_v2.py
```

3. Extract `MPLS_TP_Phase4_MILP_V2_AddOn.zip` into the root of `MPLS-AI`. When asked, allow replacement of **only** `phase4/policies.py` and `phase4/cli.py`. The ZIP also adds `tests/test_phase4_milp_v2.py` and this README.
4. Ensure the Phase 4 SciPy dependency is installed:

```cmd
python -m pip install -r requirements_phase4.txt
```

## Validate

```cmd
python -m pytest -q tests/test_phase4_milp_v2.py
python -m pytest -q
```

For the matching project snapshot supplied with Phase 4, development-environment results were **9 passed** for the new tests and **131 passed** for the full suite (97 Phase 2/3 + 25 Phase 4 + 9 V2). Your local results must be checked independently, especially if your project has been modified.

The new tests include a **150-node, seed-2029, request-R0005 regression**: both available candidates have an exactly equal primary utilization score (`4/9`) but working paths of 11 and 59 hops. V2 selects the 11-hop path regardless of candidate ordering. This does not pre-judge the admission result of the complete 50-request sequence.

## Run a small smoke benchmark first

```cmd
python run_phase4_benchmark.py --config config/debug_12_dual.json --requests 10 --seeds 2027 --headrooms 10 --algorithms greedy milp milp_v2 --max-candidates 2 --out outputs\phase4\v2_smoke_12
```

The `--algorithms` option explicitly requests both V1 and V2. The *default* remains the original `greedy cspf milp`, so existing scripts remain unchanged unless you opt in to `milp_v2`.

## Re-run the two investigated divergence cases, preserving V1 results

**60-node, 20% headroom, seed 2028:**

```cmd
python run_phase4_benchmark.py --config config/primary_60_dual.json --requests 50 --seeds 2028 --headrooms 20 --algorithms greedy milp milp_v2 --max-candidates 2 --out outputs\phase4\v2_case_60_h20_s2028
```

**150-node, 10% headroom, seed 2029:**

```cmd
python run_phase4_benchmark.py --config config/optional_150_dual.json --requests 50 --seeds 2029 --headrooms 10 --algorithms greedy milp milp_v2 --max-candidates 2 --out outputs\phase4\v2_case_150_h10_s2029
```

These commands write **new directories** and do not overwrite historical benchmark data. The 150-node full comparison is more computationally demanding; the patch was validated on the first five requests and automated regression tests, not claimed as having completed the full 50-request comparison in the development environment.

## Final primary-comparison extension (after case-study checks)

```cmd
python run_phase4_benchmark.py --config config/primary_60_dual.json --requests 100 --seeds 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 --headrooms 10 --algorithms greedy cspf milp milp_v2 --max-candidates 2 --out outputs\phase4\v2_heavy_60_15seeds
```

For a defensible *final* Phase 4 dataset, rerun other relevant headroom and scalability conditions with V2 included, rather than mixing V1 outputs with V2 as if they were the same algorithm. Report the version, solver configuration, and the measured selection-time change. Retain V1 as a transparent implementation-sensitive comparator, and use V2 as the corrected conventional MILP baseline when comparing to DDQN.

## Outputs to share for review

From each new benchmark directory, provide:

- `phase4_runs.csv`
- `phase4_summary.csv`
- `phase4_paired_deltas.csv`
- `phase4_provenance.json` (records the V2 objective definition)

If a case-study admission count remains different, also provide the per-run `decisions.csv` files.

## Interpretation boundaries

- Online algorithms see only the current request and previously accepted requests; no future demand is known.
- Every policy uses the same capped candidate pool and Phase 3 validation; V2 does not expand the routing action space.
- A corrected tie-break *may or may not* change final admission counts. Do not claim an improvement without running the complete paired comparison.
- One topology seed and synthetic demand streams do not establish real-world reliability or railway safety.
- Supervisor approval of the modeling assumptions remains pending.
