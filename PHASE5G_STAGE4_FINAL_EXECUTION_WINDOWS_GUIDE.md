# Phase 5G Stage 4 — Authorized final execution (Windows)

**Status:** The Stage 4 development-evaluator evidence passed an independent technical audit. The separately supplied **`stage4_AUDIT_RELEASE.json`** authorizes *only* the original, exactly hashed Stage 4 source package and Stage 3 freeze. **Do not use the release unless all local Windows tests/preflight below pass.** Only one prospectively frozen final seed grid is permitted; do not tune/retrain/change source or model selection using the final results.

## Step 1 — Place the release file in your project root

Download `stage4_AUDIT_RELEASE.json` and place it immediately beside `run_phase5g_stage4.py`:

```text
C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI\stage4_AUDIT_RELEASE.json
```

Do **not** replace any `phase5g_stage4/*.py`, `phase5g_stage2/*.py`, Stage 3 frozen files or older training outputs. If the evaluator rejects the release, the correct action is to **stop and report the discrepancy**, not to edit hashes.

## Step 2 — Windows verification gate (mandatory)

In Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python -m pytest -q tests/test_phase5g_stage4.py
python -m pytest -q
python run_phase5g_stage3.py verify --root outputs\phase5g\stage3_FROZEN
python run_phase5g_stage4.py preflight
```

Expect **22 Stage 4 tests**, **338 total tests if the previous 316 are unchanged**, `PHASE5G_STAGE3_VERIFY_OK` and `PHASE5G_STAGE4_PREFLIGHT_OK`. If any fail, stop and send the full output. Preserve your Windows verification transcript with the evidence.

## Step 3 — Run the complete prespecified held-out evaluation exactly once

Read `PHASE5G_STAGE3_PROTOCOL_LOCK.md` and the independent Stage 4 audit before proceeding. When all preceding checks pass, execute:

```cmd
python run_phase5g_stage4.py final --out outputs\phase5g\stage4_final --confirm-final --audit-release stage4_AUDIT_RELEASE.json
```

The command evaluates exactly the locked **8 × 20 × 3 × 6 = 2,880** cases and saves each **six-policy cohort** as a separate atomic, checksummed compressed JSON file. It prints only one cohort-completion line at a time; **do not examine interim scores and retune anything**. Checkpoints, source and protocol are locked at the top of every invocation. Do not copy different checkpoint files into the frozen folder or restart using different seeds/conditions.

**If the computer or process stops:** retain the existing folder and rerun the **identical command** with `--resume`:

```cmd
python run_phase5g_stage4.py final --out outputs\phase5g\stage4_final --confirm-final --audit-release stage4_AUDIT_RELEASE.json --resume
```

The program re-verifies saved signatures and reads any existing whole-cohort outputs. It refuses to overwrite an existing final experiment without `--resume`, rejects wrong source/protocol/software settings on resume, and checks completed cohort checksums. **Never delete results or change the experiment after viewing intermediate results.** If the program detects a corrupt cohort, stop and preserve the original file for audit rather than silently replacing it.

Expected completion message:

```text
PHASE5G_STAGE4_FINAL_COMPLETE cohorts=480 rows=2880 role=final final_data_accessed=true
PHASE5G_STAGE4_FINAL_RESULT_READY execute report-final using the same audit release
```

## Step 4 — Compute the predeclared statistical analysis

Only after all 480 cohorts finish:

```cmd
python run_phase5g_stage4.py report-final --root outputs\phase5g\stage4_final --out outputs\phase5g\stage4_final_report --audit-release stage4_AUDIT_RELEASE.json
```

Expected:

```text
PHASE5G_STAGE4_ANALYSIS_COMPLETE cohorts=480 policy_rows=2880 primary_tests=2 secondary_tests=8 statistical_unit=20_topologies
```

Your output folder contains `complete_policy_rows.csv`, `primary_cluster_inference.csv`, `secondary_cluster_inference.csv`, `exploratory_robustness.csv`, `policy_summary.csv`, and `report_manifest.json`. Confidence intervals account for 20 paired topology clusters. The two co-primary tests and eight secondary tests have **separate Holm correction families**; exploratory interactions have no confirmatory significance status. Never interpret a selected Stage 2 validation score or single development cohort as a final result.

## Step 5 — Preserve and upload evidence

Preserve the complete `stage4_final` directory and the original source/freeze ZIPs unchanged. The final data comprise a substantial number of archived request-level traces; upload the **compact final reporting archive** first:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'outputs\phase5g\stage4_final\run_manifest.json','outputs\phase5g\stage4_final_report\*','stage4_AUDIT_RELEASE.json' -DestinationPath 'Phase5G_Stage4_Final_Reports.zip' -Force"
```

Also preserve full cohort evidence. For manageable transfers, package **one experimental cell per ZIP**. Paste this single command into Command Prompt (PowerShell runs inside it):

```cmd
powershell -NoProfile -Command "$cells=@('H_BASE','L_BASE','H_BW','L_BW','H_HOLD','L_HOLD','H_K2','L_K2'); foreach($cell in $cells){ $files=Get-ChildItem -LiteralPath 'outputs\phase5g\stage4_final\cohorts' -Filter ($cell+'__*.json.gz'); if($files.Count -ne 60){throw ('Incomplete cell '+$cell)}; Compress-Archive -LiteralPath $files.FullName -DestinationPath ('Phase5G_Stage4_'+$cell+'_Cohorts.zip') -Force }"
```

This produces **eight cell ZIPs, 60 cohort artifacts each**. Upload `Phase5G_Stage4_Final_Reports.zip` plus the cell ZIPs (multiple messages are fine). Do not omit any unfavourable policy or experimental cell. Save the original CMD output for the tests, preflight, completed final evaluation and statistical report.

## Scientific boundaries

These are *synthetic* scenario-aware end-to-end protected-pair admission experiments: **not** a standards-compliant MPLS-TP MSRP implementation, a real Irish Rail topology or measurements of physical restoration latency. The protection abstraction still requires supervisor acceptance. Results may show the neural networks improve, tie or underperform the strongest conventional optimizer; report all outcomes without tuning on held-out data.
