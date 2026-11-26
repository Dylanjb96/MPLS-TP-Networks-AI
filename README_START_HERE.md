# Phase 5G Stage 2 — Source-history audit kit (read-only)

Your six runs are complete according to their saved metadata, and the primary frozen model hash is consistent, but the **training source snapshots are not the same**. The old `phase5g_stage2/__init__.py` and `workflow.py` hashes exactly match the original Stage 2 package. The user's current source differs (not a newline conversion). The audit kit supplies the original code **for read-only comparison**; do not copy it into your project source directories.

Extract the ZIP into the top level of `MPLS-AI`, producing `audit_phase5g_stage2_source_drift.py` and a folder `original_reference`. In Command Prompt:

```cmd
cd "C:\Users\Dylan B\Desktop\MPLS AI\MPLS-AI"
python audit_phase5g_stage2_source_drift.py
```

Upload **`Phase5G_Stage2_Source_Drift_Evidence.zip`** from the project directory. The report shows exact source differences and committed Git historical snapshots, when available. The archive contains **only** Phase 5G Stage 2 source code and `training_provenance.json` for the six runs, not model weights or held-out data. No sources or training results are changed.

DO NOT try to fix the problem by copying the old sources over the current ones or editing provenance hashes. We must determine whether source changes affected training before accepting a six-arm factorial model selection. If the changes were behavioral, the statistically fair corrective action may require retraining the affected runs under one frozen source version and documenting the original set as exploratory.
