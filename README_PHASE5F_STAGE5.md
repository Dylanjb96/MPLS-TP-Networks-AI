# Phase 5F Stage 5 — Final locked evaluation add-on

**START HERE:** `PHASE5F_STAGE5_START_TO_FINISH_WINDOWS_GUIDE.md` and `PHASE5F_STAGE5_PROTOCOL_LOCK.md`.

Install by extracting this ZIP over the **already working Stage 4 MPLS-AI project**. This release adds **only new Stage 5 files**. It bundles the **official frozen Stage 4 selection**, `C_enhanced` seed27 checkpoint and `D_full_hybrid` seed7 checkpoint. It does not replace or modify old source files, training results or models.

Commands (from project root):

```cmd
python -m pytest -q tests/test_phase5f_stage5.py
python -m pytest -q
python run_phase5f_stage5.py preflight
python run_phase5f_stage5.py devcheck --out outputs\phase5f\stage5_devcheck
```

**Only after reviewing the protocol**, run heldout exactly once:

```cmd
python run_phase5f_stage5.py final --out outputs\phase5f\stage5_final --confirm-final
```

If interrupted, `final --out outputs\phase5f\stage5_final --resume --confirm-final`. After full completion, `verify --root outputs\phase5f\stage5_final`.

Protection = **synthetic end-to-end link-disjoint single-link-failure surrogate, not MSRP or deployed Irish Rail system**. No training or threshold adjustment is permitted after frozen checkpoint selection.
