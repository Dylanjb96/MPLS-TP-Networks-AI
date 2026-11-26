# Phase 5F Stage 5 — Build and development verification

**Package version:** `PHASE5F_STAGE5_LOCKED_HELDOUT_V1`.

**Local development environment:** CPython 3.13.5; NetworkX 3.6.1; NumPy 2.3.5; PyTorch 2.10.0+cpu; SciPy 1.17.0. The user Windows environment previously confirmed Python 3.11.9 / NetworkX 3.5 / NumPy 2.2.6 / PyTorch 2.7.1+cu118. **Stage 5 Windows cross-environment confirmation remains pending.**

**Completed checks:** Stage 4 frozen selection file and both exact checkpoint SHA-256 values verified. Both PyTorch actor checkpoints loaded with `torch.load(...,weights_only=True)`, their source/code/actor dimensions checked, and strict state_dict compatibility confirmed. The full Stage 5 preflight passed with exactly eight fixed cells, twenty topologies, three streams, five policies, 480 matched cohorts and 2,400 policy episodes.

**Development-only integration:** `run_phase5f_stage5.py devcheck --out ...` passed on already-used **60-node topology seed 301 / stream seed 72000 / load 2.5 / dynamic / K6**. All five policies executed, and the independent single-fault path-reservation oracle reconciled all traces. Selected C weighted admission **0.986111**; greedy weighted admission **0.988095** on that single known development cohort. These numbers are **not** Stage 5 held-out results and are not evidence of population performance.

**Automated tests:** 21 Stage 5 tests passed; a combined Stage 1–5 add-on test suite passed **77/77**. Tests cover source/checkpoint integrity, model loading, policy masking, development single-fault oracle, forced rejections, trace modification detection, experimental grid completeness, cluster-level statistics, Holm correction, factorial contrasts, one-time execution safeguards, and refusal to overwrite completed outputs.

**Critical boundary:** No protected network topology with seeds 401–420 and no demand stream with seeds 73000–73002 was generated, simulated or otherwise evaluated in preparing this package. Some automated statistics fixtures use these *integers as artificial labels only*, never to create a graph or demand. The entire true final evaluation remains **not executed** and should be opened only once by the user after accepting the frozen protocol.

**Deferred:** The actual 2,400 full-policy Stage 5 episodes, their cross-environment replay, topology-clustered inference and dissertation conclusions. Do not infer actual Irish Rail or MSRP performance from the surrogate.
