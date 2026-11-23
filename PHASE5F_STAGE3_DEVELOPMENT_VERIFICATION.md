# Phase 5F — Stage 3 Engineering Verification

## Built and locally tested

An independent Stage 3 development package was built on top of the exact previously distributed Stage 1 and Stage 2 source ZIP contents. It introduces `phase5f_stage3/` with a fixed-width six-action candidate encoder, candidate-shared actor MLP, state-value critic, masked categorical action distribution, on-policy PPO clipping, optional greedy behaviour-cloning penalty, sequential episode simulator integration, training smoke, validation comparisons, manifests and decision traces.

**Local execution:** Python 3.13.5, NetworkX 3.6.1, NumPy 2.3.5, PyTorch 2.10.0+cpu. The user has Python 3.11.9 / NetworkX 3.5 / NumPy 2.2.6 / PyTorch 2.7.1+cu118; the exact Windows tests remain to be run.

Verification performed:

- `python -m pytest -q tests/test_phase5f_stage3.py`: **13 passed**.
- `python -m pytest -q tests/test_phase5f.py tests/test_phase5f_stage2.py tests/test_phase5f_stage3.py`: **38 passed** (13 Stage 1 + 12 Stage 2 + 13 Stage 3).
- `python run_phase5f_stage3.py preflight`: **passed**; 60-node sample contains six structural and six currently feasible protected routes.
- Four-episode 60-node smoke with training topology seeds 201/202, training demand seeds 71000/71001, validation topology seed 301, validation demand seed 72000, load 2.5, K=6, static+dynamic traffic: **completed**. Training records show **473 stochastic discretionary actions** and validation output contains six matched policy-by-stream rows (greedy, footprint and hybrid × two traffic modes).
- Explicit adverse tests: masked infeasible actions, forced-only rejection, action override possibility, no-future-feature exposure when expiry is modified, candidate permutation-equivariance, PPO weight updates, exact ledger recomputation, provenance markers, held-out seed rejection and overwrite refusal.

**This is not evidence of beneficial learning**. The local four-episode run is only an engineering smoke test. It should not be presented as a new evaluation result. No final held-out topology or traffic stream was accessed, no full multi-seed training was performed, and no standardized MSRP or railway deployment assertion is supported.

**Cross-machine status:** Stage 2 physical-topology/path fingerprints were previously reproduced across Windows and Linux for the Stage 2 configurations. **Stage 3 learning results have not been independently reproduced on Windows yet**; numerical identity of trained neural model weights across CPU and GPU is not guaranteed by that earlier check.
