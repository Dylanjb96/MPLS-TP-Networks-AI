# Phase 3 — From beginning to end | protected LSP planning surrogate

**Project title:** Resilience-Aware LSP Provisioning in Interconnected Railway-Like MPLS-TP Networks.  
**Scope:** Offline, entirely synthetic, deterministic preplanned **ring-segment scenario surrogate**. **This is NOT an RFC 8227 MSRP implementation** and does not establish railway safety, operator performance, service recovery milliseconds or real hardware behaviour.

## 1. What is already implemented

- Builds directly on your existing Phase 2 `railway_tp` topology and synthetic service-request generator (the five previously validated configurations remain present).
- Enumerates policy-limited route candidates by stitching arcs **only at a shared main/sub-ring gateway**. Each segment travels on one of the two fixed ring arcs.
- Preplans the route for **every** single physical-link failure. An affected ring segment switches to its opposite arc; an unaffected segment is unchanged; **the gateway sequence cannot change** in this v1 abstraction.
- Rejects any plan that loops, uses a failed link, fails to join legal segments or lacks complete failure-state paths.
- Checks **both service directions** against *separate directed capacity budgets* on each physical edge. Each admitted service consumes `bandwidth_mbps` in each direction (symmetric full-duplex service).
- Applies configurable integer headroom: usable capacity per directed arc = `floor(raw_capacity_mbps * (100 - headroom_percent) / 100)`.
- Audits **normal state and every one-link failure** against all previously admitted services and the prospective service; accepts only when every scenario passes.
- Stores reservations *externally* to the original graph attributes. Failed admission is atomic and leaves committed services unchanged.
- Exports requests, decisions, fully preplanned paths, numerical summary, and SHA-256 manifest.
- Includes full Phase 2 tests and independently testable Phase 3 behaviour, positive/negative fault/capacity fixtures.

## 2. Windows Command Prompt: step-by-step

Extract the package into a **new folder** first (e.g. `MPLS-AI-Phase3`) to avoid overwriting your working Phase 2 installation.

```cmd
cd "C:\Users\<your-windows-user>\Desktop\MPLS AI\MPLS-AI-Phase3"
python -m venv .env
.env\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

If you prefer to add Phase 3 to the old working `MPLS-AI` folder, extract **only** the add-on ZIP over that existing root; the `phase3` folder must sit beside `railway_tp`. Keep a backup before merging.

```cmd
python -m phase3.cli --config config/debug_12_dual.json --out outputs\phase3 --requests 25 --seed 2026 --headroom 10
python -m phase3.cli --config config/primary_60_dual.json --out outputs\phase3 --requests 50 --seed 2026 --headroom 10
python run_phase3_examples.py
```

Optional headroom sensitivity tests (fresh folder, same fixed demand seed):

```cmd
python -m phase3.cli --config config/primary_60_dual.json --out outputs\phase3_h0 --requests 50 --seed 2026 --headroom 0
python -m phase3.cli --config config/primary_60_dual.json --out outputs\phase3_h10 --requests 50 --seed 2026 --headroom 10
python -m phase3.cli --config config/primary_60_dual.json --out outputs\phase3_h20 --requests 50 --seed 2026 --headroom 20
```

## 3. What to inspect in each exported case

For example, files beginning with `phase3_12_dual_s42_d2026_h10` appear under the selected `--out` folder:

| File suffix | What it proves / contains |
|---|---|
| `_requests.csv` | Exact ordered synthetic request inputs (IDs, bandwidth Mb/s, priority) |
| `_decisions.csv` | Admission/rejection, reason code, plan chosen, bottleneck diagnostic |
| `_accepted_protection_plans.json` | Per-service working and precomputed fault-state physical-node paths |
| `_summary.json` | Admission ratios, worst recorded scenario, directed peak load, final-state hash |
| `_manifest.json` | SHA-256 digests of the four exported evidence files |

**Note:** The `phase3` Python code is new; the earlier Phase 2 `railway_tp` module remains the topology source. Phase 2 `reserved_mbps` graph attributes remain zero as a deliberate separation; the Phase 3 ledger is authoritative for this surrogate.

## 4. Tiny reproducible test from Python

```python
from railway_tp.topology import TopologyConfig, make_topology
from railway_tp.requests import ServiceRequest
from phase3 import ProtectionSimulator

G, rings = make_topology(TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=6,subring_count=1))
sim = ProtectionSimulator(G, rings, headroom_percent=10)
req = ServiceRequest('S-001',6,7,200,'operational',3,'within-subring')
candidates = sim.candidates(req)
print('Candidates:',len(candidates))
result = sim.admit(req,candidates[0] if candidates else None)
print(result.as_dict())
print('Ledger:',sim.ledger_summary())
```

## 5. Mandatory Phase 3 acceptance evidence

- All original Phase 2 tests still pass.
- `tests/test_phase3.py` passes, including negative fixtures for over-capacity, invalid fault routes and illegal gateways.
- Each admitted plan has exactly `physical_link_count` failure states and one no-fault path; both forward and exact reverse are evaluated.
- An already-admitted service remains committed after a later rejection; graph reservations unchanged.
- 12-node and 60-node cases run and write all five expected outputs; 100/150-node cases are optional engineering smoke checks.
- Every experimental claim says **"in the synthetic scenario-based surrogate"**, never hardware validation or MSRP compliance.
- Supervisor reviews the provisional `phase3_design_decisions.json` before its use as the final dissertation methodology.

## 6. Model boundaries and research integrity

1. The ring-segment surrogate uses fixed gateway choices; a true dual-node ring protection mechanism might change egress gateway after some faults. This implementation does not.
2. All single physical-link failures are checked independently. Multiple failures and node outages are excluded.
3. No actual RFC 8227 ring tunnels, labels, OAM, RPS, wrap/short-wrap/steering state machines or switching latency are implemented.
4. No live topology, real traffic measurements, vendor management behaviour, packet-level loss, queueing or safety qualification are included.
5. Rate budget is decimal integer Mb/s. A service of rate `b` reserves `b` in *each* direction. No real CIR/CBS policing model is claimed.
6. The demonstration policy is first feasible candidate in hop-based order; it is **not** a CSPF/MILP/DDQN comparison and cannot support conclusions about learned routing superiority.
7. Request admission is persistent until the episode ends; no release, preemption or reconfiguration.

## 7. Exact boundary of Phase 3 vs Phases 4–6

Phase 3 establishes the **common environment, policy-valid candidates, mandatory capacity protection checker, test fixtures, and reproducible request episodes**. Phase 4 then adds independent greedy/CSPF/MILP choice algorithms; Phase 5 adds constrained DDQN using the same checker; Phase 6 measures repeated-seed performance and confidence intervals.

Run the same 60-node/50-request workload at all three headroom levels with one command:

```cmd
python run_phase3_sensitivity.py
```

This creates exploratory single-seed evidence under `outputs\phase3_sensitivity`; it is not a substitute for repeated-seed statistics in Phase 6.