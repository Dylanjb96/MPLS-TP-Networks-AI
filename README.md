# MPLS-TP-Networks-AI 
# Complete Phase 2 | Synthetic interconnected railway-like MPLS-TP network

**Phase boundary:** This project finishes **Phase 2: design, generate, document, and structurally validate a reproducible synthetic interconnected-ring topology**, and prepares its inputs for Phase 3. It does **not** claim to have implemented MPLS-TP shared-ring protection (RFC 8227), configured working/protection LSPs, bandwidth admission, 50-ms switching, railway safety, or any Belden XTran / TXCare operational behaviour.

The source `MPLS_TP_Phase_1_Complete_Research_Pack(1).docx` and the accompanying Phase 2–6 roadmap motivate the synthetic 12-node debug and 60-node primary study. All capacities, distances, class rates, priorities, topologies, and counts in this package are **explicit assumptions**, not operator data.

## 1. Installation

Use Python **3.10 or newer** from the project directory.

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m railway_tp.cli --config config/debug_12_dual.json --out outputs
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m railway_tp.cli --config config/debug_12_dual.json --out outputs
```

If your system prevents PowerShell script activation, use `.venv\Scripts\python.exe` rather than changing your system security policy.

## 2. Generate the **complete Phase 2 evidence bundle**

```bash
python run_all_examples.py
```

This rebuilds the 12-node dual-gateway, 12-node single-gateway contrast, 60-node primary, and 100-node scale cases. Optional 150-node run:

```bash
python -m railway_tp.cli --config config/optional_150_dual.json --out outputs
```

A custom topology can be created with `python -m railway_tp.cli --nodes 60 --gateway-mode dual --seed 123 --out outputs`. For traceable academic results prefer a checked-in JSON configuration file.

## 3. Phase 2 delivery checklist: begin to end

| Stage | Action | Evidence / pass criterion |
|---|---|---|
| 0 | Fix scope and units | `config/*.json` contains explicit parameters in Mb/s and km. |
| 1 | Build 12-node debug topology | PNG and JSON generated; 6 main and 6 new sub nodes, two rings. |
| 2 | Define gateway architectures | Dual interconnection as primary; single gateway only as a structural contrast. |
| 3 | Validate ring construction | Every ring-specific physical subgraph is a simple cycle; node and edge membership match. |
| 4 | Validate capacities and links | Each physical edge has one ID and one capacity; multi-ring shared edge not double-counted. |
| 5 | Generate candidate graph paths | Sample main-main, main-sub, within-sub and (on 60 nodes) sub-sub paths; separate ring arcs. |
| 6 | Generate synthetic service arrivals | Seeded request CSV + JSON with class and reserved rate assumptions; no requests yet admitted. |
| 7 | Exhaust single physical-link failures | Check each edge deletion preserves graph connectivity; save failure-case CSV. |
| 8 | Compare interconnection architectures | Node deletion diagnostics show single-gateway point of failure; do not mistake this for a protocol performance test. |
| 9 | Scale to 60 and 100 nodes | Repeat all structural checks, export figures/inventories/manifest. |
| 10 | Re-run automated tests | `python -m pytest -q` returns zero failures. |
| 11 | Freeze evidence and limits | Keep manifests, code version, inputs and assessor/supervisor decisions. |
| 12 | Gate to Phase 3 | Select a *specific* documented protection behaviour and a compatible directed/resource model before implementing admission. |

## 4. Network representations

- Each physical transport node is a node of an undirected NetworkX `Graph`. Numeric node IDs are artificial.
- Main ring: an ordered cycle of `main_nodes` nodes.
- Each sub-ring: its new nodes plus **one** (single) or **two** (dual) gateway nodes, which physically belong to both rings.
- In dual mode a gateway-to-gateway main-ring physical edge also belongs to the sub-ring. There is just **one physical edge** in the graph and one capacity record; *this is not a second parallel capacity resource*.
- Main-ring links start with 10,000 Mb/s; new sub-ring links with 1,000 Mb/s. The shared main/sub link keeps the main-ring capacity. All initial reservations equal zero.
- All links have the same illustrative 2.0 km length; this is **not geography, propagation delay, or a transmission-performance prediction**.
- `seed` rotates the gateway placements around the main ring and fixes the request sequence and plot layout, not an operational traffic distribution.
- `main-to-main`, `main-to-subring`, `within-subring`, and `subring-to-subring` are classifications of endpoints; gateway endpoints are categorized as main-ring endpoints for that purpose.

### Current test graph sizes

| Case | Physical nodes | Physical links | Ring count | Shared physical edges | All single-link graph checks | All single-node graph checks (diagnostic only) |
|---|---:|---:|---:|---:|---|---|
| Debug dual | 12 | 13 | 2 | 1 | Pass (13/13) | Pass (12/12) |
| Contrast single | 12 | 13 | 2 | 0 | Pass (13/13) | **Fail** at gateway (expected) |
| Primary dual | 60 | 64 | 5 | 4 | Pass (64/64) | Pass (60/60) |
| Scale dual | 100 | 106 | 7 | 6 | Pass (106/106) | Pass (100/100) |

These are **computed structural findings for this generator** under this configuration. They do not establish a protected LSP, reserve bandwidth after a fault, or demonstrate compliant MSRP switching.

## 5. Output files explained

For every run a prefix such as `phase2_60_dual_s42_` identifies its variant:

| Suffix | Contents |
|---|---|
| `topology.json` | Graph metadata, ordered ring membership, all nodes and unique physical links. |
| `diagram.png` | Physical schematic, not geographical. |
| `validation.json` | Structural proof checks: ring cycles, memberships, gateways, bridge count and connectivity. |
| `route_examples.json` | K-shortest *global graph paths*, plus two within-ring physical arcs. **Not protected LSPs.** |
| `requests.json` | Reproducible fictional service-request sequence. |
| `nodes.csv` and `links.csv` | Inspectable topology inventory. |
| `ring_inventory.csv` | Ordered node sequences per ring. |
| `link_failure_checks.csv` | One row for each physical edge deletion. |
| `node_failure_DIAGNOSTIC.csv` | Architecture-level node deletion comparison only. |
| `synthetic_demand.csv` | 50 illustrative unadmitted request records. |
| `manifest.json` | Frozen settings, summary, class counts, file SHA-256 digests and gate result. |

## 6. Project structure

```text
MPLS_TP_Phase2_Complete/
  config/                   # versioned 12, 60, 100 and optional 150 cases
  railway_tp/
    topology.py             # graph, rings, gateways, fault diagnostics, validations
    routes.py               # K graph paths, two ring arcs, endpoint categories
    requests.py             # seeded fictional demand (not yet admitted)
    visualize.py            # schematic PNG
    export.py               # JSON/CSV inventories + file hashes
    cli.py                  # one-command evidence generation
  tests/test_phase2.py      # 20 automated checks with parametrized cases
  outputs/                  # generated images, manifests, CSV and JSON
  docs/                     # findings and Phase 3 requirements
  run_all_examples.py       # generates four evidence variants
  requirements.txt
  README.md
```

## 7. Why these checks matter and what they do **not** prove

`nx.is_connected(G)` guarantees there is some graph path between any two nodes; `nx.bridges(G)` finds edges whose deletion disconnects the graph. Exhaustively deleting each link and rechecking connectedness verifies only **physical graph reachability**. Real MPLS-TP service protection depends on configured ring tunnels, failure-detection behaviour, switching semantics, inter-ring stitching, planned resources, capacity and control-plane assumptions; none of those has been implemented here. In particular, *do not describe the K shortest arbitrary graph paths as RFC 8227 compliant protected LSPs*.

The explicit Phase 3 gate is to choose one abstraction: (A) a documented segment-by-segment ring-switching model derived carefully from RFC 8227 (e.g. steering) plus inter-ring gateway behaviour, or (B) a simpler disjoint-path surrogate, named honestly as such. The routing/capacity algorithms must use identical feasible candidates and a common post-fault capacity checker. Avoid claiming RFC 8227 compatibility if using (B).

## 8. How to transition into Phase 3

1. Create a `ProtectionPolicy` specification: permitted fault set, ring direction, inter-ring stitching/gateway failover, dedicated versus shared resource semantics, bidirectional service demands and headroom.
2. Implement `precompute_forwarding(request, plan, fault)` to produce a **defined path** for each specified fault, rather than dynamically calculating any convenient new shortest path after a fault.
3. Implement independent `validate_plan(admitted, candidate)` to check endpoint validity, legal ring segments, and all normal/post-fault capacity constraints (`sum b_i x_i,e,f <= usable_C_e`).
4. Add hand-worked fixtures for failure adjacent to each endpoint, failed shared gateway link, failed sub-ring link, inter-ring request, and insufficient bandwidth.
5. Keep the same graph fixtures and request seeds as the input for subsequent greedy/CSPF/MILP/DDQN comparisons. The Phase 2 package intentionally contains **no policy yet**.

## 9. References / caveats

- IETF RFC 8227 §4.4.1–4.4.4: interconnected single/dual node architectures and ring-tunnel protection behaviour: https://www.rfc-editor.org/info/rfc8227/
- IETF RFC 6372: survivability vocabulary: https://www.rfc-editor.org/info/rfc6372/
- NetworkX: `is_connected` https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.is_connected.html
- NetworkX: `shortest_simple_paths` https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html