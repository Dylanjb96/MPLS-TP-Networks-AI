# Phase 2: six recommended improvements - evidence summary

**Scope:** Reproducible synthetic railway-like **physical topology** study only. No MPLS-TP shared-ring protection switching, LSP bandwidth admission, loss, recovery latency or operational performance is simulated.

## 1. Multiple seeds and topology sensitivity

- 15 60-node configurations tested (5 seeds x 3 explicit ring layouts); 15/15 passed ring-cycle and individual-link graph connectivity checks.

- Seed alone primarily rotates gateway positions; it does **not** provide genuinely independent traffic/network structures. Shape variations vary main-ring size and sub-ring count.

- Sources: `01_topology_sweep.csv`, `01_sweep_summary.csv`, `01_seed_fingerprint_counts.png`.

## 2. Single vs dual gateway comparisons

- 12 nodes: single gateway 1/12 node removals disconnect; dual gateway 0/12. Articulation points: 1 single vs 0 dual.

- 60 nodes: single gateway 4/60 node removals disconnect; dual gateway 0/60. Articulation points: 4 single vs 0 dual.

- This is a structural single-point-of-failure diagnostic, consistent with RFC 8227 §4.4; **not** proof of protocol recovery.

- Sources: `02_gateway_comparison.csv`, `02_node_failure_details.csv`, PNG charts.

## 3. Graph-path characteristics and inter-ring routes

- main-to-main: n=12, mean shortest hop count=1.667, mean gateway transits=0, local edge connectivity >= 2.

- within-subring: n=35, mean shortest hop count=3.457, mean gateway transits=0.171, local edge connectivity >= 2.

- main-to-subring: n=35, mean shortest hop count=7.286, mean gateway transits=2.486, local edge connectivity >= 2.

- subring-to-subring: n=35, mean shortest hop count=10.057, mean gateway transits=4.029, local edge connectivity >= 2.

- K-shortest paths and local edge connectivity are graph metrics; neither identifies legal working/protection LSPs under MSRP.

- Sources: `03_route_sample_details.csv`, `03_route_category_summary.csv`, `03_most_common_shortest_path_edges.csv`.

## 4. Modelling assumptions and Phase 3 choices

- Current implementation: one undirected NetworkX edge per physical resource, two ring memberships can share one edge, capacities in Mb/s, no reservations.

- Proposed for Phase 3 (NOT IMPLEMENTED): **independent_per_direction** link-direction capacity accounting, explicit headroom scenarios and pre-defined fault-forwarding semantics.

- No claim of RFC 8227 conformance until a precise ring-tunnel and protection operation is defined and checked.

- Sources: `phase2_assumptions.json`, `04_assumptions_validation.json`.

## 5. Synthetic request distributions

- Three disjoint random seeds for train/validation/test; each has uniform-endpoint, category-balanced and inter-ring-heavy variants. All requests are **unadmitted**; summed Mbps is offered requested bandwidth, not traffic throughput or reserved link load.

- Train examples: uniform endpoint counts main-main=15, sub-sub=239; balanced main-main=125, sub-sub=125.

- No measured railway request distribution is asserted; categories are design parameters for future sensitivity tests.

- Sources: nine `05_demand_*_*.csv` datasets and `05_demand_summaries.csv`.

## 6. Reproducible evidence and audit

- 5/5 original topology manifests had a structural pass and all listed file SHA-256 digests verified in this run.

- Audit does not prove paper approval, source-code correctness outside tested scope, or research reproducibility on other machines.

- Sources: `06_original_manifest_integrity_audit.csv`, `06_environment_and_provenance.json`, project tests.

## Ready-for-Phase-3 gate (not a Phase 3 implementation)

1. Supervisor approves topology abstraction and the gateway interconnection comparison.

2. Explicitly decide per-direction versus aggregate physical-link capacity, and link headroom handling.

3. Pick a protection model: ring-tunnel-specific MSRP or clearly labelled simplified surrogate; set allowed faults and preconfigured failure-state forwarding.

4. Preserve baseline input data, configurations, code commit, and seed splits before Phase 3 admission experiments.

## References

- IETF RFC 8227, section 4.4, interconnected ring protection: https://www.rfc-editor.org/info/rfc8227/

- NetworkX articulation points: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.articulation_points.html

- NetworkX edge connectivity: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.connectivity.connectivity.edge_connectivity.html

