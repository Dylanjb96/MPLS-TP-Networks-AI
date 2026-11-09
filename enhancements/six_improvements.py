#Six Phase 2 enhancements to the existing railway_tp network generator.

#Outputs are synthetic, offline GRAPH diagnostics only.  Not MSRP LSP switching,
#capacity admission, railway functional safety certification, or operator telemetry.
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from railway_tp.topology import TopologyConfig, make_topology, validate_topology
from railway_tp.routes import candidate_paths, classify_request
from railway_tp.requests import CLASSES, generate_requests, requests_to_dict
from railway_tp.export import _write_json, write_table, checksum

BASE_CASES = {
    'debug_12_dual': 'debug_12_dual.json',
    'contrast_12_single': 'contrast_12_single.json',
    'primary_60_dual': 'primary_60_dual.json',
    'scale_100_dual': 'scale_100_dual.json',
    'optional_150_dual': 'optional_150_dual.json',
}
CATEGORIES = ('main-to-main', 'within-subring', 'main-to-subring', 'subring-to-subring')


def load_config(path: Path) -> TopologyConfig:
    return TopologyConfig(**json.loads(path.read_text(encoding='utf-8')))


def graph_fingerprint(G: nx.Graph) -> str:
    # Physical edges uniquely identify this labelled network (node ids preserved).
    parts = [f'{min(u,v)}:{max(u,v)}:{d["capacity_mbps"]}'
             for u, v, d in G.edges(data=True)]
    return hashlib.sha256('|'.join(sorted(parts)).encode('utf-8')).hexdigest()


def sweep_topologies(root: Path, out: Path, seeds: tuple[int, ...] = (42, 43, 44, 45, 46)) -> list[dict]:
    """Variation in both seed AND main/sub-ring shape; seed alone just rotates gateways."""
    base = load_config(root / 'config' / 'primary_60_dual.json')
    shapes = [('primary', 12, 4), ('fewer_rings', 12, 3), ('wider_main', 16, 4)]
    rows = []
    for label, main_nodes, sub_count in shapes:
        for seed in seeds:
            cfg = replace(base, seed=seed, main_nodes=main_nodes, subring_count=sub_count)
            G, rings = make_topology(cfg)
            v = validate_topology(G, rings)
            row = {
                'shape': label, 'seed': seed, 'nodes': G.number_of_nodes(),
                'links': G.number_of_edges(), 'rings': len(rings),
                'main_nodes': main_nodes, 'subring_count': sub_count,
                'gateways': ';'.join(str(n) for n, d in sorted(G.nodes(data=True)) if d['gateway']),
                'all_cycles': v['all_rings_valid_cycles'],
                'single_link_connected': v['all_single_link_removals_preserve_graph_connectivity'],
                'single_node_connected_DIAGNOSTIC': v['all_single_node_removals_preserve_graph_connectivity_DIAGNOSTIC'],
                'articulation_points': len(list(nx.articulation_points(G))),
                'graph_edge_connectivity': nx.edge_connectivity(G),
                'fingerprint': graph_fingerprint(G),
            }
            rows.append(row)
    write_table(out / '01_topology_sweep.csv', rows, list(rows[0]))
    shape_summary = []
    for label, _, _ in shapes:
        subset = [r for r in rows if r['shape'] == label]
        shape_summary.append({
            'shape': label, 'runs': len(subset), 'distinct_labelled_graphs': len({r['fingerprint'] for r in subset}),
            'all_link_scenarios_passed': all(r['single_link_connected'] for r in subset),
            'all_cycles_passed': all(r['all_cycles'] for r in subset),
            'mean_link_count': mean(r['links'] for r in subset),
        })
    write_table(out / '01_sweep_summary.csv', shape_summary, list(shape_summary[0]))
    return rows


def _disconnect_pairs(component_sizes: list[int]) -> int:
    # Unordered pair of surviving endpoints disconnected by the failed node.
    n = sum(component_sizes)
    return (n * n - sum(k * k for k in component_sizes)) // 2


def gateway_comparison(root: Path, out: Path) -> list[dict]:
    """Paired physical graphs with identical node counts and ring sizes; only gateway mode differs."""
    cases = [
        ('12', load_config(root / 'config' / 'debug_12_dual.json')),
        ('60', load_config(root / 'config' / 'primary_60_dual.json')),
    ]
    rows, detail = [], []
    for case, base in cases:
        for mode in ('single', 'dual'):
            G, rings = make_topology(replace(base, gateway_mode=mode))
            arts = set(nx.articulation_points(G))
            disconnected = 0
            max_cut_pairs = 0
            gateway_failures = 0
            for n in sorted(G.nodes):
                H = G.copy()
                H.remove_node(n)
                components = [sorted(c) for c in nx.connected_components(H)]
                sizes = sorted([len(c) for c in components], reverse=True)
                cut_pairs = _disconnect_pairs(sizes)
                if len(components) > 1:
                    disconnected += 1
                    if G.nodes[n]['gateway']:
                        gateway_failures += 1
                max_cut_pairs = max(max_cut_pairs, cut_pairs)
                detail.append({
                    'case_nodes': G.number_of_nodes(), 'gateway_mode': mode,
                    'failed_node': n, 'is_gateway': G.nodes[n]['gateway'],
                    'is_articulation_point': n in arts,
                    'remnant_connected': len(components) == 1,
                    'component_sizes': ';'.join(map(str, sizes)),
                    'disconnected_surviving_pairs': cut_pairs,
                    'disconnected_surviving_pair_pct': round(100 * cut_pairs / math.comb(G.number_of_nodes()-1, 2), 3),
                })
            rows.append({
                'case_nodes': G.number_of_nodes(), 'gateway_mode': mode,
                'node_count': G.number_of_nodes(), 'physical_link_count': G.number_of_edges(),
                'shared_physical_edges': sum(len(d['rings'])>1 for _, _, d in G.edges(data=True)),
                'articulation_point_count': len(arts),
                'articulation_nodes': ';'.join(map(str, sorted(arts))),
                'node_failures_causing_disconnection': disconnected,
                'gateway_failures_causing_disconnection': gateway_failures,
                'node_failure_disconnection_pct': round(100 * disconnected / G.number_of_nodes(), 2),
                'max_disconnected_endpoint_pairs_on_node_failure': max_cut_pairs,
                'global_edge_connectivity': nx.edge_connectivity(G),
            })
    write_table(out / '02_gateway_comparison.csv', rows, list(rows[0]))
    write_table(out / '02_node_failure_details.csv', detail, list(detail[0]))
    return rows


def endpoint_pools(G: nx.Graph, rings: dict[str, list[int]]) -> dict[str, list[tuple[int, int]]]:
    """All admissible endpoint pairs; classify by endpoints, not forwarding path."""
    main_only = [n for n in rings['main'] if G.nodes[n]['rings'] == ['main']]
    subs = {ring: [n for n in nodes if 'main' not in G.nodes[n]['rings']]
            for ring, nodes in rings.items() if ring != 'main'}
    sub_nodes = [n for arr in subs.values() for n in arr]
    res = {
        'main-to-main': [(u, v) for u in main_only for v in main_only if u != v],
        'main-to-subring': [(u, v) for u in main_only for v in sub_nodes],
        'within-subring': [(u, v) for arr in subs.values() for u in arr for v in arr if u != v],
        'subring-to-subring': [(u, v) for a, arr1 in subs.items() for b, arr2 in subs.items()
                               if a != b for u in arr1 for v in arr2],
    }
    for cat, pairs in res.items():
        if not pairs:
            raise ValueError(f'not enough endpoints to generate {cat} on this topology')
        assert all(classify_request(G, u, v) == cat for u, v in pairs)
    return res


def _edges(path: list[int]) -> set[tuple[int, int]]:
    return {(min(u, v), max(u, v)) for u, v in zip(path, path[1:])}


def route_analysis(root: Path, out: Path, samples_per_category: int = 35) -> tuple[list[dict], list[dict]]:
    cfg = load_config(root / 'config' / 'primary_60_dual.json')
    G, rings = make_topology(cfg)
    rng = random.Random(1042)
    pools = endpoint_pools(G, rings)
    detail, usage = [], Counter()
    for cat in CATEGORIES:
        options = pools[cat]
        # Sample without replacement where possible to avoid duplicate endpoint pairs.
        pairs = rng.sample(options, min(samples_per_category, len(options)))
        for src, dst in pairs:
            paths = candidate_paths(G, src, dst, k=5)
            if not paths:
                raise RuntimeError('Unexpected disconnected graph')
            shortest = paths[0]
            for edge in _edges(shortest):
                usage[edge] += 1
            # This is a graph-theoretic count; NOT protected LSP redundancy.
            connectivity = nx.edge_connectivity(G, src, dst)
            min_link_capacity = min(G.edges[u, v]['capacity_mbps'] for u, v in zip(shortest, shortest[1:]))
            detail.append({
                'endpoint_category': cat, 'source': src, 'destination': dst,
                'shortest_hops': len(shortest)-1,
                'shortest_km': sum(G.edges[u, v]['length_km'] for u, v in zip(shortest, shortest[1:])),
                'candidate_graph_paths_up_to_5': len(paths),
                'second_candidate_extra_hops': (len(paths[1])-len(shortest)) if len(paths)>1 else '',
                'local_edge_connectivity': connectivity,
                'gateways_on_shortest_graph_path': sum(bool(G.nodes[n]['gateway']) for n in shortest[1:-1]),
                'minimum_edge_capacity_mbps_on_shortest_path': min_link_capacity,
                'shortest_graph_path': '-'.join(map(str, shortest)),
            })
    write_table(out / '03_route_sample_details.csv', detail, list(detail[0]))
    summary = []
    for cat in CATEGORIES:
        data = [r for r in detail if r['endpoint_category'] == cat]
        summary.append({
            'endpoint_category': cat, 'sampled_pairs': len(data),
            'mean_shortest_hops': round(mean(r['shortest_hops'] for r in data), 3),
            'max_shortest_hops': max(r['shortest_hops'] for r in data),
            'mean_gateway_transits_on_shortest_path': round(mean(r['gateways_on_shortest_graph_path'] for r in data), 3),
            'minimum_observed_local_edge_connectivity': min(r['local_edge_connectivity'] for r in data),
            'mean_local_edge_connectivity': round(mean(r['local_edge_connectivity'] for r in data), 3),
            'pct_with_at_least_5_graph_candidates': round(100*sum(r['candidate_graph_paths_up_to_5']==5 for r in data)/len(data), 1),
        })
    write_table(out / '03_route_category_summary.csv', summary, list(summary[0]))
    top_edges=[]
    for (u, v), count in usage.most_common(15):
        d=G.edges[u, v]
        top_edges.append({
            'link_id': d['physical_link_id'], 'source': u, 'destination': v,
            'rings': ';'.join(d['rings']),
            'shortest_path_occurrences_in_sample': count,
            'capacity_mbps': d['capacity_mbps'],
        })
    write_table(out / '03_most_common_shortest_path_edges.csv', top_edges, list(top_edges[0]))
    return detail, summary


def validate_assumptions(root: Path, out: Path) -> dict:
    config_file = root / 'enhancements' / 'phase2_assumptions.json'
    assumptions = json.loads(config_file.read_text(encoding='utf-8'))
    full_duplex = assumptions['phase3_proposed_capacity_semantics']['mode']
    assert full_duplex in {'independent_per_direction', 'aggregate_shared_link'}
    assert all(0 <= x < 1 for x in assumptions['phase3_proposed_capacity_semantics']['headroom_fractions'])
    assert assumptions['current_phase2_graph']['resource_bookkeeping'] == 'one_physical_edge_per_link_id'
    cfg=load_config(root/'config'/'primary_60_dual.json')
    G,rings=make_topology(cfg)
    sample = next((u,v,d) for u,v,d in G.edges(data=True) if len(d['rings']) > 1)
    u,v,d=sample
    result = {
       'model_decisions_documented': True,
       'existing_phase2_network_graph': 'undirected physical connectivity graph',
       'phase3_proposed_direction_model': full_duplex,
       'example_shared_link_id': d['physical_link_id'],
       'example_shared_link_rings': d['rings'],
       'example_physical_link_capacity_mbps': d['capacity_mbps'],
       'example_unique_physical_link_count': sum(1 for _,_,e in G.edges(data=True)
                                                 if e['physical_link_id']==d['physical_link_id']),
       'phase3_protection_scheme_frozen': False,
       'warning': 'These Phase 3 settings are a PROPOSAL, not implemented capacity checks or RFC 8227 protection.',
    }
    _write_json(out / '04_assumptions_validation.json', result)
    return result


def _make_synthetic_record(G: nx.Graph, src: int, dst: int, rng: random.Random, tag: str, index: int, seed: int, split: str) -> dict:
    name = rng.choices(list(CLASSES), weights=[2,5,3])[0]
    rule = CLASSES[name]
    return {'request_id': f'{split.upper()}_{tag}_{index:05d}',
            'source': src, 'destination': dst,
            'bandwidth_mbps': rng.choice(rule['bandwidth_mbps']),
            'service_class': name, 'priority_weight': rule['priority_weight'],
            'endpoint_category': classify_request(G, src, dst),
            'scenario': tag, 'split': split, 'seed': seed, 'is_synthetic': True}


def balanced_requests(G: nx.Graph, rings: dict[str,list[int]], count: int, seed: int, split: str,
                      interring_heavy: bool = False) -> list[dict]:
    if count < len(CATEGORIES):
        raise ValueError('count must be >= number of categories')
    rng=random.Random(seed)
    pools=endpoint_pools(G, rings)
    quotas = ([0.10,0.10,0.35,0.45] if interring_heavy else [0.25]*4)
    counts=[int(count*q) for q in quotas]
    for i in range(count-sum(counts)):
        counts[i % 4] += 1
    tag='interring_heavy' if interring_heavy else 'balanced'
    records = []
    for cat, n in zip(CATEGORIES, counts):
        for _ in range(n):
            src,dst = rng.choice(pools[cat])
            records.append(_make_synthetic_record(G,src,dst,rng,tag,len(records)+1,seed,split))
    rng.shuffle(records)
    return records


def demand_study(root: Path, out: Path, requests_per_split: int = 500) -> tuple[list[dict], list[dict]]:
    cfg=load_config(root/'config'/'primary_60_dual.json')
    G,rings=make_topology(cfg)
    datasets=[]
    seeds={'train': 101, 'validation': 202, 'test': 303}
    headers=['request_id','source','destination','bandwidth_mbps','service_class','priority_weight',
             'endpoint_category','scenario','split','seed','is_synthetic']
    for split, seed in seeds.items():
        # Existing generator: random endpoints (not necessarily balanced across categories).
        uniform = requests_to_dict(generate_requests(G, requests_per_split, seed))
        uniform_records=[]
        for r in uniform:
            r.update({'request_id':f'{split.upper()}_uniform_{r["request_id"]}',
                      'scenario':'uniform_endpoints','split':split,'seed':seed,'is_synthetic':True})
            uniform_records.append(r)
        balanced = balanced_requests(G,rings,requests_per_split,seed+1000,split)
        heavy = balanced_requests(G,rings,requests_per_split,seed+2000,split,interring_heavy=True)
        for scenario, records in [('uniform_endpoints',uniform_records),('balanced',balanced),('interring_heavy',heavy)]:
            write_table(out/f'05_demand_{scenario}_{split}.csv',records,headers)
            datasets.extend(records)
    by_key=defaultdict(list)
    for r in datasets:
        by_key[(r['scenario'],r['split'])].append(r)
    summary=[]
    for (scenario,split),records in sorted(by_key.items()):
        classes=Counter(r['service_class'] for r in records)
        cats=Counter(r['endpoint_category'] for r in records)
        summary.append({
            'scenario':scenario,'split':split,'request_count':len(records),
            'total_offered_request_bandwidth_mbps':sum(r['bandwidth_mbps'] for r in records),
            'mean_requested_bandwidth_mbps':round(mean(r['bandwidth_mbps'] for r in records),2),
            'critical_requests':classes['critical'], 'operational_requests':classes['operational'],
            'noncritical_requests':classes['non_critical'],
            **{'category_'+c:cats[c] for c in CATEGORIES},
            'unique_source_destination_pairs':len({(r['source'],r['destination']) for r in records}),
        })
    write_table(out/'05_demand_summaries.csv',summary,list(summary[0]))
    return datasets, summary


def evidence_audit(root: Path,out: Path,create_missing: bool = True) -> list[dict]:
    existing = root / 'outputs'
    audits=[]
    for name,config_name in BASE_CASES.items():
        cfg=load_config(root/'config'/config_name)
        stem=f'phase2_{cfg.nodes}_{cfg.gateway_mode}_s{cfg.seed}'
        path=existing/f'{stem}_manifest.json'
        if not path.exists() and create_missing:
            # The original four-case run_all_examples deliberately omits optional 150.
            from railway_tp.cli import run
            run(cfg,existing,request_count=50)
        if not path.exists():
            audits.append({'configuration':name,'manifest_present':False,'structural_pass':False,
                'listed_output_files':0,'verified_sha256':0,'bad_or_missing_files':'manifest absent',
                'fully_validated':False})
            continue
        m=json.loads(path.read_text(encoding='utf-8'))
        hashes=m['output_files_sha256']; bad=[]; good=0
        for file_name, expected in hashes.items():
            f=existing/file_name
            if f.exists() and checksum(f)==expected:
                good+=1
            else:
                bad.append(file_name)
        ok=bool(m['passed_structural_gate'] and not bad and len(hashes)>=11)
        audits.append({'configuration':name,'manifest_present':True,
              'structural_pass':bool(m['passed_structural_gate']),
              'listed_output_files':len(hashes),'verified_sha256':good,
              'bad_or_missing_files':';'.join(bad), 'fully_validated':ok})
    write_table(out/'06_original_manifest_integrity_audit.csv',audits,list(audits[0]))
    metadata={
      'python':sys.version.split()[0], 'platform':platform.platform(),
      'networkx':nx.__version__, 'matplotlib':matplotlib.__version__,
      'phase2_project_root_relative':'.',
      'audit_scope':'Original 5 topology manifest files and associated sha256 outputs only',
      'git_commit':'Record separately from your local git repository before final submission',
      'supervisor_approval':'not claimed; obtain outside this script',
      'important_limitation':'Graph topology and synthetic request inventory ONLY; not service protection feasibility',
    }
    _write_json(out/'06_environment_and_provenance.json',metadata)
    return audits


def save_charts(out:Path, gateway_rows:list[dict], route_summaries:list[dict], demand_summary:list[dict], seed_rows:list[dict]) -> None:
    # Separate simple charts; no hardcoded color palettes.
    for size in (12,60):
        data=[r for r in gateway_rows if r['case_nodes']==size]
        fig, ax=plt.subplots(figsize=(7,4.2))
        ax.bar([r['gateway_mode'] for r in data],[r['node_failures_causing_disconnection'] for r in data])
        ax.set_ylabel('Node removals that disconnect graph')
        ax.set_title(f'{size}-node single vs dual gateway (graph only)')
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(out/f'02_gateway_{size}_node_cuts.png',dpi=150)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar([r['endpoint_category'].replace('-','\n') for r in route_summaries],
           [r['mean_shortest_hops'] for r in route_summaries])
    ax.set_ylabel('Mean shortest graph-path hops')
    ax.set_title('60-node topology: graph route length by endpoint category')
    fig.tight_layout();fig.savefig(out/'03_mean_route_hops.png',dpi=150);plt.close(fig)
    train=[r for r in demand_summary if r['split']=='train']
    for scenario in ('uniform_endpoints','balanced','interring_heavy'):
        row=next(r for r in train if r['scenario']==scenario)
        fig,ax=plt.subplots(figsize=(9,4.2))
        ax.bar([c.replace('-','\n') for c in CATEGORIES],[row['category_'+c] for c in CATEGORIES])
        ax.set_title(f'Synthetic train demand endpoints: {scenario}')
        ax.set_ylabel('Number of unadmitted requests')
        fig.tight_layout();fig.savefig(out/f'05_demand_{scenario}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2))
    grouped=defaultdict(set)
    for r in seed_rows:grouped[r['shape']].add(r['fingerprint'])
    ax.bar(list(grouped),[len(x) for x in grouped.values()])
    ax.set_title('Distinct labelled graphs by seed (not independent topology types)')
    ax.set_ylabel('Distinct labelled graph fingerprints')
    fig.tight_layout();fig.savefig(out/'01_seed_fingerprint_counts.png',dpi=150);plt.close(fig)


def write_report(out:Path, seeds:list[dict], gateways:list[dict], routes:list[dict],
                 demand:list[dict], audit:list[dict], assumptions:dict) -> None:
    n_total=len(seeds)
    passes=sum(bool(r['all_cycles'] and r['single_link_connected']) for r in seeds)
    verified=sum(r['fully_validated'] for r in audit)
    get=lambda size, mode: next(r for r in gateways if r['case_nodes']==size and r['gateway_mode']==mode)
    uniform_train=next(r for r in demand if r['scenario']=='uniform_endpoints' and r['split']=='train')
    balanced_train=next(r for r in demand if r['scenario']=='balanced' and r['split']=='train')
    s=[]
    s.append('# Phase 2: six recommended improvements - evidence summary\n')
    s.append('**Scope:** Reproducible synthetic railway-like **physical topology** study only. '
             'No MPLS-TP shared-ring protection switching, LSP bandwidth admission, loss, '
             'recovery latency or operational performance is simulated.\n')
    s.append('## 1. Multiple seeds and topology sensitivity\n')
    s.append(f'- {n_total} 60-node configurations tested (5 seeds x 3 explicit ring layouts); {passes}/{n_total} passed ring-cycle and individual-link graph connectivity checks.\n')
    s.append('- Seed alone primarily rotates gateway positions; it does **not** provide genuinely independent traffic/network structures. Shape variations vary main-ring size and sub-ring count.\n')
    s.append('- Sources: `01_topology_sweep.csv`, `01_sweep_summary.csv`, `01_seed_fingerprint_counts.png`.\n')
    s.append('## 2. Single vs dual gateway comparisons\n')
    for size in (12,60):
        a,b=get(size,'single'),get(size,'dual')
        s.append(f'- {size} nodes: single gateway {a["node_failures_causing_disconnection"]}/{size} node removals disconnect; dual gateway {b["node_failures_causing_disconnection"]}/{size}. '
                 f'Articulation points: {a["articulation_point_count"]} single vs {b["articulation_point_count"]} dual.\n')
    s.append('- This is a structural single-point-of-failure diagnostic, consistent with RFC 8227 §4.4; **not** proof of protocol recovery.\n')
    s.append('- Sources: `02_gateway_comparison.csv`, `02_node_failure_details.csv`, PNG charts.\n')
    s.append('## 3. Graph-path characteristics and inter-ring routes\n')
    for row in routes:
        s.append(f'- {row["endpoint_category"]}: n={row["sampled_pairs"]}, mean shortest hop count={row["mean_shortest_hops"]}, '
                 f'mean gateway transits={row["mean_gateway_transits_on_shortest_path"]}, '
                 f'local edge connectivity >= {row["minimum_observed_local_edge_connectivity"]}.\n')
    s.append('- K-shortest paths and local edge connectivity are graph metrics; neither identifies legal working/protection LSPs under MSRP.\n')
    s.append('- Sources: `03_route_sample_details.csv`, `03_route_category_summary.csv`, `03_most_common_shortest_path_edges.csv`.\n')
    s.append('## 4. Modelling assumptions and Phase 3 choices\n')
    s.append('- Current implementation: one undirected NetworkX edge per physical resource, two ring memberships can share one edge, capacities in Mb/s, no reservations.\n')
    s.append(f'- Proposed for Phase 3 (NOT IMPLEMENTED): **{assumptions["phase3_proposed_direction_model"]}** link-direction capacity accounting, explicit headroom scenarios and pre-defined fault-forwarding semantics.\n')
    s.append('- No claim of RFC 8227 conformance until a precise ring-tunnel and protection operation is defined and checked.\n')
    s.append('- Sources: `phase2_assumptions.json`, `04_assumptions_validation.json`.\n')
    s.append('## 5. Synthetic request distributions\n')
    s.append('- Three disjoint random seeds for train/validation/test; each has uniform-endpoint, category-balanced and inter-ring-heavy variants. '
             'All requests are **unadmitted**; summed Mbps is offered requested bandwidth, not traffic throughput or reserved link load.\n')
    s.append(f'- Train examples: uniform endpoint counts main-main={uniform_train["category_main-to-main"]}, '
             f'sub-sub={uniform_train["category_subring-to-subring"]}; '
             f'balanced main-main={balanced_train["category_main-to-main"]}, sub-sub={balanced_train["category_subring-to-subring"]}.\n')
    s.append('- No measured railway request distribution is asserted; categories are design parameters for future sensitivity tests.\n')
    s.append('- Sources: nine `05_demand_*_*.csv` datasets and `05_demand_summaries.csv`.\n')
    s.append('## 6. Reproducible evidence and audit\n')
    s.append(f'- {verified}/{len(audit)} original topology manifests had a structural pass and all listed file SHA-256 digests verified in this run.\n')
    s.append('- Audit does not prove paper approval, source-code correctness outside tested scope, or research reproducibility on other machines.\n')
    s.append('- Sources: `06_original_manifest_integrity_audit.csv`, `06_environment_and_provenance.json`, project tests.\n')
    s.append('## Ready-for-Phase-3 gate (not a Phase 3 implementation)\n')
    s.append('1. Supervisor approves topology abstraction and the gateway interconnection comparison.\n')
    s.append('2. Explicitly decide per-direction versus aggregate physical-link capacity, and link headroom handling.\n')
    s.append('3. Pick a protection model: ring-tunnel-specific MSRP or clearly labelled simplified surrogate; set allowed faults and preconfigured failure-state forwarding.\n')
    s.append('4. Preserve baseline input data, configurations, code commit, and seed splits before Phase 3 admission experiments.\n')
    s.append('## References\n')
    s.append('- IETF RFC 8227, section 4.4, interconnected ring protection: https://www.rfc-editor.org/info/rfc8227/\n')
    s.append('- NetworkX articulation points: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.articulation_points.html\n')
    s.append('- NetworkX edge connectivity: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.connectivity.connectivity.edge_connectivity.html\n')
    (out/'PHASE2_SIX_IMPROVEMENTS_EVIDENCE_REPORT.md').write_text('\n'.join(s)+'\n',encoding='utf-8')


def run_all(root:Path, output_dir:Path|None=None, create_missing:bool=True,
            requests_per_split:int=500,samples_per_category:int=35) -> dict:
    root=root.resolve()
    output_dir=(output_dir or root/'outputs'/'phase2_enhancements').resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    seeds=sweep_topologies(root,output_dir)
    gateways=gateway_comparison(root,output_dir)
    _, routes=route_analysis(root,output_dir,samples_per_category=samples_per_category)
    assumptions=validate_assumptions(root,output_dir)
    _, demand=demand_study(root,output_dir,requests_per_split=requests_per_split)
    audits=evidence_audit(root,output_dir,create_missing=create_missing)
    save_charts(output_dir,gateways,routes,demand,seeds)
    write_report(output_dir,seeds,gateways,routes,demand,audits,assumptions)
    final={
        'scope':'Synthetic physical-graph topology and unadmitted demand; NOT protocol or service survivability.',
        'topology_sweep_runs':len(seeds),
        'topology_sweep_cycle_and_link_checks_passed':sum(bool(r['all_cycles'] and r['single_link_connected']) for r in seeds),
        'gateway_architectures_compared':len(gateways),
        'route_categories_sampled':len(routes),
        'demand_datasets':len({(r['scenario'],r['split']) for r in demand}),
        'requests_per_dataset':requests_per_split,
        'original_manifest_audits':len(audits),
        'original_manifest_integrity_passes':sum(r['fully_validated'] for r in audits),
        'output_directory':str(output_dir.relative_to(root)) if output_dir.is_relative_to(root) else str(output_dir),
        'report_file':'PHASE2_SIX_IMPROVEMENTS_EVIDENCE_REPORT.md',
    }
    _write_json(output_dir/'FINAL_SIX_IMPROVEMENTS_SUMMARY.json',final)
    return final
