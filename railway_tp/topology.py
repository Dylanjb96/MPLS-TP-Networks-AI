#Fully synthetic, topology-only railway-like interconnected-ring model.

#This deliberately does NOT implement MPLS-TP shared ring protection,
#label/tunnel switching, service admission, post-fault reservations, or QoS.
#RFC 8227 section 4.4 motivates the SINGLE- and DUAL-interconnection shapes.

from __future__ import annotations
from dataclasses import dataclass
import random
import networkx as nx

@dataclass(frozen=True)
class TopologyConfig:
    nodes: int = 12
    seed: int = 42
    gateway_mode: str = 'dual'
    main_capacity_mbps: int = 10_000
    sub_capacity_mbps: int = 1_000
    main_nodes: int | None = None
    subring_count: int | None = None
    length_km: float = 2.0

    def __post_init__(self):
        if self.nodes < 12 or self.nodes > 500:
            raise ValueError('nodes must be in 12..500')
        if self.gateway_mode not in {'single', 'dual'}:
            raise ValueError('gateway_mode must be single or dual')
        if self.main_capacity_mbps <= 0 or self.sub_capacity_mbps <= 0:
            raise ValueError('capacities must be positive')
        if self.length_km <= 0:
            raise ValueError('length_km must be positive')
        if self.main_nodes is not None and not (4 <= self.main_nodes <= self.nodes - 3):
            raise ValueError('main_nodes must leave room for at least 3 sub-ring nodes')
        if self.subring_count is not None and self.subring_count < 1:
            raise ValueError('subring_count must be >= 1')


def _sizes(total: int, n: int) -> list[int]:
    q, r = divmod(total, n)
    return [q + int(i < r) for i in range(n)]


def _topology_shape(cfg: TopologyConfig) -> tuple[int,int,list[int]]:
    main_n = cfg.main_nodes if cfg.main_nodes is not None else (6 if cfg.nodes <= 20 else max(10, round(cfg.nodes * 0.2)))
    sub_count = cfg.subring_count if cfg.subring_count is not None else (1 if cfg.nodes <= 20 else min(6, max(3, main_n // 3)))
    if sub_count > main_n // 2:
        raise ValueError('subring_count exceeds disjoint gateway pair placement capacity')
    counts = _sizes(cfg.nodes - main_n, sub_count)
    if min(counts) < 2:
        raise ValueError('each sub-ring needs at least 2 new nodes')
    if cfg.gateway_mode == 'single' and min(counts) < 2:
        raise ValueError('single-gateway ring requires two subring nodes')
    return main_n,sub_count,counts


def make_topology(cfg: TopologyConfig) -> tuple[nx.Graph, dict[str, list[int]]]:
    """Generate one main ring plus synthetic interconnected sub-rings.

    In dual mode, each sub-ring and main ring share two physical gateway
    nodes and the main-ring edge between them. That shared edge is a SINGLE
    physical resource; it is recorded in both ring membership lists but
    its capacity is counted once. This is a *topology abstraction only*.
    """
    main_n, sub_count, counts = _topology_shape(cfg)
    rng = random.Random(cfg.seed)
    offset = rng.randrange(main_n)
    # Disjoint adjacent gateway pairs, rotated by seed.
    gateway_starts = [(offset + (2 * i)) % main_n for i in range(sub_count)]
    G = nx.Graph()
    main = list(range(main_n))
    ring_map: dict[str,list[int]] = {'main': main}
    for n in main:
        G.add_node(n, role='main_transport', rings=['main'], gateway=False, synthetic=True)

    def add_link(a: int,b: int,ring: str,capacity: int):
        if a == b: raise ValueError('self-edge')
        if G.has_edge(a,b):
            if ring not in G.edges[a,b]['rings']:
                G.edges[a,b]['rings'].append(ring)
        else:
            G.add_edge(a,b, rings=[ring], capacity_mbps=capacity,
                       reserved_mbps=0, status='active', length_km=cfg.length_km,
                       physical_link_id=f'E{min(a,b):03d}_{max(a,b):03d}', synthetic=True)

    for i,u in enumerate(main):
        add_link(u, main[(i + 1) % main_n], 'main',cfg.main_capacity_mbps)
    next_node = main_n
    for i,count in enumerate(counts):
        ring_id = f'sub_{i+1}'
        new_nodes = list(range(next_node,next_node+count))
        next_node += count
        for n in new_nodes:
            G.add_node(n, role='sub_transport', rings=[ring_id], gateway=False, synthetic=True)
        ga = gateway_starts[i]
        gateways = [ga, (ga+1)%main_n] if cfg.gateway_mode=='dual' else [ga]
        ring = [ga] + new_nodes + gateways[1:]
        ring_map[ring_id] = ring
        for g in gateways:
            G.nodes[g]['gateway'] = True
            G.nodes[g]['role'] = 'interconnection_gateway'
            if ring_id not in G.nodes[g]['rings']:
                G.nodes[g]['rings'].append(ring_id)
        for j,u in enumerate(ring):
            add_link(u,ring[(j+1)%len(ring)],ring_id,cfg.sub_capacity_mbps)

    assert G.number_of_nodes() == cfg.nodes
    G.graph.update({'synthetic': True, 'model': 'Phase 2 physical graph, NOT MSRP',
                    'gateway_mode': cfg.gateway_mode, 'seed': cfg.seed,
                    'requested_nodes': cfg.nodes, 'main_nodes': main_n,
                    'subring_count': sub_count, 'units': {'capacity':'Mb/s','length':'km'},
                    'physical_resource_policy': 'shared physical edge stored once'})
    return G, ring_map


def ring_subgraph(G: nx.Graph, ring_id: str) -> nx.Graph:
    return nx.Graph(((u,v,dict(d)) for u,v,d in G.edges(data=True) if ring_id in d['rings']))


def ring_is_valid_cycle(G: nx.Graph, ring_nodes: list[int],ring_id: str) -> bool:
    if len(ring_nodes) < 3 or len(set(ring_nodes))!=len(ring_nodes): return False
    R = ring_subgraph(G,ring_id)
    return (set(R.nodes)==set(ring_nodes) and nx.is_connected(R) and
            R.number_of_edges()==len(ring_nodes) and
            all(R.degree(n)==2 for n in R.nodes) and
            all(G.has_edge(u,ring_nodes[(i+1)%len(ring_nodes)])
                for i,u in enumerate(ring_nodes)))


def _components_after(G:nx.Graph, *, edge:tuple[int,int]|None=None, node:int|None=None) -> dict:
    H=G.copy()
    if edge is not None: H.remove_edge(*edge)
    if node is not None: H.remove_node(node)
    cc=list(nx.connected_components(H))
    return {'connected': len(cc)==1, 'component_count':len(cc),
            'component_sizes':sorted((len(s) for s in cc),reverse=True)}


def link_failure_connectivity(G:nx.Graph)->list[dict]:
    return [{'failed_link':[u,v],**_components_after(G,edge=(u,v))}
            for u,v in sorted((min(u,v),max(u,v)) for u,v in G.edges)]


def node_failure_connectivity(G:nx.Graph)->list[dict]:
    """DIAGNOSTIC ONLY; not a resilience objective or an MSRP simulation."""
    return [{'failed_node':n,'gateway':bool(G.nodes[n]['gateway']),
             **_components_after(G,node=n)} for n in sorted(G.nodes)]


def validate_topology(G:nx.Graph,rings:dict[str,list[int]])->dict:
    cycles={name:ring_is_valid_cycle(G,nodes,name) for name,nodes in rings.items()}
    failures=link_failure_connectivity(G)
    node_failures=node_failure_connectivity(G)
    shared_links = [(min(u,v),max(u,v)) for u,v,d in G.edges(data=True) if len(d['rings'])>1]
    membership_consistent = all(set(G.nodes[n]['rings'])=={r for r,ns in rings.items() if n in ns} for n in G.nodes)
    bandwidth_valid=all(isinstance(d['capacity_mbps'],int) and d['capacity_mbps']>0 and d['reserved_mbps']==0 for _,_,d in G.edges(data=True))
    all_ids_unique=len({d['physical_link_id'] for _,_,d in G.edges(data=True)})==G.number_of_edges()
    pairs={r:sorted(set(nodes)&set(rings['main'])) for r,nodes in rings.items() if r!='main'}
    expected_gateways=2 if G.graph['gateway_mode']=='dual' else 1
    gateway_counts_valid=all(len(g)==expected_gateways for g in pairs.values())
    return {'node_count':G.number_of_nodes(),'link_count':G.number_of_edges(),
            'ring_count':len(rings),'subring_count':len(rings)-1,
            'connected':nx.is_connected(G), 'all_rings_valid_cycles':all(cycles.values()),
            'ring_validity':cycles, 'node_ring_membership_consistent':membership_consistent,
            'gateway_counts_valid':gateway_counts_valid, 'gateway_node_pairs':pairs,
            'capacity_units':'Mb/s','positive_capacity_and_zero_initial_reservations':bandwidth_valid,
            'unique_physical_link_ids':all_ids_unique, 'shared_physical_link_count':len(shared_links),
            'bridge_count':len(list(nx.bridges(G))),
            'all_single_link_removals_preserve_graph_connectivity':all(r['connected'] for r in failures),
            'physical_link_failures_checked':len(failures),
            'all_single_node_removals_preserve_graph_connectivity_DIAGNOSTIC':all(r['connected'] for r in node_failures),
            'node_failures_checked_DIAGNOSTIC':len(node_failures),
            'IMPORTANT': 'Graph-only tests. NO MPLS-TP protection switching, service continuity, bandwidth assurance, or railway-safety result.'}