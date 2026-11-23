"""Seeded synthetic national-scale multiregional transport graph (not a map).

Topology supports regional rings, a redundant backbone, and synthetic crosslinks.
The national reference is the approximate 500 nodes in Irish Rail's 2023
replacement programme, NOT actual network shape, capacity or deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import networkx as nx

@dataclass(frozen=True)
class NetworkSpec:
    nodes: int = 60
    seed: int = 101
    backbone_capacity_mbps: int = 10_000
    regional_capacity_mbps: int = 1_000
    crosslink_capacity_mbps: int = 2_000

    def __post_init__(self):
        if not 30 <= self.nodes <= 500:
            raise ValueError('nodes must be 30..500 in Phase 5F stage 1')
        if any(getattr(self, f) <= 0 for f in ('backbone_capacity_mbps',
                                            'regional_capacity_mbps',
                                            'crosslink_capacity_mbps')):
            raise ValueError('all link capacities must be positive')


def build_network(spec: NetworkSpec) -> nx.Graph:
    rng = random.Random(spec.seed)
    count = spec.nodes
    backbone_size = min(count // 4, max(12, round(count * .16)))
    region_count = max(3, round(count / 12))
    region_count = min(region_count, backbone_size)
    room = count - backbone_size
    if room < region_count * 3:
        raise ValueError('insufficient nodes for regional ring cycles')
    split, rem = divmod(room, region_count)
    sizes = [split + (j < rem) for j in range(region_count)]
    graph = nx.Graph()
    for u in range(backbone_size):
        graph.add_node(u, layer='backbone', region=-1, synthetic=True)
    serial = 0

    def edge(u, v, capacity, layer, distance):
        nonlocal serial
        if u == v:
            raise ValueError('self-link')
        if graph.has_edge(u, v):
            # Shared gateway/backbone edge remains ONE physical resource.
            return
        serial += 1
        graph.add_edge(u, v, physical_link_id=f'F{serial:05d}',
                       capacity_mbps=int(capacity), length_km=float(distance),
                       layer=layer, synthetic=True)

    for u in range(backbone_size):
        edge(u, (u + 1) % backbone_size, spec.backbone_capacity_mbps,
             'backbone', 8.0 + rng.random() * 4.0)
    regions = []
    offset = backbone_size
    for rid, n_reg in enumerate(sizes):
        g0 = (rid * backbone_size) // region_count
        g1 = (g0 + 1 + (rid % 2)) % backbone_size
        local = list(range(offset, offset + n_reg))
        offset += n_reg
        for node in local:
            graph.add_node(node, layer='regional', region=rid, synthetic=True)
        route = [g0, *local, g1]
        for u, v in zip(route, route[1:]):
            edge(u, v, spec.regional_capacity_mbps, 'regional_ring',
                 2.5 + rng.random() * 2.5)
        # The regional closing edge is distinct from the backbone ring.
        edge(g1, g0, spec.regional_capacity_mbps, 'regional_ring_closure',
             3.0 + rng.random() * 2)
        regions.append(tuple(local))
    assert offset == count
    # Sparingly provisioned regional cross-corridors create meaningful path
    # alternatives, rather than merely raising the candidate cap.
    for rid in range(0, region_count - 1, 2):
        u = regions[rid][len(regions[rid]) // 2]
        v = regions[rid+1][len(regions[rid+1]) // 2]
        edge(u, v, spec.crosslink_capacity_mbps, 'regional_crosslink',
             7.0 + rng.random() * 8)
    # Additional backbone bypass corridors (independent physical links).
    stride = max(3, backbone_size // 10)
    for u in range(0, backbone_size, stride):
        v = (u + max(3, backbone_size // 4)) % backbone_size
        edge(u, v, spec.backbone_capacity_mbps, 'backbone_bypass',
             15.0 + rng.random() * 15)
    graph.graph.update(synthetic=True, name='PHASE5F_HIERARCHICAL_SYNTHETIC_V1',
                       protection_model='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP',
                       node_count=count, topology_seed=spec.seed,
                       backbone_nodes=tuple(range(backbone_size)),
                       regions=tuple(regions), use_case='research_only')
    validate_network(graph)
    return graph


def validate_network(graph: nx.Graph):
    if not nx.is_connected(graph):
        raise ValueError('network disconnected')
    if graph.number_of_nodes() != graph.graph.get('node_count'):
        raise ValueError('node count mismatch')
    ids = [d['physical_link_id'] for *_, d in graph.edges(data=True)]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate physical link ids')
    if any(d['capacity_mbps'] <= 0 for *_, d in graph.edges(data=True)):
        raise ValueError('nonpositive capacity')
    if list(nx.bridges(graph)):
        raise ValueError('single physical link bridge found')
    return True
