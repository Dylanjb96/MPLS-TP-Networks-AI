"""Synthetic *structurally seeded* multiregional transport graphs.

Not a map of Irish Rail, not a deployed MPLS-TP engineering configuration.
Every ring and backbone is constructed to maintain basic bridge-free redundancy.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import random
import networkx as nx


def edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class NetworkSpec:
    nodes: int = 60
    seed: int = 101
    backbone_capacity_mbps: int = 10_000
    regional_capacity_mbps: int = 1_000
    crosslink_capacity_mbps: int = 2_000

    def __post_init__(self) -> None:
        if not 30 <= self.nodes <= 500:
            raise ValueError('nodes must be in the predeclared range 30..500')
        if min(self.backbone_capacity_mbps,self.regional_capacity_mbps,
               self.crosslink_capacity_mbps) <= 0:
            raise ValueError('invalid link capacity')


def build_network(spec: NetworkSpec) -> nx.Graph:
    rng = random.Random(spec.seed)
    n = spec.nodes
    backbone_size = min(n // 4, max(12, round(n * .16)))
    region_count = min(backbone_size, max(3, round(n / 12)))
    room = n - backbone_size
    if room < 3*region_count:
        raise ValueError('insufficient nodes for regional ring cycles')
    split, rem = divmod(room,region_count)
    sizes=[split+(r<rem) for r in range(region_count)]
    g=nx.Graph()
    for u in range(backbone_size):
        g.add_node(u, layer='backbone', region=-1, synthetic=True)

    def add_edge(u,v,capacity,layer,nominal_length):
        if u==v or g.has_edge(u,v):
            return
        # length is routing relevant, as are capacity and topology.
        length = round(nominal_length*(.8+.4*rng.random()),7)
        g.add_edge(u,v,capacity_mbps=int(capacity),length_km=length,
                   layer=layer, synthetic=True)

    # Bidirectional (undirected physical resource) backbone ring.
    for u in range(backbone_size):
        add_edge(u,(u+1)%backbone_size,spec.backbone_capacity_mbps,
                 'backbone', 10)

    regions=[]
    offset=backbone_size
    for rid,n_reg in enumerate(sizes):
        locals_=list(range(offset,offset+n_reg))
        offset+=n_reg
        for u in locals_:
            g.add_node(u,layer='regional',region=rid,synthetic=True)
        # Topology seed now changes BOTH gateway selection and local connectivity.
        anchor=(rid*backbone_size)//region_count
        gateway0=(anchor+rng.randrange(-1,2))%backbone_size
        gateway1=(gateway0+rng.randrange(1,max(2,backbone_size//3)+1))%backbone_size
        permuted=list(locals_)
        rng.shuffle(permuted)
        # The regional route and closing link form a redundant cycle, whether
        # the closing physical edge is new or shares an existing backbone edge.
        path=[gateway0,*permuted,gateway1]
        for u,v in zip(path,path[1:]):
            add_edge(u,v,spec.regional_capacity_mbps,'regional_ring',3.5)
        add_edge(gateway0,gateway1,spec.regional_capacity_mbps,
                 'regional_ring_closure',4)
        regions.append(tuple(locals_))

    for rid in range(0,region_count-1,2):
        # Seed affects physical cross-corridor endpoints too.
        add_edge(rng.choice(regions[rid]),rng.choice(regions[rid+1]),
                 spec.crosslink_capacity_mbps,'regional_crosslink',12)

    stride=max(3,backbone_size//10)
    for u in range(0,backbone_size,stride):
        options=[v for v in range(backbone_size) if v!=u and
                 min((v-u)%backbone_size,(u-v)%backbone_size)>=3 and
                 not g.has_edge(u,v)]
        if options:
            v=rng.choice(options)
            add_edge(u,v,spec.backbone_capacity_mbps,'backbone_bypass',21)

    # Assign physical IDs in canonical edge order (not insertion order).
    for ix,(u,v) in enumerate(sorted(edge(u,v) for u,v in g.edges()),start=1):
        g.edges[u,v]['physical_link_id']=f'F{ix:05d}'
    g.graph.update(synthetic=True,name='PHASE5F_HIERARCHICAL_SYNTHETIC_STAGE2',
                   protection_model='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP',
                   node_count=n,topology_seed=spec.seed,
                   backbone_nodes=tuple(range(backbone_size)),regions=tuple(regions),
                   use_case='synthetic_research_only')
    validate_network(g)
    return g


def validate_network(g: nx.Graph) -> bool:
    if len(g)!=g.graph.get('node_count'):
        raise ValueError('node count mismatch')
    if not nx.is_connected(g):
        raise ValueError('disconnected network')
    if list(nx.bridges(g)):
        raise ValueError('physical link bridge')
    ids=[d['physical_link_id'] for _,_,d in g.edges(data=True)]
    if len(ids)!=len(set(ids)) or len(ids)!=g.number_of_edges():
        raise ValueError('duplicate or missing physical link IDs')
    if any(d['capacity_mbps']<=0 or d['length_km']<=0 for _,_,d in g.edges(data=True)):
        raise ValueError('bad capacity or routing weight')
    if any(g.nodes[u].get('synthetic') is not True for u in g.nodes):
        raise ValueError('unspecified node')
    return True


def fingerprint(g: nx.Graph, *, structural_only: bool = False) -> str:
    # SHA covers physical graph (including routing-relevant lengths) and nodes.
    data={'nodes': [(int(u),g.nodes[u]['layer'],g.nodes[u]['region'])
                    for u in sorted(g.nodes)],
          'edges': []}
    for u,v in sorted(edge(u,v) for u,v in g.edges()):
        d=g.edges[u,v]
        row=[u,v,d['capacity_mbps'],d['layer']]
        if not structural_only:
            row.extend([d['physical_link_id'],d['length_km']])
        data['edges'].append(row)
    payload=json.dumps(data,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def edge_set(g):
    return frozenset(edge(u,v) for u,v in g.edges())
