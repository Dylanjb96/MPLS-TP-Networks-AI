"""A second synthetic topology family: double-corridor ladder and local cycles.

This is deliberately different from the original hierarchical/backbone-ring
family, but it remains synthetic and does not represent Irish Rail geometry.
"""
from __future__ import annotations
import random
import networkx as nx
from phase5f_stage2.topology import NetworkSpec,build_network as build_hierarchy,validate_network,fingerprint


def build_network(nodes:int,seed:int,family:str='ladder') -> nx.Graph:
    if family == 'hierarchical':
        return build_hierarchy(NetworkSpec(nodes=nodes,seed=seed))
    if family != 'ladder':
        raise ValueError('unknown synthetic family')
    if not 30<=nodes<=500:
        raise ValueError('nodes out of range')
    rng=random.Random(seed)
    columns=max(6,round(nodes*.08))
    b=2*columns
    rest=nodes-b
    nr=min(columns,max(3,round(rest/13)))
    if rest<3*nr: raise ValueError('insufficient local nodes')
    sizes=[rest//nr+int(r<rest%nr) for r in range(nr)]
    g=nx.Graph()
    for u in range(b):
        g.add_node(u,layer='backbone',region=-1,synthetic=True)

    def connect(u,v,capacity,layer,distance):
        if u==v or g.has_edge(u,v):return
        d=round(distance*(.84+.32*rng.random()),7)
        g.add_edge(u,v,capacity_mbps=capacity,length_km=d,
                   layer=layer,synthetic=True)

    # Each rail forms a separate cycle; cross-rungs create many alternative
    # paths with substantially different cut and betweenness profiles.
    for row in (0,1):
        off=row*columns
        for i in range(columns):
            connect(off+i,off+(i+1)%columns,8000,'corridor_rail',9)
    for i in range(columns):
        connect(i,columns+i,4000,'ladder_rung',7)
    # Strongly sparse interregion access; link disjointness nevertheless holds.
    offset=b; regions=[]
    for rid,size in enumerate(sizes):
        local=list(range(offset,offset+size));offset+=size
        for v in local:
            g.add_node(v,layer='regional',region=rid,synthetic=True)
        a=(rid*columns//nr)%columns
        c=(a+(1 if rid%2==0 else 2))%columns
        gateway1=a
        gateway2=columns+c
        rng.shuffle(local)
        seq=[gateway1,*local,gateway2]
        for u,v in zip(seq,seq[1:]):connect(u,v,1000,'regional_chain',4)
        # Gateway endpoints are already connected via ladder: each local
        # chain is part of a physical cycle, so there are no bridges.
        regions.append(tuple(local))
    for rid in range(nr-1):
        if rid%3==0:
            connect(rng.choice(regions[rid]),rng.choice(regions[rid+1]),
                    1200,'corridor_crosslink',15)
    for idx,(u,v) in enumerate(sorted((min(u,v),max(u,v)) for u,v in g.edges()),1):
        g[u][v]['physical_link_id']=f'G{idx:05d}'
    g.graph.update(synthetic=True,node_count=nodes,topology_seed=seed,
                   name='PHASE5G_SYNTHETIC_CORRIDOR_LADDER_NOT_REAL_RAILWAY',
                   backbone_nodes=tuple(range(b)),regions=tuple(regions),
                   protection_model='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP',
                   use_case='synthetic_research_only')
    validate_network(g)
    return g
