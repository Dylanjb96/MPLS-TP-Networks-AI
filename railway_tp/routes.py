#Route candidates & ring arcs. NO protected LSP admission.

#Global simple paths can traverse rings in ways not allowed under MSRP.
#For Phase 3, constrained candidates must explicitly map to ring tunnels.
from __future__ import annotations
from itertools import islice
import networkx as nx


def candidate_paths(G:nx.Graph,source:int,target:int,k:int=5,weight:str|None=None)->list[list[int]]:
    if source==target: return []
    if k<1: raise ValueError('k must be positive')
    if source not in G or target not in G: raise KeyError('unknown endpoint')
    try:
        return list(islice(nx.shortest_simple_paths(G,source,target,weight=weight),k))
    except nx.NetworkXNoPath:
        return []


def ring_arcs(rings:dict[str,list[int]],ring_id:str,source:int,target:int)->list[list[int]]:
    """Two opposite physical arcs within one ordered ring (not ring-tunnel forwarding)."""
    if ring_id not in rings: raise KeyError('unknown ring')
    nodes=rings[ring_id]
    if source not in nodes or target not in nodes: raise ValueError('both endpoints must belong to ring')
    if source==target: return [[source]]
    out=[]
    for direction in (1,-1):
        i=nodes.index(source)
        path=[source]
        while path[-1]!=target:
            i=(i+direction)%len(nodes)
            path.append(nodes[i])
        out.append(path)
    return out


def path_edges(path:list[int])->list[tuple[int,int]]:
    return list(zip(path,path[1:]))


def classify_request(G:nx.Graph,source:int,destination:int)->str:
    #Based on ENDPOINTS, not on the intermediate route taken.
    if source not in G or destination not in G: raise KeyError('unknown endpoint')
    src_rings=set(G.nodes[source]['rings'])
    dst_rings=set(G.nodes[destination]['rings'])
    src_main='main' in src_rings
    dst_main='main' in dst_rings
    if src_main and dst_main: return 'main-to-main'
    if src_main != dst_main: return 'main-to-subring'
    if src_rings & dst_rings: return 'within-subring'
    return 'subring-to-subring'
