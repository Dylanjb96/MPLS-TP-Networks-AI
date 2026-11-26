"""Stable integer-weight candidate generation and exact footprint tie-break.

No hidden future request information; candidates are generated from the physical
synthetic topology, not from observed end-of-sequence outcomes.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import heapq
import json
import networkx as nx
from phase5f.protection import ProtectedPair, verify_pair, path_edges
from .topology import edge


def routing_units(u, v, attrs):
    # Stable integer routing weights avoid platform-dependent floating sums in
    # path-order comparisons. Resolution ~ 1e-6 km with deterministic tie salt.
    a,b=edge(u,v)
    salt=int.from_bytes(hashlib.sha256(f'{a},{b}'.encode()).digest()[:4],'big') % 1_000_000
    return int(round(float(attrs['length_km'])*1_000_000))*1_000_000 + salt


def weighted_copy(g):
    h=g.copy()
    for u,v,d in h.edges(data=True):
        d['stable_routing_units']=routing_units(u,v,d)
    return h


def shortest_stable(g,source,destination):
    if source not in g or destination not in g:
        return None
    heap=[(0,(source,),source)]
    seen=set()
    while heap:
        cost,path,u=heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u==destination:
            return path
        for v in sorted(g.neighbors(u)):
            if v not in seen:
                heapq.heappush(heap,(cost+g[u][v]['stable_routing_units'],path+(v,),v))
    return None


def generate_pairs(g,source,destination,max_candidates=6,max_working_paths=40):
    if max_candidates < 1 or max_working_paths < max_candidates:
        raise ValueError('invalid k or working-path limit')
    if source==destination or source not in g or destination not in g:
        return tuple()
    h=weighted_copy(g)
    result=[]
    seen=set()
    try:
        gen=nx.shortest_simple_paths(h,source,destination,weight='stable_routing_units')
        for ix,work in enumerate(gen):
            if ix>=max_working_paths:
                break
            work=tuple(work)
            fallback=h.copy()
            fallback.remove_edges_from(path_edges(work))
            backup=shortest_stable(fallback,source,destination)
            if backup is None:
                continue
            key=(work,backup)
            if key in seen:
                continue
            seen.add(key)
            cid='F2-'+hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()[:20]
            pair=ProtectedPair(work,backup,cid)
            if not verify_pair(g,source,destination,pair):
                raise AssertionError('pair is not link-disjoint')
            result.append(pair)
            if len(result)>=max_candidates:
                break
    except nx.NetworkXNoPath:
        pass
    return tuple(result)


def exact_footprint(g, plan:ProtectedPair):
    return sum((Fraction(1,int(g.edges[u,v]['capacity_mbps']))
                for path in (plan.working,plan.backup)
                for u,v in zip(path,path[1:])), Fraction(0))


def choice(policy,feasible,plans,g,rng):
    if policy=='greedy':
        return feasible[0]
    if policy=='random_feasible':
        return rng.choice(feasible)
    if policy=='footprint':
        return min(feasible,key=lambda i:(exact_footprint(g,plans[i]),
                                          plans[i].working,plans[i].backup))
    raise ValueError('unsupported control '+policy)


def candidates_digest(inventories):
    records=[[(x.candidate_id,x.working,x.backup) for x in cohort]
              for cohort in inventories]
    return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()
