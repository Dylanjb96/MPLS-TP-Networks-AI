"""Independent end-to-end disjoint path-pair scenario surrogate, NOT MSRP.

Each accepted symmetric service reserves both directions on its working path
for normal/other failure states. When any working path link fails, it uses its
preselected edge-disjoint backup end-to-end. This is NOT existing Phase 3's
ring-local opposite-arc protection and must never be presented as identical.

A vectorized [failure scenario, directional arc] capacity ledger makes a 500-node
pilot feasible; we independently recompute scenario loads in unit tests.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import time
import networkx as nx
import numpy as np
from .traffic import Demand


def edge(u, v):
    return (u, v) if u < v else (v, u)


def path_edges(path):
    return tuple(edge(u, v) for u, v in zip(path, path[1:]))

@dataclass(frozen=True)
class ProtectedPair:
    working: tuple[int, ...]
    backup: tuple[int, ...]
    candidate_id: str


def verify_pair(graph: nx.Graph, src: int, dst: int, p: ProtectedPair):
    for path in (p.working, p.backup):
        if (len(path) < 2 or path[0] != src or path[-1] != dst or
            len(path) != len(set(path)) or
            any(not graph.has_edge(u, v) for u, v in zip(path, path[1:]))):
            return False
    if set(path_edges(p.working)) & set(path_edges(p.backup)):
        return False
    return True


def generate_pairs(graph: nx.Graph, source: int, destination: int,
                   max_candidates: int = 6, max_working_paths: int = 40):
    if max_candidates < 1 or max_working_paths < max_candidates:
        raise ValueError('invalid candidate generation limits')
    if source not in graph or destination not in graph or source == destination:
        return tuple()
    result = []
    seen = set()
    try:
        generator = nx.shortest_simple_paths(graph, source, destination,
                                              weight='length_km')
        for path_no, working in enumerate(generator):
            if path_no >= max_working_paths: break
            work = tuple(working)
            backup_graph = graph.copy()
            backup_graph.remove_edges_from(path_edges(work))
            try:
                backup = tuple(nx.shortest_path(backup_graph, source, destination,
                                                weight='length_km'))
            except nx.NetworkXNoPath:
                continue
            key = (work, backup)
            if key in seen: continue
            seen.add(key)
            digest = hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()[:16]
            plan = ProtectedPair(work, backup, 'FP-'+digest)
            if not verify_pair(graph, source, destination, plan):
                raise AssertionError('candidate generator produced an invalid pair')
            result.append(plan)
            if len(result) >= max_candidates: break
    except nx.NetworkXNoPath:
        pass
    return tuple(result)


class ScenarioLedger:
    def __init__(self, graph: nx.Graph, headroom_percent: int = 10):
        if not 0 <= headroom_percent < 100:
            raise ValueError('invalid headroom')
        self.graph = graph
        self.edge_to_i = {edge(u, v): i for i, (u, v) in
                          enumerate(sorted(edge(u, v) for u, v in graph.edges()))}
        self.m = len(self.edge_to_i)
        self.loads = np.zeros((self.m + 1, self.m * 2), dtype=np.int64)
        cap = np.zeros(self.m * 2, dtype=np.int64)
        for (u,v), i in self.edge_to_i.items():
            c = graph.edges[u,v]['capacity_mbps'] * (100-headroom_percent) // 100
            cap[2*i:2*i+2] = c
        self.cap = cap
        self.active: dict[str, tuple[Demand, ProtectedPair]] = {}

    def _indices(self, path):
        # Symmetric reservation in both directions; edge-disjoint paths.
        return np.array([2*self.edge_to_i[edge(u,v)] + direction
                         for u,v in zip(path, path[1:]) for direction in (0,1)],
                        dtype=np.int64)

    def _partition(self, plan: ProtectedPair):
        failed_on_work = {self.edge_to_i[e]+1 for e in path_edges(plan.working)}
        affected = np.array(sorted(failed_on_work), dtype=np.int64)
        unaffected = np.array([i for i in range(self.m+1)
                               if i not in failed_on_work], dtype=np.int64)
        return unaffected, affected, self._indices(plan.working), self._indices(plan.backup)

    def feasible(self, demand: Demand, plan: ProtectedPair):
        if demand.request_id in self.active:
            return False
        if not verify_pair(self.graph, demand.source, demand.destination, plan):
            return False
        other, changed, wi, bi = self._partition(plan)
        b = demand.bandwidth_mbps
        if wi.size and np.any(self.loads[np.ix_(other,wi)] + b > self.cap[wi]):
            return False
        if bi.size and np.any(self.loads[np.ix_(changed,bi)] + b > self.cap[bi]):
            return False
        return True

    def _update(self, demand: Demand, plan: ProtectedPair, delta: int):
        other, changed, wi, bi = self._partition(plan)
        b = int(demand.bandwidth_mbps) * delta
        self.loads[np.ix_(other, wi)] += b
        self.loads[np.ix_(changed, bi)] += b

    def admit(self, demand: Demand, plan: ProtectedPair):
        if not self.feasible(demand, plan):
            return False
        self._update(demand, plan, 1)
        self.active[demand.request_id] = demand, plan
        return True

    def release_expired(self, now: int):
        count = 0
        for key, (d,p) in tuple(self.active.items()):
            if d.expiry <= now:
                self._update(d,p,-1)
                del self.active[key]
                count += 1
        if (self.loads < 0).any():
            raise AssertionError('negative reservation after release')
        return count

    def peak_utilization(self):
        return float(np.max(self.loads / self.cap[None,:]))

    def state_digest(self):
        items = [(key,d.bandwidth_mbps,p.candidate_id)
                 for key,(d,p) in sorted(self.active.items())]
        return hashlib.sha256(json.dumps(items,separators=(',', ':')).encode()).hexdigest()

    def verify_full_scenario_recompute(self):
        fresh = np.zeros_like(self.loads)
        for d,p in self.active.values():
            other,changed,wi,bi = self._partition(p)
            fresh[np.ix_(other, wi)] += d.bandwidth_mbps
            fresh[np.ix_(changed, bi)] += d.bandwidth_mbps
        return bool(np.array_equal(fresh,self.loads) and (self.loads>=0).all() and
                    (self.loads<=self.cap[None,:]).all())
