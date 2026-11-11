#An explicit, reproducible MPLS-TP-*inspired* protected-provisioning abstraction.

#Model: preplanned ring-local opposite-arc substitution on single physical-link
#failures, with the inter-ring gateway sequence fixed. This is *not* MSRP:
#there are no labels, ring tunnels, RPS, wrap/short-wrap procedures or timing.

#Topology is the existing Phase 2 NetworkX graph. Every physical edge is
#unique, even when it is a member of two rings. Each direction has an
#independent integer-Mb/s capacity. Each symmetric service reserves b Mb/s
#in BOTH directions on every arc used in each individual fault scenario.
#No-fault and all one-link failures are considered simultaneously as a
#robust-feasibility constraint; loads are not summed across scenarios.

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import product
from typing import Iterable
import hashlib
import json
import networkx as nx
from railway_tp.routes import ring_arcs
from railway_tp.requests import ServiceRequest

SCHEME = 'RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP'
NORMAL = 'normal'


def edge_key(u: int, v: int) -> tuple[int,int]:
    return min(u,v),max(u,v)


def physical_failures(G: nx.Graph) -> tuple[str,...]:
    ids = [d['physical_link_id'] for _,_,d in G.edges(data=True)]
    if len(ids)!=len(set(ids)):
        raise ValueError('DUPLICATE_PHYSICAL_LINK_ID')
    return tuple(sorted(ids))


def _links(G:nx.Graph) -> dict[str,tuple[int,int]]:
    return {d['physical_link_id']:edge_key(u,v) for u,v,d in G.edges(data=True)}


def _path_edges(nodes:tuple[int,...]) -> list[tuple[int,int]]:
    return list(zip(nodes,nodes[1:]))


@dataclass(frozen=True)
class PathSegment:
    ring_id: str
    working: tuple[int,...]


@dataclass(frozen=True)
class ProtectedPlan:
    candidate_id: str
    service_id: str
    segments: tuple[PathSegment,...]
    normal_path: tuple[int,...]
    fault_paths: dict[str,tuple[int,...]] = field(default_factory=dict)
    scheme: str = SCHEME


@dataclass(frozen=True)
class AdmissionResult:
    feasible: bool
    reason_codes: tuple[str,...]
    service_id: str
    candidate_id: str | None
    worst_fault_id: str | None = None
    worst_arc: tuple[int,int] | None = None
    worst_utilization: float | None = None
    capacity_margin_mbps: int | None = None
    details: str = ''

    def as_dict(self)->dict:
        return {'feasible':self.feasible, 'reason_codes':list(self.reason_codes),
                'service_id':self.service_id, 'candidate_id':self.candidate_id,
                'worst_fault_id':self.worst_fault_id,
                'worst_arc':list(self.worst_arc) if self.worst_arc else None,
                'worst_utilization':self.worst_utilization,
                'capacity_margin_mbps':self.capacity_margin_mbps,
                'details':self.details}


def _ring_options(rings:dict[str,list[int]], name:str, a:int,b:int):
    if a==b: return (tuple([a]),)
    return tuple(tuple(path) for path in ring_arcs(rings,name,a,b))


def _segment_specs(G:nx.Graph,rings:dict[str,list[int]],src:int,dst:int):
    """Return the permitted sequences (ring id, ingress, egress).

    No inter-ring transitions except at shared *main/sub* gateway nodes.
    Main-member endpoints are considered main endpoints for v1. Each subring
    appears at most once in the sequence and gateway nodes do not change
    at fail time. Arbitrary graph paths and sub-sub direct hopping excluded.
    """
    if src not in G or dst not in G or src==dst:
        return []
    A=set(G.nodes[src]['rings']); B=set(G.nodes[dst]['rings'])
    src_sub=sorted(A-{'main'}); dst_sub=sorted(B-{'main'})
    src_main='main' in A; dst_main='main' in B
    if src_main and dst_main:
        return [(('main',src,dst),)]
    if not src_main and not dst_main and set(src_sub)&set(dst_sub):
        return [((next(iter(sorted(set(src_sub)&set(dst_sub)))),src,dst),)]
    def gateway_set(r:str) -> list[int]:
        return sorted(set(rings[r])&set(rings['main']))
    options=[]
    if not src_main and dst_main:
        for r in src_sub:
            for a in gateway_set(r):
                parts=[(r,src,a)]
                if a!=dst: parts.append(('main',a,dst))
                options.append(tuple(parts))
    elif src_main and not dst_main:
        for r in dst_sub:
            for b in gateway_set(r):
                parts=[]
                if src!=b: parts.append(('main',src,b))
                parts.append((r,b,dst))
                options.append(tuple(parts))
    elif not src_main and not dst_main:
        for r1 in src_sub:
            for r2 in dst_sub:
                if r1==r2: continue
                for a in gateway_set(r1):
                    for b in gateway_set(r2):
                        parts=[(r1,src,a)]
                        if a!=b: parts.append(('main',a,b))
                        parts.append((r2,b,dst))
                        options.append(tuple(parts))
    return options


def _concat_segments(segments:tuple[PathSegment,...], *, fault:str|None,
                     link_by_id:dict[str,tuple[int,int]],rings:dict[str,list[int]])->tuple[int,...]:
    """Deterministic: only affected working segments switch to the opposite arc.

    Unaffected segments remain unchanged (even when fault is on their spare
    opposite arc). No post-failure route search or gateway relocation.
    """
    out=[]
    failed=link_by_id.get(fault) if fault is not None else None
    for s in segments:
        work=s.working
        if failed is not None and any(edge_key(u,v)==failed for u,v in _path_edges(work)):
            a,b=work[0],work[-1]
            arcs=_ring_options(rings,s.ring_id,a,b)
            alternatives=[p for p in arcs if p!=work]
            if len(alternatives)!=1:
                raise ValueError('NO_OPPOSITE_ARC')
            work=alternatives[0]
        if out and out[-1]!=work[0]:
            raise ValueError('ILLEGAL_GATEWAY_TRANSITION')
        out.extend(work if not out else work[1:])
    return tuple(out)


def _valid_simple(G:nx.Graph, path:tuple[int,...],src:int,dst:int,
                  failed:tuple[int,int]|None=None)->bool:
    return (len(path)>=2 and path[0]==src and path[-1]==dst and
            len(set(path))==len(path) and
            all(G.has_edge(u,v) and (failed is None or edge_key(u,v)!=failed)
                for u,v in _path_edges(path)))


def _candidate_id(segments:tuple[PathSegment,...])->str:
    body=[(s.ring_id,list(s.working)) for s in segments]
    return 'P-'+hashlib.sha256(json.dumps(body,separators=(',',':')).encode()).hexdigest()[:12]


def _make_candidate(G:nx.Graph,rings:dict[str,list[int]], req:ServiceRequest,
                    segments:tuple[PathSegment,...])->ProtectedPlan|None:
    by_id=_links(G)
    try:
        normal=_concat_segments(segments,fault=None,link_by_id=by_id,rings=rings)
        if not _valid_simple(G,normal,req.source,req.destination): return None
        fpaths={}
        for fault in sorted(by_id):
            p=_concat_segments(segments,fault=fault,link_by_id=by_id,rings=rings)
            if not _valid_simple(G,p,req.source,req.destination,failed=by_id[fault]):
                return None
            fpaths[fault]=p
    except (ValueError,KeyError,IndexError):
        return None
    return ProtectedPlan(_candidate_id(segments),req.request_id,segments,normal,fpaths)


def generate_candidates(G:nx.Graph,rings:dict[str,list[int]],req:ServiceRequest,
                        max_candidates:int=12)->list[ProtectedPlan]:
    """Enumerate 2^segments ring-local paths with gateway variants.

    All returned plans have preplanned *simple* normal and every single-link
    failure state. Capacity is NOT checked here: the independent validator
    makes the admission decision using the current admitted-service set.
    """
    if max_candidates<1:raise ValueError('max_candidates must be >= 1')
    all_plans=[]
    for specs in _segment_specs(G,rings,req.source,req.destination):
        choices=[_ring_options(rings,r,a,b) for r,a,b in specs]
        for combination in product(*choices):
            segs=tuple(PathSegment(r,tuple(path)) for (r,_,_),path in zip(specs,combination))
            plan=_make_candidate(G,rings,req,segs)
            if plan is not None: all_plans.append(plan)
    distinct={p.candidate_id:p for p in all_plans}
    return sorted(distinct.values(), key=lambda p:(len(p.normal_path),p.normal_path,p.candidate_id))[:max_candidates]


def validate_structure(G:nx.Graph,rings:dict[str,list[int]],req:ServiceRequest,
                       plan:ProtectedPlan)->tuple[bool,str]:
    """Strict checker recomputes all scenario paths from segments, not trusting plan fields."""
    if plan.scheme != SCHEME or plan.service_id != req.request_id:
        return False,'PLAN_ID_OR_SCHEME_MISMATCH'
    if req.source not in G or req.destination not in G or req.source==req.destination or req.bandwidth_mbps<=0:
        return False,'INVALID_REQUEST'
    allowed=_segment_specs(G,rings,req.source,req.destination)
    declared=tuple((s.ring_id,s.working[0],s.working[-1]) for s in plan.segments if s.working)
    if len(declared)!=len(plan.segments) or declared not in allowed:
        return False,'ILLEGAL_RING_OR_GATEWAY_SEQUENCE'
    for s in plan.segments:
        if s.ring_id not in rings or s.working not in _ring_options(rings,s.ring_id,s.working[0],s.working[-1]):
            return False,'INVALID_RING_ARC'
    expected=_make_candidate(G,rings,req,plan.segments)
    if expected is None:
        return False,'UNPROTECTED_OR_LOOPING_FAULT_PATH'
    if (expected.candidate_id!=plan.candidate_id or expected.normal_path!=plan.normal_path or
        expected.fault_paths != plan.fault_paths):
        return False,'FAULT_PLAN_NOT_PRECOMPUTED_AS_DECLARED'
    return True,'OK'


class ProtectionSimulator:
    """Atomic protected-service admission, independent of any choice algorithm.

    Admission attempts never change the Phase2 NetworkX graph. The committed
    service store is changed ONLY after a complete successful audit.
    """
    def __init__(self,G:nx.Graph,rings:dict[str,list[int]],headroom_percent:int=10,
                 arc_capacity_overrides:dict[tuple[int,int],int]|None=None):
        if not isinstance(headroom_percent,int) or not 0<=headroom_percent<100:
            raise ValueError('headroom_percent must be integer 0..99')
        if not nx.is_connected(G): raise ValueError('initial graph disconnected')
        self.G=G
        self.rings=rings
        self.headroom_percent=headroom_percent
        # An optional directional physical-rate ceiling (e.g., test fixture or
        # deliberately asymmetric link assumption), never duplicate an edge per ring.
        self.arc_capacity_overrides=dict(arc_capacity_overrides or {})
        for (u,v),rate in self.arc_capacity_overrides.items():
            if not G.has_edge(u,v) or not isinstance(rate,int) or rate<=0 or rate>G.edges[u,v]['capacity_mbps']:
                raise ValueError('INVALID_DIRECTIONAL_CAPACITY_OVERRIDE')
        self.failures=physical_failures(G)
        self.by_id=_links(G)
        self._accepted:dict[str,tuple[ServiceRequest,ProtectedPlan]]={}
        self._check_topology()

    def _check_topology(self):
        from railway_tp.topology import validate_topology
        summary=validate_topology(self.G,self.rings)
        for flag in ('all_rings_valid_cycles','node_ring_membership_consistent','unique_physical_link_ids'):
            if not summary[flag]:raise ValueError(f'INVALID_TOPOLOGY_{flag}')
        if any(not isinstance(d['capacity_mbps'],int) or d['capacity_mbps']<=0
               for _,_,d in self.G.edges(data=True)):
            raise ValueError('INVALID_CAPACITY')

    def available_capacity(self,u:int,v:int)->int:
        c=self.arc_capacity_overrides.get((u,v),self.G.edges[u,v]['capacity_mbps'])
        return (c*(100-self.headroom_percent))//100

    def candidates(self,req:ServiceRequest,max_candidates:int=12)->list[ProtectedPlan]:
        return generate_candidates(self.G,self.rings,req,max_candidates)

    def _ledger(self,pairs:Iterable[tuple[ServiceRequest,ProtectedPlan]]):
        """One scenario at a time, both directions, no cross-scenario load sum."""
        ledgers={scenario:defaultdict(int) for scenario in (NORMAL,*self.failures)}
        for req,plan in pairs:
            for scenario,ledger in ledgers.items():
                path=plan.normal_path if scenario==NORMAL else plan.fault_paths[scenario]
                for u,v in _path_edges(path):
                    ledger[(u,v)]+=req.bandwidth_mbps
                    ledger[(v,u)]+=req.bandwidth_mbps
        return ledgers

    def audit(self,req:ServiceRequest,plan:ProtectedPlan|None)->AdmissionResult:
        if req.request_id in self._accepted:
            return AdmissionResult(False,('DUPLICATE_SERVICE_ID',),req.request_id,plan.candidate_id if plan else None)
        if not isinstance(req.bandwidth_mbps,int) or req.bandwidth_mbps<=0 or req.source==req.destination or req.source not in self.G or req.destination not in self.G:
            return AdmissionResult(False,('INVALID_REQUEST',),req.request_id,plan.candidate_id if plan else None)
        if plan is None:
            return AdmissionResult(False,('NO_POLICY_VALID_PROTECTED_PLAN',),req.request_id,None)
        valid,reason=validate_structure(self.G,self.rings,req,plan)
        if not valid:
            return AdmissionResult(False,(reason,),req.request_id,plan.candidate_id)
        ledgers=self._ledger(list(self._accepted.values())+[(req,plan)])
        worst_ratio=-1.0; worst_margin=None; worst_scenario=None; worst_arc=None; overloads=[]
        for scenario,loads in ledgers.items():
            failed=self.by_id.get(scenario)
            for u,v,d in self.G.edges(data=True):
                for a,b in ((u,v),(v,u)):
                    load=loads.get((a,b),0)
                    cap=self.available_capacity(a,b)
                    if failed is not None and edge_key(a,b)==failed:
                        if load>0: overloads.append((scenario,(a,b),load,0))
                        continue
                    ratio=load/cap if cap else float('inf')
                    margin=cap-load
                    if ratio>worst_ratio or (ratio==worst_ratio and (worst_margin is None or margin<worst_margin)):
                        worst_ratio=ratio;worst_margin=margin;worst_scenario=scenario;worst_arc=(a,b)
                    if load>cap: overloads.append((scenario,(a,b),load,cap))
        if overloads:
            f,arc,load,cap=overloads[0]
            return AdmissionResult(False,('POSTFAULT_CAPACITY_EXCEEDED' if f!=NORMAL else 'NORMAL_CAPACITY_EXCEEDED',),
                                   req.request_id,plan.candidate_id,f,arc,round(worst_ratio,6),worst_margin,
                                   f'First failed budget: scenario={f}, arc={arc}, load={load}, usable={cap} Mb/s; evaluated all faults')
        return AdmissionResult(True,('OK',),req.request_id,plan.candidate_id,
                               worst_scenario,worst_arc,round(worst_ratio,6),worst_margin,
                               f'Checked normal and all {len(self.failures)} unique physical-link failure scenarios; both directions')

    def admit(self,req:ServiceRequest,plan:ProtectedPlan|None)->AdmissionResult:
        result=self.audit(req,plan)
        if result.feasible:
            self._accepted[req.request_id]=(req,plan)
        return result

    def select_greedy(self,req:ServiceRequest,max_candidates:int=12)->AdmissionResult:
        """Demonstration ONLY; Phase4 algorithm comparison is not implemented."""
        candidates=self.candidates(req,max_candidates)
        for c in candidates:
            result=self.audit(req,c)
            if result.feasible:
                return self.admit(req,c)
        if not candidates:
            return self.admit(req,None)
        return self.audit(req,candidates[0])

    def accepted(self)->list[tuple[ServiceRequest,ProtectedPlan]]:
        return list(self._accepted.values())

    def state_digest(self)->str:
        records=[(k,v[0].bandwidth_mbps,v[1].candidate_id) for k,v in sorted(self._accepted.items())]
        return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()

    def ledger_summary(self)->dict:
        current=self._ledger(self._accepted.values())
        peak=[]
        for sc,loads in current.items():
            failed=self.by_id.get(sc)
            for (u,v),load in loads.items():
                if failed == edge_key(u,v) and load>0:
                    raise AssertionError('Committed path traverses failed link')
                if failed == edge_key(u,v): continue
                peak.append((load/self.available_capacity(u,v),sc,u,v,load))
        biggest=max(peak,default=(0,NORMAL,None,None,0))
        return {'admitted_services':len(self._accepted),'physical_faults':len(self.failures),
                'scenarios_checked':len(current),'headroom_percent':self.headroom_percent,
                'peak_utilization':round(biggest[0],6),
                'peak_scenario':biggest[1],'peak_directed_arc':[biggest[2],biggest[3]],
                'peak_reserved_mbps':biggest[4],
                'scope':'scenario-based robust directed-capacity check, not MSRP reservation signalling'}