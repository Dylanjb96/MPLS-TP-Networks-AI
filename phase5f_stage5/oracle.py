"""Second-implementation single-fault capacity checker for *recorded* decisions.

Independent of ScenarioLedger._partition/_indices; symmetric directions are
identical by Stage 1 contract, so the oracle uses one column per physical link.
On a fault to any working edge, it swaps the entire working path to backup.
This is still a comparative single-link-failure surrogate, NOT an MSRP oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np
from phase5f.protection import verify_pair


def undirected(u,v):return (min(u,v),max(u,v))

@dataclass
class _Active:
    expiry:int
    candidate:object
    bandwidth:int

class SingleFaultOracle:
    def __init__(self,graph,headroom=10):
        if not 0<=headroom<100:raise ValueError('bad headroom')
        self.idx={e:i for i,e in enumerate(sorted(undirected(u,v) for u,v in graph.edges()))}
        self.links=len(self.idx)
        self.cap=np.array([graph[u][v]['capacity_mbps']*(100-headroom)//100
                           for u,v in self.idx],dtype=np.int64)
        self.loads=np.zeros((self.links+1,self.links),dtype=np.int64)
        self.active={}
        self.graph=graph

    def _ids(self,path):return np.array([self.idx[undirected(u,v)] for u,v in zip(path,path[1:])],dtype=np.int64)

    def _delta(self,p,bw,sign):
        """Direct scenario semantics: all scenarios use working unless its link fails."""
        wi=self._ids(p.working);bi=self._ids(p.backup)
        self.loads[:,wi]+=sign*bw
        for failed in wi:
            self.loads[failed+1,wi]-=sign*bw
            self.loads[failed+1,bi]+=sign*bw
        return wi,bi

    def release(self,now):
        released=0
        for key,item in list(self.active.items()):
            if item.expiry<=now:
                self._delta(item.candidate,item.bandwidth,-1)
                del self.active[key];released+=1
        if np.any(self.loads<0):raise AssertionError('oracle negative reservation')
        return released

    def feasible(self,d,p):
        if d.request_id in self.active or not verify_pair(self.graph,d.source,d.destination,p):return False
        wi=self._ids(p.working);bi=self._ids(p.backup);bw=d.bandwidth_mbps
        failed_rows=wi+1
        working_rows=np.ones(self.links+1,dtype=bool)
        working_rows[failed_rows]=False
        if np.any(self.loads[np.ix_(working_rows,wi)]+bw>self.cap[wi]):return False
        if np.any(self.loads[np.ix_(failed_rows,bi)]+bw>self.cap[bi]):return False
        return True

    def admit(self,d,p):
        if not self.feasible(d,p):raise AssertionError('oracle rejects selected path')
        self._delta(p,int(d.bandwidth_mbps),1)
        if np.any(self.loads>self.cap) or np.any(self.loads<0):
            raise AssertionError('oracle scenario reservation exceeds capacity')
        self.active[d.request_id]=_Active(d.expiry,p,int(d.bandwidth_mbps))

    def digest(self):
        return hashlib.sha256(self.loads.tobytes()).hexdigest()


def verify_trace(graph,demands,inventory,records,k,headroom=10, *,verify_rejections=True):
    if len(records)!=len(demands) or len(inventory)!=len(demands):
        raise AssertionError('oracle missing decision records')
    oracle=SingleFaultOracle(graph,headroom)
    admitted=0;overrides=0;discretionary=0;released=0
    for pos,(d,plans,r) in enumerate(zip(demands,inventory,records)):
        if int(r['request_index'])!=pos or str(r['request_id'])!=str(d.request_id):
            raise AssertionError('oracle decision/request mismatch')
        released+=oracle.release(d.arrival)
        available=plans[:k]
        if len(available)!=int(r['structural_candidates']):
            raise AssertionError('oracle candidate prefix mismatch')
        picked=int(r['chosen_index']);greedy=int(r['greedy_index'])
        # Sample the complete candidate-feasibility mask independently every
        # 64th decision. All chosen routes and every rejection are checked.
        if pos%64==0:
            feasible_indices=[i for i,p in enumerate(available) if oracle.feasible(d,p)]
            if (len(feasible_indices)!=int(r['feasible_candidates']) or
                greedy!=(feasible_indices[0] if feasible_indices else -1)):
                raise AssertionError('independent candidate feasibility mask mismatch')
        if picked<0:
            if int(r['admitted'])!=0 or greedy!=-1:raise AssertionError('invalid rejection trace')
            if verify_rejections and any(oracle.feasible(d,p) for p in available):
                raise AssertionError('non-forced rejection detected by independent oracle')
        else:
            if picked>=len(available) or greedy<0 or int(r['admitted'])!=1:
                raise AssertionError('invalid chosen route')
            if r['chosen_candidate_id']!=available[picked].candidate_id:
                raise AssertionError('chosen path identifier mismatch')
            oracle.admit(d,available[picked]);admitted+=1
            overrides+=int(picked!=greedy)
        if int(r['overridden'])!=int(picked>=0 and picked!=greedy):
            raise AssertionError('wrong override trace flag')
        discretionary+=int(r['feasible_candidates']>1)
        if int(r['active_after'])!=len(oracle.active) or int(r['cumulative_admitted'])!=admitted:
            raise AssertionError('active/accepted count differs from direct oracle')
    return dict(accepted=admitted,departures=released,active_end=len(oracle.active),
                overrides=overrides,discretionary=discretionary,
                oracle_state_sha256=oracle.digest())
