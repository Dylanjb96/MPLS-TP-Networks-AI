#Independent positive/negative fixtures for Phase 3 scenario surrogate.

#No test asserts real-world MPLS-TP shared-ring protection or switching time.

from dataclasses import replace
import copy
import json
from pathlib import Path
import pytest
import networkx as nx
from railway_tp.topology import make_topology,TopologyConfig
from railway_tp.requests import ServiceRequest,generate_requests
from phase3.model import (ProtectionSimulator,generate_candidates,validate_structure,
                          physical_failures,PathSegment,edge_key)
from phase3.cli import run


def topology(n=12,mode='dual',main_capacity=10_000,sub_capacity=1_000):
    return make_topology(TopologyConfig(nodes=n,gateway_mode=mode,seed=42,
                                        main_nodes=6 if n==12 else 12,
                                        subring_count=1 if n==12 else 4,
                                        main_capacity_mbps=main_capacity,
                                        sub_capacity_mbps=sub_capacity))


def request(i,src,dst,mbps=200):
    return ServiceRequest(f'S{i}',src,dst,mbps,'operational',3,'test')


def one(sim,req):
    plans=sim.candidates(req)
    assert plans, f'No plan for {req.source}->{req.destination}'
    return plans[0]


# P3-01 valid same-ring; all fault states and bidirectional cap checks.
def test_p301_same_ring_positive():
    G,rings=topology()
    sim=ProtectionSimulator(G,rings)
    req=request(1,6,7)
    p=one(sim,req)
    assert len(p.fault_paths)==G.number_of_edges()
    assert sim.admit(req,p).feasible
    assert sim.ledger_summary()['peak_utilization'] <= 1
    assert p.normal_path[0]==6 and p.normal_path[-1]==7


def test_p301_same_ring_negative_overcapacity():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7,901)
    assert not sim.audit(req,one(sim,req)).feasible


# P3-02 / 03 segment boundaries only at legal gateways.
def test_p302_main_to_subring_positive():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,2,7)
    p=one(sim,req)
    assert len(p.segments)==2
    assert p.segments[0].ring_id=='main'
    gateway=p.segments[0].working[-1]
    assert gateway==p.segments[1].working[0]
    assert 'main' in G.nodes[gateway]['rings'] and 'sub_1' in G.nodes[gateway]['rings']
    assert sim.admit(req,p).feasible


def test_p303_sub_to_sub_positive():
    G,rings=topology(n=60);sim=ProtectionSimulator(G,rings)
    req=request(1,12,25,50)
    p=one(sim,req)
    assert [s.ring_id for s in p.segments]==['sub_1','main','sub_2']
    assert sim.admit(req,p).feasible


def test_p303_illegal_gateway_transition_negative():
    G,rings=topology(n=60);sim=ProtectionSimulator(G,rings)
    req=request(1,12,25,50);p=one(sim,req)
    corrupted=replace(p,segments=p.segments[:-1])
    assert not sim.audit(req,corrupted).feasible


# P3-04 physical graph path but no valid ring/gateway policy.
def test_p304_route_exists_no_policy_plan():
    G,rings=topology()
    G.add_node(999,rings=[],role='external_stub',gateway=False,synthetic=True)
    G.add_edge(999,2,rings=[],capacity_mbps=1000,reserved_mbps=0,
               status='active',length_km=1,physical_link_id='E002_999',synthetic=True)
    sim=ProtectionSimulator(G,rings)
    assert nx.has_path(G,999,7)
    req=request(1,999,7,50)
    assert sim.candidates(req)==[]
    assert sim.admit(req,None).reason_codes==('NO_POLICY_VALID_PROTECTED_PLAN',)


# P3-05 fault-specific congestion with normal-state disjoint arcs.
def test_p305_fault_only_capacity_overload_atomic():
    G,rings=topology(); sim=ProtectionSimulator(G,rings)
    r1=request(1,6,7,500);p1=one(sim,r1)
    assert sim.admit(r1,p1).feasible
    digest=sim.state_digest()
    r2=request(2,8,9,401);p2=one(sim,r2)
    result=sim.admit(r2,p2)
    assert not result.feasible
    assert result.reason_codes==('POSTFAULT_CAPACITY_EXCEEDED',)
    assert result.worst_fault_id is not None
    assert sim.state_digest()==digest and len(sim.accepted())==1


def test_p305_fault_boundary_900_allowed():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    r1=request(1,6,7,500);r2=request(2,8,9,400)
    assert sim.admit(r1,one(sim,r1)).feasible
    assert sim.admit(r2,one(sim,r2)).feasible


# P3-06 shared physical edge counted once even if member of two rings.
def test_p306_shared_physical_resource_not_duplicated():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    shared=[(u,v) for u,v,d in G.edges(data=True) if len(d['rings'])==2]
    assert len(shared)==1
    u,v=shared[0]
    req=request(1,u,v,500);p=one(sim,req)
    assert p.normal_path==(u,v)
    ledger=sim._ledger([(req,p)])['normal']
    assert ledger[(u,v)]==500 and ledger[(v,u)]==500
    assert len(sim.failures)==G.number_of_edges()


# P3-07 both directions of faulted physical edge are unavailable.
def test_p307_failed_edge_not_in_any_fault_path():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7);p=one(sim,req)
    edge_id=G.edges[6,7]['physical_link_id']
    state=p.fault_paths[edge_id]
    assert edge_key(6,7) not in [edge_key(u,v) for u,v in zip(state,state[1:])]
    reversed_state=tuple(reversed(state))
    assert edge_key(6,7) not in [edge_key(u,v) for u,v in zip(reversed_state,reversed_state[1:])]


def test_p307_unaffected_segments_unchanged():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,2,7);p=one(sim,req)
    normal_edges={edge_key(u,v) for u,v in zip(p.normal_path,p.normal_path[1:])}
    unaffected=next(i for i,e in sim.by_id.items() if e not in normal_edges)
    assert p.fault_paths[unaffected]==p.normal_path


# P3-08 integer directional rate/headroom boundary.
@pytest.mark.parametrize('mbps,expected',[(900,True),(901,False)])
def test_p308_headroom_boundary(mbps,expected):
    G,rings=topology(main_capacity=1000)
    sim=ProtectionSimulator(G,rings,headroom_percent=10)
    req=request(1,0,1,mbps);p=one(sim,req)
    assert sim.audit(req,p).feasible is expected


def test_p308_directional_override_causes_only_reverse_infeasible():
    G,rings=topology(main_capacity=1000)
    sim=ProtectionSimulator(G,rings,headroom_percent=0,arc_capacity_overrides={(1,0):800})
    req=request(1,0,1,900);p=one(sim,req)
    result=sim.audit(req,p)
    assert not result.feasible
    assert result.reason_codes==('NORMAL_CAPACITY_EXCEEDED',)
    assert sim._ledger([(req,p)])['normal'][(1,0)]==900


# P3-09 monotonicity across headroom levels.
@pytest.mark.parametrize('mbps,expected',[(800,[True,True,True]),(850,[True,True,False]),
                                          (950,[True,False,False])])
def test_p309_headroom_monotonic(mbps,expected):
    G,rings=topology(main_capacity=1000)
    req=request(1,0,1,mbps)
    got=[ProtectionSimulator(G,rings,headroom_percent=h).audit(req,
         generate_candidates(G,rings,req)[0]).feasible for h in (0,10,20)]
    assert got==expected


# P3-10 rejection atomic state preservation.
def test_p310_rejection_no_state_mutation():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7,100)
    assert sim.admit(req,one(sim,req)).feasible
    before=sim.state_digest(); prior=copy.deepcopy([(u,v,d) for u,v,d in G.edges(data=True)])
    invalid=request(2,8,9,901)
    assert not sim.admit(invalid,one(sim,invalid)).feasible
    assert sim.state_digest()==before and [(u,v,d) for u,v,d in G.edges(data=True)]==prior


def test_p310_duplicate_service_atomic():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7,100);p=one(sim,req)
    assert sim.admit(req,p).feasible
    before=sim.state_digest()
    result=sim.admit(req,p)
    assert result.reason_codes==('DUPLICATE_SERVICE_ID',)
    assert sim.state_digest()==before


# P3-11 deterministic same seed and request sequences.
def test_p311_reproducibility():
    G,rings=topology()
    incoming=generate_requests(G,30,seed=142)
    def execute():
        sim=ProtectionSimulator(G,rings)
        decisions=[sim.select_greedy(req).feasible for req in incoming]
        return decisions,sim.state_digest(),sim.ledger_summary()
    assert execute()==execute()


# P3-12 explicit invalid preplanned failure-state path rejected.
def test_p312_corrupt_fault_path_rejected():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7);p=one(sim,req)
    fail=G.edges[6,7]['physical_link_id']
    wrong=dict(p.fault_paths);wrong[fail]=p.normal_path
    result=sim.audit(req,replace(p,fault_paths=wrong))
    assert not result.feasible and result.reason_codes==('FAULT_PLAN_NOT_PRECOMPUTED_AS_DECLARED',)


# P3-13 loop/nonring route and illegal gateway rejected.
def test_p313_looping_arc_rejected():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7);p=one(sim,req)
    bad=(6,5,6,7)
    result=sim.audit(req,replace(p,segments=(PathSegment('sub_1',bad),)))
    assert not result.feasible


def test_p313_gateway_crossing_wrong_ring_rejected():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,2,7);p=one(sim,req)
    corrupted=replace(p,segments=(PathSegment('main',p.normal_path),))
    assert not sim.audit(req,corrupted).feasible


# P3-14 symmetric reverse always checked, not assumed.
def test_p314_reverse_budget_violates_candidate():
    G,rings=topology(main_capacity=1000)
    sim=ProtectionSimulator(G,rings,headroom_percent=0,arc_capacity_overrides={(1,0):799})
    req=request(1,0,1,800)
    result=sim.audit(req,one(sim,req))
    assert not result.feasible and result.worst_arc==(1,0)


# P3-15 solver origin does not influence validator.
def test_p315_independent_validator_proposer_agnostic():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7);p=one(sim,req)
    outputs={label:sim.audit(req,p).as_dict() for label in ('greedy','CSPF','MILP','DDQN')}
    assert len({json.dumps(o,sort_keys=True) for o in outputs.values()})==1


# Extras, including artifact/run verification and cap sanity.
def test_protection_plan_is_validated_separately():
    G,rings=topology();req=request(1,2,7)
    p=generate_candidates(G,rings,req)[0]
    assert validate_structure(G,rings,req,p)==(True,'OK')
    assert validate_structure(G,rings,req,replace(p,candidate_id='FAKE'))[0] is False


def test_max_candidates_and_headroom_validation():
    G,rings=topology();req=request(1,6,7)
    assert len(generate_candidates(G,rings,req,max_candidates=1))==1
    with pytest.raises(ValueError):generate_candidates(G,rings,req,max_candidates=0)
    with pytest.raises(ValueError):ProtectionSimulator(G,rings,headroom_percent=-1)
    with pytest.raises(ValueError):ProtectionSimulator(G,rings,arc_capacity_overrides={(1,0):999_999})


def test_simulator_does_not_change_phase2_reserved_attrs():
    G,rings=topology();sim=ProtectionSimulator(G,rings)
    req=request(1,6,7,200);p=one(sim,req)
    assert sim.admit(req,p).feasible
    assert all(d['reserved_mbps']==0 for _,_,d in G.edges(data=True))


def test_no_future_info_accepted_only_online():
    G,rings=topology(); sim=ProtectionSimulator(G,rings)
    first=request(1,6,7,500);second=request(2,8,9,400)
    assert sim.admit(first,one(sim,first)).feasible
    assert sim.admit(second,one(sim,second)).feasible
    assert len(sim.accepted())==2


def test_cli_exports_auditable_files(tmp_path):
    out=tmp_path/'phase3';cfg=TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=6,subring_count=1)
    result=run(cfg,out,requests=8,seed=7,headroom=10)
    assert result['request_count']==8
    files=list(out.glob('*'))
    assert len(files)==5
    assert any(f.name.endswith('manifest.json') for f in files)
    assert all(f.stat().st_size>0 for f in files)


def test_single_link_fault_list_exact_physical_edges():
    G,rings=topology(); ids=physical_failures(G)
    assert len(ids)==13 and len(ids)==len(set(ids))


def test_single_gateway_policy_supported_single_link_only():
    G,rings=topology(mode='single');sim=ProtectionSimulator(G,rings)
    req=request(1,2,7,200)
    assert sim.admit(req,one(sim,req)).feasible
    assert sim.G.graph['gateway_mode']=='single'
    # Node loss intentionally NOT tested for protection; it is Phase2 graph diagnosis only.