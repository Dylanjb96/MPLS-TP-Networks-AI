#Phase 4 independent policy, MILP formulation and experiment integrity tests.
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from railway_tp.topology import TopologyConfig, make_topology
from railway_tp.requests import generate_requests
from phase3.model import NORMAL, ProtectionSimulator
from phase4.policies import (POLICIES, choose, choose_cspf, choose_greedy,
                             choose_milp, _milp_select)
from phase4.cli import benchmark, run_one


@pytest.fixture(scope='module')
def network12():
    config=TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=6,
                          subring_count=1,main_capacity_mbps=10000,
                          sub_capacity_mbps=1000,length_km=2.0)
    G,rings=make_topology(config)
    return config,G,rings


@pytest.fixture(scope='module')
def network60():
    config=TopologyConfig(nodes=60,seed=42,gateway_mode='dual',main_nodes=12,
                          subring_count=4,main_capacity_mbps=10000,
                          sub_capacity_mbps=1000,length_km=2.0)
    G,rings=make_topology(config)
    return config,G,rings


@pytest.mark.parametrize('policy', POLICIES)
def test_policy_commits_only_valid_plan(network12,policy):
    _,G,rings=network12
    sim=ProtectionSimulator(G,rings,10)
    req=generate_requests(G,1,seed=2026)[0]
    candidates=sim.candidates(req,100)
    outcome=choose(sim,req,candidates,policy)
    assert outcome.result.feasible
    assert len(sim.accepted())==1
    assert sim.audit(req,candidates[0]).reason_codes == ('DUPLICATE_SERVICE_ID',)
    assert sim.ledger_summary()['peak_utilization'] <= 1.0 + 1e-9


@pytest.mark.parametrize('policy',POLICIES)
def test_invalid_request_cannot_commit(network12,policy):
    _,G,rings=network12
    sim=ProtectionSimulator(G,rings,10)
    req=generate_requests(G,1,seed=55)[0]
    invalid=replace(req,source=10000)
    prior=sim.state_digest()
    outcome=choose(sim,invalid,[],policy)
    assert outcome.result.feasible is False
    assert sim.state_digest()==prior


@pytest.mark.parametrize('policy',POLICIES)
def test_impossible_bandwidth_atomic(network12,policy):
    _,G,rings=network12
    sim=ProtectionSimulator(G,rings,10)
    req=generate_requests(G,1,seed=26)[0]
    req=replace(req,bandwidth_mbps=20000)
    cands=sim.candidates(req,100)
    assert cands
    prior=sim.state_digest()
    outcome=choose(sim,req,cands,policy)
    assert not outcome.result.feasible
    assert len(sim.accepted()) == 0
    assert sim.state_digest()==prior


def test_first_feasible_matches_phase3(network60):
    _,G,rings=network60
    A=ProtectionSimulator(G,rings,10)
    B=ProtectionSimulator(G,rings,10)
    for req in generate_requests(G,12,seed=2030):
        original=A.select_greedy(req,12)
        fresh=choose_greedy(B,req,B.candidates(req,12)).result
        assert (original.feasible,original.candidate_id,original.reason_codes) == \
               (fresh.feasible,fresh.candidate_id,fresh.reason_codes)
    assert A.state_digest()==B.state_digest()


def test_cspf_selects_shortest_robust_feasible(network60):
    _,G,rings=network60
    sim=ProtectionSimulator(G,rings,10)
    for req in generate_requests(G,8,seed=2027):
        choices=sim.candidates(req,12)
        expected=[p for p in choices if sim.audit(req,p).feasible]
        choice=choose_cspf(sim,req,choices)
        if expected:
            assert choice.result.feasible
            assert len(choice.plan.normal_path)==min(len(p.normal_path) for p in expected)
        else:
            assert not choice.result.feasible


def _max_util(sim, req, plan):
    ledgers=sim._ledger(sim.accepted()+[(req,plan)])
    return max((load/sim.available_capacity(a,b)
                for scenario, arcs in ledgers.items()
                for (a,b),load in arcs.items()),default=0.0)


def test_online_milp_minmax_matches_exhaustive_candidates(network60):
    _,G,rings=network60
    sim=ProtectionSimulator(G,rings,10)
    for req in generate_requests(G,7,seed=2027):
        candidates=sim.candidates(req,20)
        feasible=[p for p in candidates if sim.audit(req,p).feasible]
        outcome=choose_milp(sim,req,candidates,timeout_sec=4)
        if feasible:
            assert outcome.result.feasible
            assert outcome.solver_status == 'OPTIMAL_ONLINE_MINIMAX'
            oracle = min(_max_util(
                # The chosen candidate has just been committed; reconstruct
                # previous state by excluding current request for comparison.
                _sim_except(G,rings,sim,req.request_id), req, candidate)
                for candidate in feasible)
            previous = _sim_except(G,rings,sim,req.request_id)
            chosen_score=_max_util(previous,req,outcome.plan)
            assert chosen_score <= oracle + 1e-6
        else:
            assert not outcome.result.feasible
            assert outcome.solver_status=='INFEASIBLE'


def _sim_except(G,rings,sim,service_id):
    copy=ProtectionSimulator(G,rings,sim.headroom_percent)
    for old_req,old_plan in sim.accepted():
        if old_req.request_id!=service_id:
            assert copy.admit(old_req,old_plan).feasible
    return copy


def test_candidate_equal_pool_policy_identifiers(network60):
    _,G,rings=network60
    req=generate_requests(G,1,seed=2027)[0]
    ids=[]
    for p in POLICIES:
        sim=ProtectionSimulator(G,rings,10)
        ids.append([c.candidate_id for c in sim.candidates(req,2)])
    assert ids[0]==ids[1]==ids[2]
    assert len(ids[0])==2


def test_milp_nonoptimal_fallback_is_labelled(network12,monkeypatch):
    import phase4.policies as mod
    _,G,rings=network12
    sim=ProtectionSimulator(G,rings,10)
    req=generate_requests(G,1,seed=2026)[0]
    monkeypatch.setattr(mod, '_milp_select',
                        lambda sim,req,candidates,timeout_sec: (None,'SOLVER_NONOPTIMAL_STATUS_1',None))
    outcome=choose_milp(sim,req,sim.candidates(req,12))
    assert outcome.result.feasible
    assert 'FALLBACK_GREEDY_SOLVER_NONOPTIMAL_STATUS_1'==outcome.solver_status


def test_milp_timeout_validation(network12):
    _,G,rings=network12
    sim=ProtectionSimulator(G,rings,10)
    req=generate_requests(G,1,seed=2026)[0]
    with pytest.raises(ValueError,match='timeout'):
        choose_milp(sim,req,sim.candidates(req,12),timeout_sec=0)


def test_run_files_sha_consistency(network12,tmp_path):
    cfg,G,_=network12
    reqs=generate_requests(G,5,seed=2027)
    output=tmp_path/'one'
    row=run_one(cfg,reqs,'milp',10,2,output)
    assert row['requests']==5
    assert row['admitted']+row['rejected']==5
    manifest=json.loads((output/'manifest.json').read_text())
    for filename,expected in manifest['sha256'].items():
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest()==expected
    assert row['fallback_count']==0


def test_paired_benchmark_identical_requests(network12,tmp_path):
    cfg,G,_=network12
    output=tmp_path/'benchmark'
    result=benchmark(cfg,6,[2027,2028],[10],list(POLICIES),output,2)
    assert result['runs']==6
    rows=list(__import__('csv').DictReader((output/'phase4_runs.csv').open()))
    assert len(rows)==6
    assert len(set(r['request_sha256'] for r in rows if r['demand_seed']=='2027'))==1
    assert len(set(r['request_sha256'] for r in rows if r['demand_seed']=='2028'))==1
    assert all(int(r['admitted'])+int(r['rejected'])==6 for r in rows)
    assert (output/'phase4_paired_deltas.csv').exists()
    with pytest.raises(FileExistsError):
        benchmark(cfg,6,[2027],[10],['greedy'],output,2)


@pytest.mark.parametrize('bad_kwargs',[
    {'requests':0}, {'seeds':[2027,2027]}, {'headrooms':[10,10]},
    {'policies':['milp','milp']}, {'max_candidates':0},
    {'headrooms':[100]}, {'milp_timeout':0}, {'policies':['clairvoyant']}
])
def test_reject_invalid_experimental_design(network12,tmp_path,bad_kwargs):
    cfg,_,_=network12
    params=dict(cfg=cfg,requests=3,seeds=[2027],headrooms=[10],
                policies=['greedy'],output=tmp_path/'bad',max_candidates=2,
                milp_timeout=5.0)
    params.update(bad_kwargs)
    with pytest.raises(ValueError):
        benchmark(**params)
