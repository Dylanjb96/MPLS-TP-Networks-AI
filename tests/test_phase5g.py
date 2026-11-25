"""Phase 5G development-only unit/integration tests (no reserved final data)."""
from __future__ import annotations
import csv,json
import networkx as nx
import numpy as np
import pytest
import torch
from phase5f.protection import ScenarioLedger,verify_pair
from phase5f.traffic import stream_digest
from phase5f_stage2.candidates import generate_pairs
from phase5f_stage2.topology import fingerprint
from phase5g.protocol import check_seed_role
from phase5g.topology import build_network
from phase5g.optimization import scenario_minimax_choice,objective
from phase5g.gnn import graph_observation,GraphCandidateActorCritic,tensors,ppo_update
from phase5g.workflow import (make_cohort,simulate_minimax,simulate_gnn,
                               gnn_smoke,paired_dev_pilot)
from phase5f_stage5.oracle import verify_trace
from phase5f_stage5.freeze import verify_freeze


def _basic(family='ladder',mode='dynamic',load=1.6):
    return make_cohort(nodes=60,topology_seed=501,stream_seed=74000,
         family=family,load=load,mode=mode)


def test_disallow_all_legacy_and_g_final_seed_roles():
    for role in ('dev','train','validation'):
        for t,s in ((401,74000),(501,73000),(801,74000),(501,77000),
                    (601,75000),(701,76000),(501,74000)):
            if (role,t,s) in (('dev',501,74000),('train',601,75000),('validation',701,76000)):
                continue
            with pytest.raises(ValueError):check_seed_role(role,t,s)
    for t,s,role in ((501,74000,'dev'),(601,75000,'train'),(701,76000,'validation')):
        check_seed_role(role,t,s)
    with pytest.raises(ValueError):check_seed_role('final',801,77000)


@pytest.mark.parametrize('n',[60,150,250,500])
def test_ladder_connected_no_bridges_unique_ids(n):
    for seed in (501,502):
        g=build_network(n,seed,'ladder')
        assert len(g)==n and nx.is_connected(g) and not list(nx.bridges(g))
        assert len({x['physical_link_id'] for *_,x in g.edges(data=True)})==g.number_of_edges()
    assert fingerprint(build_network(n,501,'ladder'))!=fingerprint(build_network(n,502,'ladder'))


def test_structural_family_difference():
    a=build_network(60,501,'hierarchical');b=build_network(60,501,'ladder')
    assert fingerprint(a,structural_only=True)!=fingerprint(b,structural_only=True)
    assert a.graph['name']!=b.graph['name']


def test_invalid_family_and_spec():
    with pytest.raises(ValueError):build_network(60,501,'fantasy')
    with pytest.raises(ValueError):build_network(29,501,'ladder')


def test_protected_candidates_are_structurally_valid():
    for family in ('hierarchical','ladder'):
        g,d,inv=_basic(family)
        assert len(inv)==len(d)
        for dem,plans in zip(d[:20],inv[:20]):
            assert plans
            for p in plans:
                assert verify_pair(g,dem.source,dem.destination,p)


def test_common_traffic_static_dynamic_except_expiry():
    _,d1,_=_basic('hierarchical','static')
    _,d2,_=_basic('hierarchical','dynamic')
    assert stream_digest(d1,include_expiry=False)==stream_digest(d2,include_expiry=False)
    assert stream_digest(d1)!=stream_digest(d2)


def test_stress_preserves_source_dest_class_and_priority():
    kw=dict(nodes=60,topology_seed=501,stream_seed=74000,
            family='ladder',load=1.6,mode='dynamic')
    _,a,_=make_cohort(stress='base',**kw)
    _,b,_=make_cohort(stress='bandwidth_150',**kw)
    _,c,_=make_cohort(stress='short_hold',**kw)
    assert all((x.source,x.destination,x.priority_weight,x.service_class)==
               (y.source,y.destination,y.priority_weight,y.service_class)==
               (z.source,z.destination,z.priority_weight,z.service_class)
               for x,y,z in zip(a,b,c))
    assert all(y.bandwidth_mbps>=x.bandwidth_mbps for x,y in zip(a,b))
    assert all(z.expiry<=x.expiry for x,z in zip(a,c))
    with pytest.raises(ValueError):make_cohort(stress='short_hold',mode='static',
         nodes=60,topology_seed=501,stream_seed=74000)


def test_prospective_fiveg_seeds_are_blocked():
    with pytest.raises(ValueError):make_cohort(nodes=60,topology_seed=801,stream_seed=77000)
    with pytest.raises(ValueError):make_cohort(nodes=60,topology_seed=401,stream_seed=73000)


def test_minimax_exact_argmin_finite():
    g,d,inv=_basic()
    ledger=ScenarioLedger(g,10)
    for dem,plans in zip(d,inv):
        feasible=[i for i,p in enumerate(plans) if ledger.feasible(dem,p)]
        if len(feasible)>1:
            chosen=scenario_minimax_choice(ledger,dem,plans,feasible)
            costs=[objective(ledger,dem,plans[i]) for i in feasible]
            assert costs[feasible.index(chosen)]==min(costs)
            assert all(np.isfinite(float(x[0])) and np.isfinite(float(x[1])) for x in costs)
            return
    raise AssertionError('no actionable minmax state')


def test_minimax_trace_independent_oracle():
    g,d,inv=_basic()
    summary,trace=simulate_minimax(g,d,inv)
    oracle=verify_trace(g,d,inv,trace,6)
    assert oracle['accepted']==summary['admitted']
    assert len(trace)==summary['requests']==96
    assert summary['admitted']+summary['rejected']==96


def test_gnn_action_mask_and_finite_output():
    g,d,inv=_basic()
    led=ScenarioLedger(g,10)
    feasible=[i for i,p in enumerate(inv[0]) if led.feasible(d[0],p)]
    obs=graph_observation(g,led,d[0],inv[0],feasible)
    model=GraphCandidateActorCritic(16)
    logits,value=model(tensors(obs))
    assert logits.shape==(6,) and torch.isfinite(logits[torch.as_tensor(obs['mask'])]).all()
    assert torch.isfinite(value)
    assert model.decide(obs,greedy_idx=feasible[0])['action'] in feasible
    obs['mask'][feasible[1:][0]]=False
    assert model.decide(obs,greedy_idx=feasible[0])['action'] in [i for i in feasible if obs['mask'][i]]


def test_gnn_graph_node_permutation_equivariance():
    g,d,inv=_basic()
    led=ScenarioLedger(g,10)
    feasible=[i for i,p in enumerate(inv[0]) if led.feasible(d[0],p)]
    obs=graph_observation(g,led,d[0],inv[0],feasible)
    model=GraphCandidateActorCritic(16).eval()
    a=model(tensors(obs))[0]
    n=len(obs['node']);rng=np.random.default_rng(47)
    perm=rng.permutation(n);rev=np.argsort(perm)
    new=dict(obs)
    new['node']=obs['node'][perm]
    new['idx']=rev[obs['idx']]
    b=model(tensors(new))[0]
    assert torch.allclose(a,b,atol=1e-6)


def test_gnn_nodal_observation_distinguishes_network_families():
    obs=[]
    for family in ('hierarchical','ladder'):
        g,d,inv=_basic(family)
        l=ScenarioLedger(g)
        feasible=[i for i,p in enumerate(inv[0]) if l.feasible(d[0],p)]
        obs.append(graph_observation(g,l,d[0],inv[0],feasible))
    assert obs[0]['edge'].shape!=obs[1]['edge'].shape or not np.array_equal(obs[0]['edge'],obs[1]['edge'])


def test_gnn_capacity_feasibility_oracle():
    g,d,inv=_basic()
    torch.set_num_threads(1);torch.manual_seed(7)
    model=GraphCandidateActorCritic(16)
    summary,trace,_=simulate_gnn(g,d,inv,model,stochastic=False)
    oracle=verify_trace(g,d,inv,trace,6)
    assert oracle['accepted']==summary['admitted']
    assert summary['feasibility_violations']==0


def test_gnn_clipped_update_finite():
    g,d,inv=_basic()
    torch.set_num_threads(1);torch.manual_seed(7)
    model=GraphCandidateActorCritic(16)
    _,_,samples=simulate_gnn(g,d,inv,model,stochastic=True)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    params=[x.detach().clone() for x in model.parameters()]
    result=ppo_update(model,opt,samples[:12],epochs=1,batch_size=4)
    assert result['samples']==min(12,len(samples)) and np.isfinite(result['loss'])
    assert any(not torch.equal(a,b) for a,b in zip(params,model.parameters()))


def test_preflight_frozen_selected_c_sha():
    assert verify_freeze()['selected']['checkpoint_sha256']=='e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9'


def test_dev_pilot_outputs_and_no_overwrite(tmp_path):
    out=tmp_path/'pilot'
    rows=paired_dev_pilot(out,sizes=(60,),families=('ladder',),
        topology_seeds=(501,),stream_seeds=(74000,),load=1.6,
        policies=('greedy','scenario_minimax','C_frozen'))
    assert len(rows)==3
    assert {r['request_sha256'] for r in rows}.__len__()==1
    assert len(list(csv.DictReader((out/'phase5g_dev_pilot.csv').open())))==3
    meta=json.loads((out/'phase5g_dev_manifest.json').read_text())
    assert meta['role']=='PHASE5G_DEVELOPMENT_ONLY_NOT_CONFIRMATORY'
    with pytest.raises(FileExistsError):paired_dev_pilot(out,sizes=(60,),families=('ladder',),
        topology_seeds=(501,),stream_seeds=(74000,),load=1.6,
        policies=('greedy',))


def test_dev_pilot_rejects_final_before_creating_output(tmp_path):
    out=tmp_path/'bad'
    with pytest.raises(ValueError):paired_dev_pilot(out,topology_seeds=(801,),stream_seeds=(77000,))
    assert not out.exists()


def test_gnn_smoke_test_outputs_and_oracle(tmp_path):
    out=tmp_path/'smoke'
    h,v=gnn_smoke(out,episodes=1)
    assert len(h)==1 and len(v)==4
    assert {r['policy'] for r in v}=={'greedy','footprint','scenario_minimax','GNN_PPO'}
    assert json.loads((out/'gnn_smoke_manifest.json').read_text())['role']=='DEVELOPMENT_ONLY_NO_FINAL_SEEDS'
    assert (out/'gnn_engineering_smoke.pt').is_file()
