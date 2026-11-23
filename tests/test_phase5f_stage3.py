"""Stage 3 regression tests: causal observation, masking, feasibility, seed isolation."""
from pathlib import Path
import csv
import json
import numpy as np
import pytest
import torch
from phase5f.traffic import generate_demands
from phase5f.protection import ScenarioLedger
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs
from phase5f_stage3.features import encode,MAX_ACTIONS,CAND_DIM,GLOBAL_DIM
from phase5f_stage3.agent import CandidateActorCritic,ppo_update
from phase5f_stage3.workflow import run_episode,run_smoke,make_cohort,_validate_seed_role


@pytest.fixture(scope='module')
def fixture_stream():
    g=build_network(NetworkSpec(nodes=60,seed=201))
    demands=generate_demands(g,28,71000,True)
    inventory=tuple(generate_pairs(g,d.source,d.destination,6,40) for d in demands)
    return g,demands,inventory


def test_feature_shapes_mask_feasible_only(fixture_stream):
    g,ds,inv=fixture_stream
    ledger=ScenarioLedger(g,10)
    idx=[i for i,p in enumerate(inv[0]) if ledger.feasible(ds[0],p)]
    c,global_,mask=encode(g,ledger,ds[0],inv[0],idx,dynamic=True)
    assert c.shape==(MAX_ACTIONS,CAND_DIM) and global_.shape==(GLOBAL_DIM,)
    assert mask.shape==(MAX_ACTIONS,) and mask.sum()==len(idx)
    assert all(mask[i] for i in idx)
    assert np.all(c[~mask]==0)
    assert np.isfinite(c).all() and np.isfinite(global_).all()


def test_features_never_expose_future_expiry_or_suffix(fixture_stream):
    from dataclasses import replace
    g,ds,inv=fixture_stream
    ledger=ScenarioLedger(g,10)
    indices=[i for i,p in enumerate(inv[0]) if ledger.feasible(ds[0],p)]
    first=encode(g,ledger,ds[0],inv[0],indices,dynamic=True)
    changed=replace(ds[0],expiry=999999)
    second=encode(g,ledger,changed,inv[0],indices,dynamic=True)
    for a,b in zip(first,second):assert np.array_equal(a,b)


def test_model_valid_mask_and_gradient(fixture_stream):
    g,ds,inv=fixture_stream
    led=ScenarioLedger(g,10)
    feasible=[i for i,p in enumerate(inv[0]) if led.feasible(ds[0],p)]
    obs=encode(g,led,ds[0],inv[0],feasible)
    model=CandidateActorCritic(hidden=32)
    result=model.decide(obs,greedy_idx=feasible[0],deterministic=True)
    assert result['action'] in feasible
    c,glob,mask=obs
    logits,value=model(torch.tensor(c)[None],torch.tensor(glob)[None],torch.tensor(mask)[None])
    assert logits.shape==(1,MAX_ACTIONS) and value.shape==(1,)
    assert (logits[0,~torch.tensor(mask)]<-1e8).all()


def test_permutation_invariant_shared_scorer():
    torch.manual_seed(11)
    model=CandidateActorCritic(hidden=32)
    c=torch.randn(1,MAX_ACTIONS,CAND_DIM)
    gl=torch.randn(1,GLOBAL_DIM)
    mask=torch.tensor([[True,False,True,True,False,True]])
    logits,v=model(c,gl,mask)
    perm=[5,3,2,0,4,1]
    logits2,v2=model(c[:,perm],gl,mask[:,perm])
    assert torch.allclose(logits2,logits[:,perm])
    assert torch.allclose(v,v2)


def test_forced_rejection_only_and_recompute(fixture_stream):
    g,ds,inv=fixture_stream
    for policy in ('greedy','footprint'):
        stats,obs,trace=run_episode(g,ds,inv,policy=policy,k=6,mode='dynamic',trace=True)
        assert stats['admitted']+stats['rejected']==len(ds)
        assert stats['feasibility_violations']==0
        assert len(trace)==len(ds)
        assert stats['overrides']==sum(row['overridden'] for row in trace)
        assert all(row['admitted'] or row['feasible_candidates']==0 for row in trace)


def test_model_can_actually_override_and_still_pass_feasibility(fixture_stream):
    g,ds,inv=fixture_stream
    class OverridePolicy:
        def decide(self,obs,*,greedy_idx,**kwargs):
            indices=np.flatnonzero(obs[2]).tolist()
            alt=indices[-1]
            return dict(action=alt,log_prob=0.0,value=0.0,greedy_probability=0.1)
    res,_,trace=run_episode(g,ds,inv,policy='hybrid',model=OverridePolicy(),
                           k=6,mode='dynamic',trace=True)
    assert res['overrides']>0
    assert res['feasibility_violations']==0
    assert all(row['chosen_index']==-1 or row['chosen_index']<6 for row in trace)


def test_ppo_update_on_policy_gradient(fixture_stream):
    torch.manual_seed(13)
    g,ds,inv=fixture_stream
    model=CandidateActorCritic(hidden=32)
    stats,history,_=run_episode(g,ds,inv,policy='hybrid',model=model,
                                k=6,mode='dynamic',stochastic=True)
    assert len(history)>0
    assert all(0<=o['return']<=1.0000001 for o in history)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0003)
    prev=[p.detach().clone() for p in model.parameters()]
    update=ppo_update(model,optimizer,history,epochs=1,batch_size=32,seed=5)
    assert update['n']==len(history)
    assert np.isfinite(update['loss'])
    assert any(not torch.equal(a,b) for a,b in zip(prev,model.parameters()))


@pytest.mark.parametrize('role,badtop,badstream',[
    ('train',(401,),(71000,)),('train',(201,),(73000,)),
    ('validation',(401,),(72000,)),('validation',(301,),(73000,)),
])
def test_heldout_seeds_rejected(role,badtop,badstream):
    with pytest.raises(ValueError):
        _validate_seed_role(role,badtop,badstream)


def test_cohort_repeat_has_exact_same_candidates():
    a=make_cohort(60,201,71000,1.6,'static',6,40)
    b=make_cohort(60,201,71000,1.6,'static',6,40)
    assert a is b
    assert fingerprint(a[0])==fingerprint(b[0])
    assert a[2]==b[2]


def test_smoke_creates_separate_outputs_and_frozen_provenance(tmp_path):
    out=tmp_path/'stage3_dev'
    result=run_smoke(out,nodes=60,episodes=2,train_topologies=(201,),
        train_streams=(71000,),val_topologies=(301,),val_streams=(72000,),
        load=1.6,modes=('static',),k=6,agent_seed=7)
    assert result['final_data_accessed'] is False
    assert result['training_actions']>0
    assert result['validation_rows']==3
    assert (out/'smoke_model.pt').exists()
    assert (out/'validation_hybrid_decisions.csv').exists()
    with (out/'validation_summary.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==3
    assert {r['policy'] for r in rows}=={'greedy','footprint','hybrid'}
    assert all(int(r['feasibility_violations'])==0 for r in rows)
    data=json.loads((out/'stage3_smoke_manifest.json').read_text())
    assert data['protection_model_approved'] is False
    assert data['msrp_compliant'] is False
    assert not data['final_data_accessed']
    with pytest.raises(FileExistsError):
        run_smoke(out,nodes=60,episodes=1)
