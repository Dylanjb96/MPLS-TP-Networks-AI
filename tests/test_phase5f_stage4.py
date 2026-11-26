"""Phase 5F Stage 4 development-only safety, reproducibility and integration tests."""
from pathlib import Path
import csv,json,torch,numpy as np,pytest
from phase5f_stage4 import VERSION
from phase5f_stage4.ablation import CONFIGS,transform,MaskedFeaturePolicy,mask_training_observations
from phase5f_stage4.workflow import (TrainPlan,CohortCache,run_training,validate_role_settings,
                                     _schedule,_val_means,_source_hashes)
from phase5f_stage4.selection import select_models
from phase5f_stage3.agent import CandidateActorCritic
from phase5f_stage3.features import CAND_DIM,GLOBAL_DIM,MAX_ACTIONS,encode
from phase5f.protection import ScenarioLedger
from phase5f.traffic import generate_demands
from phase5f_stage2.topology import build_network,NetworkSpec
from phase5f_stage2.candidates import generate_pairs


def _small(config='A_standard',seed=7,episodes=2):
    return TrainPlan(config=config,agent_seed=seed,episodes=episodes,validate_every=1,
        patience=0,train_sizes=(60,),train_loads=(1.6,),train_modes=('static',),
        train_topologies=(201,),train_streams=(71000,),
        val_topologies=(301,),val_streams=(72000,),val_size=60,val_load=1.6,val_mode='static')


def test_factorial_four_arms_exact():
    assert [(c.guidance,c.enhanced) for c in CONFIGS.values()]==[
        (False,False),(True,False),(False,True),(True,True)]


def test_feature_ablation_and_no_greedy_leakage():
    c=np.ones((MAX_ACTIONS,CAND_DIM),np.float32)
    g=np.ones((GLOBAL_DIM,),np.float32)
    m=np.array([True,False,True,False,False,False])
    a=transform((c,g,m),CONFIGS['A_standard'])
    b=transform((c,g,m),CONFIGS['B_guided'])
    enhanced=transform((c,g,m),CONFIGS['C_enhanced'])
    full=transform((c,g,m),CONFIGS['D_full_hybrid'])
    assert np.all(a[0][:,12]==0) and np.all(enhanced[0][:,12]==0)
    assert b[0][0,12]==1 and full[0][0,12]==1
    assert a[0][0,5]==0 and b[0][0,7]==0
    assert enhanced[0][0,5]==1 and full[0][0,7]==1
    assert np.all(a[0][~m]==0)
    assert not np.shares_memory(c,a[0])


def test_feature_transform_is_same_in_training_and_online():
    obs=(np.ones((6,17),np.float32),np.ones(8,np.float32),
         np.array([True,True,False,False,False,False]))
    hist=[dict(t=0,c=obs[0],g=obs[1],mask=obs[2],action=0,old_logp=0.0,greedy=0,**{"return":0.5})]
    for cfg in CONFIGS.values():
        transformed=transform(obs,cfg)
        masked=mask_training_observations(hist,cfg)[0]
        for expected,actual in zip(transformed,(masked['c'],masked['g'],masked['mask'])):
            assert np.array_equal(expected,actual)


@pytest.mark.parametrize('role,topo,stream',[('train',401,71000),('train',201,73000),
  ('validation',401,72000),('validation',301,73000),
  ('train',301,71000),('validation',201,72000)])
def test_reserved_or_cross_role_seeds_rejected(role,topo,stream):
    with pytest.raises(ValueError):
        if role=='train':validate_role_settings((topo,),(stream,),(301,),(72000,))
        else:validate_role_settings((201,),(71000,),(topo,),(stream,))


def test_plan_defaults_frozen_and_heldout_disjoint():
    p=TrainPlan('D_full_hybrid',7)
    p.validate()
    assert p.val_size==500 and p.val_load==4.0 and p.val_mode=='dynamic'
    assert set(p.train_topologies).isdisjoint(set(range(401,421)))
    assert set(p.val_streams).isdisjoint(set(range(73000,73003)))
    assert set(p.train_sizes)=={60,150,250,500}
    assert set(p.train_loads)=={1.6,2.5,4.0}


def test_schedule_includes_all_load_mode_size_combinations():
    schedule=_schedule(TrainPlan('D_full_hybrid',7))
    assert len(schedule)==96
    assert len({(r['nodes'],r['load'],r['mode']) for r in schedule[:24]})==24
    assert len({r['topology_seed'] for r in schedule})==20
    assert len({r['stream_seed'] for r in schedule})==8
    assert _schedule(TrainPlan('D_full_hybrid',7))==schedule


def test_cohort_cache_hashes_reproduced_and_heldout_blocked(tmp_path):
    c=CohortCache(tmp_path/'cache')
    first=c.get(60,201,71000,1.6,'static',6,40,'train')
    second=c.get(60,201,71000,1.6,'static',6,40,'train')
    assert first[3]==second[3] and first[2]==second[2]
    assert c.hits==1 and c.misses==1
    with pytest.raises(ValueError):c.get(60,401,73000,1.6,'static',6,40,'train')
    with pytest.raises(ValueError):c.get(60,401,73000,1.6,'static',6,40,'validation')


def test_validation_summary_checks():
    rows=[]
    for policy,v in [('greedy',.6),('footprint',.7),('hybrid',.8)]:
        rows.append(dict(policy=policy,weighted_admission=v))
    assert _val_means(rows)=={'greedy':.6,'footprint':.7,'hybrid':.8}
    with pytest.raises(AssertionError):_val_means(rows[:2])


def test_training_complete_evidence_and_no_overwrite(tmp_path):
    out=tmp_path/'A_standard_seed7'
    m=run_training(out,tmp_path/'cache',_small())
    assert m['completed_episodes']==2 and m['validation_evaluations']==2
    assert m['best_episode'] in (1,2) and m['checkpoint_sha256']
    assert m['final_test_seeds_accessed'] is False
    with (out/'validation_summary.csv').open(newline='') as f: val=list(csv.DictReader(f))
    assert len(val)==6
    assert all(int(x['decision_records'])==96 for x in val if x['policy']=='hybrid')
    assert all(int(x['feasibility_violations'])==0 for x in val)
    ckpt=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False)
    assert ckpt['config']=='A_standard' and ckpt['train_role']=='DEVELOPMENT_TRAINING_VALIDATION_ONLY'
    with pytest.raises(FileExistsError):run_training(out,tmp_path/'cache',_small())
    with pytest.raises(ValueError):run_training(out,tmp_path/'cache',_small(),resume=True)


def test_resume_mid_training_reproducible(monkeypatch,tmp_path):
    import phase5f_stage4.workflow as workflow
    # Compare full three-episode run to an interrupted+resumed run using identical
    # predeclared settings and seed. Crash immediately after episode 1 is persisted.
    p=_small(config='D_full_hybrid',seed=17,episodes=3)
    original=workflow.ppo_update
    out=tmp_path/'interrupted'
    calls={'n':0}
    def interrupt(*args,**kwargs):
        calls['n']+=1
        if calls['n']==2:raise KeyboardInterrupt('simulated CPU interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(workflow,'ppo_update',interrupt)
    with pytest.raises(KeyboardInterrupt):workflow.run_training(out,tmp_path/'cache',p)
    assert (out/'latest_state.pt').exists()
    monkeypatch.setattr(workflow,'ppo_update',original)
    resumed=workflow.run_training(out,tmp_path/'cache',p,resume=True)
    assert resumed['completed_episodes']==3
    assert len(list(csv.DictReader((out/'training_history.csv').open())))==3


def test_select_requires_all_twelve(tmp_path):
    with pytest.raises(FileNotFoundError):
        select_models(tmp_path/'not_exists',tmp_path/'selection')


def test_stage3_model_and_ablation_valid_action():
    graph=build_network(NetworkSpec(nodes=60,seed=201))
    d=generate_demands(graph,1,71000,False)[0]
    plans=generate_pairs(graph,d.source,d.destination,6,40)
    ledger=ScenarioLedger(graph,10)
    feasible=[i for i,p in enumerate(plans) if ledger.feasible(d,p)]
    obs=encode(graph,ledger,d,plans,feasible)
    model=CandidateActorCritic(hidden=32)
    for cfg in CONFIGS.values():
        choice=MaskedFeaturePolicy(model,cfg).decide(obs,greedy_idx=feasible[0])
        assert choice['action'] in feasible


def test_resumed_weights_match_uninterrupted_training(tmp_path,monkeypatch):
    import phase5f_stage4.workflow as workflow
    plan=_small('B_guided',27,episodes=3)
    completed=workflow.run_training(tmp_path/'full',tmp_path/'cache',plan)
    original=workflow.ppo_update
    calls=[0]
    def fail_after_first(*args,**kwargs):
        calls[0]+=1
        if calls[0]==2:raise KeyboardInterrupt('injected interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(workflow,'ppo_update',fail_after_first)
    with pytest.raises(KeyboardInterrupt):
        workflow.run_training(tmp_path/'resumed',tmp_path/'cache',plan)
    monkeypatch.setattr(workflow,'ppo_update',original)
    resumed=workflow.run_training(tmp_path/'resumed',tmp_path/'cache',plan,resume=True)
    for name in ('training_history.csv','validation_summary.csv','validation_checkpoints.csv'):
        assert (tmp_path/'full'/name).read_bytes()==(tmp_path/'resumed'/name).read_bytes()
    full_model=torch.load(tmp_path/'full'/'best_model.pt',map_location='cpu',weights_only=False)
    resumed_model=torch.load(tmp_path/'resumed'/'best_model.pt',map_location='cpu',weights_only=False)
    assert full_model['episode']==resumed_model['episode']
    for k,v in full_model['state_dict'].items():
        assert torch.equal(v,resumed_model['state_dict'][k])
