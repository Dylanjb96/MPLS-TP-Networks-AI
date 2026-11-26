"""Development-only regression tests for Phase 5G Stage 2 matched training.

None of the tests open or generate the reserved final held-out seeds.
"""
from pathlib import Path
import csv,json,torch,pytest
from unittest.mock import patch
from phase5g.protocol import check_seed_role
from phase5g.workflow import make_cohort
from phase5f.protection import ScenarioLedger
from phase5f_stage3.features import encode
from phase5g.gnn import graph_observation
from phase5g_stage2.workflow import (
 Plan,ARCHITECTURES,MODEL_SEEDS,schedule,CohortCache,build_model,param_count,
 simulate_mlp,forward_one,matched_ppo_update,reduce_samples,
 run_training,select_models,status,sources,preflight
)


def short_plan(model='gnn',seed=7,epochs=2):
    return Plan(model,seed,episodes=epochs,validate_every=1,
       train_sizes=(60,),train_loads=(1.6,),train_modes=('dynamic',),
       train_topologies=(601,602),train_streams=(75000,75001),
       val_topologies=(701,),val_streams=(76000,),
       val_size=60,val_load=1.6,max_update_samples=8)


def test_preflight():
    preflight()


def test_seed_partition_positive_and_negative():
    check_seed_role('train',601,75000)
    check_seed_role('validation',701,76000)
    for t,s in [(801,77000),(401,73000),(601,76000),(701,75000)]:
        with pytest.raises(ValueError):check_seed_role('train',t,s)
    for bad in ('final','heldout','test'):
        with pytest.raises(ValueError):check_seed_role(bad,801,77000)


def test_plan_forbids_ladder_and_protection_changes():
    short_plan().validate()
    with pytest.raises(ValueError):short_plan().__class__(**{**short_plan().description(),'family':'ladder'}).validate()
    with pytest.raises(ValueError):Plan('gnn',7,k=2).validate()
    with pytest.raises(ValueError):Plan('gnn',7,val_topologies=(801,)).validate()
    with pytest.raises(ValueError):Plan('gnn',8).validate()


def test_schedules_are_identical_between_neural_arms():
    g=short_plan('gnn',7,epochs=3)
    m=short_plan('mlp',7,epochs=3)
    assert schedule(g)==schedule(m)
    assert len(schedule(g))==3
    assert schedule(short_plan('gnn',17,epochs=3))!=schedule(g)


def test_model_parameter_counts_differ_by_design():
    m=build_model('mlp',7);g=build_model('gnn',7)
    assert param_count(m)==15758 and param_count(g)==15778
    assert abs(param_count(m)-param_count(g))/param_count(g)<0.002


def test_feasible_mask_forward_graph_and_tabular():
    graph,demands,inv=make_cohort(nodes=60,topology_seed=501,stream_seed=74000,
                                   role='dev',load=1.6)
    led=ScenarioLedger(graph,10)
    for d,plans in zip(demands,inv):
        feasible=[i for i,p in enumerate(plans) if led.feasible(d,p)]
        if len(feasible)>1:
            for modelname in ARCHITECTURES:
                model=build_model(modelname,7)
                observation=(graph_observation(graph,led,d,plans,feasible,k=6,dynamic=True)
                    if modelname=='gnn' else encode(graph,led,d,plans,feasible,k=6,dynamic=True))
                logits,v=forward_one(model,observation,modelname)
                assert logits.shape==(6,) and bool(torch.isfinite(v))
                assert model.decide(observation,greedy_idx=feasible[0])['action'] in feasible
            break
    else:pytest.fail('no discretionary candidates')


def test_ladder_not_permitted_in_training_cache(tmp_path):
    cache=CohortCache(tmp_path)
    with pytest.raises(ValueError):
        cache.get(nodes=60,topology_seed=601,stream_seed=75000,
             load=1.6,mode='dynamic',role='train',family='ladder')


def test_training_cache_reuse_and_tamper_detection(tmp_path):
    cache=CohortCache(tmp_path)
    x=dict(nodes=60,topology_seed=601,stream_seed=75000,load=1.6,mode='dynamic',role='train')
    g,d,inv,h=cache.get(**x)
    assert cache.misses==1 and cache.hits==0 and len(d)==96
    g2,d2,inv2,h2=cache.get(**x)
    assert cache.hits==1 and h==h2 and len(inv)==len(inv2)
    filename=list(tmp_path.glob('*.json.gz'))[0]
    filename.write_bytes(b'broken')
    with pytest.raises(Exception):cache.get(**x)


def test_mlp_stochastic_rollout_and_onpolicy_update():
    graph,d,inv=make_cohort(nodes=60,topology_seed=601,stream_seed=75000,
                           role='train',load=1.6,mode='dynamic')
    m=build_model('mlp',7)
    summary,trace,samples=simulate_mlp(graph,d,inv,m,mode='dynamic',stochastic=True,trace=True)
    assert summary['admitted']<=96 and summary['feasibility_violations']==0
    assert len(trace)==96 and len(samples)==summary['discretionary']
    subset=reduce_samples(samples,8,seed=7)
    opt=torch.optim.Adam(m.parameters(),lr=.0003)
    result=matched_ppo_update(m,opt,subset,architecture='mlp',seed=7)
    assert result['samples']<=8 and abs(result['loss'])<1e4


def test_gnn_matched_update_finite():
    from phase5g.workflow import simulate_gnn
    graph,d,inv=make_cohort(nodes=60,topology_seed=601,stream_seed=75000,
                           role='train',load=1.6,mode='dynamic')
    m=build_model('gnn',17)
    summary,_,samples=simulate_gnn(graph,d,inv,m,stochastic=True,mode='dynamic',trace=False)
    assert len(samples)==summary['discretionary']
    opt=torch.optim.Adam(m.parameters(),lr=.0003)
    update=matched_ppo_update(m,opt,reduce_samples(samples,4,17),architecture='gnn',seed=17)
    assert update['samples']==4 and abs(update['loss'])<1e4


def test_quick_training_gnn_and_provenance(tmp_path):
    out=tmp_path/'gnn'
    result=run_training(out,short_plan('gnn'),cache_root=tmp_path/'cache')
    assert result['status']=='TRAIN_OK' and result['episodes']==2
    assert len(list(csv.DictReader((out/'validation_summary.csv').open())))==10
    assert status(out)['status']=='complete'
    meta=json.loads((out/'training_provenance.json').read_text())
    assert meta['final_seeds_accessed'] is False
    assert meta['source_sha256']==sources()
    assert Path(out/meta['best_checkpoint']).is_file()
    assert (out/meta['best_trace']).is_file()
    result2=run_training(out,short_plan('gnn'),cache_root=tmp_path/'cache',resume=True)
    assert result2['status']=='ALREADY_COMPLETE'


def test_quick_training_mlp(tmp_path):
    out=tmp_path/'mlp'
    r=run_training(out,short_plan('mlp'),cache_root=tmp_path/'cache')
    assert r['status']=='TRAIN_OK' and status(out)['episode']==2
    assert json.loads((out/'training_provenance.json').read_text())['param_count']<20000


def test_run_rejects_overwrite(tmp_path):
    o=tmp_path/'run'
    run_training(o,short_plan('mlp',7,epochs=1),cache_root=tmp_path/'cache')
    with pytest.raises(FileExistsError):run_training(o,short_plan('mlp',7,epochs=1),cache_root=tmp_path/'cache')


def test_safe_resume_after_interrupted_episode(tmp_path):
    import phase5g_stage2.workflow as w
    o=tmp_path/'resume';pl=short_plan('mlp',7,epochs=2)
    real=w.matched_ppo_update
    calls={'n':0}
    def fail_second(*args,**kwargs):
        calls['n']+=1
        if calls['n']==2:raise RuntimeError('simulated power failure')
        return real(*args,**kwargs)
    with patch.object(w,'matched_ppo_update',side_effect=fail_second):
        with pytest.raises(RuntimeError,match='power failure'):
            run_training(o,pl,cache_root=tmp_path/'cache')
    assert status(o)['episode']==1
    resumed=run_training(o,pl,cache_root=tmp_path/'cache',resume=True)
    assert resumed['status']=='TRAIN_OK' and status(o)['episode']==2


def test_selection_requires_complete_six_arm_matrix(tmp_path):
    with pytest.raises(ValueError,match='missing completed run'):
        select_models(tmp_path/'no_runs',tmp_path/'selection')


def test_plan_signature_detects_changed_source(tmp_path):
    out=tmp_path/'r'
    run_training(out,short_plan('mlp',7,epochs=1),cache_root=tmp_path/'cache')
    bad=short_plan('mlp',7,epochs=2)
    with pytest.raises(RuntimeError,match='resumption'):
        run_training(out,bad,cache_root=tmp_path/'cache',resume=True)

