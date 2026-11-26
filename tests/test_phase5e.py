"""Phase 5E research integrity and small-scale end-to-end smoke tests.

All tests use 12-node development streams. NO held-out requests are generated.
"""
from __future__ import annotations
from dataclasses import asdict
import copy,csv,json
from pathlib import Path
import numpy as np
import pytest
from railway_tp.topology import TopologyConfig
from phase5.environment import Observation,OnlineEpisode,prepare_stream
from phase5e.agent import HybridConfig,ResidualPPO
from phase5e.policy import ResidualEpisode,gate_action,controls_action,feature_names
from phase5e.workflow import (cohort_guard,probe,train,select,evaluate,
                             statistics_report,_ci95,_holm,_pvalue,validate_env)

CFG=TopologyConfig(**json.loads((Path(__file__).resolve().parents[1]/'config/debug_12_dual.json').read_text()))


def test_guard_forbids_previous_and_future_heldout():
    cohort_guard([11000,11001],[22000,22001])
    for bad in (31000,41027,51000,61000,61059,62000):
        with pytest.raises(ValueError):cohort_guard([11000],[bad])
    with pytest.raises(ValueError):cohort_guard([11000],[20000])
    with pytest.raises(ValueError):cohort_guard([11000],[11000])
    with pytest.raises(ValueError):cohort_guard([11000,11000],[22000])
    with pytest.raises(ValueError):validate_env(100,10,1)


def test_config_validation_and_feature_dims():
    with pytest.raises(ValueError):HybridConfig(seed=-1)
    with pytest.raises(ValueError):HybridConfig(gate_threshold=2)
    with pytest.raises(ValueError):HybridConfig(greedy_bc_coef=-.1)
    assert len(feature_names(True))==34
    assert len(feature_names(False))==27


class FakeNative:
    def __init__(self,mask):
        self.k=len(mask)-1;self.mask=np.asarray(mask,dtype=bool)
        self.pos=0;self.done=False;self.native_actions=[]
    def observe(self):
        rows=np.tile(np.arange(26,dtype=np.float32),(self.k+1,1))
        return Observation(rows,self.mask,'R0001')
    def step(self,a):
        if not self.mask[a]:raise ValueError('illegal')
        self.native_actions.append(a);self.done=True
        return 1.,True,{'action':a,'admitted':int(a!=self.k)}
    def summary(self):return {'admitted':sum(x!=self.k for x in self.native_actions)}


@pytest.mark.parametrize('mask,mapping',[
   ([True,True,False,False],[0,1]),
   ([False,True,True,False],[1,2]),
   ([True,False,True,False],[0,2]),
   ([False,False,False,True],[3]),
])
def test_residual_mapping_and_forced_reject(mask,mapping):
    native=FakeNative(mask)
    env=ResidualEpisode(native,enhanced=False)
    obs=env.observe()
    assert list(obs.mapping)==mapping
    assert obs.mask[0]
    assert obs.discretionary==(len(mapping)>1)
    assert obs.rows.shape==(native.k,27)
    assert obs.rows[0,-1]==1
    idx=len(mapping)-1
    env.step(idx)
    assert native.native_actions==[mapping[idx]]
    assert env.override_count==int(idx!=0)
    with pytest.raises(ValueError):env.observe()


def test_invalid_residual_choice_rejected_without_mutation():
    n=FakeNative([True,False,False,False])
    env=ResidualEpisode(n,enhanced=False)
    with pytest.raises(ValueError):env.step(1)
    assert n.native_actions==[]


def test_gate_probability_difference_and_no_forced_override():
    mask=np.array([True,True])
    assert gate_action(np.array([0.,0.]),mask,0.)[0]==0
    assert gate_action(np.array([0.,2.]),mask,.4)[0]==1
    assert gate_action(np.array([0.,2.]),mask,.9)[0]==0
    assert gate_action(np.array([100.,200.]),np.array([True,False]),0.)==(0,0.)


def test_real_integration_feasibility_and_enhanced_features():
    p=prepare_stream(CFG,10,11000,2)
    native=OnlineEpisode(CFG,p,10,2)
    residual=ResidualEpisode(native,enhanced=True)
    while not residual.done:
        obs=residual.observe()
        assert obs.rows.shape==(2,34)
        assert obs.mask[0]
        assert np.isfinite(obs.rows).all()
        # Deliberately try alternatives when feasible; Phase-3 admits or
        # rejects every decision through the same existing validator.
        a=1 if obs.discretionary else 0
        r,done,record=residual.step(a)
        assert record['admitted']==int(record['action']!=2)
    assert residual.summary()['admitted']==len(native.sim.accepted())


def test_hybrid_safe_save_and_load(tmp_path):
    cfg=HybridConfig(seed=7,enhanced=True,warmstart_steps=0)
    model=ResidualPPO(2,cfg)
    p=tmp_path/'best.pt'
    model.save(p,{'version':'smoke'})
    restored,meta=ResidualPPO.load(p)
    assert meta=={'version':'smoke'}
    assert restored.names==model.names
    with pytest.raises(ValueError):restored.act(np.zeros((2,34),np.float32),np.array([False,False]))


def test_statistics_helpers():
    assert _ci95([1,1,1])==(1,1,1)
    assert _pvalue([0,0,0])==1
    assert _holm([.04,.01,.02])==pytest.approx([.04,.03,.04])


def test_probe_cannot_open_heldout(tmp_path):
    with pytest.raises(ValueError):
        probe(CFG,tmp_path/'seal',requests=8,seed_start=61000,seed_count=1)
    assert not (tmp_path/'seal').exists()


@pytest.fixture(scope='module')
def model_run_factory(tmp_path_factory):
    root=tmp_path_factory.mktemp('phase5e_test_runs')
    trainroot=root/'train';trainroot.mkdir()
    for seed in (7,17):
        out=trainroot/f'enhanced_seed{seed}'
        train(CFG,out,HybridConfig(seed=seed,enhanced=True,
              warmstart_steps=2,batch=8,ppo_epochs=1),
              requests=8,k=2,train_start=11000,train_count=2,
              val_start=22000,val_count=2,episodes=2,
              validate_every=1,patience=3)
    selection=root/'selection'
    sel=select(selection,trainroot,min_seeds=2)
    return root,sel


def test_train_select_checkpoint_provenance(model_run_factory):
    root,selection=model_run_factory
    assert len(selection['training_seeds'])==2
    assert selection['validation_seeds']==[22000,22001]
    assert selection['selected_seed'] in (7,17)
    assert len(list((root/'train').glob('*/best_model.pt')))==2
    assert (root/'selection'/'selection_trials.csv').exists()
    from phase5c.dataset import checksum
    assert checksum(Path(selection['selected_checkpoint']))==selection['checkpoint_sha256']
    with pytest.raises(FileExistsError):select(root/'selection',root/'train',min_seeds=2)


def test_development_evaluate_statistics_plots_and_refuse_overwrite(model_run_factory,tmp_path):
    root,selection=model_run_factory
    out=tmp_path/'evaluation'
    result=evaluate(CFG,out,root/'selection'/'selection_manifest.json',
                    requests=8,k=2,role='development',seed_start=23000,seed_count=2)
    assert result['runs']==16
    with (out/'phase5e_runs.csv').open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==16
    by={(int(r['demand_seed']),r['policy']):r for r in rows}
    for seed in (23000,23001):
        assert by[seed,'greedy']['request_sha256']==by[seed,'capacity_override']['request_sha256']
        assert len({r['request_sha256'] for r in rows if int(r['demand_seed'])==seed})==1
        assert int(by[seed,'hybrid_gated']['admitted'])<=8
    st=statistics_report(out,tmp_path/'stats')
    assert Path(st['report']).exists()
    with (tmp_path/'stats'/'paired_summary.csv').open(newline='',encoding='utf8') as f:
        paired=list(csv.DictReader(f))
    assert len(paired)==7
    from phase5e.workflow import plots
    images=plots(tmp_path/'stats',tmp_path/'plots')['figures']
    assert len(images)==3 and all(Path(p).is_file() for p in images)
    with pytest.raises(FileExistsError):evaluate(CFG,out,root/'selection'/'selection_manifest.json',
        requests=8,k=2,role='development',seed_start=23000,seed_count=2)


def test_primary_seal_refuses_early_or_wrong_design(model_run_factory,tmp_path):
    root,_=model_run_factory
    output=tmp_path/'sealed'
    # Current prototype 12-node setup cannot masquerade as the primary 60-node design.
    with pytest.raises(ValueError):
        evaluate(CFG,output,root/'selection'/'selection_manifest.json',
                 requests=8,k=2,role='primary',seed_start=61000,seed_count=60)
    with pytest.raises(ValueError):
        evaluate(CFG,output,root/'selection'/'selection_manifest.json',
                 requests=8,k=2,role='development',seed_start=61000,seed_count=1)
    assert not output.exists()


def test_manifest_hash_mismatch_rejected_before_output(model_run_factory,tmp_path):
    root,manifest=model_run_factory
    broken=dict(manifest)
    broken['checkpoint_sha256']='0'*64
    p=tmp_path/'broken.json';p.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='hash'):
        evaluate(CFG,tmp_path/'no_output',p,requests=8,k=2,
                 role='development',seed_start=23000,seed_count=1)
    assert not (tmp_path/'no_output').exists()


def test_preflight_readonly_source_check(tmp_path):
    from phase5e.workflow import preflight,hashes
    from phase5d import VERSION as PHASE5D
    r=preflight(CFG)
    assert r['test_seeds_generated']==0
    assert 'phase3/model.py' in r['current_source_sha256']
    prior={'version':PHASE5D,'topology_config':asdict(CFG),
        'phase3_through_phase5_source_sha256':{'phase3/model.py':hashes()['phase3/model.py']},
        'additional_source_sha256':{'phase5e/policy.py':hashes()['phase5e/policy.py']}}
    p=tmp_path/'phase5d_manifest.json';p.write_text(json.dumps(prior))
    assert preflight(CFG,p)['phase5d_source_provenance_confirmed'] is True
    prior['phase3_through_phase5_source_sha256']['phase3/model.py']='0'*64
    p.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match='differs'):preflight(CFG,p)


def test_legacy_greedy_and_residual_greedy_are_identical(tmp_path):
    from phase5.experiment import _run_one
    prep=prepare_stream(CFG,20,23000,2)
    result=_run_one(CFG,prep,'greedy',10,2,None,tmp_path/'old_greedy',False,5.,23000)
    env=ResidualEpisode(OnlineEpisode(CFG,prep,10,2),enhanced=False)
    while not env.done:env.step(0)
    assert result['state_digest']==env.summary()['state_digest']
    assert result['admitted']==env.summary()['admitted']


def test_k_four_probe_is_real_inventory_not_assumed(tmp_path):
    from phase5e.workflow import probe
    probe(CFG,tmp_path/'k4',requests=8,k=4,seed_start=23010,seed_count=1)
    p=json.loads((tmp_path/'k4'/'probe_manifest.json').read_text())
    inv={int(k):v for k,v in p['candidate_inventory'].items()}
    assert sum(inv.values())==8
    assert p['k_above_two_actually_used']==any(k>2 for k in inv)
