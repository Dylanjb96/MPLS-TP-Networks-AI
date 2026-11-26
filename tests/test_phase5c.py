"""Phase 5C regression tests. No historical held-out seeds are used for tuning."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from railway_tp.topology import TopologyConfig
from phase5.environment import FEATURE_NAMES,OnlineEpisode,prepare_stream
from phase5c.common import RLConfig,make_agent,logits_and_prob,CheckpointMixin
from phase5c.dataset import build_dataset,load_dataset,forbidden
from phase5c.workflow import train_offline,train_ppo,run_policy,evaluate
from phase5c.selection import select_models

@pytest.fixture(scope='module')
def cfg():return TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=4,
    subring_count=2,main_capacity_mbps=10000,sub_capacity_mbps=1000,length_km=2)


def sample_data(n=32):
    rng=np.random.default_rng(42);count=len(FEATURE_NAMES)
    rows=rng.normal(size=(n,3,count)).astype(np.float32)
    mask=np.tile(np.array([True,True,False]),(n,1))
    mask[::5]=[False,False,True]
    actions=np.asarray([0 if i%5 else 2 for i in range(n)])
    nxt=rows.copy();nmask=mask.copy()
    done=np.zeros(n,dtype=bool);done[::7]=True
    return dict(rows=rows,mask=mask,action=actions,reward=np.ones(n,dtype=np.float32),
                next_rows=nxt,next_mask=nmask,done=done)


def test_masks_normalized_and_illegal_zero():
    logits=torch.tensor([[1.,-10.,1.],[20.,50.,20.]])
    masks=torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
    p,lp=logits_and_prob(logits,masks)
    assert torch.allclose(p.sum(-1),torch.ones(2))
    assert p[0,1]==0 and p[1,0]==0
    with pytest.raises(ValueError):logits_and_prob(logits,torch.zeros_like(masks))


@pytest.mark.parametrize('algorithm',['bcq','cql','awac','iql'])
def test_offline_update_gradients_and_legal_actions(algorithm,tmp_path):
    data=sample_data()
    model=make_agent(RLConfig(algorithm=algorithm,batch=8),len(FEATURE_NAMES),3)
    before=[p.detach().clone() for p in model.q.parameters()]
    loss=model.update(data,np.arange(8))
    assert np.isfinite(loss) and model.gradient_steps==1
    assert any(not torch.allclose(a,b) for a,b in zip(before,model.q.parameters()))
    for i in range(8):
        a=model.act(data['rows'][i],data['mask'][i])
        assert data['mask'][i,a]
    path=tmp_path/'best.pt';model.save(path,{'algorithm':algorithm})
    new,meta=CheckpointMixin.load(path)
    assert meta['algorithm']==algorithm
    for i in range(4):assert new.act(data['rows'][i],data['mask'][i])==model.act(data['rows'][i],data['mask'][i])


def test_bcq_behavior_support_excludes_unsupported_action():
    model=make_agent(RLConfig(algorithm='bcq',behavior_threshold=0.95),len(FEATURE_NAMES),3)
    for p in model.policy.parameters():p.data.zero_()
    assert model.act(np.zeros((3,len(FEATURE_NAMES)),np.float32),
                     np.array([True,False,False]))==0


def test_split_guards():
    with pytest.raises(ValueError):forbidden([10000],[10000])
    with pytest.raises(ValueError):forbidden([31000],[20000])
    with pytest.raises(ValueError):forbidden([51000],[20000])


def test_dataset_record_and_checksum(cfg,tmp_path):
    folder=tmp_path/'data'
    meta=build_dataset(cfg,folder,requests=4,k=2,seed_start=10000,seed_count=1)
    data,info=load_dataset(folder)
    assert meta['rows']==8 and data['rows'].shape==(8,3,len(FEATURE_NAMES))
    assert int((data['behavior_id']==0).sum())==4
    assert np.all(data['mask'][np.arange(8),data['action']])
    assert info['stream_sha256']==meta['stream_sha256']
    with (folder/'offline_transitions.npz').open('ab') as file:file.write(b'tamper')
    with pytest.raises(ValueError,match='sha256'):load_dataset(folder)


def test_ppo_masking_and_training(cfg,tmp_path):
    model=make_agent(RLConfig(algorithm='ppo',batch=4,ppo_epochs=1),len(FEATURE_NAMES),3)
    env=OnlineEpisode(cfg,prepare_stream(cfg,6,10000,2),10,2)
    result=model.train_episode(env)
    assert result['requests']==6
    assert model.gradient_steps>=1
    for mask in (np.array([1,1,0],dtype=bool),np.array([0,0,1],dtype=bool)):
        assert mask[model.act(np.zeros((3,len(FEATURE_NAMES)),np.float32),mask,greedy=True)]
    path=tmp_path/'ppo.pt';model.save(path,{'algorithm':'ppo'})
    loaded,meta=CheckpointMixin.load(path)
    assert meta['algorithm']=='ppo'
    assert loaded.act(np.zeros((3,len(FEATURE_NAMES)),np.float32),
                      np.array([True,False,False]),greedy=True)==0


def test_train_offline_and_checkpoint(cfg,tmp_path):
    ds=tmp_path/'dataset'
    build_dataset(cfg,ds,requests=4,k=2,seed_start=10000,seed_count=1)
    output=tmp_path/'cql'
    result=train_offline(cfg,output,RLConfig(algorithm='cql',batch=4),ds,
          requests=4,k=2,val_seed_start=20000,val_seed_count=1,
          updates=4,validate_every=2,patience=3)
    assert result['best_gradient_step'] in (2,4)
    assert (output/'best_model.pt').exists()
    assert result['offline_dataset_npz_sha256']


def test_train_ppo_and_checkpoint(cfg,tmp_path):
    result=train_ppo(cfg,tmp_path/'ppo',RLConfig(algorithm='ppo',ppo_epochs=1),
         requests=4,k=2,train_seed_start=10000,train_seed_count=1,
         val_seed_start=20000,val_seed_count=1,
         episodes=2,validate_every=1,patience=3)
    assert result['best_episode'] in (1,2)
    assert (tmp_path/'ppo'/'best_model.pt').exists()


def test_selection_and_evaluate_smoke(cfg,tmp_path):
    run_dirs=[]
    for algo in ('ppo','bcq','cql','awac','iql'):
        for seed in (7,17,27):
            folder=tmp_path/f'{algo}_{seed}';folder.mkdir()
            model=make_agent(RLConfig(algorithm=algo,seed=seed),len(FEATURE_NAMES),3)
            model.save(folder/'best_model.pt',{'version':'PHASE5C_DISCRETE_RL_V1',
                'algorithm':algo,'feature_version':'PHASE5_STATE_CANDIDATE_FEATURES_V1',
                'feature_names':list(FEATURE_NAMES),'topology_config':asdict(cfg),
                'requests_per_seed':2,'headroom_percent':10,'max_candidates':2,
                'train_seeds':[10000],'val_seeds':[20000]})
            info={'algorithm':algo,'config':asdict(model.config),
                  'training_seeds':[10000],'validation_seeds':[20000],
                  'topology':asdict(cfg),'requests_per_seed':2,'headroom':10,
                  'max_candidates':2,'best_validation_score':[0.1+seed/100000,0.1]}
            (folder/'training_provenance.json').write_text(json.dumps(info))
            run_dirs.append(folder)
    selected=select_models(tmp_path/'selected',run_dirs,3)
    assert len(selected['winners'])==5
    manifest=json.loads((tmp_path/'selected'/'selection_manifest.json').read_text())
    checkpoints={x['algorithm']:Path(x['selected_checkpoint']) for x in manifest['winners']}
    with pytest.raises(ValueError,match='previously examined'):
        evaluate(cfg,tmp_path/'bad',checkpoints,requests=2,k=2,test_seed_start=41000,test_seed_count=1)
    results=evaluate(cfg,tmp_path/'evaluation',checkpoints,requests=2,k=2,
                     test_seed_start=30010,test_seed_count=1)
    assert results['runs']==9
    assert (tmp_path/'evaluation'/'phase5_runs.csv').exists()


def test_awac_offline_to_online_finetune(cfg,tmp_path):
    from phase5c.workflow import finetune_awac
    dataset=tmp_path/'offline'
    build_dataset(cfg,dataset,requests=4,k=2,seed_start=10000,seed_count=1)
    initial=tmp_path/'awac_initial'
    train_offline(cfg,initial,RLConfig(algorithm='awac',batch=4),dataset,
                  requests=4,k=2,val_seed_start=20000,val_seed_count=1,
                  updates=2,validate_every=1,patience=3)
    final=tmp_path/'awac_finetuned'
    p=finetune_awac(cfg,final,initial/'best_model.pt',dataset,
                    requests=4,k=2,train_seed_start=10000,train_seed_count=1,
                    val_seed_start=20000,val_seed_count=1,episodes=2,
                    updates_per_episode=2,validate_every=1,patience=3)
    assert p['training_regime']=='offline_to_online'
    assert p['online_training_transitions']==8
    m,meta=CheckpointMixin.load(final/'best_model.pt')
    assert meta['training_regime']=='offline_to_online'
    assert meta['algorithm']=='awac'
