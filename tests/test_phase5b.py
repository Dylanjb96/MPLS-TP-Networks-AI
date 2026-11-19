"""Phase5B deterministic mask, gradient, checkpoint, and no-leakage smoke tests."""
import json
import numpy as np
from pathlib import Path
import pytest
import torch

from railway_tp.topology import TopologyConfig
from phase5.agent import Transition
from phase5.environment import FEATURE_NAMES, OnlineEpisode, prepare_stream
from phase5b.sac import SACConfig, MaskedDiscreteSAC, masked_distribution
from phase5b.workflow import guard_splits, train_sac, evaluate_models


def cfg():
    return TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=6,
                          subring_count=1,main_capacity_mbps=10000,sub_capacity_mbps=1000)


def test_mask_distribution():
    logits=torch.tensor([[2.,9.,-1.],[0.,1.,2.]])
    mask=torch.tensor([[True,False,True],[False,True,False]])
    p,lp=masked_distribution(logits,mask)
    assert p[0,1].item()==0
    assert p[1,1].item()==1
    assert torch.allclose(p.sum(dim=-1),torch.ones(2))
    with pytest.raises(ValueError):
        masked_distribution(logits,torch.zeros_like(mask))


def test_sac_mask_and_update():
    a=MaskedDiscreteSAC(len(FEATURE_NAMES),SACConfig(batch_size=4,learning_starts=4,
                  train_every=1,replay_capacity=24,seed=17,target_tau=1.0))
    rows=np.ones((3,len(FEATURE_NAMES)),dtype=np.float32)*0.2
    mask=np.array([False,True,False])
    assert all(a.act(rows,mask)==1 for _ in range(20))
    for i in range(8):
        a.remember(Transition(rows.copy(),mask.copy(),1,1.0,
            rows.copy(),mask.copy(),i%2==0))
        a.learn()
    assert a.gradient_steps==5
    assert np.isfinite(a.last_loss)
    assert all(torch.equal(x,y) for x,y in zip(a.critic1.parameters(),a.target1.parameters()))


def test_terminal_mask_is_safe():
    a=MaskedDiscreteSAC(len(FEATURE_NAMES),SACConfig(batch_size=2,learning_starts=2,
                                                     train_every=1,replay_capacity=8))
    rows=np.ones((3,len(FEATURE_NAMES)),dtype=np.float32)
    mask=np.array([True,False,False])
    for _ in range(2):
        a.remember(Transition(rows.copy(),mask.copy(),0,1.,
            np.zeros_like(rows),np.zeros(3,dtype=bool),True))
        loss=a.learn()
    assert a.gradient_steps>=1
    assert np.isfinite(loss)


def test_sac_checkpoint_roundtrip(tmp_path):
    a=MaskedDiscreteSAC(len(FEATURE_NAMES),SACConfig(seed=11))
    rows=np.zeros((3,len(FEATURE_NAMES)),np.float32)
    mask=np.array([True,True,False])
    path=tmp_path/'sac.pt'
    a.save(path,{'note':'test'})
    clone,meta=MaskedDiscreteSAC.load(path)
    assert meta['note']=='test'
    assert a.act(rows,mask,greedy=True)==clone.act(rows,mask,greedy=True)


def test_reject_historical_seeds():
    with pytest.raises(ValueError):
        guard_splits({31000},{20000})
    with pytest.raises(ValueError):
        guard_splits({10000},{20000},{31005})
    with pytest.raises(ValueError):
        guard_splits({10000},{10000})
    with pytest.raises(ValueError):
        guard_splits({41000},{20000})
    guard_splits({10000},{20000},{41000})


def test_small_train_and_eval(tmp_path):
    c=cfg()
    tr=tmp_path/'sac_train'
    p=train_sac(c,tr,requests=5,train_seed_count=2,val_seed_count=2,
         train_episodes=3,validate_every=1,patience=10,
         agent_config=SACConfig(seed=7,batch_size=2,learning_starts=2,
                                train_every=1,replay_capacity=16))
    assert (tr/'best_model.pt').exists()
    assert p['completed_episodes']==3
    old=tmp_path/'ddqn_train'
    from phase5.experiment import train
    from phase5.agent import AgentConfig
    train(c,old,requests=5,train_seed_count=2,val_seed_count=2,
          train_episodes=2,validate_every=1,patience=10,
          agent_config=AgentConfig(seed=7,batch_size=2,learning_starts=2,
                                   train_every=1,replay_capacity=16))
    with pytest.raises(ValueError,match='held-out|historical'):
        evaluate_models(c,tmp_path/'forbidden',{'original_ddqn':old/'best_model.pt'},
                        requests=5,test_seed_start=31000,test_seed_count=1)
    out=tmp_path/'eval'
    result=evaluate_models(c,out,{'original_ddqn':old/'best_model.pt',
                    'discrete_sac':tr/'best_model.pt'},requests=5,
                    test_seed_start=41000,test_seed_count=2)
    assert result['runs']==12
    import csv
    with (out/'phase5_runs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==12
    for seed in (41000,41001):
        rs=[r for r in rows if int(r['demand_seed'])==seed]
        assert len({r['request_sha256'] for r in rs})==1
        assert all(int(r['admitted'])+int(r['rejected'])==5 for r in rs)
    from phase5.statistics import summarize
    d=summarize(out,tmp_path/'stats')
    assert d['seeds']==2
