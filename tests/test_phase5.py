"""Phase 5 regression tests; no trained-performance claim from smoke tests."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from phase3.model import ProtectionSimulator
from railway_tp.topology import TopologyConfig, make_topology
from phase5.environment import (FEATURE_NAMES, FEATURE_VERSION, OnlineEpisode,
                                prepare_stream, stream_sha256)
from phase5.agent import AgentConfig, MaskedDDQN, Transition
from phase5.experiment import train, evaluate
from phase5.statistics import summarize


@pytest.fixture(scope='module')
def conf():
    return TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=6,
                          subring_count=1,main_capacity_mbps=10000,sub_capacity_mbps=1000)


def test_prepared_request_is_reproducible(conf):
    a=prepare_stream(conf,8,2027,2)
    b=prepare_stream(conf,8,2027,2)
    assert a.request_sha256==b.request_sha256
    assert [tuple(p.candidate_id for p in c) for c in a.candidate_sets] == [
           tuple(p.candidate_id for p in c) for c in b.candidate_sets]


def test_mask_reject_never_allowed_if_feasible(conf):
    env=OnlineEpisode(conf,prepare_stream(conf,4,2027,2),10,2)
    s=env.observe()
    assert s.rows.shape==(3,len(FEATURE_NAMES))
    assert s.mask[:2].any() and not s.mask[2]
    before=env.sim.state_digest()
    with pytest.raises(ValueError):env.step(2)
    with pytest.raises(ValueError):env.step(20)
    assert env.sim.state_digest()==before
    reward,done,row=env.step(int(np.flatnonzero(s.mask)[0]))
    assert reward==row['priority_weight']/5 and row['admitted']==1
    assert not done and len(env.sim.accepted())==1


def test_all_infeasible_forces_only_rejection(conf):
    env=OnlineEpisode(conf,prepare_stream(conf,3,2027,2),98,2)
    # First request may be infeasible depending on synthetic traffic; if not,
    # increase demand in first request without changing its source/destination.
    from dataclasses import replace
    p=env.prepared
    altered=replace(p,requests=(replace(p.requests[0],bandwidth_mbps=999999),)+p.requests[1:])
    env=OnlineEpisode(conf,altered,10,2)
    obs=env.observe()
    assert list(obs.mask)==[False,False,True]
    original=env.sim.state_digest()
    with pytest.raises(ValueError):env.step(0)
    reward,done,row=env.step(2)
    assert reward==0 and not row['admitted'] and not done
    assert original==env.sim.state_digest()


def test_agent_respects_mask_and_replay_updates():
    conf=AgentConfig(seed=9,batch_size=4,learning_starts=4,train_every=1,
                     replay_capacity=16,target_sync=1)
    agent=MaskedDDQN(len(FEATURE_NAMES),conf)
    rows=np.zeros((3,len(FEATURE_NAMES)),np.float32)
    mask=np.asarray([False,True,False])
    assert all(agent.act(rows,mask)==1 for _ in range(10))
    for i in range(5):
        agent.remember(Transition(rows,mask,1,1.0,rows,mask,i==4))
        agent.learn()
    assert agent.gradient_steps>=1 and agent.last_loss is not None
    assert all(torch.equal(a,b) for a,b in zip(agent.online.parameters(),agent.target.parameters()))


def test_checkpoint_safe_roundtrip(tmp_path):
    agent=MaskedDDQN(len(FEATURE_NAMES),AgentConfig(seed=9))
    x=np.zeros((3,len(FEATURE_NAMES)),np.float32)
    mask=np.array([True,True,False])
    path=tmp_path/'model.pt'; agent.save(path,{'note':'test'})
    clone,meta=MaskedDDQN.load(path)
    assert meta['note']=='test'
    assert agent.act(x,mask,greedy=True)==clone.act(x,mask,greedy=True)


def test_training_rejects_overlap_before_creating_files(tmp_path,conf):
    with pytest.raises(ValueError,match='overlap'):
        train(conf,tmp_path/'invalid',requests=3,train_seed_start=10000,
              train_seed_count=2,val_seed_start=10001,val_seed_count=2,
              train_episodes=1)
    assert not (tmp_path/'invalid').exists()


def test_tiny_train_eval_report_and_no_leakage(tmp_path,conf):
    run=tmp_path/'train'
    train(conf,run,requests=5,train_seed_start=10000,train_seed_count=2,
          val_seed_start=20000,val_seed_count=2,train_episodes=3,
          validate_every=1,patience=20,
          agent_config=AgentConfig(seed=8,batch_size=2,learning_starts=2,
                                   train_every=1,replay_capacity=32,epsilon_decay_steps=10))
    assert (run/'best_model.pt').exists()
    with pytest.raises(ValueError,match='overlap'):
        evaluate(run/'best_model.pt',conf,tmp_path/'bad',requests=5,
                 test_seed_start=10000,test_seed_count=1)
    assert not (tmp_path/'bad').exists()
    evaluation=tmp_path/'evaluation'
    result=evaluate(run/'best_model.pt',conf,evaluation,requests=5,
                    test_seed_start=30000,test_seed_count=2,
                    policies=('greedy','cspf','milp_v2','ddqn'),save_plans=True)
    assert result['runs']==8
    summary=summarize(evaluation,tmp_path/'statistics')
    assert summary['seeds']==2
    assert (evaluation/'runs'/'seed_30000_ddqn'/'accepted_protection_plans.json').exists()
    import csv
    with (evaluation/'phase5_runs.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==8
    for seed in [30000,30001]:
        subset=[r for r in rows if int(r['demand_seed'])==seed]
        assert len({x['request_sha256'] for x in subset})==1
        assert len({x['model'] for x in subset})==1
        assert all(int(r['admitted'])+int(r['rejected'])==5 for r in subset)
