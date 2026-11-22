"""New Phase5D checks: seed guards, identical-state decisions, isolated branching.

Tests use debug 12-node synthetic streams only; no 51000/61000-series tests.
"""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import pytest

from railway_tp.topology import TopologyConfig
from phase5.environment import OnlineEpisode,prepare_stream
from phase5c.common import RLConfig,make_agent
from phase5d.analysis import (development_seed_guard,greedy_action,
    behavior,counterfactual,milp_trace,load_selected_models,_branch)
from phase5d.cli import parser


@pytest.fixture(scope='module')
def cfg():
    return TopologyConfig(nodes=12,seed=42,gateway_mode='dual',main_nodes=4,
        subring_count=2,main_capacity_mbps=10000,sub_capacity_mbps=1000,length_km=2)


def csv_rows(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def test_development_only_seeds():
    assert development_seed_guard([20000,20001])=='DEVELOPMENT_ONLY'
    assert development_seed_guard([21000,21029])=='DEVELOPMENT_ONLY'
    with pytest.raises(ValueError,match='only training/development'):
        development_seed_guard([41027])
    with pytest.raises(ValueError,match='sealed'):
        development_seed_guard([61000],retrospective=True)
    assert development_seed_guard([41027],retrospective=True).startswith('RETROSPECTIVE')
    with pytest.raises(ValueError,match='retrospective mode only permits'):
        development_seed_guard([41027,77777],retrospective=True)
    with pytest.raises(ValueError,match='unique'):
        development_seed_guard([20000,20000])


def test_greedy_mask_rejection_forced(cfg):
    env=OnlineEpisode(cfg,prepare_stream(cfg,6,20000,2),10,2)
    while not env.done:
        obs=env.observe(); action=greedy_action(obs,2)
        assert obs.mask[action]
        if action==2:assert int(obs.mask[:2].sum())==0
        _,_,record=env.step(action)
        assert record['admitted']==int(action!=2)


def test_behavior_baseline_smoke(cfg,tmp_path):
    out=tmp_path/'behavior'
    result=behavior(cfg,out,{},requests=5,seeds=[20000,20001])
    assert result['trajectory_records']==10
    assert csv_rows(out/'same_state_choices.csv')==[]
    own=csv_rows(out/'trajectory_choices.csv')
    assert len(own)==10 and {r['policy'] for r in own}=={'greedy'}
    manifest=json.loads((out/'diagnostic_manifest.json').read_text())
    assert manifest['seeds']==[20000,20001]
    assert len(manifest['request_sha256_by_seed'])==2
    with pytest.raises(FileExistsError):behavior(cfg,out,{},requests=5,seeds=[20000])


def test_behavior_with_frozen_model(cfg,tmp_path):
    model=make_agent(RLConfig(algorithm='cql',seed=7),actions=3)
    result=behavior(cfg,tmp_path/'b',{'cql':model},requests=6,seeds=[20000])
    same=csv_rows(tmp_path/'b'/'same_state_choices.csv')
    own=csv_rows(tmp_path/'b'/'trajectory_choices.csv')
    assert len(same)==6 and len(own)==12
    for r in same:
        assert int(r['chosen_action']) in (0,1,2)
        assert int(r['agreed_with_greedy']) in (0,1)
    assert {r['policy'] for r in csv_rows(tmp_path/'b'/'policy_summary.csv')}=={'greedy','cql'}


def test_branch_copy_does_not_mutate_parent(cfg):
    import copy
    env=OnlineEpisode(cfg,prepare_stream(cfg,5,20000,2),10,2)
    before=env.sim.state_digest()
    copy_env=copy.deepcopy(env)
    outcome=_branch(copy_env,greedy_action(copy_env.observe(),2),3)
    assert outcome['steps']==3
    assert env.pos==0 and env.sim.state_digest()==before
    assert copy_env.pos==3


def test_counterfactual_smoke(cfg,tmp_path):
    out=tmp_path/'cf'
    result=counterfactual(cfg,out,requests=8,seeds=[21000],max_branches=3,horizon=8)
    assert result['branch_points']>=0
    rows=csv_rows(out/'counterfactual_branches.csv')
    for r in rows:
        assert r['branch_start_sha256']
        assert int(r['rollout_steps'])>=1
        assert int(r['request_index'])<8
    meta=json.loads((out/'diagnostic_manifest.json').read_text())
    assert meta['future_requests_used_for_retrospective_rollout'] is True
    with pytest.raises(ValueError,match='sealed'):
        counterfactual(cfg,tmp_path/'blocked',requests=2,seeds=[61000])
    assert not (tmp_path/'blocked').exists()


def test_milp_trace_explicit_retrospective(cfg,tmp_path):
    with pytest.raises(ValueError,match='retrospective'):
        milp_trace(cfg,tmp_path/'denied',requests=3,seed=41027)
    result=milp_trace(cfg,tmp_path/'trace',requests=3,seed=41027,retrospective=True)
    assert result['not_a_new_heldout_test']
    assert len(csv_rows(tmp_path/'trace'/'milp_greedy_trace.csv'))==6
    assert (tmp_path/'trace'/'trace_summary.json').exists()


def test_model_loading_rejects_mismatched_checkpoint_hash(cfg,tmp_path):
    ck=tmp_path/'fake.pt';ck.write_bytes(b'fake')
    manifest={'winners':[{'algorithm':x,'selected_checkpoint':str(ck),
                         'checkpoint_sha256':'0'*64} for x in ('ppo','bcq','cql','awac','iql')]}
    path=tmp_path/'selection.json';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='SHA-256'):
        load_selected_models(path,None,None,cfg,3,10,2,[20000])


def test_cli_has_all_three_commands():
    from phase5d.cli import parser
    p=parser()
    for command in ('behavior','counterfactual','milp-trace'):
        ns=p.parse_args([command,'--out','x']+(['--retrospective'] if command=='milp-trace' else []))
        assert ns.command==command


def test_all_seven_frozen_models_load_and_behavior(cfg,tmp_path):
    from phase5.environment import FEATURE_NAMES,FEATURE_VERSION
    from phase5.agent import AgentConfig,MaskedDDQN
    from phase5b.sac import SACConfig,MaskedDiscreteSAC
    from phase5c.dataset import checksum
    metadata={'feature_version':FEATURE_VERSION,'feature_names':list(FEATURE_NAMES),
              'topology_config':asdict(cfg),'requests_per_seed':4,
              'headroom_percent':10,'max_candidates':2,
              'train_seeds':[10000],'val_seeds':[20000]}
    winners=[]
    for label in ('ppo','bcq','cql','awac','iql'):
        agent=make_agent(RLConfig(algorithm=label),len(FEATURE_NAMES),3)
        p=tmp_path/(label+'.pt');agent.save(p,{'algorithm':label,**metadata})
        winners.append({'algorithm':label,'selected_checkpoint':str(p),
                        'checkpoint_sha256':checksum(p)})
    selection=tmp_path/'sel.json';selection.write_text(json.dumps({'winners':winners}))
    ddqn=tmp_path/'ddqn.pt'
    MaskedDDQN(len(FEATURE_NAMES),AgentConfig()).save(ddqn,metadata)
    sac=tmp_path/'sac.pt'
    MaskedDiscreteSAC(len(FEATURE_NAMES),SACConfig()).save(sac,metadata)
    models,hashes=load_selected_models(selection,ddqn,sac,cfg,4,10,2,[20000])
    assert len(models)==7 and len(hashes)==7
    result=behavior(cfg,tmp_path/'all_models',models,requests=4,seeds=[20000],model_hashes=hashes)
    assert result['same_state_comparisons']==28
    assert result['trajectory_records']==32
    assert len(csv_rows(tmp_path/'all_models'/'policy_summary.csv'))==8
