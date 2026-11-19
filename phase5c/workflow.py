"""Reproducible Phase5C train/validate/evaluate pipeline.

No retroactive modification of Phases 3--5B. Evaluation is one-shot on fresh
51000-series seeds after validation-only model selection; prior held-out test
sets are forbidden here. Baseline policy runs use the unmodified Phase5 harness.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import random
import statistics as py_statistics
import numpy as np
import torch
from railway_tp.topology import TopologyConfig
from phase3.model import SCHEME
from phase5.environment import FEATURE_NAMES,FEATURE_VERSION,OnlineEpisode,prepare_stream
from phase5.experiment import (new_output,dump_csv,dump_json,source_hashes,_run_one)
from phase5.agent import MaskedDDQN
from phase5b.sac import MaskedDiscreteSAC
from phase5.statistics import summarize
from . import VERSION,ALGORITHMS
from .common import RLConfig,make_agent,CheckpointMixin
from .dataset import checksum,forbidden,load_dataset

EXAMINED_TEST=set(range(31000,31030))|set(range(41000,41030))|set(range(2027,2042))


def run_policy(model,cfg,prep,headroom,k):
    env=OnlineEpisode(cfg,prep,headroom,k)
    while not env.done:
        obs=env.observe()
        env.step(model.act(obs.rows,obs.mask,greedy=True))
    return env.summary()


def _validate(model,cfg,val_streams,headroom,k):
    summaries=[run_policy(model,cfg,p,headroom,k) for p in val_streams.values()]
    return (py_statistics.mean(r['weighted_admission_ratio'] for r in summaries),
            py_statistics.mean(r['admission_ratio'] for r in summaries))


def _common_meta(cfg,requests,headroom,k,train_seeds,val_seeds,config,extra):
    return {'version':VERSION,'algorithm':config.algorithm,'feature_version':FEATURE_VERSION,
            'feature_names':list(FEATURE_NAMES),'topology_config':asdict(cfg),
            'requests_per_seed':requests,'headroom_percent':headroom,'max_candidates':k,
            'train_seeds':sorted(train_seeds),'val_seeds':sorted(val_seeds),
            'training_seed_is_agent_seed':config.seed,
            'protection_model':SCHEME,'protection_model_approved':False,
            'reject_policy':'FORCED_ONLY_NO_VOLUNTARY_REJECTION',
            'objective':'discounted_sum_priority_weight_over_5',**extra}


def train_ppo(cfg:TopologyConfig,out:Path,config:RLConfig,*,requests=100,headroom=10,k=2,
              train_seed_start=10000,train_seed_count=80,val_seed_start=20000,
              val_seed_count=10,episodes=160,validate_every=10,patience=8):
    if config.algorithm!='ppo':raise ValueError('only ppo')
    if min(requests,k,train_seed_count,val_seed_count,episodes,validate_every,patience)<1:
        raise ValueError('positive counts required')
    train_seeds=set(range(train_seed_start,train_seed_start+train_seed_count))
    val_seeds=set(range(val_seed_start,val_seed_start+val_seed_count))
    forbidden(train_seeds,val_seeds)
    started=time.perf_counter()
    new_output(out)
    model=make_agent(config,len(FEATURE_NAMES),k+1)
    validation={s:prepare_stream(cfg,requests,s,k) for s in sorted(val_seeds)}
    train_hashes={}
    rows=[];best=(-1.,-1.);best_ep=0;stale=0
    for ep in range(1,episodes+1):
        seed=train_seed_start+(ep-1)%train_seed_count
        prep=prepare_stream(cfg,requests,seed,k)
        train_hashes[str(seed)]=prep.request_sha256
        env=OnlineEpisode(cfg,prep,headroom,k)
        metrics=model.train_episode(env)
        row={'episode':ep,'train_seed':seed,'training_admission':metrics['admission_ratio'],
             'training_weighted':metrics['weighted_admission_ratio'],
             'gradient_steps':model.gradient_steps,'loss':model.last_loss,
             'validation_weighted':'','validation_admission':'','best_checkpoint':0}
        if ep%validate_every==0 or ep==episodes:
            val=_validate(model,cfg,validation,headroom,k)
            row['validation_weighted']=round(val[0],8)
            row['validation_admission']=round(val[1],8)
            score=(round(val[0],12),round(val[1],12))
            if score>best:
                best=score;best_ep=ep;stale=0;row['best_checkpoint']=1
                model.save(out/'best_model.pt',_common_meta(cfg,requests,headroom,k,
                    train_seeds,val_seeds,config,{'best_episode':ep,'training_regime':'on_policy',
                    'best_validation_weighted_admission':val[0],
                    'best_validation_admission':val[1]}))
            else:stale+=1
        rows.append(row)
        print(f'phase5c ppo ep={ep} train={metrics["admitted"]}/{requests} '
              f'val={row["validation_weighted"]}',flush=True)
        if stale>=patience:break
    model.save(out/'last_model.pt',{'version':VERSION,'note':'not validation-selected'})
    dump_csv(out/'training_history.csv',rows)
    provenance={'version':VERSION,'algorithm':'ppo','training_regime':'on_policy','config':asdict(config),'topology':asdict(cfg),
        'requests_per_seed':requests,'headroom':headroom,'max_candidates':k,
        'training_seeds':sorted(train_seeds),'train_request_sha256_seen':train_hashes,
        'validation_seeds':sorted(val_seeds),
        'val_request_sha256':{str(s):p.request_sha256 for s,p in validation.items()},
        'completed_episodes':len(rows),'best_episode':best_ep,'best_validation_score':best,
        'gradient_steps':model.gradient_steps,'wall_seconds':round(time.perf_counter()-started,3),
        'python_version':platform.python_version(),'torch_version':torch.__version__,
        'protection_model':SCHEME,
        'protection_model_approved':False,'source_sha256':source_hashes()}
    dump_json(out/'training_provenance.json',provenance)
    return provenance


def train_offline(cfg:TopologyConfig,out:Path,config:RLConfig,dataset_dir:Path,*,
                  requests=100,headroom=10,k=2,val_seed_start=20000,val_seed_count=10,
                  updates=3000,validate_every=300,patience=8):
    if config.algorithm=='ppo':raise ValueError('ppo requires train-ppo')
    if min(requests,k,val_seed_count,updates,validate_every,patience)<1:
        raise ValueError('positive counts required')
    data,manifest=load_dataset(dataset_dir)
    if (manifest['topology']!=asdict(cfg) or manifest['headroom']!=headroom or
        manifest['max_candidates']!=k or manifest['requests_per_seed']!=requests):
        raise ValueError('offline dataset environment mismatch')
    train_seeds=set(manifest['seeds']);val_seeds=set(range(val_seed_start,val_seed_start+val_seed_count))
    forbidden(train_seeds,val_seeds)
    started=time.perf_counter()
    new_output(out)
    model=make_agent(config,len(FEATURE_NAMES),k+1)
    val={s:prepare_stream(cfg,requests,s,k) for s in sorted(val_seeds)}
    rng=np.random.default_rng(config.seed)
    rows=[];best=(-1.,-1.);best_step=0;stale=0
    for step in range(1,updates+1):
        ids=rng.integers(0,len(data['action']),size=config.batch)
        loss=model.update(data,ids)
        if step%validate_every==0 or step==updates:
            score_raw=_validate(model,cfg,val,headroom,k)
            score=(round(score_raw[0],12),round(score_raw[1],12))
            row={'gradient_step':step,'loss':round(loss,8),
                'validation_weighted':round(score_raw[0],8),
                'validation_admission':round(score_raw[1],8),'best_checkpoint':0}
            if score>best:
                best=score;best_step=step;stale=0;row['best_checkpoint']=1
                model.save(out/'best_model.pt',_common_meta(cfg,requests,headroom,k,
                    train_seeds,val_seeds,config,{
                     'best_gradient_step':step,'training_regime':'offline',
                     'best_validation_weighted_admission':score_raw[0],
                     'best_validation_admission':score_raw[1],
                     'offline_dataset_sha256':manifest['npz_sha256'],
                     'offline_dataset_manifest_sha256':checksum(dataset_dir/'offline_dataset_manifest.json')}))
            else:stale+=1
            rows.append(row)
            print(f'phase5c {config.algorithm} update={step} '
                  f'val_weighted={row["validation_weighted"]}',flush=True)
            if stale>=patience:break
    model.save(out/'last_model.pt',{'version':VERSION,'note':'not validation-selected'})
    dump_csv(out/'training_history.csv',rows)
    prov={'version':VERSION,'algorithm':config.algorithm,'training_regime':'offline','config':asdict(config),
          'topology':asdict(cfg),'requests_per_seed':requests,'headroom':headroom,
          'max_candidates':k,'offline_dataset_manifest_sha256':
          checksum(dataset_dir/'offline_dataset_manifest.json'),
          'offline_dataset_npz_sha256':manifest['npz_sha256'],
          'offline_dataset_behavior_counts':manifest['behavior_counts'],
          'training_seeds':sorted(train_seeds),'validation_seeds':sorted(val_seeds),
          'val_request_sha256':{str(s):p.request_sha256 for s,p in val.items()},
          'best_gradient_step':best_step,'best_validation_score':best,
          'completed_gradient_steps':model.gradient_steps,
          'wall_seconds':round(time.perf_counter()-started,3),
          'python_version':platform.python_version(),'torch_version':torch.__version__,
          'protection_model':SCHEME,'protection_model_approved':False,
          'source_sha256':source_hashes()}
    dump_json(out/'training_provenance.json',prov)
    return prov


def load_policy(label:str,path:Path):
    if label in ALGORITHMS:
        agent,meta=CheckpointMixin.load(path)
        if meta['algorithm']!=label:raise ValueError('wrong Phase5C algorithm checkpoint')
    elif label=='original_ddqn':agent,meta=MaskedDDQN.load(path)
    elif label=='discrete_sac':agent,meta=MaskedDiscreteSAC.load(path)
    else:raise ValueError('unknown model label '+label)
    return agent,meta


def _run_rl_one(cfg,prep,label,model,headroom,k,out,save_plans,seed):
    # Reuse vetted Phase5 feasibility, hashes, detailed logs and statistics.
    row=_run_one(cfg,prep,'ddqn',headroom,k,model,out,save_plans,5.0,seed)
    row['policy']=label;row['rl_model_family']=label
    csv_path=out/'decisions.csv'
    with csv_path.open(newline='',encoding='utf8') as f:
        rd=csv.DictReader(f);fields=list(rd.fieldnames);records=list(rd)
    for r in records:
        if r['solver_status']=='MASKED_DDQN_ACTION':
            r['solver_status']='MASKED_'+label.upper()+'_ACTION'
    dump_csv(csv_path,records,fields)
    jpath=out/'summary.json'
    summary=json.loads(jpath.read_text(encoding='utf8'))
    summary['policy']=label;summary['rl_model_family']=label
    summary['solver_status_counts']=dict(Counter(r['solver_status'] for r in records))
    dump_json(jpath,summary)
    mpath=out/'manifest.json'
    m=json.loads(mpath.read_text(encoding='utf8'))
    m['policy']=label
    for file in ('summary.json','decisions.csv'):
        m['files_sha256'][file]=checksum(out/file)
    dump_json(mpath,m)
    return row


def evaluate(cfg:TopologyConfig,out:Path,checkpoints:dict[str,Path],*,requests=100,headroom=10,
             k=2,test_seed_start=51000,test_seed_count=30,save_plans=False):
    if not checkpoints or min(requests,k,test_seed_count)<1:
        raise ValueError('invalid eval parameters')
    tests=set(range(test_seed_start,test_seed_start+test_seed_count))
    if tests & EXAMINED_TEST:raise ValueError('previously examined test seeds cannot be Phase5C fresh test')
    models={};metas={};hashes={}
    for label,path in checkpoints.items():
        model,meta=load_policy(label,path)
        if (meta.get('feature_version')!=FEATURE_VERSION or
            meta.get('feature_names')!=list(FEATURE_NAMES) or
            meta.get('topology_config')!=asdict(cfg) or
            meta.get('requests_per_seed')!=requests or
            meta.get('headroom_percent')!=headroom or
            meta.get('max_candidates')!=k):
            raise ValueError('checkpoint environment/feature mismatch for '+label)
        train=(set(meta['train_seeds']) if 'train_seeds' in meta else
               set(range(meta['train_seed_start'],meta['train_seed_start']+meta['train_seed_count'])))
        val=(set(meta['val_seeds']) if 'val_seeds' in meta else
             set(range(meta['val_seed_start'],meta['val_seed_start']+meta['val_seed_count'])))
        if tests & (train|val):raise ValueError('test overlaps train/validation')
        if label=='ppo' and not hasattr(model,'train_episode'):raise AssertionError('ppo mismatch')
        models[label]=model;metas[label]=meta;hashes[label]=checksum(path)
    new_output(out)
    policies=['greedy','cspf','milp_v2','random_feasible']+list(checkpoints)
    rows=[]
    for seed in sorted(tests):
        prepared=prepare_stream(cfg,requests,seed,k)
        for label in policies:
            folder=out/'runs'/f'seed_{seed}_{label}'
            row=(_run_rl_one(cfg,prepared,label,models[label],headroom,k,folder,save_plans,seed)
                 if label in models else
                 _run_one(cfg,prepared,label,headroom,k,None,folder,save_plans,5.0,seed))
            row['demand_seed']=seed;rows.append(row)
            print(f'phase5c eval seed={seed} policy={label} admitted={row["admitted"]}',flush=True)
    dump_csv(out/'phase5_runs.csv',rows)
    by={(r['demand_seed'],r['policy']):r for r in rows}; paired=[]
    for seed in sorted(tests):
        g=by[seed,'greedy']
        for label in policies:
            if label=='greedy':continue
            v=by[seed,label]
            if v['request_sha256']!=g['request_sha256']:
                raise AssertionError('request mismatch')
            paired.append({'demand_seed':seed,'policy':label,'baseline':'greedy',
               'delta_admitted':v['admitted']-g['admitted'],
               'delta_admission_ratio':round(v['admission_ratio']-g['admission_ratio'],6),
               'delta_weighted_ratio':round(v['weighted_admission_ratio']-g['weighted_admission_ratio'],6),
               'delta_avg_selection_ms':round(v['avg_selection_ms']-g['avg_selection_ms'],6)})
    dump_csv(out/'phase5_paired_deltas.csv',paired)
    dump_json(out/'phase5_provenance.json',{'version':VERSION,
        'dataset_role':'NEW_HELD_OUT_AFTER_PHASE5B','test_seeds':sorted(tests),
        'topology_config':asdict(cfg),'requests_per_seed':requests,'headroom_percent':headroom,
        'max_candidates':k,'policies':policies,'checkpoint_sha256':hashes,
        'checkpoint_metadata':metas,'source_sha256':source_hashes(),
        'phase5c_source_sha256':{
           n:checksum(Path(__file__).parent/n) for n in
           ('dataset.py','common.py','offline_agents.py','ppo.py','workflow.py')},
        'request_sha256_by_seed':{str(seed):by[seed,'greedy']['request_sha256'] for seed in tests},
        'protected_scenario_model':SCHEME,'protection_model_approved':False,
        'no_future_request_oracle':True,'forced_reject_only':True,
        'note':'Phase5/Phase5B previously examined tests are not used for new tuning.'})
    return {'runs':len(rows),'seeds':len(tests),'policies':policies}


def statistics(evaluation_dir:Path,out:Path):
    response=summarize(evaluation_dir,out)
    report=out/'phase5_statistical_report.md'
    txt=report.read_text(encoding='utf8').replace(
        '# Phase 5 — Held-out masked DDQN versus conventional baselines',
        '# Phase 5C — New held-out on-policy and offline deep RL comparison')
    report.write_text(txt+'\nPhase5C uses NEW held-out demand seeds; previous Phase5/5B test seeds excluded.\n'
                      'Offline methods use a fixed training-only transition dataset; '
                      'masked PPO collects on-policy simulation experience.\n',encoding='utf8')
    return response


def finetune_awac(cfg:TopologyConfig,out:Path,offline_checkpoint:Path,dataset_dir:Path,*,
                 requests=100,headroom=10,k=2,train_seed_start=10000,train_seed_count=80,
                 val_seed_start=20000,val_seed_count=10,episodes=80,
                 updates_per_episode=64,validate_every=10,patience=8,
                 exploration_epsilon=0.1,online_fraction=0.5):
    """Optional AWAC *offline-to-online* stage using only TRAIN simulation seeds.

    Mixes logged transitions with newly collected simulation transitions for
    advantage-weighted policy updates. No held-out test data is observed.
    The initial policy must come from an offline AWAC checkpoint.
    """
    from .dataset import load_dataset
    if (min(requests,k,train_seed_count,val_seed_count,episodes,updates_per_episode,
            validate_every,patience)<1 or
            not 0 <= exploration_epsilon <= 1 or not 0 <= online_fraction <= 1):
        raise ValueError('invalid AWAC finetune design')
    train_seeds=set(range(train_seed_start,train_seed_start+train_seed_count))
    val_seeds=set(range(val_seed_start,val_seed_start+val_seed_count))
    from .dataset import forbidden
    forbidden(train_seeds,val_seeds)
    model,meta=CheckpointMixin.load(offline_checkpoint)
    if model.config.algorithm!='awac' or meta['algorithm']!='awac':
        raise ValueError('offline AWAC checkpoint is required')
    data,manifest=load_dataset(dataset_dir)
    if (meta['offline_dataset_sha256']!=manifest['npz_sha256'] or
        meta['topology_config']!=asdict(cfg) or
        meta['requests_per_seed']!=requests or meta['max_candidates']!=k or
        meta['headroom_percent']!=headroom or
        set(meta['train_seeds'])!=train_seeds or set(meta['val_seeds'])!=val_seeds):
        raise ValueError('AWAC offline checkpoint / training split mismatch')
    new_output(out)
    rng=np.random.default_rng(model.config.seed+761431)
    py_rng=random.Random(model.config.seed+19361)
    online={key:[] for key in ('rows','mask','action','reward','next_rows','next_mask','done')}
    validation={s:prepare_stream(cfg,requests,s,k) for s in sorted(val_seeds)}
    rows=[];best=(-1.,-1.);best_ep=0;stale=0
    for ep in range(1,episodes+1):
        seed=train_seed_start+(ep-1)%train_seed_count
        stream=prepare_stream(cfg,requests,seed,k)
        env=OnlineEpisode(cfg,stream,headroom,k)
        while not env.done:
            ob=env.observe()
            legal=np.flatnonzero(ob.mask).tolist()
            action=(py_rng.choice(legal) if py_rng.random()<exploration_epsilon
                    else model.act(ob.rows,ob.mask,greedy=True))
            reward,done,_=env.step(action)
            nxt=env.observe() if not done else None
            batch_transition={'rows':ob.rows.copy(),'mask':ob.mask.copy(),
                'action':action,'reward':reward,
                'next_rows':nxt.rows.copy() if nxt else np.zeros_like(ob.rows),
                'next_mask':nxt.mask.copy() if nxt else np.eye(1,k+1,k,dtype=bool)[0],
                'done':done}
            for key,val in batch_transition.items():online[key].append(val)
        n_online=len(online['action'])
        for _ in range(updates_per_episode):
            n_on=int(round(model.config.batch*online_fraction))
            n_on=min(n_on,n_online)
            n_off=model.config.batch-n_on
            off_ids=rng.integers(0,len(data['action']),size=n_off)
            on_ids=rng.integers(0,n_online,size=n_on)
            mixed={key:np.concatenate((data[key][off_ids],
                      np.asarray([online[key][int(i)] for i in on_ids],dtype=data[key].dtype)),axis=0)
                   for key in ('rows','mask','action','reward','next_rows','next_mask','done')}
            model.update(mixed,np.arange(model.config.batch))
        row={'episode':ep,'train_seed':seed,'online_training_transitions':n_online,
            'training_admission':env.summary()['admission_ratio'],
            'training_weighted':env.summary()['weighted_admission_ratio'],
            'gradient_steps_after_offline_pretraining':model.gradient_steps,
            'validation_weighted':'','validation_admission':'','best_checkpoint':0}
        if ep%validate_every==0 or ep==episodes:
            score_raw=_validate(model,cfg,validation,headroom,k)
            row['validation_weighted']=round(score_raw[0],8)
            row['validation_admission']=round(score_raw[1],8)
            score=(round(score_raw[0],12),round(score_raw[1],12))
            if score>best:
                best=score;best_ep=ep;stale=0;row['best_checkpoint']=1
                model.save(out/'best_model.pt',_common_meta(cfg,requests,headroom,k,
                    train_seeds,val_seeds,model.config,{
                    'training_regime':'offline_to_online',
                    'best_episode':ep,'best_validation_weighted_admission':score_raw[0],
                    'best_validation_admission':score_raw[1],
                    'offline_dataset_sha256':manifest['npz_sha256'],
                    'offline_dataset_manifest_sha256':checksum(dataset_dir/'offline_dataset_manifest.json'),
                    'initial_offline_checkpoint_sha256':checksum(offline_checkpoint)}))
            else:stale+=1
        rows.append(row)
        print(f'phase5c awac hybrid episode={ep} val={row["validation_weighted"]}',flush=True)
        if stale>=patience:break
    model.save(out/'last_model.pt',{'version':VERSION,'note':'not validation-selected'})
    dump_csv(out/'training_history.csv',rows)
    prov={'version':VERSION,'algorithm':'awac','training_regime':'offline_to_online',
       'config':asdict(model.config),'topology':asdict(cfg),
       'requests_per_seed':requests,'headroom':headroom,'max_candidates':k,
       'training_seeds':sorted(train_seeds),'validation_seeds':sorted(val_seeds),
       'val_request_sha256':{str(s):p.request_sha256 for s,p in validation.items()},
       'offline_dataset_npz_sha256':manifest['npz_sha256'],
       'initial_offline_checkpoint_sha256':checksum(offline_checkpoint),
       'exploration_epsilon':exploration_epsilon,'online_fraction':online_fraction,
       'updates_per_episode':updates_per_episode,
       'completed_episodes':len(rows),'best_episode':best_ep,'best_validation_score':best,
       'gradient_steps_after_offline_pretraining':model.gradient_steps,
       'online_training_transitions':len(online['action']),
       'protection_model':SCHEME,'protection_model_approved':False,
       'source_sha256':source_hashes()}
    dump_json(out/'training_provenance.json',prov)
    return prov
