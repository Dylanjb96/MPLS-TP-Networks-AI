"""Prospective Phase5B training and reporting; never reuse inspected Phase5 held-out seeds."""
from __future__ import annotations
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
from phase3.model import SCHEME
from phase5.agent import AgentConfig, MaskedDDQN
from phase5.environment import FEATURE_NAMES, FEATURE_VERSION, OnlineEpisode, prepare_stream
from phase5.experiment import (_do_episode, _run_one, dump_csv, dump_json, new_output,
                               source_hashes, train as train_ddqn)
from phase5.statistics import summarize
from railway_tp.topology import TopologyConfig
from .sac import SACConfig, MaskedDiscreteSAC
from . import VERSION

# 31000..31029 were ALREADY examined in the original Phase 5 held-out analysis.
# They are irrevocably unsuitable for Phase5B parameter or model selection.
HISTORICAL = set(range(2027,2042)) | set(range(30000,30002)) | set(range(31000,31030))
RESERVED_FUTURE_TEST = set(range(41000,41030))


def guard_splits(training: set[int], validation:set[int], testing:set[int] | None = None):
    if not training or not validation or (training & validation):
        raise ValueError('training/validation sets must be nonempty and disjoint')
    if (training | validation) & (HISTORICAL | RESERVED_FUTURE_TEST):
        raise ValueError('historical and reserved future test seeds are forbidden for training/validation')
    if testing is not None and ((testing & (training | validation | HISTORICAL)) or not testing):
        raise ValueError('new held-out seeds overlap historical, training or validation seeds')


def diagnose_choice_opportunities(cfg:TopologyConfig, output:Path, *,
                                  requests=100,headroom=10,max_candidates=2,
                                  seed_start=20000,seed_count=10):
    """Greedy-ledger diagnostic: how often are >=2 valid actions actually available?"""
    seeds=set(range(seed_start,seed_start+seed_count))
    if not seeds or seeds & (HISTORICAL | RESERVED_FUTURE_TEST):
        raise ValueError('diagnostic must not reuse previously examined test seeds')
    new_output(output)
    rows=[]
    for seed in sorted(seeds):
        prepared=prepare_stream(cfg,requests,seed,max_candidates)
        env=OnlineEpisode(cfg,prepared,headroom,max_candidates)
        counts=Counter()
        while not env.done:
            obs=env.observe()
            n=int(obs.mask[:max_candidates].sum())
            counts[n]+=1
            action=int(np.flatnonzero(obs.mask)[0])
            env.step(action)
        rows.append({'demand_seed':seed,'requests':requests,
                     'no_feasible':counts[0], 'exactly_one_feasible':counts[1],
                     'two_or_more_feasible':sum(v for k,v in counts.items() if k>=2),
                     'admitted_greedy':env.summary()['admitted'],
                     'request_sha256':prepared.request_sha256})
    dump_csv(output/'choice_opportunities.csv',rows)
    totals={k:sum(r[k] for r in rows) for k in
            ('requests','no_feasible','exactly_one_feasible','two_or_more_feasible')}
    totals.update({'seeds':len(rows),'model':SCHEME,
                   'interpretation':'Counts are under greedy reservation trajectory; other policies may see different legal sets.'})
    dump_json(output/'choice_opportunities_summary.json',totals)
    return totals

def train_sac(cfg:TopologyConfig, output:Path, *, requests=100, headroom=10, max_candidates=2,
              train_seed_start=10000, train_seed_count=80, val_seed_start=20000,
              val_seed_count=10, train_episodes=160, validate_every=10,
              patience=8, agent_config:SACConfig|None=None):
    if min(requests,max_candidates,train_seed_count,val_seed_count,train_episodes,validate_every,patience)<1:
        raise ValueError('all counts must be positive')
    training=set(range(train_seed_start,train_seed_start+train_seed_count))
    validation=set(range(val_seed_start,val_seed_start+val_seed_count))
    guard_splits(training,validation)
    new_output(output)
    ac=agent_config or SACConfig()
    agent=MaskedDiscreteSAC(len(FEATURE_NAMES),ac)
    validation_streams={s:prepare_stream(cfg,requests,s,max_candidates) for s in sorted(validation)}
    histories=[]; training_hashes={}; best=(-1.,-1.); best_ep=None; stale=0
    for ep in range(1,train_episodes+1):
        seed=train_seed_start+(ep-1)%train_seed_count
        prepared=prepare_stream(cfg,requests,seed,max_candidates)
        training_hashes[str(seed)]=prepared.request_sha256
        data,updates=_do_episode(agent,cfg,prepared,headroom,max_candidates,True)
        row={'episode':ep,'train_demand_seed':seed,
             'train_admission_ratio':round(data['admission_ratio'],6),
             'train_weighted_admission_ratio':round(data['weighted_admission_ratio'],6),
             'training_loss_last':agent.last_loss if agent.last_loss is not None else '',
             'updates_in_episode':updates,'total_gradient_steps':agent.gradient_steps,
             'validation_weighted_admission':'','validation_admission':'','checkpoint_best':0}
        if ep%validate_every==0 or ep==train_episodes:
            checked=[_do_episode(agent,cfg,p,headroom,max_candidates,False)[0]
                     for p in validation_streams.values()]
            weighted=statistics.mean(x['weighted_admission_ratio'] for x in checked)
            raw=statistics.mean(x['admission_ratio'] for x in checked)
            row['validation_weighted_admission']=round(weighted,6)
            row['validation_admission']=round(raw,6)
            score=(round(weighted,12),round(raw,12))
            if score>best:
                best=score;best_ep=ep;stale=0;row['checkpoint_best']=1
                agent.save(output/'best_model.pt',{
                    'version':VERSION,'algorithm':'discrete_sac',
                    'feature_version':FEATURE_VERSION,'feature_names':list(FEATURE_NAMES),
                    'topology_config':asdict(cfg),'requests_per_seed':requests,
                    'headroom_percent':headroom,'max_candidates':max_candidates,
                    'train_seed_start':train_seed_start,'train_seed_count':train_seed_count,
                    'val_seed_start':val_seed_start,'val_seed_count':val_seed_count,
                    'best_episode':ep,'best_validation_weighted_admission':weighted,
                    'best_validation_admission':raw,
                    'training_seed_is_agent_seed':ac.seed,'model':SCHEME,
                    'protection_model_approved':False,
                    'reject_policy':'FORCED_ONLY_NO_VOLUNTARY_REJECTION',
                    'objective':'discounted_sum_priority_weight_over_5',
                    'sac_entropy_alpha':ac.alpha})
            else:stale+=1
        histories.append(row)
        print(f'sac episode={ep:03d} seed={seed} admitted={data["admitted"]}/{requests} '
              f'weighted={data["weighted_admission_ratio"]:.3f} '+
              (f'validation={row["validation_weighted_admission"]}' if row['validation_weighted_admission']!='' else ''),
              flush=True)
        if stale>=patience:break
    agent.save(output/'last_model.pt',{'version':VERSION,'note':'last not best'})
    dump_csv(output/'training_history.csv',histories)
    provenance={
        'model':VERSION,'algorithm':'discrete_sac',
        'protection_model':SCHEME,'protection_model_approved':False,
        'cfg':asdict(cfg),'headroom_percent':headroom,'max_candidates':max_candidates,
        'requests_per_seed':requests,'train_seed_start':train_seed_start,
        'train_seed_count':train_seed_count,'val_seed_start':val_seed_start,
        'val_seed_count':val_seed_count,'train_seed_hashes':training_hashes,
        'val_seed_hashes':{str(s):p.request_sha256 for s,p in validation_streams.items()},
        'requested_episodes':train_episodes,'completed_episodes':len(histories),
        'best_episode':best_ep,'best_validation_score':best,
        'gradient_steps':agent.gradient_steps,'agent_config':asdict(ac),
        'feature_names':list(FEATURE_NAMES),
        'source_sha256':source_hashes(),
        'phase5b_source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ('sac.py','workflow.py')},
        'no_historical_test_seeds_used_for_training_or_selection':True}
    dump_json(output/'training_provenance.json',provenance)
    return provenance


def train_model(algorithm, cfg, output, **kwargs):
    seeds=set(range(kwargs['train_seed_start'],kwargs['train_seed_start']+kwargs['train_seed_count']))
    val=set(range(kwargs['val_seed_start'],kwargs['val_seed_start']+kwargs['val_seed_count']))
    guard_splits(seeds,val)
    if algorithm=='ddqn':return train_ddqn(cfg,output,**kwargs)
    if algorithm=='discrete_sac':return train_sac(cfg,output,**kwargs)
    raise ValueError('invalid algorithm')


def load_model(algorithm:str, path:Path):
    if algorithm in ('original_ddqn','tuned_ddqn'):
        model,meta=MaskedDDQN.load(path)
    elif algorithm=='discrete_sac':
        model,meta=MaskedDiscreteSAC.load(path)
    else:raise ValueError(f'unknown policy {algorithm}')
    return model,meta


def _run_rl_one(cfg,prepared,policy,model,headroom,k,output,save_plans,seed):
    """Use the same independent Phase3 commit oracle but label the algorithm accurately."""
    from phase5.experiment import _run_one
    # _run_one executes policy='ddqn' for any agent implementing .act(...,greedy=True).
    # Rewrite the per-run policy name and output metadata to the actual model family.
    row=_run_one(cfg,prepared,'ddqn',headroom,k,model,output,save_plans,5.0,seed)
    row['policy']=policy
    row['rl_model_family']=policy
    records_path=output/'decisions.csv'
    with records_path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames
        records=list(reader)
    status_label={'discrete_sac':'MASKED_DISCRETE_SAC_ACTION',
                  'tuned_ddqn':'MASKED_TUNED_DDQN_ACTION',
                  'original_ddqn':'MASKED_ORIGINAL_DDQN_ACTION'}[policy]
    for record in records:
        if record['solver_status']=='MASKED_DDQN_ACTION':
            record['solver_status']=status_label
    dump_csv(records_path,records,fields)
    summary_path=output/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary['policy']=policy
    summary['rl_model_family']=policy
    summary['solver_status_counts']=dict(Counter(record['solver_status'] for record in records))
    dump_json(summary_path,summary)
    # Every modified output file receives a matching content hash in its manifest.
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['policy']=policy
    for fname in ('summary.json','decisions.csv'):
        manifest['files_sha256'][fname]=hashlib.sha256((output/fname).read_bytes()).hexdigest()
    dump_json(manifest_path,manifest)
    return row


def evaluate_models(cfg,output,checkpoints:dict[str,Path],*,requests=100,headroom=10,
                    max_candidates=2,test_seed_start=41000,test_seed_count=30,
                    save_plans=False):
    if (min(requests,max_candidates,test_seed_count)<1 or 'greedy' in checkpoints or
        any(k not in ('original_ddqn','tuned_ddqn','discrete_sac') for k in checkpoints)):
        raise ValueError('invalid evaluation arguments')
    testing=set(range(test_seed_start,test_seed_start+test_seed_count))
    if testing & HISTORICAL:
        raise ValueError('previously inspected seed was requested as new held-out test')
    models={}; metas={}; hashes={}
    for name,path in checkpoints.items():
        model,meta=load_model(name,path)
        train=set(range(meta['train_seed_start'],meta['train_seed_start']+meta['train_seed_count']))
        val=set(range(meta['val_seed_start'],meta['val_seed_start']+meta['val_seed_count']))
        guard_splits(train,val,testing)
        if (meta.get('feature_version')!=FEATURE_VERSION or
            meta.get('feature_names')!=list(FEATURE_NAMES) or
            meta.get('max_candidates')!=max_candidates or
            meta.get('topology_config')!=asdict(cfg) or
            meta.get('requests_per_seed')!=requests or
            meta.get('headroom_percent')!=headroom):
            raise ValueError(f'checkpoint {name}: config/feature mismatch')
        models[name]=model;metas[name]=meta
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    new_output(output)
    policies=['greedy','cspf','milp_v2','random_feasible']+list(checkpoints)
    rows=[]
    for seed in sorted(testing):
        prepared=prepare_stream(cfg,requests,seed,max_candidates)
        for policy in policies:
            run_folder=output/'runs'/f'seed_{seed}_{policy}'
            if policy in models:
                row=_run_rl_one(cfg,prepared,policy,models[policy],headroom,max_candidates,
                                run_folder,save_plans,seed)
            else:
                row=_run_one(cfg,prepared,policy,headroom,max_candidates,None,run_folder,
                             save_plans,5.,seed)
            row['demand_seed']=seed
            rows.append(row)
            print(f'phase5b seed={seed} policy={policy} admitted={row["admitted"]}/{requests}',flush=True)
    dump_csv(output/'phase5_runs.csv',rows)
    by={(r['demand_seed'],r['policy']):r for r in rows}
    paired=[]
    for seed in sorted(testing):
        g=by[seed,'greedy']
        for policy in policies:
            if policy=='greedy':continue
            p=by[seed,policy]
            if p['request_sha256']!=g['request_sha256']:
                raise AssertionError('unpaired demand')
            paired.append({'demand_seed':seed,'policy':policy,'baseline':'greedy',
                           'delta_admitted':p['admitted']-g['admitted'],
                           'delta_admission_ratio':round(p['admission_ratio']-g['admission_ratio'],6),
                           'delta_weighted_ratio':round(p['weighted_admission_ratio']-g['weighted_admission_ratio'],6),
                           'delta_avg_selection_ms':round(p['avg_selection_ms']-g['avg_selection_ms'],6)})
    dump_csv(output/'phase5_paired_deltas.csv',paired)
    dump_json(output/'phase5_provenance.json',{
        'version':VERSION,'dataset_role':'NEW_HELD_OUT_AFTER_HISTORICAL_PHASE5',
        'previously_inspected_test_seeds':sorted(HISTORICAL & set(range(31000,31030))),
        'config':asdict(cfg),'requests_per_seed':requests,'headroom_percent':headroom,
        'max_candidates':max_candidates,'seeds':sorted(testing),
        'policies':policies,'checkpoints_sha256':hashes,'checkpoint_metadata':metas,
        'source_sha256':source_hashes(),
        'phase5b_source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ('sac.py','workflow.py')},
        'protection_model':SCHEME,'protection_model_approved':False,
        'no_future_request_oracle':True,'forced_reject_only':True,
        'note':'Historical Phase5 held-out results are reported separately, not used for Phase5B tuning.'})
    return {'runs':len(rows),'seeds':len(testing),'policies':policies}
