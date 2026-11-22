"""Phase 5E experiment management: prospectively separated datasets.

Primary claim remains unchanged 60-node, k=2, 100 requests, 10% headroom.
Training and validation ONLY use declared development streams. A NEW 61000-61059
heldout is opened only after the validation-only model-selection manifest exists.
A separate 62000-series sensitivity may use different k; never pool its results
with the fixed primary scenario or select models using test metrics.
"""
from __future__ import annotations
from collections import Counter,defaultdict,OrderedDict
from dataclasses import asdict
import csv,hashlib,json,platform,random,statistics,time
from pathlib import Path
import numpy as np
import torch
import scipy
from scipy import stats as scipy_stats
from railway_tp.topology import TopologyConfig,make_topology
from railway_tp.requests import generate_requests
from phase3.model import SCHEME,ProtectionSimulator
from phase5.environment import OnlineEpisode,PreparedStream,prepare_stream,stream_sha256
from phase5.experiment import new_output,dump_csv,dump_json,source_hashes,_run_one
from phase5c.dataset import checksum
from phase5c.workflow import load_policy,_run_rl_one
from .policy import ResidualEpisode,feature_names,controls_action
from .agent import HybridConfig,ResidualPPO
from . import VERSION

PRIOR_TEST=set(range(31000,31030))|set(range(41000,41030))|set(range(51000,51030))|set(range(2027,2042))
PRIMARY_TEST=set(range(61000,61060))
SENSITIVITY_TEST=set(range(62000,62060))
TEST_FORBIDDEN=PRIOR_TEST|PRIMARY_TEST|SENSITIVITY_TEST
PREVIOUS_DEV=set(range(10000,10080))|set(range(20000,20010))|set(range(21000,21030))
SOURCE_FILES=('phase3/model.py','phase4/policies.py','phase5/environment.py',
    'phase5/experiment.py','phase5/agent.py','phase5c/ppo.py',
    'phase5e/policy.py','phase5e/agent.py','phase5e/workflow.py',
    'railway_tp/topology.py','railway_tp/requests.py')


def hashes():
    root=Path(__file__).resolve().parents[1]
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in SOURCE_FILES if (root/n).is_file()}


def cohort_guard(train_seeds,val_seeds):
    t,v=set(train_seeds),set(val_seeds)
    if (not t or not v or len(t)!=len(list(train_seeds)) or
        len(v)!=len(list(val_seeds)) or t&v or (t|v)&TEST_FORBIDDEN):
        raise ValueError('invalid/overlapping training or validation cohorts, or use of any held-out seeds')
    if any(s<10000 or s>=30000 for s in t|v):
        raise ValueError('development seeds must be in [10000,29999]')
    if any(s in PREVIOUS_DEV for s in v):
        raise ValueError('new Phase 5E validation cannot reuse previously examined development seeds')


def prepare_for_headroom(cfg,requests,seed,k,headroom):
    """Use the exact Phase-5 standard stream at 10%; for sensitivity cases
    generate policy-valid candidates with the matching Phase-3 headroom.
    """
    if headroom==10:return prepare_stream(cfg,requests,seed,k)
    graph,rings=make_topology(cfg)
    incoming=tuple(generate_requests(graph,requests,seed=seed))
    t0=time.perf_counter()
    sim=ProtectionSimulator(graph,rings,headroom)
    plans=tuple(tuple(sim.candidates(r,k)) for r in incoming)
    return PreparedStream(incoming,plans,stream_sha256(incoming),time.perf_counter()-t0)


def validate_env(requests,headroom,k):
    if requests<1 or k<2 or k>8 or not 0<=headroom<100:
        raise ValueError('invalid requests/headroom/k (k must be 2..8 for Phase 5E)')


def _evaluate_model(model,cfg,prepared,headroom,k,threshold):
    data=[]
    for seed,prep in prepared.items():
        env=ResidualEpisode(OnlineEpisode(cfg,prep,headroom,k),enhanced=model.config.enhanced)
        while not env.done:
            obs=env.observe();env.step(model.act(obs.rows,obs.mask,threshold=threshold))
        data.append(env.summary())
    return {'weighted':statistics.mean(v['weighted_admission_ratio'] for v in data),
            'admission':statistics.mean(v['admission_ratio'] for v in data),
            'overrides':sum(v['overrides'] for v in data),
            'discretionary':sum(v['discretionary'] for v in data)}


def train(cfg:TopologyConfig,out:Path,config:HybridConfig,*,requests=100,
          headroom=10,k=2,train_start=11000,train_count=80,
          val_start=22000,val_count=20,episodes=160,validate_every=10,patience=8):
    validate_env(requests,headroom,k)
    if min(train_count,val_count,episodes,validate_every,patience)<1:
        raise ValueError('positive training counts required')
    train_seeds=list(range(train_start,train_start+train_count))
    val_seeds=list(range(val_start,val_start+val_count))
    cohort_guard(train_seeds,val_seeds)
    if out.exists():raise FileExistsError('refusing to overwrite '+str(out))
    t0=time.perf_counter()
    model=ResidualPPO(k,config)
    val={s:prepare_for_headroom(cfg,requests,s,k,headroom) for s in val_seeds}
    # Keep at most four expensive prepared streams in memory.
    cache: OrderedDict[int,object]=OrderedDict()
    training_hashes={}
    def train_prep(seed):
        if seed not in cache:
            cache[seed]=prepare_for_headroom(cfg,requests,seed,k,headroom)
            training_hashes[str(seed)]=cache[seed].request_sha256
            if len(cache)>4:cache.popitem(last=False)
        else:cache.move_to_end(seed)
        return cache[seed]
    # Imitation warm start on TRAINING-ONLY states, never validation/test.
    warm=[]
    warm_n=min(4,train_count)
    for seed in train_seeds[:warm_n]:
        env=ResidualEpisode(OnlineEpisode(cfg,train_prep(seed),headroom,k),enhanced=config.enhanced)
        while not env.done:
            o=env.observe()
            if o.discretionary:warm.append((o.rows.copy(),o.mask.copy()))
            env.step(0)
    warm_observations=model.warmstart(warm)
    new_output(out)
    best=(-1.,-1.);best_ep=None;stale=0
    history=[]
    for ep in range(1,episodes+1):
        seed=train_seeds[(ep-1)%train_count]
        env=ResidualEpisode(OnlineEpisode(cfg,train_prep(seed),headroom,k),enhanced=config.enhanced)
        metrics=model.train_episode(env)
        row={'episode':ep,'train_demand_seed':seed,'train_weighted':metrics['weighted_admission_ratio'],
             'train_admission':metrics['admission_ratio'],'train_overrides':metrics['overrides'],
             'train_discretionary':metrics['discretionary'],'gradient_steps':model.gradient_steps,
             'loss':model.last_loss,'validation_weighted':'','validation_admission':'',
             'validation_overrides':'','best_checkpoint':0}
        if ep%validate_every==0 or ep==episodes:
            results=_evaluate_model(model,cfg,val,headroom,k,config.gate_threshold)
            score=(round(results['weighted'],12),round(results['admission'],12))
            row.update(validation_weighted=results['weighted'],
                       validation_admission=results['admission'],
                       validation_overrides=results['overrides'])
            if score>best:
                best=score;best_ep=ep;stale=0;row['best_checkpoint']=1
                meta={'version':VERSION,'algorithm':'greedy_guided_residual_ppo',
                      'configuration':asdict(config),'feature_names':feature_names(config.enhanced),
                      'topology':asdict(cfg),'requests':requests,'headroom':headroom,'k':k,
                      'train_seeds':train_seeds,'val_seeds':val_seeds,
                      'train_request_sha256_seen':dict(training_hashes),
                      'val_request_sha256':{str(s):p.request_sha256 for s,p in val.items()},
                      'best_episode':best_ep,'best_validation_weighted':results['weighted'],
                      'best_validation_admission':results['admission'],
                      'gate_threshold':config.gate_threshold,'protection_model':SCHEME,
                      'protection_model_approved':False,
                      'objective':'sum_priority_weight_divided_by_5_gamma_'+str(config.gamma),
                      'no_future_oracle':True,'source_sha256':hashes()}
                model.save(out/'best_model.pt',meta)
            else:stale+=1
        history.append(row)
        print(f'phase5e train ep={ep} init={config.seed} seed={seed} '
              f'weighted={metrics["weighted_admission_ratio"]:.4f} '
              f'val={row["validation_weighted"]}',flush=True)
        if stale>=patience:break
    model.save(out/'last_model.pt',{'version':VERSION,'note':'NOT selected for evaluation'})
    dump_csv(out/'training_history.csv',history)
    prov={'version':VERSION,'algorithm':'greedy_guided_residual_ppo','config':asdict(config),
          'topology':asdict(cfg),'requests':requests,'headroom':headroom,'k':k,
          'train_seeds':train_seeds,'val_seeds':val_seeds,'training_hashes_seen':training_hashes,
          'val_request_sha256':{str(s):p.request_sha256 for s,p in val.items()},
          'episodes_completed':len(history),'best_episode':best_ep,
          'best_validation_score':list(best),'gradient_steps':model.gradient_steps,
          'warmstart_training_choice_states':warm_observations,
          'source_sha256':hashes(), 'protection_model':SCHEME,
          'protection_model_approved':False,'no_test_oracle':True,
          'wall_seconds':round(time.perf_counter()-t0,3),
          'python_version':platform.python_version(),'torch_version':torch.__version__}
    dump_json(out/'training_provenance.json',prov)
    return {'run':str(out),'best_episode':best_ep,'best_validation_score':list(best),
            'episodes_completed':len(history),'warmstart_states':warm_observations}


def select(out:Path,runs_root:Path,min_seeds=3):
    if min_seeds<2:raise ValueError('at least two training initializations necessary')
    dirs=sorted(set(p.parent for p in runs_root.glob('**/training_provenance.json')))
    if not dirs:raise ValueError('no Phase 5E training records found')
    grouped=defaultdict(list);cohort=None;common_hashes=None;input_rows=[]
    for folder in dirs:
        source=folder/'training_provenance.json';pt=folder/'best_model.pt'
        if not pt.is_file():raise FileNotFoundError(pt)
        prov=json.loads(source.read_text(encoding='utf-8'))
        if prov.get('version')!=VERSION or prov.get('algorithm')!='greedy_guided_residual_ppo':
            raise ValueError('unexpected non-Phase5E train run in selection root')
        c=dict(prov['config']);seed=c.pop('seed')
        spec=(prov['topology'],prov['requests'],prov['headroom'],prov['k'],
              prov['train_seeds'],prov['val_seeds'],prov['val_request_sha256'])
        if cohort is None:cohort=spec
        elif cohort!=spec:raise ValueError('training/validation environment or hashes inconsistent')
        if common_hashes is None:common_hashes=prov['source_sha256']
        elif common_hashes!=prov['source_sha256']:
            raise ValueError('source code differed across training runs')
        model,meta=ResidualPPO.load(pt)
        if (meta.get('best_episode')!=prov['best_episode'] or
            list(prov['best_validation_score'])!=[
              round(meta['best_validation_weighted'],12),
              round(meta['best_validation_admission'],12)] or
            meta['configuration']!=prov['config']):
            raise ValueError('provenance/checkpoint mismatch in '+str(folder))
        if (meta.get('train_seeds')!=prov['train_seeds'] or
            meta.get('val_seeds')!=prov['val_seeds'] or
            meta.get('val_request_sha256')!=prov['val_request_sha256'] or
            meta.get('source_sha256')!=prov['source_sha256']):
            raise ValueError('checkpoint provenance/training cohort mismatch in '+str(folder))
        with (folder/'training_history.csv').open(newline='',encoding='utf-8') as f:
            history=list(csv.DictReader(f))
        if len(history)!=prov['episodes_completed'] or not history:
            raise ValueError('training history length mismatch in '+str(folder))
        best_rows=[r for r in history if r['best_checkpoint']=='1']
        if not best_rows or int(best_rows[-1]['episode'])!=prov['best_episode']:
            raise ValueError('best checkpoint log flag mismatch in '+str(folder))
        if (abs(float(best_rows[-1]['validation_weighted'])-prov['best_validation_score'][0])>1e-10 or
            abs(float(best_rows[-1]['validation_admission'])-prov['best_validation_score'][1])>1e-10):
            raise ValueError('best validation metrics mismatch in '+str(folder))
        # Only config parameters, not random seed, define a trial group.
        key=json.dumps(c,sort_keys=True)
        record={'seed':seed,'checkpoint':str(pt),'checkpoint_sha256':checksum(pt),
                'provenance_sha256':checksum(source),'score':prov['best_validation_score'],
                'best_episode':prov['best_episode']}
        grouped[key].append(record)
        input_rows.append((key,seed))
    trials=[];options=[]
    for key,records in grouped.items():
        ss=[r['seed'] for r in records]
        if len(records)<min_seeds or len(ss)!=len(set(ss)):
            raise ValueError('need >=min_seeds DISTINCT initializations for EVERY configuration')
        w=statistics.mean(r['score'][0] for r in records)
        a=statistics.mean(r['score'][1] for r in records)
        trials.append({'config_json':key,'initializations':len(records),
                       'agent_seeds':','.join(map(str,sorted(ss))),
                       'mean_validation_weighted':w,'mean_validation_admission':a})
        options.append((round(w,12),round(a,12),key,records))
    # A deterministic key tie-break, not a scientific claim of superiority.
    _,_,winning_key,records=max(options,key=lambda v:(v[0],v[1],v[2]))
    chosen=max(records,key=lambda r:(round(r['score'][0],12),
                                     round(r['score'][1],12),-r['seed']))
    new_output(out)
    dump_csv(out/'selection_trials.csv',sorted(trials,key=lambda r:r['config_json']))
    manifest={'version':VERSION,'selection_rule':'validation-only grouped mean weighted admission; '
              'single checkpoint weighted score and lowest agent seed on tie',
              'winner_config':json.loads(winning_key),'winner_mean_weighted':statistics.mean(
                  r['score'][0] for r in records),
              'winner_mean_admission':statistics.mean(r['score'][1] for r in records),
              'selected_seed':chosen['seed'],'selected_best_episode':chosen['best_episode'],
              'selected_checkpoint':chosen['checkpoint'],
              'checkpoint_sha256':chosen['checkpoint_sha256'],
              'training_provenance_sha256':chosen['provenance_sha256'],
              'training_seeds':cohort[4],'validation_seeds':cohort[5],
              'validation_request_sha256':cohort[6],
              'topology':cohort[0],'requests':cohort[1],'headroom':cohort[2],'k':cohort[3],
              'source_sha256':common_hashes,'initializations_per_config':min_seeds,
              'models_compared':len(grouped),'heldout_test_never_consulted':True,
              'protection_model':SCHEME,'protection_model_approved':False}
    dump_json(out/'selection_manifest.json',manifest)
    return manifest


def _run_residual_one(cfg,prep,label,model,out,headroom,k,seed,*,gate=True,
                      threshold=None,control=None,save_plans=False):
    new_output(out)
    env=ResidualEpisode(OnlineEpisode(cfg,prep,headroom,k),enhanced=(model.config.enhanced if model else False))
    timing=0.;forward=0.;rows=[];rng=random.Random(88437+int(seed))
    while not env.done:
        before=env.env.sim.state_digest()
        begin=time.perf_counter()
        obs=env.observe()
        inference_ms=''
        if model is not None:
            t1=time.perf_counter()
            chosen=model.act(obs.rows,obs.mask,gate=gate,threshold=threshold)
            inference_ms=(time.perf_counter()-t1)*1000
            forward+=inference_ms/1000
        else:chosen=controls_action(control,obs,rng,threshold=threshold if threshold is not None else .10)
        _,_,rec=env.step(chosen)
        elapsed=time.perf_counter()-begin
        timing+=elapsed
        rows.append({**rec,'demand_seed':seed,'policy':label,
                     'request_index':len(rows),'state_before_sha256':before,
                     'state_after_sha256':env.env.sim.state_digest(),
                     'decision_ms':round(elapsed*1000,6),
                     'model_inference_ms':round(inference_ms,6) if model else ''})
    summary=env.summary()
    summary.update({'policy':label,'demand_seed':seed,'request_sha256':prep.request_sha256,
                   'avg_selection_ms':timing/len(prep.requests)*1000,
                   'avg_model_only_ms':forward/len(prep.requests)*1000 if model else '',
                   'candidate_generation_seconds_shared':prep.generation_seconds,
                   'no_future_request_oracle':True,
                   'protection_model':SCHEME,'protection_model_approved':False})
    dump_csv(out/'requests.csv',[asdict(r) for r in prep.requests])
    dump_csv(out/'decisions.csv',rows)
    dump_json(out/'summary.json',summary)
    if save_plans:
        dump_json(out/'accepted_protection_plans.json',{
            'protection_model':SCHEME,
            'admitted':[{'request_id':r.request_id,'candidate_id':p.candidate_id,
                         'working_forward':list(p.normal_path),
                         'failure_state_forward_paths':{f:list(v) for f,v in p.fault_paths.items()}}
                        for r,p in env.env.sim.accepted()]})
    dump_json(out/'manifest.json',{'version':VERSION,'policy':label,
             'request_sha256':prep.request_sha256,'files_sha256':{
                 x:checksum(out/x) for x in (['requests.csv','decisions.csv','summary.json']+
                    (['accepted_protection_plans.json'] if save_plans else []))}})
    return summary


def probe(cfg:TopologyConfig,out:Path,*,requests=100,headroom=10,k=2,
          seed_start=23000,seed_count=10,capacity_threshold=.05):
    """Development-only random and capacity controls; not model selection."""
    validate_env(requests,headroom,k)
    seeds=list(range(seed_start,seed_start+seed_count))
    if not seeds or set(seeds)&TEST_FORBIDDEN or any(not 10000<=s<30000 for s in seeds):
        raise ValueError('probe may only use declared development seeds')
    new_output(out)
    results=[];hashes_by_seed={};inventory=Counter()
    for seed in seeds:
        prep=prepare_for_headroom(cfg,requests,seed,k,headroom)
        hashes_by_seed[str(seed)]=prep.request_sha256
        inventory.update(len(c) for c in prep.candidate_sets)
        for mode in ('greedy','random_override','capacity_override'):
            row=_run_residual_one(cfg,prep,mode,None,
                   out/'runs'/f'seed_{seed}_{mode}',headroom,k,seed,
                   control=mode,threshold=capacity_threshold)
            results.append(row)
            print(f'phase5e probe seed={seed} mode={mode} '
                  f'admitted={row["admitted"]} overrides={row["overrides"]}',flush=True)
    dump_csv(out/'probe_runs.csv',results)
    by={(r['demand_seed'],r['policy']):r for r in results}
    comparisons=[]
    for seed in seeds:
        for mode in ('random_override','capacity_override'):
            alt=by[seed,mode];g=by[seed,'greedy']
            comparisons.append({'demand_seed':seed,'alternative':mode,
                'admitted_delta':alt['admitted']-g['admitted'],
                'weighted_delta':alt['weighted_admission_ratio']-g['weighted_admission_ratio'],
                'overrides':alt['overrides'],'discretionary':alt['discretionary']})
    dump_csv(out/'probe_paired.csv',comparisons)
    summary={}
    for mode in ('random_override','capacity_override'):
        sub=[r for r in comparisons if r['alternative']==mode]
        summary[mode]={'wins':sum(r['weighted_delta']>1e-12 for r in sub),
                       'losses':sum(r['weighted_delta']<-1e-12 for r in sub),
                       'ties':sum(abs(r['weighted_delta'])<=1e-12 for r in sub),
                       'mean_weighted_difference':statistics.mean(r['weighted_delta'] for r in sub)}
    dump_json(out/'probe_manifest.json',{'version':VERSION,
        'role':'DEVELOPMENT_ONLY_NO_MODEL_SELECTION','topology':asdict(cfg),
        'requests':requests,'headroom':headroom,'k':k,'seeds':seeds,
        'request_sha256':hashes_by_seed,'probe_summary':summary,
        'candidate_inventory':{str(key):val for key,val in sorted(inventory.items())},
        'k_above_two_actually_used':any(key>2 for key in inventory),
        'source_sha256':hashes(),'protection_model':SCHEME,
        'protection_model_approved':False})
    return summary


def _checkpoint_compatible(model,meta,cfg,requests,headroom,k):
    expected={'version':VERSION,'topology':asdict(cfg),'requests':requests,
              'headroom':headroom,'k':k,'feature_names':feature_names(model.config.enhanced),
              'protection_model':SCHEME,'protection_model_approved':False}
    for key,value in expected.items():
        if meta.get(key)!=value:raise ValueError('selected checkpoint mismatch for '+key)
    cohort_guard(meta['train_seeds'],meta['val_seeds'])
    # Source code must remain frozen between training and evaluation.
    for name,recorded in meta['source_sha256'].items():
        if hashes().get(name)!=recorded:
            raise ValueError('source changed since training: '+name)


def _load_phase5c_selection(path,cfg,requests,headroom,k):
    if k!=2:raise ValueError('Phase5C checkpoints only support k=2')
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj['topology']!=asdict(cfg) or obj['headroom']!=headroom or obj['max_candidates']!=k or obj['requests_per_seed']!=requests:
        raise ValueError('legacy selection manifest environment mismatch')
    winners=obj['winners']
    if {x['algorithm'] for x in winners}!={'ppo','bcq','cql','awac','iql'}:
        raise ValueError('expected five original frozen Phase5C policies')
    paths={}
    for x in winners:
        pt=Path(x['selected_checkpoint'])
        if checksum(pt)!=x['checkpoint_sha256']:
            raise ValueError('Phase5C model changed since selection: '+x['algorithm'])
        paths[x['algorithm']]=pt
    return paths


def evaluate(cfg:TopologyConfig,out:Path,selection:Path,*,requests=100,headroom=10,k=2,
             seed_start=61000,seed_count=60,role='primary',
             phase5c_selection=None,original_ddqn=None,discrete_sac=None,
             save_plans=False,milp_timeout=5.0,capacity_threshold=.05):
    validate_env(requests,headroom,k)
    if role not in ('primary','sensitivity','development'):raise ValueError('unknown evaluation role')
    seeds=list(range(seed_start,seed_start+seed_count))
    if not seeds:raise ValueError('seed count must be positive')
    tests=set(seeds)
    if role=='primary':
        if (seed_start!=61000 or seed_count!=60 or requests!=100 or headroom!=10 or k!=2 or
            cfg.nodes!=60 or cfg.seed!=42 or cfg.gateway_mode!='dual'):
            raise ValueError('primary evaluation strictly requires the registered 60-node / k=2 / 61000-61059 protocol')
    elif role=='sensitivity':
        if (seed_start!=62000 or seed_count!=60 or k==2):
            raise ValueError('sensitivity evaluation requires k>2 and registered seeds 62000-62059')
    else:
        if tests & TEST_FORBIDDEN or any(not 10000<=s<30000 for s in tests):
            raise ValueError('development evaluation must not inspect ANY heldout set')
    chosen_manifest=json.loads(selection.read_text(encoding='utf-8'))
    if chosen_manifest.get('version')!=VERSION or chosen_manifest.get('models_compared',0)<1:
        raise ValueError('Phase5E validation-only selection manifest required')
    path=Path(chosen_manifest['selected_checkpoint'])
    if checksum(path)!=chosen_manifest['checkpoint_sha256']:
        raise ValueError('selected hybrid model hash mismatch')
    agent,meta=ResidualPPO.load(path)
    _checkpoint_compatible(agent,meta,cfg,requests,headroom,k)
    if tests & (set(meta['train_seeds'])|set(meta['val_seeds'])):
        raise ValueError('evaluation requests overlap training/validation')
    if (chosen_manifest['winner_config']!={key:value for key,value in meta['configuration'].items() if key!='seed'} or
        chosen_manifest['selected_seed']!=meta['configuration']['seed']):
        raise ValueError('selected config or seed does not match checkpoint')
    legacy={}
    if phase5c_selection is not None:legacy.update(_load_phase5c_selection(
       phase5c_selection,cfg,requests,headroom,k))
    if original_ddqn is not None:legacy['original_ddqn']=Path(original_ddqn)
    if discrete_sac is not None:legacy['discrete_sac']=Path(discrete_sac)
    legacy_models={}; legacy_hashes={}
    for name,pt in legacy.items():
        model,old_meta=load_policy(name,pt)
        from phase5.environment import FEATURE_NAMES,FEATURE_VERSION
        expected={'topology_config':asdict(cfg),'requests_per_seed':requests,
                  'headroom_percent':headroom,'max_candidates':k,
                  'feature_version':FEATURE_VERSION,'feature_names':list(FEATURE_NAMES)}
        for key,value in expected.items():
            if old_meta.get(key)!=value:raise ValueError('legacy checkpoint mismatch '+name+' '+key)
        old_train=set(old_meta['train_seeds']) if 'train_seeds' in old_meta else set(range(
            old_meta['train_seed_start'],old_meta['train_seed_start']+old_meta['train_seed_count']))
        old_val=set(old_meta['val_seeds']) if 'val_seeds' in old_meta else set(range(
            old_meta['val_seed_start'],old_meta['val_seed_start']+old_meta['val_seed_count']))
        if tests&(old_train|old_val):raise ValueError('legacy checkpoint train/validation leak')
        legacy_models[name]=model;legacy_hashes[name]=checksum(pt)
    # All checks complete BEFORE creating evidence directory.
    new_output(out)
    labels=['greedy','cspf','milp_v2','random_feasible',
            'random_override','capacity_override','hybrid_gated','hybrid_ungated']+list(legacy)
    runs=[];requests_hashes={}
    for seed in seeds:
        prepared=prepare_stream(cfg,requests,seed,k)
        requests_hashes[str(seed)]=prepared.request_sha256
        for label in labels:
            folder=out/'runs'/f'seed_{seed}_{label}'
            if label in ('greedy','cspf','milp_v2','random_feasible'):
                row=_run_one(cfg,prepared,label,headroom,k,None,folder,save_plans,milp_timeout,seed)
            elif label in legacy:
                row=_run_rl_one(cfg,prepared,label,legacy_models[label],headroom,k,folder,save_plans,seed)
            else:
                row=_run_residual_one(cfg,prepared,label,
                    agent if label.startswith('hybrid_') else None,
                    folder,headroom,k,seed,
                    gate=(label!='hybrid_ungated'),
                    control=label if not label.startswith('hybrid_') else None,
                    threshold=(capacity_threshold if label=='capacity_override' else None),
                    save_plans=save_plans)
            row['demand_seed']=seed;row['policy']=label
            # Independently reconstruct full-precision metrics from decisions.
            with (folder/'decisions.csv').open(newline='',encoding='utf-8') as f:
                actions=list(csv.DictReader(f))
            if len(actions)!=requests:raise AssertionError('missing request records')
            total_w=sum(r.priority_weight for r in prepared.requests)
            admitted=sum(int(r['admitted']) for r in actions)
            weighted=sum(int(r['admitted'])*int(r['priority_weight']) for r in actions)/total_w
            if admitted!=int(row['admitted']):raise AssertionError('admission count mismatch in '+label)
            row['admission_ratio']=admitted/requests
            row['weighted_admission_ratio']=weighted
            row['request_sha256']=prepared.request_sha256
            row['model']=SCHEME
            runs.append(row)
            print(f'phase5e {role} seed={seed} policy={label} admitted={admitted} '
                  f'weighted={weighted:.5f}',flush=True)
    dump_csv(out/'phase5e_runs.csv',runs)
    look={(int(r['demand_seed']),r['policy']):r for r in runs}
    paired=[]
    for seed in seeds:
        base=look[seed,'greedy']
        for label in labels:
            if label=='greedy':continue
            alt=look[seed,label]
            if base['request_sha256']!=alt['request_sha256']:
                raise AssertionError('paired stream hash mismatch')
            paired.append({'demand_seed':seed,'policy':label,
                'admission_delta':alt['admission_ratio']-base['admission_ratio'],
                'weighted_delta':alt['weighted_admission_ratio']-base['weighted_admission_ratio'],
                'admitted_count_delta':alt['admitted']-base['admitted'],
                'greedy_state_digest':base['state_digest'],
                'policy_state_digest':alt['state_digest']})
    dump_csv(out/'phase5e_paired_deltas.csv',paired)
    manifest={'version':VERSION,'role':role,'selection_manifest_sha256':checksum(selection),
              'selected_hybrid_checkpoint_sha256':checksum(path),'legacy_checkpoint_sha256':legacy_hashes,
              'model_selection_locked':True,'test_seeds':seeds,'requests':requests,
              'headroom':headroom,'k':k,'topology':asdict(cfg),'policies':labels,
              'request_sha256_by_seed':requests_hashes,'source_sha256':hashes(),
              'no_future_request_oracle':True,'selection_target':'priority_weighted_admission',
              'no_retraining_after_heldout':True,'protection_model':SCHEME,
              'protection_model_approved':False,'forbidden_claim':'MSRP-compliance/restoration-time'}
    dump_json(out/'evaluation_manifest.json',manifest)
    return {'runs':len(runs),'policies':labels,'seeds':len(seeds),
            'heldout':role!='development','out':str(out)}


def _ci95(values):
    a=np.asarray(values,dtype=float);n=len(a);mu=float(a.mean())
    if n<2:return (mu,float('nan'),float('nan'))
    sd=float(a.std(ddof=1))
    if sd==0:return (mu,mu,mu)
    half=float(scipy_stats.t.ppf(.975,n-1))*sd/np.sqrt(n)
    return mu,mu-half,mu+half


def _pvalue(diffs):
    a=np.asarray(diffs,dtype=float)
    if len(a)<2:return 1.0
    if np.allclose(a,a[0],atol=1e-15,rtol=0):
        return 1.0 if abs(float(a[0]))<1e-15 else 0.0
    return float(scipy_stats.ttest_1samp(a,0.0).pvalue)


def _holm(pvalues):
    """Holm family-wise adjusted two-sided p-values, monotone by sorted rank."""
    n=len(pvalues);out=[0.0]*n
    sorted_ids=sorted(range(n),key=lambda i:pvalues[i]);running=0.0
    for rank,i in enumerate(sorted_ids):
        running=max(running,min(1.0,(n-rank)*pvalues[i]))
        out[i]=running
    return out


def statistics_report(eval_dir:Path,out:Path):
    source=eval_dir/'phase5e_runs.csv'
    manifest=json.loads((eval_dir/'evaluation_manifest.json').read_text(encoding='utf-8'))
    with source.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    seeds=manifest['test_seeds'];labels=manifest['policies'];expected=len(seeds)*len(labels)
    if len(rows)!=expected:raise ValueError('unexpected number of evaluation rows')
    by={}
    for r in rows:
        key=(int(r['demand_seed']),r['policy'])
        if key in by:raise ValueError('duplicate seed/policy')
        by[key]=r
    summary=[]
    for label in labels:
        rs=[by[s,label] for s in seeds]
        weighted=[float(r['weighted_admission_ratio']) for r in rs]
        ordinary=[float(r['admission_ratio']) for r in rs]
        mean,lower,upper=_ci95(weighted)
        raw,raw_lo,raw_hi=_ci95(ordinary)
        summary.append({'policy':label,'admitted_total':sum(int(r['admitted']) for r in rs),
                        'offered_total':sum(int(r['requests']) for r in rs),
                        'mean_weighted':mean,'weighted_ci_low':lower,'weighted_ci_high':upper,
                        'mean_admission':raw,'admission_ci_low':raw_lo,'admission_ci_high':raw_hi,
                        'mean_selection_ms':statistics.mean(float(r['avg_selection_ms']) for r in rs),
                        'same_state_as_greedy':sum(r['state_digest']==by[s,'greedy']['state_digest']
                                                   for s,r in zip(seeds,rs))})
    comparisons=[]
    for label in labels:
        if label=='greedy':continue
        dw=[float(by[s,label]['weighted_admission_ratio'])-
            float(by[s,'greedy']['weighted_admission_ratio']) for s in seeds]
        da=[float(by[s,label]['admission_ratio'])-
            float(by[s,'greedy']['admission_ratio']) for s in seeds]
        w,wlo,whi=_ci95(dw); a,alo,ahi=_ci95(da)
        comparisons.append({'policy':label,'seeds':len(seeds),
           'mean_weighted_delta_pp':100*w,'weighted_ci_low_pp':100*wlo,
           'weighted_ci_high_pp':100*whi,'weighted_p_unadjusted':_pvalue(dw),
           'mean_admission_delta_pp':100*a,'admission_ci_low_pp':100*alo,
           'admission_ci_high_pp':100*ahi,'admission_p_unadjusted':_pvalue(da),
           'positive_weighted_seeds':sum(x>1e-12 for x in dw),
           'negative_weighted_seeds':sum(x<-1e-12 for x in dw),
           'zero_weighted_seeds':sum(abs(x)<=1e-12 for x in dw)})
    for item,adj in zip(comparisons,_holm([x['weighted_p_unadjusted'] for x in comparisons])):
        item['weighted_p_holm']=adj
    for item,adj in zip(comparisons,_holm([x['admission_p_unadjusted'] for x in comparisons])):
        item['admission_p_holm']=adj
    new_output(out)
    dump_csv(out/'condition_summary.csv',summary)
    dump_csv(out/'paired_summary.csv',comparisons)
    txt=['# Phase 5E: Greedy-Guided Residual Reinforcement Learning — statistical report','',
      f'- Evaluation role: **{manifest["role"]}**.',
      f'- Matched demand sequences: **{len(seeds)}**; offered requests per sequence: **{manifest["requests"]}**.',
      '- Main inferential unit: demand seed, NOT individual requests.',
      '- All comparisons share candidate generation, fixed topology, forced-only rejection and the same single-link scenario surrogate.',
      '- Nominal CIs: two-sided 95% Student-t intervals across demand seeds.',
      '- P-values: paired, two-sided t-tests; Holm family-wise adjustment across all alternatives to greedy.',
      '- Zero-width confidence intervals indicate zero *observed* sample variance, not population equivalence.',
      '- Time reflects measured selection/provisioning (not restoration or protection switching).',
      '- Previously inspected test streams were never used for the hybrid selection.',
      '- Do not modify selected checkpoints or thresholds after examining this evaluation.','',
      '## Admission results','',
      '| Algorithm | Admitted / offered | Weighted admission | Ordinary admission | Selection ms/request |','|---|---:|---:|---:|---:|']
    for r in summary:
        txt.append(f'| {r["policy"]} | {r["admitted_total"]}/{r["offered_total"]} | '
          f'{r["mean_weighted"]:.3%} [{r["weighted_ci_low"]:.3%}, {r["weighted_ci_high"]:.3%}] | '
          f'{r["mean_admission"]:.3%} | {r["mean_selection_ms"]:.2f} |')
    txt+=['','## Paired changes vs greedy','',
      '| Algorithm | Weighted difference pp (95% CI) | Holm-adjusted p | Positive / negative / zero seeds |',
      '|---|---:|---:|---:|']
    for r in comparisons:
        txt.append(f'| {r["policy"]} | {r["mean_weighted_delta_pp"]:+.3f} '
          f'[{r["weighted_ci_low_pp"]:+.3f}, {r["weighted_ci_high_pp"]:+.3f}] | '
          f'{r["weighted_p_holm"]:.4g} | {r["positive_weighted_seeds"]} / '
          f'{r["negative_weighted_seeds"]} / {r["zero_weighted_seeds"]} |')
    txt+=['','## Interpretation boundaries','',
      '- A hybrid improvement is demonstrated only if measured in its pre-specified condition against same-condition greedy, with uncertainty assessed.',
      '- Enhanced-state and candidate-count ablations must not be conflated with changes to learning architecture.',
      '- Primary 60-node, k=2 inference cannot establish results across railway topologies.',
      '- No full MPLS-TP MSRP protocol, restoration-time, safety or railway deployment claim.',
      '- Negative and null results are scientifically informative; do not select variants based on test metrics.','']
    (out/'phase5e_statistical_report.md').write_text('\n'.join(txt),encoding='utf-8')
    dump_json(out/'statistics_manifest.json',{'version':VERSION,'source_sha256':checksum(source),
        'evaluation_manifest_sha256':checksum(eval_dir/'evaluation_manifest.json'),
        'role':manifest['role'],'seeds':seeds,'comparisons':len(comparisons),
        'p_adjustment':'Holm','interval':'two-sided t, seed-level, nominal 95%',
        'protection_model':SCHEME,'protection_model_approved':False})
    return {'summary':str(out/'condition_summary.csv'),
        'paired':str(out/'paired_summary.csv'),
        'report':str(out/'phase5e_statistical_report.md')}


def plots(stats_dir:Path,out:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (stats_dir/'condition_summary.csv').open(newline='',encoding='utf-8') as f:
        data=list(csv.DictReader(f))
    if not data:raise ValueError('no statistics data')
    new_output(out)
    names=[r['policy'] for r in data]
    weighted=[100*float(r['mean_weighted']) for r in data]
    ordinary=[100*float(r['mean_admission']) for r in data]
    latency=[float(r['mean_selection_ms']) for r in data]
    files=[]
    for filename,values,title,xlabel in (
       ('weighted_admission.png',weighted,'Mean priority-weighted admission','Weighted admission (%)'),
       ('ordinary_admission.png',ordinary,'Mean ordinary admission','Admission (%)'),
       ('selection_latency.png',latency,'Mean provisioning selection time','Milliseconds / request')):
        fig,ax=plt.subplots(figsize=(10,max(4,len(names)*.44)))
        ax.barh(names,values)
        ax.set_xlabel(xlabel);ax.set_title(title);ax.invert_yaxis()
        fig.tight_layout()
        p=out/filename;fig.savefig(p,dpi=160);plt.close(fig);files.append(str(p))
    return {'figures':files}


def preflight(cfg:TopologyConfig,phase5d_manifest:Path|None=None):
    """Read-only source/architecture preflight; no experiment data generated."""
    report={'version':VERSION,'protection_model':SCHEME,
         'protection_model_approved':False,'topology':asdict(cfg),
         'current_source_sha256':hashes(),
         'python_version':platform.python_version(),
         'torch_version':torch.__version__,
         'numpy_version':np.__version__,
         'scipy_version':scipy.__version__,
         'test_seeds_generated':0}
    if phase5d_manifest is not None:
        m=json.loads(phase5d_manifest.read_text(encoding='utf-8'))
        if m.get('version')!='PHASE5D_FROZEN_POLICY_DIAGNOSTICS_V1':
            raise ValueError('expected Phase5D frozen-diagnostic manifest')
        if m.get('topology_config')!=asdict(cfg):
            raise ValueError('topology differs from Phase 5D')
        previous={**m['phase3_through_phase5_source_sha256'],
                  **m.get('additional_source_sha256',{})}
        root=Path(__file__).resolve().parents[1]
        mismatches=[]
        for name,sha in previous.items():
            p=root/name
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
                mismatches.append(name)
        if mismatches:
            raise ValueError('current project differs from Phase5D provenance: '+', '.join(mismatches))
        report['phase5d_source_provenance_confirmed']=True
        report['phase5d_manifest_sha256']=checksum(phase5d_manifest)
    else: report['phase5d_source_provenance_confirmed']='not checked (no manifest supplied)'
    return report
