"""Reproducible train/validation and held-out comparison with Phase-4 V2.

No data-driven decision or hyperparameter selection is made from test seeds.
Baseline and DDQN methods see identical request streams per evaluation seed.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from dataclasses import asdict
import csv
import hashlib
import json
import platform
import random
from pathlib import Path
import statistics
import time

import numpy as np
from railway_tp.topology import TopologyConfig, make_topology
from phase3.model import ProtectionSimulator, SCHEME
from phase4.policies import choose, ALL_POLICIES
from . import VERSION
from .environment import (FEATURE_NAMES, FEATURE_VERSION, OnlineEpisode,
                          PreparedStream, prepare_stream)
from .agent import AgentConfig, MaskedDDQN, Transition


def source_hashes() -> dict[str,str]:
    root = Path(__file__).resolve().parents[1]
    files = ('phase3/model.py','phase4/policies.py','phase4/cli.py',
             'phase5/environment.py','phase5/agent.py',
             'phase5/experiment.py','phase5/statistics.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in files if (root/name).exists()}


def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def dump_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError('need rows or explicit fields')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError('Refusing to overwrite research output directory: ' + str(path))
    path.mkdir(parents=True)


def _do_episode(model: MaskedDDQN, cfg: TopologyConfig, prepared: PreparedStream,
                headroom: int, k: int, training: bool = False) -> tuple[dict, int]:
    env = OnlineEpisode(cfg, prepared, headroom, k)
    loss_count = 0
    while not env.done:
        state = env.observe()
        action = model.act(state.rows, state.mask, greedy=not training)
        reward, done, _ = env.step(action)
        if training:
            nxt = env.observe() if not done else None
            model.remember(Transition(state.rows.copy(), state.mask.copy(), action,
                  reward, nxt.rows.copy() if nxt else np.zeros_like(state.rows),
                  nxt.mask.copy() if nxt else np.zeros_like(state.mask), done))
            if model.learn() is not None:
                loss_count += 1
    return env.summary(), loss_count


def train(cfg: TopologyConfig, output: Path, *, requests: int = 100,
          headroom: int = 10, max_candidates: int = 2,
          train_seed_start: int = 10000, train_seed_count: int = 80,
          val_seed_start: int = 20000, val_seed_count: int = 10,
          train_episodes: int = 160, validate_every: int = 10,
          patience: int = 8, agent_config: AgentConfig | None = None) -> dict:
    if not all(v > 0 for v in (requests, max_candidates, train_seed_count,
                                 val_seed_count, train_episodes, validate_every, patience)):
        raise ValueError('positive counts required')
    train_seeds = set(range(train_seed_start, train_seed_start+train_seed_count))
    val_seeds = set(range(val_seed_start, val_seed_start+val_seed_count))
    if train_seeds & val_seeds:
        raise ValueError('train and validation demand seeds overlap')
    if train_seeds & set(range(2027, 2042)) or val_seeds & set(range(2027, 2042)):
        raise ValueError('Phase4 development seeds 2027..2041 are reserved for historical analysis')
    new_output(output)
    config = agent_config or AgentConfig()
    model = MaskedDDQN(len(FEATURE_NAMES), config)
    val_streams = {s:prepare_stream(cfg, requests, s, max_candidates) for s in sorted(val_seeds)}
    # Bound RAM: a prepared 100-request stream embeds every single-link fault path.
    train_streams: OrderedDict[int, PreparedStream] = OrderedDict()
    train_seed_hashes: dict[str,str] = {}
    best_score = (-1.0, -1.0)
    best_episode = None
    stagnant = 0
    rows: list[dict] = []
    for ep in range(1, train_episodes+1):
        seed = train_seed_start + ((ep-1) % train_seed_count)
        if seed not in train_streams:
            train_streams[seed] = prepare_stream(cfg, requests, seed, max_candidates)
            train_seed_hashes[str(seed)] = train_streams[seed].request_sha256
            if len(train_streams) > 4:
                train_streams.popitem(last=False)
        else:
            train_streams.move_to_end(seed)
        data, updates = _do_episode(model, cfg, train_streams[seed], headroom, max_candidates, training=True)
        row = {'episode':ep, 'train_demand_seed':seed,
               'train_admission_ratio':round(data['admission_ratio'],6),
               'train_weighted_admission_ratio':round(data['weighted_admission_ratio'],6),
               'train_total_reward':round(data['total_reward'],5),
               'epsilon_after_episode':round(model.epsilon(),6),
               'training_loss_last':model.last_loss if model.last_loss is not None else '',
               'updates_in_episode':updates, 'total_gradient_steps':model.gradient_steps,
               'validation_weighted_admission':'', 'validation_admission':'',
               'checkpoint_best':0}
        if ep % validate_every == 0 or ep == train_episodes:
            val = [_do_episode(model, cfg, v, headroom, max_candidates, False)[0]
                   for v in val_streams.values()]
            val_weighted = statistics.mean(v['weighted_admission_ratio'] for v in val)
            val_raw = statistics.mean(v['admission_ratio'] for v in val)
            row['validation_weighted_admission'] = round(val_weighted, 6)
            row['validation_admission'] = round(val_raw, 6)
            score = (round(val_weighted, 12), round(val_raw, 12))
            if score > best_score:
                best_score = score
                best_episode = ep
                stagnant = 0
                row['checkpoint_best'] = 1
                model.save(output/'best_model.pt', {
                    'version':VERSION, 'feature_version':FEATURE_VERSION,
                    'feature_names':list(FEATURE_NAMES), 'topology_config':asdict(cfg),
                    'requests_per_seed':requests, 'headroom_percent':headroom,
                    'max_candidates':max_candidates,
                    'train_seed_start':train_seed_start, 'train_seed_count':train_seed_count,
                    'val_seed_start':val_seed_start, 'val_seed_count':val_seed_count,
                    'best_episode':ep, 'best_validation_weighted_admission':val_weighted,
                    'best_validation_admission':val_raw,
                    'training_seed_is_agent_seed':config.seed,
                    'model':SCHEME,
                    'protection_model_approved':False,
                    'reject_policy':'FORCED_ONLY_NO_VOLUNTARY_REJECTION',
                    'objective':'discounted_sum_priority_weight_over_5',
                })
            else:
                stagnant += 1
        rows.append(row)
        print(f'episode={ep:03d} seed={seed} admitted={data["admitted"]}/{requests} '
              f'weighted={data["weighted_admission_ratio"]:.3f} eps={model.epsilon():.3f}' +
              (f' val={row["validation_weighted_admission"]}'
               if row['validation_weighted_admission'] != '' else ''),flush=True)
        if stagnant >= patience:
            print('Validation early stopping triggered.',flush=True)
            break
    model.save(output/'last_model.pt', {'version':VERSION,'feature_version':FEATURE_VERSION,
                                       'note':'Last network is NOT the selected best validation model'})
    dump_csv(output/'training_history.csv', rows)
    summary = {'model':VERSION, 'protection_model':SCHEME,
               'protection_model_approved':False,
               'cfg':asdict(cfg),'requests_per_seed':requests,
               'headroom_percent':headroom, 'max_candidates':max_candidates,
               'train_seed_start':train_seed_start,'train_seed_count':train_seed_count,
               'val_seed_start':val_seed_start,'val_seed_count':val_seed_count,
               'train_seed_hashes':train_seed_hashes,
               'val_seed_hashes':{str(s):x.request_sha256 for s,x in val_streams.items()},
               'requested_episodes':train_episodes, 'completed_episodes':len(rows),
               'best_episode':best_episode, 'best_validation_score':best_score,
               'gradient_steps':model.gradient_steps,'agent_config':asdict(config),
               'feature_names':list(FEATURE_NAMES),
               'source_sha256':source_hashes(),
               'no_test_seeds_used_for_training_or_selection':True}
    dump_json(output/'training_provenance.json',summary)
    return summary


def _run_one(cfg: TopologyConfig, prepared: PreparedStream, policy: str,
             headroom: int, k: int, model: MaskedDDQN | None, output: Path,
             save_plans: bool, milp_timeout: float, demand_seed: int) -> dict:
    new_output(output)
    if policy in ('ddqn', 'random_feasible'):
        if policy=='ddqn' and model is None:raise ValueError('DDQN checkpoint is mandatory')
        env=OnlineEpisode(cfg, prepared, headroom, k)
        decision_time = 0.0
        inference_only=0.0
        # Random-feasible is a diagnostic control, not a conventional optimized baseline.
        rng = random.Random(87123 + int(demand_seed))
        while not env.done:
            start=time.perf_counter()
            obs=env.observe()
            t1=time.perf_counter()
            if policy == 'ddqn':
                action=model.act(obs.rows,obs.mask,greedy=True)
                inference_only += time.perf_counter()-t1
            else:
                action=rng.choice(np.flatnonzero(obs.mask).tolist())
            _,_,record=env.step(action)
            elapsed=time.perf_counter()-start
            decision_time+=elapsed
            record['selection_ms']=round(elapsed*1000,6)
            record['inference_ms']='' # separately measured aggregate below
        sim=env.sim
        records=env.records
    else:
        graph,rings=make_topology(cfg)
        sim=ProtectionSimulator(graph,rings,headroom)
        records=[];decision_time=0.0;inference_only=0.0
        for req,plans in zip(prepared.requests,prepared.candidate_sets):
            t0=time.perf_counter()
            choice=choose(sim,req,list(plans),policy,milp_timeout_sec=milp_timeout)
            elapsed=time.perf_counter()-t0
            decision_time+=elapsed
            records.append({
                'request_id':req.request_id, 'source':req.source, 'destination':req.destination,
                'endpoint_category':req.endpoint_category,'service_class':req.service_class,
                'priority_weight':req.priority_weight,'bandwidth_mbps':req.bandwidth_mbps,
                'admitted':int(choice.result.feasible),
                'reward':req.priority_weight/5.0 if choice.result.feasible else 0.0,
                'action':'', 'feasible_actions':'','candidate_count':len(plans),
                'candidate_id':choice.plan.candidate_id if choice.plan else '',
                'working_hops':len(choice.plan.normal_path)-1 if choice.plan else '',
                'reason':';'.join(choice.result.reason_codes),
                'worst_utilization':choice.result.worst_utilization if choice.result.worst_utilization is not None else '',
                'solver_status':choice.solver_status,'selection_ms':round(elapsed*1000,6),
                'inference_ms':'',
            })
    accepted=sim.accepted()
    incoming=prepared.requests
    total_weight=sum(r.priority_weight for r in incoming)
    rejected_reasons=Counter(x['reason'] for x in records if not x['admitted'])
    solver_statuses=Counter(x['solver_status'] for x in records)
    summary={
        'policy':policy,'nodes':cfg.nodes,'topology_seed':cfg.seed,
        'headroom_percent':headroom,'max_candidates':k,
        'requests':len(incoming), 'admitted':len(accepted),
        'rejected':len(incoming)-len(accepted),
        'admission_ratio':round(len(accepted)/len(incoming),6),
        'weighted_admission_ratio':round(sum(r.priority_weight for r,_ in accepted)/total_weight,6),
        'total_reward':round(sum(r.priority_weight/5 for r,_ in accepted),6),
        'normal_capacity_rejections':rejected_reasons['NORMAL_CAPACITY_EXCEEDED'],
        'postfault_capacity_rejections':rejected_reasons['POSTFAULT_CAPACITY_EXCEEDED'],
        'no_plan_rejections':rejected_reasons['NO_POLICY_VALID_PROTECTED_PLAN'],
        'forced_rejections':solver_statuses['FORCED_REJECT'] if policy in ('ddqn','random_feasible') else '',
        'solver_fallbacks':sum(v for s,v in solver_statuses.items() if s.startswith('FALLBACK')),
        'solver_nonoptimal_statuses':sum(v for s,v in solver_statuses.items() if 'NONOPTIMAL' in s),
        'avg_selection_ms':round(decision_time/len(incoming)*1000,6),
        'avg_inference_only_ms':round(inference_only/len(incoming)*1000,6) if policy=='ddqn' else '',
        'selection_seconds':round(decision_time,6),
        'candidate_generation_seconds_shared':round(prepared.generation_seconds,6),
        'peak_scenario_utilization':sim.ledger_summary()['peak_utilization'],
        'state_digest':sim.state_digest(), 'request_sha256':prepared.request_sha256,
        'output_dir':str(output), 'model':SCHEME,
        'protection_model_approved':False,
    }
    dump_csv(output/'requests.csv',[asdict(r) for r in incoming])
    dump_csv(output/'decisions.csv',records)
    if save_plans:
        dump_json(output/'accepted_protection_plans.json',{
            'model':SCHEME,
            'admitted':[{'request':asdict(req),'candidate_id':p.candidate_id,
                         'working_forward':list(p.normal_path),
                         'failure_state_forward_paths':{f:list(path) for f,path in p.fault_paths.items()},
                         'reverse_rule':'exact reverse of forward path'} for req,p in accepted]})
    dump_json(output/'summary.json',{**summary,'solver_status_counts':dict(solver_statuses),
             'scenario_ledger':sim.ledger_summary()})
    file_names = (['requests.csv', 'decisions.csv', 'summary.json'] +
                  (['accepted_protection_plans.json'] if save_plans else []))
    dump_json(output/'manifest.json',{'files_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                                      for name in file_names},
                                       'policy':policy,'request_sha256':prepared.request_sha256,
                                       'model':SCHEME,'protection_model_approved':False,
                                       'comparison':'same fixed protection surrogate as Phase 3 and Phase 4'})
    return summary


def evaluate(checkpoint: Path, cfg: TopologyConfig, output: Path, *,
             requests: int = 100, headroom: int = 10, max_candidates: int = 2,
             test_seed_start: int = 31000, test_seed_count: int = 30,
             policies: tuple[str,...] = ('greedy','cspf','milp_v2','random_feasible','ddqn'),
             save_plans: bool = False, milp_timeout: float = 5.0,
             allow_shift: bool = False) -> dict:
    if test_seed_count < 1 or requests < 1 or len(policies) < 2 or len(set(policies)) != len(policies):
        raise ValueError('invalid number of seeds, requests or policies')
    if 'ddqn' not in policies or 'greedy' not in policies or any(
                  p not in {'ddqn','random_feasible',*ALL_POLICIES} for p in policies):
        raise ValueError('evaluate must include ddqn and greedy plus valid optional baselines')
    model, meta = MaskedDDQN.load(checkpoint)
    if (meta.get('version') != VERSION or meta.get('feature_version') != FEATURE_VERSION or
        meta.get('feature_names') != list(FEATURE_NAMES)):
        raise ValueError('checkpoint model / feature version mismatch')
    if (meta['max_candidates'] != max_candidates or
        (not allow_shift and (meta['topology_config'] != asdict(cfg) or
                              meta['requests_per_seed'] != requests or
                              meta['headroom_percent'] != headroom))):
        raise ValueError('evaluation configuration changed: use --allow-shift for labelled transfer test; k cannot change')
    test_seeds=set(range(test_seed_start,test_seed_start+test_seed_count))
    train_seeds=set(range(meta['train_seed_start'],meta['train_seed_start']+meta['train_seed_count']))
    val_seeds=set(range(meta['val_seed_start'],meta['val_seed_start']+meta['val_seed_count']))
    if test_seeds & (train_seeds | val_seeds):
        raise ValueError('held-out test seeds overlap training/validation seeds')
    if test_seeds & set(range(2027,2042)):
        raise ValueError('Phase4 development seeds 2027..2041 are not untouched held-out test data')
    new_output(output)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows=[]
    for seed in sorted(test_seeds):
        prepared=prepare_stream(cfg,requests,seed,max_candidates)
        for policy in policies:
            row=_run_one(cfg,prepared,policy,headroom,max_candidates,model,
                         output/'runs'/f'seed_{seed}_{policy}',save_plans,milp_timeout,seed)
            row['demand_seed']=seed
            rows.append(row)
            print(f'seed={seed} policy={policy:7s} admitted={row["admitted"]}/{requests} '
                  f'weighted={row["weighted_admission_ratio"]:.3f}', flush=True)
    dump_csv(output/'phase5_runs.csv',rows)
    matched={(r['demand_seed'],r['policy']):r for r in rows}
    paired=[]
    for seed in sorted(test_seeds):
        g=matched[seed,'greedy']
        for policy in policies:
            if policy=='greedy': continue
            p=matched[seed,policy]
            if p['request_sha256'] != g['request_sha256']:
                raise AssertionError('paired demand streams differ')
            paired.append({'demand_seed':seed, 'policy':policy, 'baseline':'greedy',
                           'delta_admitted':p['admitted']-g['admitted'],
                           'delta_admission_ratio':round(p['admission_ratio']-g['admission_ratio'],6),
                           'delta_weighted_ratio':round(p['weighted_admission_ratio']-g['weighted_admission_ratio'],6),
                           'delta_avg_selection_ms':round(p['avg_selection_ms']-g['avg_selection_ms'],6)})
    dump_csv(output/'phase5_paired_deltas.csv',paired)
    versions={'python':platform.python_version()}
    for name in ('torch','numpy','scipy','networkx'):
        try:
            import importlib.metadata
            versions[name]=importlib.metadata.version(name)
        except Exception:
            versions[name]='unknown'
    provenance={
        'phase5_version':VERSION, 'source_protection_model':SCHEME,
        'protection_model_approved':False,'checkpoint_sha256':sha,
        'checkpoint_metadata':meta,'config':asdict(cfg),
        'headroom_percent':headroom,'max_candidates':max_candidates,
        'requests_per_seed':requests,'seeds':sorted(test_seeds),
        'dataset_role':'held_out_test' if not allow_shift else 'held_out_test_shifted_configuration',
        'algorithm_versions':list(policies),'package_versions':versions,
        'source_sha256':source_hashes(),
        'request_sha256_by_seed':{str(r['demand_seed']):r['request_sha256'] for r in rows if r['policy']=='greedy'},
        'no_future_request_oracle':True,
        'ddqn_reject_action':'forced only; never reject while feasible',
        'comparison_measures':'same request hashes and same Phase3 feasibility checker',
        'timing_note':'candidate generation is precomputed identically outside every policy decision; selection includes feasibility checks, and DDQN feature construction + inference',
        'network_generalization_claim':False,
        'phase4_development_seeds_not_reused':True,
    }
    dump_json(output/'phase5_provenance.json',provenance)
    return {'test_seed_count':test_seed_count,'runs':len(rows),
            'paired_comparisons':len(paired),'output':str(output)}
