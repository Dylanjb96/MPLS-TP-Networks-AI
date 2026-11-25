"""Development-only paired comparison and true message-passing GNN PPO smoke.

No Phase 5G final data execution method exists. No Phase 5F Stage 5 seeds
are accepted. All new files live in separate Phase 5G directories.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import csv,hashlib,json,random,time
import numpy as np
import torch
from phase5f.traffic import generate_demands,stream_digest
from phase5f.protection import ScenarioLedger
from phase5f_stage2.topology import fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage3.workflow import run_episode
from phase5f_stage5.oracle import verify_trace
from phase5f_stage5.freeze import verify_freeze,load_models
from . import VERSION
from .protocol import check_seed_role
from .topology import build_network
from .gnn import graph_observation,GraphCandidateActorCritic,ppo_update
from .optimization import scenario_minimax_choice


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()


def _files_sha():
    root=Path(__file__).resolve().parent.parent
    rel=[*sorted(str(x.relative_to(root)).replace('\\','/') for x in (root/'phase5g').glob('*.py')),
         'run_phase5g.py']
    return {p:sha_file(root/p) for p in rel}


def _create(out):
    out=Path(out)
    if out.exists():raise FileExistsError('Refusing to overwrite experiment evidence: '+str(out))
    out.mkdir(parents=True)
    return out


def _csv(path,rows):
    if not rows:raise ValueError('cannot save empty evidence')
    keys=list(rows[0])
    if any(set(x)!=set(keys) for x in rows):raise AssertionError('CSV incompatible records')
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def _json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf8')


def make_cohort(*,nodes,topology_seed,stream_seed,family='hierarchical',load=2.5,
                mode='dynamic',stress='base',role='dev',k=6):
    check_seed_role(role,topology_seed,stream_seed)
    if nodes not in (60,150,250,500) or k not in (2,6):
        raise ValueError('undeclared topology size or candidate count')
    if mode not in ('static','dynamic') or load not in (1.6,2.5,4.0):
        raise ValueError('undeclared mode or load')
    if stress not in ('base','bandwidth_150','short_hold'):
        raise ValueError('undeclared stress mode')
    if stress=='short_hold' and mode!='dynamic':
        raise ValueError('short_hold applies to dynamic traffic only')
    g=build_network(nodes,topology_seed,family)
    demands=generate_demands(g,round(nodes*load),stream_seed,mode=='dynamic')
    if stress=='bandwidth_150':
        demands=tuple(replace(d,bandwidth_mbps=(3*d.bandwidth_mbps+1)//2) for d in demands)
    if stress=='short_hold':
        demands=tuple(replace(d,expiry=d.arrival+max(1,(d.expiry-d.arrival)//2)) for d in demands)
    inventory=tuple(generate_pairs(g,d.source,d.destination,k,40) for d in demands)
    return g,demands,inventory


def _decision_record(d,t,plans,feasible,chosen,ledger,accepted):
    greedy=feasible[0] if feasible else None
    return dict(request_id=d.request_id,request_index=t,
        service=d.service_class,weight=d.priority_weight,
        structural_candidates=len(plans),feasible_candidates=len(feasible),
        greedy_index=-1 if greedy is None else greedy,
        chosen_index=-1 if chosen is None else chosen,
        greedy_candidate_id='' if greedy is None else plans[greedy].candidate_id,
        chosen_candidate_id='' if chosen is None else plans[chosen].candidate_id,
        overridden=int(chosen is not None and chosen!=greedy),
        admitted=int(chosen is not None),active_after=len(ledger.active),
        cumulative_admitted=accepted,state_digest=ledger.state_digest(),
        greedy_probability='')


def simulate_minimax(g,demands,inventory,*,k=6,trace=True):
    led=ScenarioLedger(g,10)
    accepted=0;weighted=0;departures=0;discretionary=0;overrides=0;valid=[]
    for t,(d,full) in enumerate(zip(demands,inventory)):
        departures+=led.release_expired(d.arrival)
        plans=full[:k]
        feasible=[i for i,p in enumerate(plans) if led.feasible(d,p)]
        discretionary+=len(feasible)>1
        pick=scenario_minimax_choice(led,d,plans,feasible) if feasible else None
        if pick is not None:
            if not led.admit(d,plans[pick]):raise AssertionError('minimax candidate infeasible')
            accepted+=1;weighted+=d.priority_weight
            overrides+=pick!=feasible[0]
        if trace:valid.append(_decision_record(d,t,plans,feasible,pick,led,accepted))
    if not led.verify_full_scenario_recompute():raise AssertionError('scenario loads incorrect')
    return dict(requests=len(demands),admitted=accepted,rejected=len(demands)-accepted,
        weighted_admission=weighted/sum(x.priority_weight for x in demands),
        ordinary_admission=accepted/len(demands),discretionary=discretionary,
        overrides=overrides,departures=departures,active_end=len(led.active),
        peak_scenario_utilization=led.peak_utilization(),feasibility_violations=0),valid


def simulate_gnn(g,demands,inventory,model,*,k=6,stochastic=False,seed=7,trace=True):
    led=ScenarioLedger(g,10)
    accepted=0;weighted=0;departures=0;discretionary=0;overrides=0;records=[];samples=[]
    weights=np.zeros(len(demands),dtype=np.float64)
    for t,(d,full) in enumerate(zip(demands,inventory)):
        departures+=led.release_expired(d.arrival)
        plans=full[:k]
        feasible=[i for i,p in enumerate(plans) if led.feasible(d,p)]
        discretionary+=len(feasible)>1
        pick=feasible[0] if feasible else None
        if len(feasible)>1:
            obs=graph_observation(g,led,d,plans,feasible,k=k,dynamic=d.expiry<=len(demands))
            pred=model.decide(obs,deterministic=not stochastic,greedy_idx=feasible[0])
            pick=pred['action']
            if stochastic:
                samples.append(dict(t=t,observation=obs,action=pick,
                                    greedy=feasible[0],old_logp=pred['log_prob']))
        if pick is not None:
            if pick not in feasible or not led.admit(d,plans[pick]):
                raise AssertionError('GNN selected an infeasible candidate')
            accepted+=1;weighted+=d.priority_weight;weights[t]=d.priority_weight
            overrides+=pick!=feasible[0]
        if trace:records.append(_decision_record(d,t,plans,feasible,pick,led,accepted))
    if not led.verify_full_scenario_recompute():raise AssertionError('GNN scenario loads incorrect')
    remaining=np.cumsum(weights[::-1])[::-1]/sum(d.priority_weight for d in demands)
    for sample in samples:sample['return']=float(remaining[sample['t']])
    return dict(requests=len(demands),admitted=accepted,rejected=len(demands)-accepted,
        weighted_admission=weighted/sum(d.priority_weight for d in demands),
        ordinary_admission=accepted/len(demands),discretionary=discretionary,
        overrides=overrides,departures=departures,active_end=len(led.active),
        peak_scenario_utilization=led.peak_utilization(),feasibility_violations=0),records,samples


def paired_dev_pilot(out,*,sizes=(60,),families=('hierarchical','ladder'),
                     topology_seeds=(501,502),stream_seeds=(74000,),
                     mode='dynamic',load=2.5,stress='base',k=6,
                     policies=('greedy','footprint','scenario_minimax','C_frozen')):
    for t in topology_seeds:
        for s in stream_seeds:check_seed_role('dev',t,s)
    if len(set(families))!=len(families) or any(f not in ('hierarchical','ladder') for f in families):
        raise ValueError('invalid topology families')
    if any(x not in ('greedy','footprint','scenario_minimax','C_frozen') for x in policies):
        raise ValueError('unknown policy')
    selection=verify_freeze()
    models=load_models() if 'C_frozen' in policies else {}
    rows=[];cohort_map={};start=time.perf_counter()
    for n in sizes:
      for family in families:
       for t in topology_seeds:
        for s in stream_seeds:
            g,d,inv=make_cohort(nodes=n,family=family,topology_seed=t,stream_seed=s,
                                load=load,mode=mode,stress=stress,k=k)
            key=f'{n}/{family}/{t}/{s}/{load}/{mode}/{stress}/{k}'
            cohort_map[key]=dict(network_sha256=fingerprint(g),requests_sha256=stream_digest(d),
                                 candidates_sha256=candidates_digest(inv),count=len(d),edges=g.number_of_edges())
            for p in policies:
                begin=time.perf_counter()
                if p=='scenario_minimax':
                    summary,trace=simulate_minimax(g,d,inv,k=k)
                else:
                    original_policy='hybrid' if p=='C_frozen' else p
                    summary,_,trace=run_episode(g,d,inv,policy=original_policy,
                        model=models.get('C_enhanced') if p=='C_frozen' else None,
                        headroom=10,k=k,mode=mode,gate_threshold=0.0,
                        stochastic=False,trace=True,seed=s)
                verdict=verify_trace(g,d,inv,trace,k,headroom=10)
                if verdict['accepted']!=summary['admitted'] or verdict['overrides']!=summary['overrides']:
                    raise AssertionError('independent oracle disagreed')
                rows.append(dict(nodes=n,family=family,topology_seed=t,stream_seed=s,
                   load=load,mode=mode,stress=stress,k=k,policy=p,
                   requests=summary['requests'],admitted=summary['admitted'],
                   rejected=summary['rejected'],weighted_admission=summary['weighted_admission'],
                   discretionary=summary['discretionary'],overrides=summary['overrides'],
                   departures=summary['departures'],peak_scenario_utilization=summary['peak_scenario_utilization'],
                   scenario_oracle_sha256=verdict['oracle_state_sha256'],
                   network_sha256=fingerprint(g),request_sha256=stream_digest(d),
                   candidates_sha256=candidates_digest(inv),
                   elapsed_seconds=round(time.perf_counter()-begin,6),
                   feasibility_violations=0,decision_records=len(trace)))
    out=_create(out)
    _csv(out/'phase5g_dev_pilot.csv',rows)
    _json(out/'phase5g_dev_manifest.json',dict(version=VERSION,
         role='PHASE5G_DEVELOPMENT_ONLY_NOT_CONFIRMATORY',
         prior_phase5f_heldout_results_not_accessed_by_code=True,
         forbidden_stage5_seeds_used=False,
         selected_C_sha256=selection['selected']['checkpoint_sha256'],
         source_sha256=_files_sha(),topology_families=list(families),
         topology_seeds=list(topology_seeds),stream_seeds=list(stream_seeds),
         settings=dict(sizes=list(sizes),load=load,mode=mode,k=k,stress=stress,
                       policies=list(policies)),cohorts=cohort_map,policy_runs=len(rows),
         elapsed_seconds=round(time.perf_counter()-start,3),
         msrp_compliant=False,real_irish_rail_topology=False))
    return rows


def gnn_smoke(out,*,episodes=2,agent_seed=7,load=1.6,mode='dynamic'):
    if not 1<=episodes<=8 or load not in (1.6,2.5,4.0) or mode not in ('static','dynamic'):
        raise ValueError('invalid GNN engineering-smoke settings')
    torch.set_num_threads(1);torch.manual_seed(agent_seed)
    model=GraphCandidateActorCritic(hidden=32)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    history=[];total=0
    begin=time.perf_counter()
    for ep in range(episodes):
        topo=601+(ep%2);stream=75000+(ep%2)
        g,d,inv=make_cohort(nodes=60,topology_seed=topo,stream_seed=stream,
                            family='hierarchical',load=load,mode=mode,role='train')
        summary,_,observations=simulate_gnn(g,d,inv,model,stochastic=True,seed=agent_seed+ep)
        update=ppo_update(model,opt,observations,epochs=2,seed=agent_seed+ep)
        total+=update['samples']
        history.append(dict(episode=ep+1,topology_seed=topo,stream_seed=stream,
            admitted=summary['admitted'],weighted_admission=summary['weighted_admission'],
            discretionary=summary['discretionary'],overrides=summary['overrides'],
            sampled_actions=update['samples'],loss=update['loss']))
    g,d,inv=make_cohort(nodes=60,topology_seed=701,stream_seed=76000,
                        family='hierarchical',load=load,mode=mode,role='validation')
    validation=[]
    for p in ('greedy','footprint','scenario_minimax','GNN_PPO'):
        if p=='scenario_minimax':summary,trace=simulate_minimax(g,d,inv)
        elif p=='GNN_PPO':summary,trace,_=simulate_gnn(g,d,inv,model)
        else:summary,_,trace=run_episode(g,d,inv,policy=p,headroom=10,k=6,
                                            mode=mode,trace=True,seed=76000)
        verdict=verify_trace(g,d,inv,trace,6,headroom=10)
        if verdict['accepted']!=summary['admitted']:
            raise AssertionError('GNN smoke oracle mismatch')
        validation.append(dict(policy=p,requests=summary['requests'],
            admitted=summary['admitted'],weighted_admission=summary['weighted_admission'],
            discretionary=summary['discretionary'],overrides=summary['overrides'],
            decision_records=len(trace),oracle_sha256=verdict['oracle_state_sha256']))
    out=_create(out)
    _csv(out/'gnn_training_history.csv',history)
    _csv(out/'gnn_validation.csv',validation)
    checkpoint=out/'gnn_engineering_smoke.pt'
    torch.save(dict(role='ENGINEERING_SMOKE_NOT_FINAL',state_dict=model.state_dict(),
        agent_seed=agent_seed,episodes=episodes,hidden=32,version=VERSION,
        train_seeds='G_TRAIN',validation_seeds='G_VALIDATION'),checkpoint)
    _json(out/'gnn_smoke_manifest.json',dict(version=VERSION,role='DEVELOPMENT_ONLY_NO_FINAL_SEEDS',
       train_topologies=[601,602],train_streams=[75000,75001],
       val_topologies=[701],val_streams=[76000],
       candidate_limit=6,load=load,mode=mode,episodes=episodes,actions=total,
       checkpoint_sha256=sha_file(checkpoint),source_sha256=_files_sha(),
       elapsed_seconds=round(time.perf_counter()-begin,3),
       no_future_request_lookahead=True,msrp_compliant=False))
    return history,validation


def preflight():
    selection=verify_freeze()
    for family in ('hierarchical','ladder'):
        g,d,inv=make_cohort(nodes=60,topology_seed=501,stream_seed=74000,
                            family=family,load=1.6,mode='dynamic')
        if len(inv[0])<2:raise AssertionError('missing protected candidate choices')
        ledger=ScenarioLedger(g,10)
        # Find a request with at least two feasible alternatives.
        found=False
        for demand,plans in zip(d,inv):
            feasible=[i for i,p in enumerate(plans) if ledger.feasible(demand,p)]
            if len(feasible)>1:
                x=scenario_minimax_choice(ledger,demand,plans,feasible)
                ob=graph_observation(g,ledger,demand,plans,feasible)
                model=GraphCandidateActorCritic(hidden=32)
                pick=model.decide(ob,greedy_idx=feasible[0])['action']
                if x not in feasible or pick not in feasible:
                    raise AssertionError('preflight invalid candidate')
                found=True;break
        if not found:raise AssertionError('preflight no discretionary candidates')
        print(f'PHASE5G_PREFLIGHT_FAMILY_OK family={family} nodes=60 edges={g.number_of_edges()} '
              f'feasible={len(feasible)} graph_sha256={fingerprint(g)}')
    print('PHASE5G_PREFLIGHT_OK frozen_C_sha256='+selection['selected']['checkpoint_sha256'])
