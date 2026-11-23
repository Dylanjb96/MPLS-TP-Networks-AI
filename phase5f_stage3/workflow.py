"""Phase 5F Stage 3 DEVELOPMENT smoke workflow; never opens held-out seeds.

RL train episodes use stochastic candidate actions and complete remaining returns.
Policy selection and national-scale held-out inference are deliberately out of
scope for this installable Stage 3 engineering milestone.
"""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from phase5f.traffic import generate_demands,stream_digest
from phase5f.protection import ScenarioLedger
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest,choice
from phase5f_stage2.probe import environment
from . import VERSION
from .agent import CandidateActorCritic,ppo_update
from .features import encode,MAX_ACTIONS,CAND_DIM,GLOBAL_DIM

PROTECTION='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP'
TRAIN_TOPOLOGY=range(201,221)
TRAIN_STREAM=range(71000,71008)
VALIDATION_TOPOLOGY=range(301,311)
VALIDATION_STREAM=range(72000,72003)
RESERVED_HELDOUT_TOPOLOGY=(401,420)
RESERVED_HELDOUT_STREAM=(73000,73002)
LOADS=(1.6,2.5,4.0)


def _write_csv(path,rows):
    if not rows:
        raise ValueError('refuse empty evidence file '+str(path))
    keys=list(rows[0])
    if any(set(r)!=set(keys) for r in rows):
        raise ValueError('CSV schema mismatch')
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def _json(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n',encoding='utf8')


def source_hashes():
    root=Path(__file__).resolve().parent.parent
    paths=(list((root/'phase5f_stage3').glob('*.py'))+
           [root/'phase5f'/n for n in ('traffic.py','protection.py')]+
           [root/'phase5f_stage2'/n for n in ('topology.py','candidates.py')])
    return {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


@lru_cache(maxsize=20)
def make_cohort(nodes,toposeed,streamseed,load,mode,k,max_working_paths):
    """Pre-generate all protected path pairs; identical streams across policies."""
    if mode not in ('static','dynamic') or load not in LOADS or k not in (2,6):
        raise ValueError('undeclared ablation condition')
    graph=build_network(NetworkSpec(nodes=nodes,seed=toposeed))
    demands=generate_demands(graph,round(nodes*load),streamseed,mode=='dynamic')
    inv=tuple(generate_pairs(graph,d.source,d.destination,k,max_working_paths)
              for d in demands)
    return graph,demands,inv


def run_episode(graph,demands,inventory,*,policy='greedy',model=None,
                headroom=10,k=6,mode='static',gate_threshold=0.0,
                stochastic=False,trace=False,seed=7):
    """Execute one full sequential path-pair provisioning episode.

    Fully independent scenario-capacity recomputation verifies final active
    reservations. No voluntary service rejection or future demand lookahead.
    """
    if len(demands)!=len(inventory) or not demands:
        raise ValueError('bad episode inputs')
    if k not in (2,6): raise ValueError('undeclared candidate limit')
    if policy=='hybrid' and model is None:
        raise ValueError('hybrid requires a loaded neural actor')
    rng=random.Random(seed)
    ledger=ScenarioLedger(graph,headroom)
    sumweight=sum(d.priority_weight for d in demands)
    assert sumweight>0
    history=[]; decisions=[]; weights=np.zeros(len(demands),np.float64)
    offered=len(demands); accepted=0; deviations=0; discretionary=0; departures=0
    all_candidates=0; feascands=0
    for t,(d,full) in enumerate(zip(demands,inventory)):
        departures+=ledger.release_expired(d.arrival)
        plans=full[:k]
        feasible=[i for i,p in enumerate(plans) if ledger.feasible(d,p)]
        all_candidates+=len(plans);feascands+=len(feasible)
        discretionary+=int(len(feasible)>1)
        picked=None;greedy=feasible[0] if feasible else None
        prediction=None
        if feasible:
            if policy=='hybrid':
                if len(feasible)==1:
                    picked=greedy
                else:
                    obs=encode(graph,ledger,d,plans,feasible,k=k,dynamic=mode=='dynamic')
                    prediction=model.decide(obs,deterministic=not stochastic,
                                            greedy_idx=greedy,gate_threshold=gate_threshold)
                    picked=prediction['action']
                    if stochastic:
                        history.append(dict(t=t,c=obs[0],g=obs[1],mask=obs[2],
                                            action=picked,old_logp=prediction['log_prob'],
                                            greedy=greedy))
            elif policy in ('greedy','footprint','random_feasible'):
                picked=choice(policy,feasible,plans,graph,rng)
            else:
                raise ValueError('unknown policy '+policy)
            if picked not in feasible:
                raise AssertionError('model selected an infeasible candidate')
            deviations+=int(picked!=greedy)
            if not ledger.admit(d,plans[picked]):
                raise AssertionError('pre-feasible route failed admission')
            accepted+=1
            weights[t]=d.priority_weight
        if trace:
            decisions.append(dict(request_id=d.request_id,request_index=t,
                service=d.service_class,weight=d.priority_weight,
                structural_candidates=len(plans),feasible_candidates=len(feasible),
                greedy_index=-1 if greedy is None else greedy,
                chosen_index=-1 if picked is None else picked,
                greedy_candidate_id='' if greedy is None else plans[greedy].candidate_id,
                chosen_candidate_id='' if picked is None else plans[picked].candidate_id,
                overridden=int(picked is not None and picked!=greedy),
                admitted=int(picked is not None),
                active_after=len(ledger.active),
                cumulative_admitted=accepted,
                state_digest=ledger.state_digest(),
                greedy_probability='' if prediction is None else prediction['greedy_probability']))
    if not ledger.verify_full_scenario_recompute():
        raise AssertionError('independent scenario ledger recomputation failed')
    # Complete-sequence Monte Carlo remaining return. Skipped forced choices
    # remain in the reward suffix. Does not inform action selection online.
    if history:
        remaining=np.cumsum(weights[::-1])[::-1]/sumweight
        for h in history:
            h['return']=float(remaining[h['t']])
    summary=dict(policy=policy,requests=offered,admitted=accepted,
                 rejected=offered-accepted,ordinary_admission=accepted/offered,
                 weighted_admission=sum(weights)/sumweight,
                 discretionary=discretionary,overrides=deviations,
                 departures=departures,active_end=len(ledger.active),
                 structural_candidates_total=all_candidates,
                 feasible_candidates_total=feascands,
                 avg_feasible_candidates=feascands/offered,
                 peak_scenario_utilization=ledger.peak_utilization(),
                 state_digest=ledger.state_digest(),feasibility_violations=0,
                 decision_records=len(history))
    return summary,history,decisions


def _validate_seed_role(role,topologies,streams):
    tr=TRAIN_TOPOLOGY if role=='train' else VALIDATION_TOPOLOGY
    sr=TRAIN_STREAM if role=='train' else VALIDATION_STREAM
    if not topologies or not streams or any(x not in tr for x in topologies) or any(x not in sr for x in streams):
        raise ValueError(f'{role} seeds outside reserved {role} partition')


def run_smoke(out,*,nodes=60,episodes=4,train_topologies=(201,202),
              train_streams=(71000,71001),val_topologies=(301,),
              val_streams=(72000,),load=2.5,modes=('static','dynamic'),
              k=6,max_working_paths=40,headroom=10,agent_seed=7,
              learning_rate=0.0003,bc_coef=0.01,gate_threshold=0.1):
    """Runnable Stage 3 engineering smoke, not final multi-seed RL training.

    Creates only new output directories. Never accesses 401..420/73000..73002.
    """
    _validate_seed_role('train',train_topologies,train_streams)
    _validate_seed_role('validation',val_topologies,val_streams)
    if not 30<=nodes<=500 or not 1<=episodes<=20 or k not in (2,6):
        raise ValueError('invalid Stage 3 smoke scale')
    if not modes or any(m not in ('static','dynamic') for m in modes) or load not in LOADS:
        raise ValueError('undeclared traffic or load')
    if max_working_paths<k or not 0<=headroom<100 or not 0<=gate_threshold<=10:
        raise ValueError('invalid path generation, headroom or gate')
    out=Path(out)
    if out.exists():raise FileExistsError('would overwrite previous evidence '+str(out))
    torch.manual_seed(agent_seed)
    np.random.seed(agent_seed)
    rng=random.Random(agent_seed)
    # CPU only for deterministic Stage 3 smoke and cross-host comparability.
    model=CandidateActorCritic(hidden=64)
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_cohorts=[(t,s,m) for t in train_topologies for s in train_streams for m in modes]
    rng.shuffle(train_cohorts)
    hist=[];total_training_actions=0
    start=time.perf_counter()
    for ep in range(episodes):
        t,s,mode=train_cohorts[ep%len(train_cohorts)]
        g,d,inv=make_cohort(nodes,t,s,load,mode,k,max_working_paths)
        summary,observations,_=run_episode(g,d,inv,policy='hybrid',model=model,
            headroom=headroom,k=k,mode=mode,stochastic=True,seed=agent_seed+ep)
        update=ppo_update(model,optim,observations,epochs=2,batch_size=64,
                          entropy_coef=0.01,bc_coef=bc_coef,seed=agent_seed+ep)
        total_training_actions+=update['n']
        hist.append(dict(episode=ep+1,nodes=nodes,topology_seed=t,stream_seed=s,
            load=load,mode=mode,k=k,accepted=summary['admitted'],
            weighted_admission=summary['weighted_admission'],
            overrides=summary['overrides'],discretionary=summary['discretionary'],
            update_actions=update['n'],ppo_loss=update['loss']))
    validation=[];trace_rows=[];cohort_hashes={}
    model.eval()
    for t in val_topologies:
        for s in val_streams:
            for mode in modes:
                g,d,inv=make_cohort(nodes,t,s,load,mode,k,max_working_paths)
                cohort=f'{nodes}/{t}/{s}/{load}/{mode}/{k}'
                cohort_hashes[cohort]=dict(network_sha256=fingerprint(g),
                    requests_sha256=stream_digest(d),candidates_sha256=candidates_digest(inv))
                for policy in ('greedy','footprint','hybrid'):
                    result,_,decisions=run_episode(g,d,inv,policy=policy,model=model if policy=='hybrid' else None,
                        headroom=headroom,k=k,mode=mode,gate_threshold=gate_threshold,
                        stochastic=False,trace=True,seed=agent_seed)
                    validation.append(dict(nodes=nodes,topology_seed=t,stream_seed=s,
                        load=load,mode=mode,k=k,**result))
                    if policy=='hybrid':
                        for row in decisions:
                            trace_rows.append(dict(nodes=nodes,topology_seed=t,stream_seed=s,
                                                   load=load,mode=mode,policy=policy,**row))
    out.mkdir(parents=True,exist_ok=False)
    _write_csv(out/'training_history.csv',hist)
    _write_csv(out/'validation_summary.csv',validation)
    _write_csv(out/'validation_hybrid_decisions.csv',trace_rows)
    ckpt=out/'smoke_model.pt'
    torch.save(dict(state_dict=model.state_dict(),version=VERSION,
        agent_seed=agent_seed,feature_dims=dict(candidate=CAND_DIM,global_=GLOBAL_DIM),
        max_actions=MAX_ACTIONS,hidden=64,source_hashes=source_hashes(),
        train_role='DEVELOPMENT_ONLY_NOT_FINAL_MODEL'),ckpt)
    ck_hash=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    manifest=dict(version=VERSION,role='DEVELOPMENT_ONLY_SMOKE_NOT_FOR_FINAL_EVALUATION',
        created_after_stage2=True,synthetic_not_irish_rail=True,
        protection_model=PROTECTION,msrp_compliant=False,protection_model_approved=False,
        source_sha256=source_hashes(),software=environment(),
        train_topology_seeds=list(train_topologies),train_stream_seeds=list(train_streams),
        validation_topology_seeds=list(val_topologies),validation_stream_seeds=list(val_streams),
        reserved_heldout_topology_seeds=list(RESERVED_HELDOUT_TOPOLOGY),
        reserved_heldout_stream_seeds=list(RESERVED_HELDOUT_STREAM),
        final_data_accessed=False,no_future_oracle=True,
        load=load,modes=list(modes),k=k,nodes=nodes,headroom=headroom,
        max_working_paths=max_working_paths,
        agent_seed=agent_seed,episodes=episodes,learning_rate=learning_rate,
        bc_coef=bc_coef,gate_threshold=gate_threshold,
        training_actions=total_training_actions,validation_rows=len(validation),
        checkpoint_sha256=ck_hash,validation_cohorts=cohort_hashes,
        runtime_seconds=round(time.perf_counter()-start,3),
        note='Engineering smoke only. No final multi-seed training, model selection, inferential claims or held-out testing.')
    _json(out/'stage3_smoke_manifest.json',manifest)
    return manifest
