"""Phase 5D diagnostics. Only current-observation features go to trained policies.

Counterfactual *analysis* deliberately replays known future requests on cloned
simulation ledgers. It is a retrospective what-if, NOT an online routing oracle,
and MUST NOT be included in deployed features or used to claim an improvement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch

from railway_tp.topology import TopologyConfig
from phase3.model import SCHEME
from phase4.policies import choose_greedy, choose_milp_v2
from phase5.environment import FEATURE_NAMES, FEATURE_VERSION, OnlineEpisode, prepare_stream
from phase5.experiment import new_output, dump_csv, dump_json, source_hashes
from phase5c.dataset import checksum
from phase5c.workflow import load_policy
from . import VERSION

DEV_SEEDS = set(range(10000, 10080)) | set(range(20000, 20010)) | set(range(21000, 21030))
PREVIOUS_TEST_SEEDS = set(range(31000, 31030)) | set(range(41000, 41030)) | set(range(51000, 51030))
FUTURE_TEST_SEEDS = set(range(61000, 61060))
REQUIRED_ALGORITHMS = ('ppo', 'bcq', 'cql', 'awac', 'iql')


def development_seed_guard(seeds, *, retrospective: bool = False):
    seq = list(seeds)
    if not seq or len(set(seq)) != len(seq):
        raise ValueError('seeds must be non-empty and unique')
    if set(seq) & FUTURE_TEST_SEEDS:
        raise ValueError('Phase 5E future held-out seeds 61000-61059 are sealed')
    if not retrospective and not set(seq) <= DEV_SEEDS:
        raise ValueError('only training/development seeds 10000-10079, 20000-20009, '
                         'or 21000-21029 are allowed without --retrospective')
    if retrospective and set(seq) - (DEV_SEEDS | PREVIOUS_TEST_SEEDS):
        raise ValueError('retrospective mode only permits previously inspected 31000/41000/51000 '
                         'test sets, plus defined development seeds')
    if retrospective and set(seq) & PREVIOUS_TEST_SEEDS:
        return 'RETROSPECTIVE_ALREADY_INSPECTED_NOT_FOR_TUNING'
    return 'DEVELOPMENT_ONLY'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(cfg: TopologyConfig, count: int, headroom: int, k: int, *, seeds, role,
              request_hashes=None, checkpoints=None):
    root=Path(__file__).resolve().parents[1]
    files=['phase5c/common.py','phase5c/offline_agents.py','phase5c/ppo.py',
           'phase5c/workflow.py','phase5b/sac.py','phase5d/analysis.py',
           'railway_tp/topology.py','railway_tp/requests.py']
    return {'version':VERSION,'analysis_role':role,'seeds':list(seeds),
            'requests_per_seed':count,'headroom_percent':headroom,
            'max_candidates':k,'topology_config':asdict(cfg),
            'protection_model':SCHEME,'protection_model_approved':False,
            'reject_policy':'FORCED_ONLY_NO_VOLUNTARY_REJECTION',
            'candidate_generation_excluded_from_inference':True,
            'request_sha256_by_seed':request_hashes or {},
            'model_sha256':checkpoints or {},
            'phase3_through_phase5_source_sha256':source_hashes(),
            'additional_source_sha256':{p:_sha(root/p) for p in files if (root/p).is_file()},
            'note':'The analysis is descriptive. No weights are trained or updated.'}


def load_selected_models(selection: Path | None, original_ddqn: Path | None,
                         discrete_sac: Path | None, cfg: TopologyConfig,
                         requests: int, headroom: int, k: int, analysis_seeds):
    """Verify selected model hashes and configuration before loading weights."""
    checkpoints={}
    if selection is not None:
        manifest=json.loads(selection.read_text(encoding='utf-8'))
        winners=manifest.get('winners',[])
        if len(winners)!=5 or {w['algorithm'] for w in winners}!=set(REQUIRED_ALGORITHMS):
            raise ValueError('expected precisely five Phase 5C selected algorithms')
        for entry in winners:
            label=entry['algorithm'];path=Path(entry['selected_checkpoint'])
            if not path.is_file():
                raise FileNotFoundError(f'{label}: checkpoint not found relative to current directory: {path}')
            if checksum(path)!=entry['checkpoint_sha256']:
                raise ValueError(f'Phase 5C checkpoint SHA-256 mismatch: {label}')
            checkpoints[label]=path
    for label, path in (('original_ddqn',original_ddqn),('discrete_sac',discrete_sac)):
        if path is not None:checkpoints[label]=path
    models={};hashes={}
    for label,path in checkpoints.items():
        model,meta=load_policy(label,path)
        expected={'feature_version':FEATURE_VERSION,'feature_names':list(FEATURE_NAMES),
                  'topology_config':asdict(cfg),'requests_per_seed':requests,
                  'headroom_percent':headroom,'max_candidates':k}
        for key,value in expected.items():
            if meta.get(key)!=value:
                raise ValueError(f'{label}: checkpoint metadata mismatch for {key}')
        tr=(set(meta['train_seeds']) if 'train_seeds' in meta else
            set(range(meta['train_seed_start'],meta['train_seed_start']+meta['train_seed_count'])))
        val=(set(meta['val_seeds']) if 'val_seeds' in meta else
             set(range(meta['val_seed_start'],meta['val_seed_start']+meta['val_seed_count'])))
        # Development-seed reuse is allowed for diagnostics, but disclose it.
        if not tr or not val:
            raise ValueError(f'{label}: missing train/validation provenance')
        for member in vars(model).values():
            if isinstance(member,torch.nn.Module):member.eval()
        models[label]=model; hashes[label]=checksum(path)
    return models,hashes


def greedy_action(obs, k: int):
    candidates=np.flatnonzero(obs.mask[:k])
    return int(candidates[0]) if len(candidates) else k


def _model_scores(label, model, obs, k):
    """Read-only scores; actor probabilities are NOT success probabilities."""
    rows=torch.as_tensor(obs.rows[None],dtype=torch.float32)
    mask=torch.as_tensor(obs.mask[None],dtype=torch.bool)
    res={'score_kind':'none','score_0':'','score_1':'',
         'policy_probability_0':'','policy_probability_1':'',
         'critic_advantage_alt_minus_greedy':'', 'bcq_support_0':'','bcq_support_1':''}
    with torch.no_grad():
        if label=='original_ddqn':
            score=model.online(rows)[0];res['score_kind']='ddqn_q'
        elif label=='discrete_sac':
            score=model.actor(rows)[0];res['score_kind']='sac_actor_logit'
            if k==2 and bool(obs.mask[:2].all()):
                q=torch.minimum(model.critic1(rows),model.critic2(rows))[0]
                res['critic_advantage_alt_minus_greedy']=float((q[1]-q[0]).item())
        elif label=='ppo':
            score=model.actor(rows)[0];res['score_kind']='ppo_logit'
        elif label in ('awac','iql'):
            score=model.policy(rows)[0];res['score_kind']=label+'_actor_logit'
        elif label in ('bcq','cql'):
            score=model.q(rows)[0];res['score_kind']=label+'_q'
            if label=='bcq':
                supported=model._supported(rows,mask)[0]
                for idx in range(min(k,2)):res[f'bcq_support_{idx}']=int(supported[idx].item())
        else:
            raise ValueError('unsupported model label '+label)
        for idx in range(min(k,2)):
            if bool(obs.mask[idx]):res[f'score_{idx}']=float(score[idx].item())
        if res['score_kind'].endswith('logit'):
            legal=score.masked_fill(~mask[0],-1e9)
            probabilities=torch.softmax(legal,dim=-1)
            for idx in range(min(k,2)):
                if bool(obs.mask[idx]):res[f'policy_probability_{idx}']=float(probabilities[idx].item())
    return res


def _run_own(cfg,prep,label,model,headroom,k,seed):
    env=OnlineEpisode(cfg,prep,headroom,k)
    rows=[]
    while not env.done:
        obs=env.observe();baseline=greedy_action(obs,k)
        n=int(obs.mask[:k].sum())
        selected=baseline if model is None else int(model.act(obs.rows,obs.mask,greedy=True))
        if not obs.mask[selected]:raise AssertionError('a frozen model chose an infeasible action')
        prior=env.sim.state_digest();pos=env.pos
        reward,_,rec=env.step(selected)
        if rec['admitted'] != int(selected!=k):raise AssertionError('invalid admission')
        rows.append({'demand_seed':seed,'policy':label,'request_index':pos,
                     'request_id':obs.request_id,'feasible_candidate_count':n,
                     'both_feasible':int(n==2),'greedy_action_same_state':baseline,
                     'chosen_action':selected,'agreed_with_greedy':int(selected==baseline),
                     'admitted':rec['admitted'],'reward':reward,
                     'candidate_id':rec['candidate_id'],'reason':rec['reason'],
                     'state_before_sha256':prior,'state_after_sha256':env.sim.state_digest()})
    return rows,env.summary()


def behavior(cfg:TopologyConfig,out:Path,models:dict[str,Any],*,requests=100,
             headroom=10,k=2,seeds=(20000,),model_hashes=None,
             retrospective=False):
    if k!=2:raise ValueError('Phase 5D behavioural choice diagnostics require max_candidates=2')
    if requests<1 or not 0<=headroom<100:raise ValueError('invalid requests/headroom')
    role=development_seed_guard(seeds,retrospective=retrospective)
    new_output(out)
    same=[];trajectory=[];summaries=[];hashes={}
    for seed in seeds:
        prep=prepare_stream(cfg,requests,seed,k);hashes[str(seed)]=prep.request_sha256
        base=OnlineEpisode(cfg,prep,headroom,k)
        while not base.done:
            obs=base.observe();default=greedy_action(obs,k);n=int(obs.mask[:k].sum())
            digest=base.sim.state_digest();pos=base.pos
            for label,model in models.items():
                selected=int(model.act(obs.rows,obs.mask,greedy=True))
                if not obs.mask[selected]:raise AssertionError(label+' selected invalid candidate')
                same.append({'demand_seed':seed,'policy':label,'request_index':pos,
                             'request_id':obs.request_id,'greedy_state_before_sha256':digest,
                             'feasible_candidate_count':n,'both_feasible':int(n==2),
                             'greedy_action':default,'chosen_action':selected,
                             'agreed_with_greedy':int(selected==default),
                             'greedy_candidate_id':(prep.candidate_sets[pos][default].candidate_id if default<k else ''),
                             'chosen_candidate_id':(prep.candidate_sets[pos][selected].candidate_id if selected<k else ''),
                             **_model_scores(label,model,obs,k)})
            base.step(default)
        base_summary=base.summary()
        baseline_rows,baseline_summary=_run_own(cfg,prep,'greedy',None,headroom,k,seed)
        if baseline_summary['state_digest']!=base_summary['state_digest']:
            raise AssertionError('greedy baseline trajectory non-reproducible')
        trajectory.extend(baseline_rows)
        summaries.append({'demand_seed':seed,'policy':'greedy',**baseline_summary,
                          'state_matches_greedy':1,'own_two_feasible_agreement':1.0,
                          'first_action_divergence_index':'','first_admission_divergence_index':''})
        for label,model in models.items():
            own,summary=_run_own(cfg,prep,label,model,headroom,k,seed)
            trajectory.extend(own)
            both=[r for r in own if r['both_feasible']]
            own_agree=(sum(r['agreed_with_greedy'] for r in both)/len(both) if both else None)
            first_action=next((i for i,(a,b) in enumerate(zip(baseline_rows,own))
                               if a['chosen_action']!=b['chosen_action']),None)
            first_outcome=next((i for i,(a,b) in enumerate(zip(baseline_rows,own))
                               if a['admitted']!=b['admitted']),None)
            summaries.append({'demand_seed':seed,'policy':label,**summary,
                              'state_matches_greedy':int(summary['state_digest']==base_summary['state_digest']),
                              'own_two_feasible_agreement':own_agree if own_agree is not None else '',
                              'first_action_divergence_index':first_action if first_action is not None else '',
                              'first_admission_divergence_index':first_outcome if first_outcome is not None else ''})
        print(f'phase5d behavior seed={seed} compared_models={len(models)}',flush=True)
    summary_rows=[]
    for label in ('greedy',*models):
        s=[r for r in summaries if r['policy']==label]
        own=[r for r in trajectory if r['policy']==label]
        common=[r for r in same if r['policy']==label and r['both_feasible']]
        informative=[r for r in own if r['both_feasible']]
        summary_rows.append({'policy':label,'seeds':len(s),'requests':len(own),
                             'own_two_feasible_states':len(informative),
                             'own_candidate1_frequency_two_feasible':
                                sum(r['chosen_action']==1 for r in informative)/len(informative) if informative else '',
                             'own_agreement_two_feasible':
                                sum(r['agreed_with_greedy'] for r in informative)/len(informative) if informative else '',
                             'same_greedy_states_two_feasible':len(common),
                             'same_state_agreement_two_feasible':
                                sum(r['agreed_with_greedy'] for r in common)/len(common) if common else '',
                             'seed_final_digest_matches_greedy':sum(r['state_matches_greedy'] for r in s),
                             'admitted_total':sum(r['admitted'] for r in s),
                             'mean_weighted_admission':statistics.mean(r['weighted_admission_ratio'] for r in s)})
    dump_csv(out/'same_state_choices.csv',same,fieldnames=list(same[0]) if same else
             ['demand_seed','policy','request_index','both_feasible'])
    dump_csv(out/'trajectory_choices.csv',trajectory)
    dump_csv(out/'per_seed_summary.csv',summaries)
    dump_csv(out/'policy_summary.csv',summary_rows)
    dump_json(out/'diagnostic_manifest.json',_metadata(cfg,requests,headroom,k,seeds=seeds,
              role=role,request_hashes=hashes,checkpoints=model_hashes))
    return {'seeds':len(seeds),'models':len(models),'same_state_comparisons':len(same),
            'trajectory_records':len(trajectory),'role':role,'outputs':str(out)}


def _branch(env: OnlineEpisode, action: int, horizon: int):
    """Cloned state, chosen branch action, then deterministic first-feasible."""
    initial=env.pos;before=env.sim.state_digest()
    rewards=[];admitted=0
    while not env.done and env.pos<initial+horizon:
        obs=env.observe();a=action if env.pos==initial else greedy_action(obs,env.k)
        r,_,rec=env.step(a);rewards.append(r);admitted+=rec['admitted']
    return {'weighted_reward':sum(rewards),'admitted':admitted,
            'state_after_sha256':env.sim.state_digest(),
            'state_before_sha256':before,'steps':len(rewards)}


def counterfactual(cfg:TopologyConfig,out:Path,*,requests=100,headroom=10,k=2,
                   seeds=(21000,),max_branches=8,horizon=100):
    if k!=2:raise ValueError('exactly two candidates needed for branch comparison')
    if min(requests,max_branches,horizon)<1:raise ValueError('positive counts required')
    role=development_seed_guard(seeds)
    new_output(out);records=[];seed_hashes={}
    for seed in seeds:
        prep=prepare_stream(cfg,requests,seed,k);seed_hashes[str(seed)]=prep.request_sha256
        greedy=OnlineEpisode(cfg,prep,headroom,k);eligible=[]; snapshots={}
        while not greedy.done:
            obs=greedy.observe()
            if int(obs.mask[:k].sum())==2:
                eligible.append(greedy.pos)
                # Snapshot the reservation simulator ONCE per decision point.
                # Replaying long prefixes separately for every sampled branch
                # would multiply expensive single-failure scenario audits.
                snapshots[greedy.pos]=copy.deepcopy(greedy.sim)
            greedy.step(greedy_action(obs,k))
        # Deterministic stratified positions over the entire decision horizon;
        # avoids selecting only early uncongested route-choice opportunities.
        if len(eligible)>max_branches:
            picks=np.linspace(0,len(eligible)-1,max_branches,dtype=int).tolist()
            positions=[eligible[i] for i in sorted(set(picks))]
        else:positions=eligible
        for index in positions:
            # Start each branch from the same saved greedy-prefix ledger. No
            # preceding requests are recomputed or changed after sampling.
            snapshot=snapshots[index]
            snapshot_digest=snapshot.state_digest()
            greedy_sim=OnlineEpisode(cfg,prep,headroom,k)
            greedy_sim.sim=copy.deepcopy(snapshot)
            greedy_sim.pos=index
            alternative_sim=OnlineEpisode(cfg,prep,headroom,k)
            alternative_sim.sim=copy.deepcopy(snapshot)
            alternative_sim.pos=index
            obs=greedy_sim.observe()
            if int(obs.mask[:k].sum())!=2:raise AssertionError('branch-state drift')
            result_g=_branch(greedy_sim,0,horizon)
            result_a=_branch(alternative_sim,1,horizon)
            if result_g['state_before_sha256']!=result_a['state_before_sha256']:
                raise AssertionError('counterfactual branches did not share initial state')
            if snapshot.state_digest()!=snapshot_digest or snapshot_digest!=result_g['state_before_sha256']:
                raise AssertionError('counterfactual mutated its stored prefix state')
            records.append({'demand_seed':seed,'request_index':index,'request_id':obs.request_id,
                'greedy_candidate_id':prep.candidate_sets[index][0].candidate_id,
                'alternative_candidate_id':prep.candidate_sets[index][1].candidate_id,
                'branch_start_sha256':result_g['state_before_sha256'],
                'horizon':horizon,'rollout_steps':result_g['steps'],
                'greedy_suffix_weighted_reward':result_g['weighted_reward'],
                'alternative_suffix_weighted_reward':result_a['weighted_reward'],
                'alternative_minus_greedy_weighted_reward':result_a['weighted_reward']-result_g['weighted_reward'],
                'greedy_suffix_admitted':result_g['admitted'],
                'alternative_suffix_admitted':result_a['admitted'],
                'alternative_minus_greedy_admitted':result_a['admitted']-result_g['admitted'],
                'greedy_branch_final_sha256':result_g['state_after_sha256'],
                'alternative_branch_final_sha256':result_a['state_after_sha256']})
        print(f'phase5d counterfactual seed={seed} eligible={len(eligible)} sampled={len(positions)}',flush=True)
    dump_csv(out/'counterfactual_branches.csv',records,fieldnames=list(records[0]) if records else
             ['demand_seed','request_index','request_id'])
    positives=sum(r['alternative_minus_greedy_weighted_reward']>1e-12 for r in records)
    zeros=sum(abs(r['alternative_minus_greedy_weighted_reward'])<=1e-12 for r in records)
    negatives=len(records)-positives-zeros
    summary={'branch_points':len(records),'positive_alternative':positives,
             'tie':zeros,'negative_alternative':negatives,
             'average_alternative_minus_greedy_reward':statistics.mean(
                 r['alternative_minus_greedy_weighted_reward'] for r in records) if records else None,
             'warning':'Realized known-future matched suffix on development streams only; '
                       'NOT an online observable advantage, not evidence of general improvement.'}
    dump_json(out/'counterfactual_summary.json',summary)
    meta=_metadata(cfg,requests,headroom,k,seeds=seeds,role=role,request_hashes=seed_hashes)
    meta.update({'horizon':horizon,'max_branches_per_seed':max_branches,
                 'future_requests_used_for_retrospective_rollout':True,
                 'never_expose_future_arrivals_to_online_policy':True,
                 'selection':'deterministic evenly spaced positions over greedy two-feasible states'})
    dump_json(out/'diagnostic_manifest.json',meta)
    return summary


def milp_trace(cfg:TopologyConfig,out:Path,*,requests=100,headroom=10,k=2,
               seed=41027,retrospective=False):
    if not retrospective:raise ValueError('MILP inspected-test trace requires explicit --retrospective')
    role=development_seed_guard([seed],retrospective=True)
    if k!=2:raise ValueError('trace assumes max_candidates=2')
    prep=prepare_stream(cfg,requests,seed,k)
    new_output(out)
    traces={};finals={}
    for name,method in (('greedy',choose_greedy),('milp_v2',choose_milp_v2)):
        env=OnlineEpisode(cfg,prep,headroom,k)
        # Important: choose_* commits to env.sim directly; must never call
        # env.step() again or the request would be provisioned twice.
        rows=[]
        while not env.done:
            pos=env.pos;obs=env.observe();req=prep.requests[pos];plans=list(prep.candidate_sets[pos])
            before=env.sim.state_digest()
            valid=[int(x) for x in np.flatnonzero(obs.mask[:k])]
            choice=method(env.sim,req,plans)
            if choice.plan is not None and not choice.result.feasible:
                raise AssertionError('selected invalid plan')
            selected=next((i for i,p in enumerate(plans) if choice.plan is not None
                           and p.candidate_id==choice.plan.candidate_id),k)
            if selected==k and valid:raise AssertionError('voluntary rejection')
            if selected!=k and selected not in valid:raise AssertionError('infeasible decision')
            # We advance exactly one request; the selected policy already committed.
            rows.append({'demand_seed':seed,'policy':name,'request_index':pos,
                'request_id':req.request_id,'priority_weight':req.priority_weight,
                'feasible_actions_before':','.join(str(x) for x in valid),
                'chosen_action':selected,
                'candidate_id':choice.plan.candidate_id if choice.plan else '',
                'admitted':int(choice.result.feasible),'reason':';'.join(choice.result.reason_codes),
                'solver_status':choice.solver_status,
                'state_before_sha256':before,
                'state_after_sha256':env.sim.state_digest()})
            env.pos+=1;env._current=None;env._current_audits=()
        traces[name]=rows
        admitted=sum(r['admitted'] for r in rows)
        if admitted != len(env.sim.accepted()):raise AssertionError('trace count mismatch')
        total_weight=sum(req.priority_weight for req in prep.requests)
        finals[name]={'admitted':admitted,'weighted_admission_ratio':sum(
            prep.requests[r['request_index']].priority_weight for r in rows if r['admitted'])/total_weight,
            'state_digest':env.sim.state_digest()}
    combined=traces['greedy']+traces['milp_v2']
    greedy, milp=traces['greedy'],traces['milp_v2']
    divergences=[]
    for a,b in zip(greedy,milp):
        if a['request_id']!=b['request_id']:raise AssertionError('unmatched request streams')
        if a['chosen_action']!=b['chosen_action'] or a['admitted']!=b['admitted']:
            divergences.append({'request_index':a['request_index'],'request_id':a['request_id'],
                 'greedy_action':a['chosen_action'],'milp_v2_action':b['chosen_action'],
                 'greedy_admitted':a['admitted'],'milp_v2_admitted':b['admitted'],
                 'greedy_before':a['state_before_sha256'],'milp_before':b['state_before_sha256'],
                 'greedy_after':a['state_after_sha256'],'milp_after':b['state_after_sha256']})
    dump_csv(out/'milp_greedy_trace.csv',combined)
    dump_csv(out/'divergence_points.csv',divergences,fieldnames=list(divergences[0]) if divergences
             else ['request_index','request_id'])
    summary={'seed':seed,'request_sha256':prep.request_sha256,
             'greedy':finals['greedy'],'milp_v2':finals['milp_v2'],
             'first_choice_divergence_index':divergences[0]['request_index'] if divergences else None,
             'first_outcome_divergence_index':next((i for i,(a,b) in enumerate(zip(greedy,milp))
                 if a['admitted']!=b['admitted']),None),
             'choice_divergence_count':len(divergences),
             'admission_count_delta_milp_minus_greedy':finals['milp_v2']['admitted']-finals['greedy']['admitted'],
             'milp_solver_status_counts':dict(Counter(r['solver_status'] for r in milp)),
             'role':role,'not_a_new_heldout_test':True}
    dump_json(out/'trace_summary.json',summary)
    dump_json(out/'diagnostic_manifest.json',_metadata(cfg,requests,headroom,k,seeds=[seed],
            role=role,request_hashes={str(seed):prep.request_sha256}))
    return summary
