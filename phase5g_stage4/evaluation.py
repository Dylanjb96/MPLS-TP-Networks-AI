"""Locked Phase 5G six-policy evaluator; no final access without external audit release.

Every single cohort is an atomic, self-checksummed six-policy commitment. The
protected-route feasibility oracle is independent of the PPO reservation ledger.
No model ever accesses future request content when choosing a path.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import csv,gzip,hashlib,io,json,os,platform,time
import networkx as nx
import numpy as np
import torch
from phase5f.traffic import generate_demands,stream_digest
from phase5f_stage2.topology import fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage3.agent import CandidateActorCritic
from phase5f_stage3.workflow import run_episode
from phase5f_stage4.ablation import MaskedFeaturePolicy,CONFIGS
from phase5f_stage5.oracle import SingleFaultOracle,verify_trace
from phase5g.topology import build_network
from phase5g.workflow import make_cohort, simulate_gnn,simulate_minimax
from phase5g_stage2.workflow import build_model,simulate_mlp
from . import VERSION
from .protocol import (PROTOCOL_SHA256,STAGE3_FREEZE_MANIFEST_SHA256,
    POLICIES, CELLS, BY_CELL,CONDITIONS,FINAL_TOPOLOGIES,FINAL_STREAMS,
    validate_seeds,validate_protocol,semantic_lock,STRESS_SEMANTICS,canonical)

ROOT=Path(__file__).resolve().parent.parent
FREEZE=ROOT/'outputs'/'phase5g'/'stage3_FROZEN'
STAGE4_FILES=('phase5g_stage4/__init__.py','phase5g_stage4/protocol.py',
              'phase5g_stage4/evaluation.py','phase5g_stage4/statistics.py',
              'run_phase5g_stage4.py')
CHECKPOINT_SHA={
 'gnn_PPO':'7db8b92ab9bc1648a9224eea727efe035848543fe514c7d7c14be3ee584658c9',
 'mlp_PPO':'942c8edd516ad28b4012733bfd8504af01fde50f47444be7d15c8e02995e3363',
 'C_frozen':'e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9',
}
POLICY_FILES={'gnn_PPO':'gnn_PPO.pt','mlp_PPO':'mlp_PPO.pt','C_frozen':'C_frozen.pt'}
ROW_FIELDS=('cell','family','stress','k','nodes','mode','load','topology_seed',
 'stream_seed','policy','requests','admitted','rejected','weighted_admission',
 'ordinary_admission','discretionary','overrides','departures','active_end',
 'structural_candidates_total','feasible_candidates_total',
 'never_feasible_requests','exact_trajectory_peak_scenario_utilization',
 'end_of_episode_peak_scenario_utilization','policy_seconds','oracle_seconds',
 'candidate_generation_seconds','network_sha256','request_sha256',
 'candidates_sha256','oracle_sha256','decision_records','feasibility_violations')


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {rel:sha_file(ROOT/rel) for rel in STAGE4_FILES}


def freeze_lock():
    validate_protocol(ROOT)
    manifest_path=FREEZE/'freeze_manifest.json'
    if not manifest_path.is_file() or sha_file(manifest_path)!=STAGE3_FREEZE_MANIFEST_SHA256:
        raise RuntimeError('Stage3 freeze manifest bytes modified or not installed')
    m=json.loads(manifest_path.read_text(encoding='utf8'))
    if (m.get('role')!='FROZEN_PRE_HELDOUT_NO_EVALUATION' or
        m.get('protocol_sha256')!=PROTOCOL_SHA256 or
        m.get('locked_cell_count')!=8 or m.get('locked_policy_runs')!=2880 or
        m.get('locked_matched_cohorts')!=480 or
        m.get('final_topology_seeds')!=list(FINAL_TOPOLOGIES) or
        m.get('final_stream_seeds')!=list(FINAL_STREAMS) or
        m.get('policies')!=list(POLICIES)):
        raise RuntimeError('Stage3 frozen manifest contradicts locked evaluation design')
    for rel, expected in m['files_sha256'].items():
        if sha_file(FREEZE/rel)!=expected:
            raise RuntimeError('Stage3 frozen artifact changed: '+rel)
    for rel,expected in m['source_sha256'].items():
        if sha_file(ROOT/rel)!=expected:
            raise RuntimeError('Stage3 source changed after freeze: '+rel)
    # The Stage3 freeze tool itself remains immutable, not just its dependencies.
    if sha_file(ROOT/'run_phase5g_stage3.py')!=m['files_sha256']['freeze_sources/run_phase5g_stage3.py']:
        raise RuntimeError('Stage3 frozen executable changed')
    for policy,filename in POLICY_FILES.items():
        if sha_file(FREEZE/'models'/filename)!=CHECKPOINT_SHA[policy]:
            raise RuntimeError('Frozen checkpoint changed: '+policy)
    return m


def models_load():
    freeze_lock()
    torch.set_num_threads(1)
    models={}
    for policy,architecture,seed,ep,params in [('gnn_PPO','gnn',27,48,15778),('mlp_PPO','mlp',17,24,15758)]:
        ck=torch.load(FREEZE/'models'/POLICY_FILES[policy],map_location='cpu',weights_only=True)
        if (ck.get('role')!='PHASE5G_STAGE2_VALIDATION_SELECTED_NOT_HELDOUT' or
            ck.get('architecture')!=architecture or ck.get('agent_seed')!=seed or
            ck.get('episode')!=ep or ck.get('model_parameters')!=params):
            raise RuntimeError('wrong frozen neural model identity: '+policy)
        model=build_model(architecture,seed)
        model.load_state_dict(ck['model_state'],strict=True)
        if sum(x.numel() for x in model.parameters())!=params:
            raise RuntimeError('unexpected parameter count '+policy)
        model.eval();models[policy]=model
    c=torch.load(FREEZE/'models'/'C_frozen.pt',map_location='cpu',weights_only=True)
    if (c.get('config')!='C_enhanced' or c.get('agent_seed')!=27 or c.get('episode')!=60 or
        c.get('hidden')!=64 or c.get('max_actions')!=6 or c.get('train_role')!='DEVELOPMENT_TRAINING_VALIDATION_ONLY'):
        raise RuntimeError('wrong frozen C metadata')
    model=CandidateActorCritic(hidden=64)
    model.load_state_dict(c['state_dict'],strict=True)
    model.eval();models['C_frozen']=MaskedFeaturePolicy(model,CONFIGS['C_enhanced'])
    return models


def stresses(demands,stress):
    if stress not in STRESS_SEMANTICS:raise ValueError('unknown stress')
    if stress=='base':return tuple(demands)
    if stress=='bandwidth_150':
        return tuple(replace(d,bandwidth_mbps=(3*d.bandwidth_mbps+1)//2) for d in demands)
    return tuple(replace(d,expiry=d.arrival+max(1,(d.expiry-d.arrival)//2)) for d in demands)


def cohort(cell,*,nodes,topology_seed,stream_seed,role):
    """Strictly blocked final seeds except through role='final' after release gate."""
    validate_seeds(role,topology_seed,stream_seed)
    if role=='dev':
        graph,demands,inv=make_cohort(nodes=nodes,topology_seed=topology_seed,stream_seed=stream_seed,
            family=cell.family,load=4.0,mode='dynamic',stress=cell.stress,role='dev',k=cell.k)
    else:
        if nodes!=500:raise ValueError('Final heldout must be 500 nodes')
        graph=build_network(nodes,topology_seed,cell.family)
        original=generate_demands(graph,round(nodes*4.0),stream_seed,True)
        demands=stresses(original,cell.stress)
        inv=tuple(generate_pairs(graph,d.source,d.destination,cell.k,40) for d in demands)
    if len(demands)!=round(nodes*4.0) or len(inv)!=len(demands) or any(len(a)>cell.k for a in inv):
        raise AssertionError('unmatched cohort generation')
    return graph,demands,inv


def fast_oracle_trajectory(graph,demands,inventory,trace,k):
    """Independent physical-fault oracle with exact running congestion peak.

    At admit time only two rectangular subsets can *increase*: (i) working
    arcs for unaffected scenarios and (ii) backup arcs for faults on the
    working path. Therefore the running max can be updated by inspecting these
    subsets instead of scanning the entire (E+1)xE matrix per request.
    """
    from phase5f_stage5.oracle import undirected
    if not(len(demands)==len(inventory)==len(trace)):raise AssertionError('missing oracle trace')
    oracle=SingleFaultOracle(graph,10)
    admitted=0;overrides=0;departures=0;discretionary=0;maxu=0.0;nonfeasible=0
    for pos,(d,plans,r) in enumerate(zip(demands,inventory,trace)):
        if int(r['request_index'])!=pos or str(r['request_id'])!=str(d.request_id):
            raise AssertionError('trace mismatched to demand')
        departures+=oracle.release(d.arrival)
        opts=plans[:k]
        if len(opts)!=int(r['structural_candidates']):raise AssertionError('structural option mismatch')
        picked=int(r['chosen_index']);greedy=int(r['greedy_index'])
        if pos%64==0 or picked<0:
            feas=[i for i,p in enumerate(opts) if oracle.feasible(d,p)]
            if int(r['feasible_candidates'])!=len(feas) or greedy!=(feas[0] if feas else -1):
                raise AssertionError('oracle candidate mask mismatch')
        else:
            feas=None
        if picked<0:
            if int(r['admitted'])!=0 or greedy!=-1 or (feas is not None and feas):
                raise AssertionError('voluntary rejection')
            nonfeasible+=1
        else:
            if not (0<=picked<len(opts) and greedy>=0 and int(r['admitted'])==1):
                raise AssertionError('invalid route decision')
            candidate=opts[picked]
            if str(r['chosen_candidate_id'])!=candidate.candidate_id:
                raise AssertionError('candidate digest mismatch')
            oracle.admit(d,candidate)
            admitted+=1
            overrides+=int(picked!=greedy)
            # Exact maximum of all fault scenarios at all arrivals. No other
            # load elements can rise, so a subset max suffices after admit.
            wi=oracle._ids(candidate.working);bi=oracle._ids(candidate.backup)
            mask=np.ones(oracle.links+1,dtype=bool);mask[wi+1]=False
            if len(wi):
                maxu=max(maxu,float(np.max(oracle.loads[np.ix_(mask,wi)]/oracle.cap[wi][None,:])))
            if len(bi) and len(wi):
                maxu=max(maxu,float(np.max(oracle.loads[np.ix_(wi+1,bi)]/oracle.cap[bi][None,:])))
        discretionary+=int(r['feasible_candidates'])>1
        if int(r['overridden'])!=int(picked>=0 and picked!=greedy):
            raise AssertionError('override flag mismatch')
        if int(r['active_after'])!=len(oracle.active) or int(r['cumulative_admitted'])!=admitted:
            raise AssertionError('oracle active/admitted counters mismatch')
    endu=float(np.max(oracle.loads/oracle.cap[None,:]))
    if maxu+1e-10<endu:raise AssertionError('trajectory max smaller than end-state utilization')
    return dict(accepted=admitted,overrides=overrides,departures=departures,
        discretionary=discretionary,active_end=len(oracle.active),
        oracle_sha256=oracle.digest(),never_feasible_requests=nonfeasible,
        exact_trajectory_peak_scenario_utilization=maxu,
        end_of_episode_peak_scenario_utilization=endu)


def _summary(policy,g,d,inv,k,models):
    # For K2, candidate inventories have at most 2 pairs. All neural models
    # still require six-slot features; pass k=6 internally for padding ONLY.
    # Independent oracle always enforces the real K2 inventory size.
    t0=time.perf_counter()
    with torch.inference_mode():
        if policy=='gnn_PPO':
            stat,trace,_=simulate_gnn(g,d,inv,models['gnn_PPO'],k=6,mode='dynamic',trace=True)
        elif policy=='mlp_PPO':
            stat,trace,_=simulate_mlp(g,d,inv,models['mlp_PPO'],mode='dynamic',trace=True)
        elif policy=='scenario_minimax':
            stat,trace=simulate_minimax(g,d,inv,k=k,trace=True)
        else:
            sim='hybrid' if policy=='C_frozen' else policy
            stat,_,trace=run_episode(g,d,inv,policy=sim,
              model=models['C_frozen'] if policy=='C_frozen' else None,
              headroom=10,k=6 if policy=='C_frozen' else k,mode='dynamic',
              gate_threshold=0.0,stochastic=False,trace=True,seed=7)
    policy_time=time.perf_counter()-t0
    oracle_start=time.perf_counter()
    verdict=fast_oracle_trajectory(g,d,inv,trace,k)
    oracle_time=time.perf_counter()-oracle_start
    for field in ('admitted','overrides','departures','discretionary','active_end'):
        key={'admitted':'accepted'}.get(field,field)
        if int(stat[field])!=verdict[key]:raise AssertionError('policy/oracle disagree '+policy+' '+field)
    if len(trace)!=len(d):raise AssertionError('missing request-level decisions')
    if stat['feasibility_violations']!=0:raise AssertionError('policy reported invalid protection')
    # Cheap complete independent replay of the archived Stage5 oracle is also
    # available to audit development runs; not doubled here on final for cost.
    return stat,trace,verdict,policy_time,oracle_time


def evaluate_cohort(cell,*,nodes,topology_seed,stream_seed,role,models):
    start=time.perf_counter()
    g,d,inv=cohort(cell,nodes=nodes,topology_seed=topology_seed,stream_seed=stream_seed,role=role)
    generation_seconds=time.perf_counter()-start
    hashes=dict(network_sha256=fingerprint(g),request_sha256=stream_digest(d),
                candidates_sha256=candidates_digest(inv))
    results=[];traces={}
    for policy in POLICIES:
        stat,trace,oracle,psec,osec=_summary(policy,g,d,inv,cell.k,models)
        traces[policy]=trace
        results.append(dict(cell=cell.id,family=cell.family,stress=cell.stress,k=cell.k,
          nodes=nodes,mode='dynamic',load=4.0,topology_seed=topology_seed,stream_seed=stream_seed,
          policy=policy,requests=int(stat['requests']),admitted=int(stat['admitted']),
          rejected=int(stat['rejected']),weighted_admission=float(stat['weighted_admission']),
          ordinary_admission=float(stat['ordinary_admission']),discretionary=int(stat['discretionary']),
          overrides=int(stat['overrides']),departures=int(stat['departures']),
          active_end=int(stat['active_end']),
          structural_candidates_total=sum(len(plans) for plans in inv),
          feasible_candidates_total=sum(int(t['feasible_candidates']) for t in trace),
          never_feasible_requests=oracle['never_feasible_requests'],
          exact_trajectory_peak_scenario_utilization=oracle['exact_trajectory_peak_scenario_utilization'],
          end_of_episode_peak_scenario_utilization=oracle['end_of_episode_peak_scenario_utilization'],
          policy_seconds=round(psec,6),oracle_seconds=round(osec,6),
          candidate_generation_seconds=round(generation_seconds,6),
          **hashes,oracle_sha256=oracle['oracle_sha256'],decision_records=len(trace),
          feasibility_violations=0))
    if any(set(row)!=set(ROW_FIELDS) for row in results):raise AssertionError('result schema mismatch')
    return results,traces,hashes


def _hash(data):return hashlib.sha256(data).hexdigest()


def pack_payload(settings,rows,traces,hashes,cohort_key):
    body=dict(version=VERSION,locked_signature=settings['run_signature'],
      cohort_key=cohort_key,role=settings['role'],input_hashes=hashes,
      results=rows,traces=traces)
    core=canonical(body)
    return gzip.compress(canonical(dict(body=body,body_sha256=_hash(core))),mtime=0)


def inspect_payload(content,settings,key):
    doc=json.loads(gzip.decompress(content))
    b=doc['body']
    if (_hash(canonical(b))!=doc['body_sha256'] or b['version']!=VERSION or
        b['locked_signature']!=settings['run_signature'] or b['cohort_key']!=key or
        b['role']!=settings['role']):
        raise RuntimeError('cohort content or provenance mismatch: '+key)
    if (len(b['results'])!=6 or set(b['traces'])!=set(POLICIES) or
        [r['policy'] for r in b['results']]!=list(POLICIES)):
        raise RuntimeError('incomplete six-policy cohort: '+key)
    expected=key.split('__');cell=BY_CELL[expected[0]]
    for r in b['results']:
        if (r['cell']!=cell.id or int(r['topology_seed'])!=int(expected[1]) or
            int(r['stream_seed'])!=int(expected[2]) or int(r['k'])!=cell.k or
            r['family']!=cell.family or r['stress']!=cell.stress or
            int(r['decision_records'])!=int(r['requests']) or
            len(b['traces'][r['policy']])!=int(r['requests']) or
            any(r[name]!=b['input_hashes'][name] for name in b['input_hashes']) or
            int(r['feasibility_violations'])!=0 or
            int(r['admitted'])+int(r['rejected'])!=int(r['requests'])):
            raise RuntimeError('invalid cohort row or trace: '+key+' '+r['policy'])
    return b


def _software():
    return dict(python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,networkx=nx.__version__,platform=platform.platform())


def settings_for(*,role,nodes,cell_ids,topologies,streams):
    if role not in ('dev','final'):raise ValueError('unknown evaluation role')
    if role=='final' and (nodes!=500 or tuple(cell_ids)!=tuple(c.id for c in CELLS) or
                          tuple(topologies)!=FINAL_TOPOLOGIES or tuple(streams)!=FINAL_STREAMS):
        raise ValueError('final evaluation grid must exactly match frozen design')
    if not cell_ids or len(set(cell_ids))!=len(cell_ids) or any(c not in BY_CELL for c in cell_ids):
        raise ValueError('bad experimental cell list')
    if not topologies or not streams or len(set(topologies))!=len(topologies) or len(set(streams))!=len(streams):
        raise ValueError('duplicate or empty cohort seeds')
    if role=='dev' and nodes not in (60,150,250,500):raise ValueError('wrong dev-only node count')
    for t in topologies:
        for s in streams:validate_seeds(role,t,s)
    return dict(version=VERSION,role=role,nodes=nodes,cell_ids=list(cell_ids),
        topology_seeds=list(topologies),stream_seeds=list(streams),
        software=_software(),stage3_freeze_sha256=STAGE3_FREEZE_MANIFEST_SHA256,
        original_source_sha256=freeze_lock()['source_sha256'],
        stage4_source_sha256=source_hashes(),
        semantic_lock_sha256=semantic_lock(),
        checkpoints_sha256=CHECKPOINT_SHA,protocol_sha256=PROTOCOL_SHA256)


def _seal(settings):
    return dict(settings,run_signature=_hash(canonical(settings)))


def _cohort_key(cell,t,s):return f'{cell}__{t}__{s}'


def run_batches(out,*,role,nodes,cell_ids,topologies,streams,resume=False):
    frozen=freeze_lock()
    selected=_seal(settings_for(role=role,nodes=nodes,cell_ids=cell_ids,topologies=topologies,streams=streams))
    out=Path(out)
    if out.exists():
        if not resume:raise FileExistsError('Refusing to overwrite previous evidence; use --resume if intact')
        saved=json.loads((out/'run_manifest.json').read_text(encoding='utf8'))
        if saved!=selected:raise RuntimeError('Run software/source/experiment lock mismatch; resumption denied')
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/'run_manifest.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n',encoding='utf8')
    cohortdir=out/'cohorts';cohortdir.mkdir(exist_ok=True)
    models=models_load()  # All frozen checkpoint verification before any new topology.
    expected=[_cohort_key(c,t,s) for c in cell_ids for t in topologies for s in streams]
    actual={p.name[:-8] for p in cohortdir.glob('*.json.gz')}
    if not actual.issubset(set(expected)):
        raise RuntimeError('unexpected cohort output from another experiment; stop')
    for ix,key in enumerate(expected,1):
        f=cohortdir/(key+'.json.gz')
        if f.exists():
            inspect_payload(f.read_bytes(),selected,key)
            print('PHASE5G_STAGE4_COHORT_EXISTING_OK '+key,flush=True)
            continue
        cellid,top,stream=key.split('__');cell=BY_CELL[cellid]
        rows,traces,hashes=evaluate_cohort(cell,nodes=nodes,topology_seed=int(top),
                                           stream_seed=int(stream),role=role,models=models)
        content=pack_payload(selected,rows,traces,hashes,key)
        inspect_payload(content,selected,key)
        temp=f.with_suffix(f.suffix+'.tmp')
        with temp.open('wb') as fh:
            fh.write(content);fh.flush();os.fsync(fh.fileno())
        os.replace(temp,f)
        print(f'PHASE5G_STAGE4_COHORT_OK {ix}/{len(expected)} {key} policies=6',flush=True)
    result=summary_rows(out,require_complete=True)
    print(f'PHASE5G_STAGE4_{role.upper()}_COMPLETE cohorts={len(expected)} rows={len(result)} '
          f'role={role} final_data_accessed={str(role=="final").lower()}')
    return result


def summary_rows(out,require_complete=True):
    folder=Path(out);settings=json.loads((folder/'run_manifest.json').read_text(encoding='utf8'))
    if settings!=_seal({k:v for k,v in settings.items() if k!='run_signature'}):
        raise RuntimeError('run manifest damaged')
    expected=[_cohort_key(c,t,s) for c in settings['cell_ids'] for t in settings['topology_seeds'] for s in settings['stream_seeds']]
    actual={p.name[:-8] for p in (folder/'cohorts').glob('*.json.gz')}
    if not actual.issubset(set(expected)) or (require_complete and set(expected)!=actual):
        raise ValueError('missing or extraneous cohort artifacts')
    rows=[]
    for key in expected:
        f=folder/'cohorts'/(key+'.json.gz')
        if not f.exists():continue
        core=inspect_payload(f.read_bytes(),settings,key)
        rows.extend(core['results'])
    return rows


def write_dev_summary(out):
    root=Path(out);rows=summary_rows(root)
    if json.loads((root/'run_manifest.json').read_text())['role']!='dev':
        raise ValueError('dev reporter refuses final outputs')
    file=root/'development_summary.csv'
    if file.exists():raise FileExistsError('Refusing to overwrite development summary')
    with file.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=ROW_FIELDS);w.writeheader();w.writerows(rows)
    return rows



def replay_selected_validation(out,*,topology_seed=701,stream_seed=76000):
    """Independent Stage-2 validation trace regression; no Stage-4 final seeds.

    Compare complete 2,000-action records against Stage 3 frozen validation
    traces, including exact ledger state digests, for both selected new agents.
    """
    from phase5g.protocol import check_seed_role
    check_seed_role('validation',topology_seed,stream_seed)
    if int(topology_seed) not in (701,702,703) or int(stream_seed) not in (76000,76001):
        raise ValueError('validation-replay restricted to Stage2 selected cohorts')
    freeze_lock()
    models=models_load()
    g,d,inv=make_cohort(nodes=500,topology_seed=topology_seed,
        stream_seed=stream_seed,family='hierarchical',load=4.0,
        mode='dynamic',stress='base',role='validation',k=6)
    report=[]
    for policy in ('gnn_PPO','mlp_PPO'):
        stats,trace,oracle,psec,osec=_summary(policy,g,d,inv,6,models)
        arc=FREEZE/'traces'/((('gnn' if policy=='gnn_PPO' else 'mlp'))+'_selected_validation_trace.csv.gz')
        with gzip.open(arc,'rt',encoding='utf8',newline='') as f:
            allrows=list(csv.DictReader(f))
        expected=[r for r in allrows if int(r['topology_seed'])==topology_seed and
                  int(r['stream_seed'])==stream_seed]
        if len(expected)!=2000 or len(trace)!=2000:
            raise AssertionError('selected validation trace not complete')
        matched=('request_id','request_index','chosen_index','admitted',
                 'greedy_index','feasible_candidates','structural_candidates',
                 'overridden','state_digest','cumulative_admitted')
        for i,(current,prior) in enumerate(zip(trace,expected)):
            for field in matched:
                if str(current[field])!=str(prior[field]):
                    raise AssertionError(f'validation replay differs policy={policy} request={i} field={field}')
        report.append(dict(policy=policy,topology_seed=topology_seed,
             stream_seed=stream_seed,requests=len(d),admitted=stats['admitted'],
             weighted_admission=stats['weighted_admission'],
             oracle_sha256=oracle['oracle_sha256'],validation_trace_reproduced=True,
             source_sha256=source_hashes()))
    out=Path(out)
    if out.exists():raise FileExistsError('Refusing to overwrite validation-replay evidence')
    out.mkdir(parents=True,exist_ok=False)
    (out/'validation_replay_manifest.json').write_text(
        json.dumps(dict(role='PHASE5G_STAGE4_STAGE2_VALIDATION_REPLAY_ONLY',
                 selected_models=report,final_seeds_used=False),
                 sort_keys=True,indent=2)+'\n',encoding='utf8')
    print('PHASE5G_STAGE4_VALIDATION_REPLAY_OK policies=2 requests=2000 '
          'matched_full_decisions=True final_seeds_accessed=False')
    return report


def release_check(release_path,*,role):
    if role!='final':raise ValueError('audit release is only for final')
    source=source_hashes()
    p=Path(release_path)
    if not p.is_file():
        raise RuntimeError('STAGE4_FINAL_LOCKED: independent implementation audit release missing; final data forbidden')
    rel=json.loads(p.read_text(encoding='utf8'))
    expected={'role':'PHASE5G_STAGE4_INDEPENDENT_IMPLEMENTATION_AUDIT_RELEASE',
        'stage3_freeze_manifest_sha256':STAGE3_FREEZE_MANIFEST_SHA256,
        'stage4_source_sha256':source,'protocol_sha256':PROTOCOL_SHA256,
        'semantic_lock_sha256':semantic_lock(),
        'final_topologies':list(FINAL_TOPOLOGIES),'final_streams':list(FINAL_STREAMS),
        'passed_development_audit':True}
    if rel!=expected:
        raise RuntimeError('STAGE4_FINAL_LOCKED: release invalid or code/protocol changed')
    # A release is an explicit reproducibility/audit declaration, not a
    # cryptographic signature against the local user deliberately editing it.
    return rel
