#Preregistered once-only Stage 5 held-out execution; never trains or selects.

#All configuration/seed choices are immutable; there is no --seed/--load/--model
#option on the final command. Each entire matched cohort is atomically saved,
#so a machine interruption can be resumed without rerunning completed cohorts.

from __future__ import annotations
from pathlib import Path
import csv,gzip,hashlib,json,os,platform,time
import numpy as np
import torch
from phase5f.traffic import generate_demands,stream_digest
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage2.probe import environment
from phase5f_stage3.workflow import run_episode
from . import VERSION
from .freeze import FROZEN,sha256,verify_freeze,load_models
from .oracle import verify_trace
from .statistics import CELLS,PRIMARY,TOPOLOGIES,STREAMS,POLICIES,analyze,validate_full_grid

FINAL_ROLE='HELDOUT_EVALUATION_ONCE_ONLY_NO_TRAINING'


def hash_json(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _stage5_source_hashes():
    root=Path(__file__).resolve().parent.parent
    files=list(sorted((root/'phase5f_stage5').glob('*.py')))+[root/'run_phase5f_stage5.py']
    return {str(f.relative_to(root)).replace('\\','/'):sha256(f) for f in files}


def _json_write_atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n',encoding='utf8')
    os.replace(tmp,path)


def _gzip_write_atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=5) as f:
        json.dump(data,f,sort_keys=True,separators=(',',':'))
    os.replace(tmp,path)


def _gzip_read(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)


def _csv_atomic(path,rows):
    if not rows:raise ValueError('refusing empty evidence table')
    keys=list(rows[0]);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if any(set(row)!=set(keys) for row in rows):raise ValueError('inconsistent CSV row keys')
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def immutable_plan():
    selection=verify_freeze()
    return dict(version=VERSION,role=FINAL_ROLE,
        protocol='PHASE5F_STAGE5_PREREGISTERED_HELDOUT_V1',
        status='LOCKED_BEFORE_ANY_FINAL_TOPOLOGY_REQUEST_GENERATION',
        topology_seeds=list(TOPOLOGIES),stream_seeds=list(STREAMS),
        cells=[dict(nodes=n,k=k,traffic=mode,load=load) for n,k,mode,load in CELLS],
        primary_cell=dict(nodes=500,k=6,traffic='dynamic',load=4.0),
        policies=list(POLICIES),headroom_percent=10,max_working_paths=40,
        selection_manifest_sha256=sha256(FROZEN/'selection_manifest.json'),
        selected_C_sha256=selection['selected']['checkpoint_sha256'],
        nominated_D_sha256=selection['proposed_full_hybrid']['checkpoint_sha256'],
        baseline_rule='first feasible greedy; min footprint; seeded uniform feasible',
        deterministic_neural_inference=True,
        guidance_gate_C=0.0,guidance_gate_D=0.1,
        checkpoint_training_permitted=False,validation_based_selection_permitted=False,
        topology_cluster_count=len(TOPOLOGIES),streams_per_cluster=len(STREAMS),
        primary_contrast='C_enhanced - greedy mean topology-clustered paired weighted admission (two-sided t 95% CI)',
        secondary_family=['C_enhanced - footprint','D_full_hybrid - greedy','C_enhanced - D_full_hybrid'],
        secondary_multiplicity='Holm three contrasts',
        factorial='2 sizes x 2 candidate limits x 2 traffic modes at load 4.0; seven effects Holm adjusted',
        cluster_bootstrap_reps=10000,cluster_bootstrap_seed=5192026,
        stage4_source_sha256=selection['source_sha256'],
        stage5_source_sha256=_stage5_source_hashes(),
        frozen_runtime_environment=environment(),
        protection_model='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP',
        msrp_compliant=False,protection_model_approved=False,
        actual_irish_rail_network=False,simulator_not_railway_restoration_time=True)


def _cohort_filename(n,k,mode,t,s):
    return f'n{n}_k{k}_{mode}_t{t}_s{s}.json.gz'


def _evaluate_cell(graph,demands,inventory,*,n,k,mode,topo,stream,
                   candidates_sha,gen_seconds,models,keep_trace=False):
    #Evaluate *every* frozen policy against the exact same precomputed graph,
    #request stream and candidate prefix. The separate numerical oracle also
    #checks all admitted choices and forced rejections across time.
    
    gsha=fingerprint(graph);rsha=stream_digest(demands)
    rows=[];traces={};oracles={}
    for policy in POLICIES:
        policy_type='hybrid' if policy in models else policy
        agent=models.get(policy)
        gate=0.1 if policy=='D_full_hybrid' else 0.0
        begin=time.perf_counter()
        summary,_,decisions=run_episode(graph,demands,inventory,policy=policy_type,
            model=agent,k=k,headroom=10,mode=mode,stochastic=False,
            gate_threshold=gate,trace=True,seed=stream)
        elapsed=time.perf_counter()-begin
        verified=verify_trace(graph,demands,inventory,decisions,k,headroom=10)
        if any(verified[x]!=summary[y] for x,y in
               [('accepted','admitted'),('departures','departures'),
                ('active_end','active_end'),('overrides','overrides'),
                ('discretionary','discretionary')]):
            raise AssertionError('independent oracle/engine summary divergence')
        if summary['feasibility_violations']!=0 or summary['rejected']!=summary['requests']-summary['admitted']:
            raise AssertionError('invalid simulator result')
        row=dict(summary,nodes=n,topology_seed=topo,stream_seed=stream,load=4.0,mode=mode,k=k,
            policy=policy,decision_records=len(decisions),
            network_sha256=gsha,request_sha256=rsha,candidates_sha256=candidates_sha,
            scenario_oracle_sha256=verified['oracle_state_sha256'],
            generation_seconds=round(gen_seconds,6),simulation_seconds=round(elapsed,6))
        rows.append(row)
        oracles[policy]=dict(final_scenario_load_sha256=verified['oracle_state_sha256'],
                             accepted=verified['accepted'],departures=verified['departures'])
        if keep_trace and policy in ('C_enhanced','D_full_hybrid'):
            traces[policy]=decisions
    return rows,traces,oracles


def _cohort_validate(content,plan_sha,expect_cell=None):
    if content.get('role')!=FINAL_ROLE or content.get('plan_sha256')!=plan_sha:
        raise RuntimeError('saved cohort has different locked protocol')
    if expect_cell is not None and tuple(content['cell'])!=tuple(expect_cell):
        raise RuntimeError('saved cohort cell differs from filename identity')
    rows=content.get('rows',[])
    if len(rows)!=len(POLICIES) or [r['policy'] for r in rows]!=list(POLICIES):
        raise RuntimeError('corrupt cohort: missing/misordered policies')
    if len({str(r['network_sha256']) for r in rows})!=1 or len({str(r['request_sha256']) for r in rows})!=1 or len({str(r['candidates_sha256']) for r in rows})!=1:
        raise RuntimeError('corrupt cohort: policies used different streams/routes')
    for r in rows:
        if (int(r['requests'])!=round(int(r['nodes'])*4.0) or
            int(r['admitted'])+int(r['rejected'])!=int(r['requests']) or
            int(r['feasibility_violations'])!=0 or
            int(r['decision_records'])!=int(r['requests'])):
            raise RuntimeError('corrupt cohort: counts/trace/feasibility')
    return rows


def _expected_files(root):
    for t in TOPOLOGIES:
        for s in STREAMS:
            for n,k,mode,load in CELLS:
                yield root/'cohorts'/_cohort_filename(n,k,mode,t,s), (n,k,mode,load,t,s)


def _aggregate(root,plan_sha):
    all_rows=[]
    for path,expected in _expected_files(root):
        if not path.is_file():raise RuntimeError('incomplete final: '+str(path))
        cohort=_gzip_read(path)
        all_rows.extend(_cohort_validate(cohort,plan_sha,expected))
    validate_full_grid(all_rows)
    summary,primary,factorial,paired=analyze(all_rows)
    _csv_atomic(root/'heldout_runs.csv',all_rows)
    _csv_atomic(root/'heldout_cell_descriptives.csv',summary)
    _csv_atomic(root/'heldout_primary_inference.csv',primary)
    _csv_atomic(root/'heldout_factorial_effects.csv',factorial)
    _csv_atomic(root/'heldout_topology_paired_deltas.csv',paired)
    selection=verify_freeze()
    manifest=dict(version=VERSION,role=FINAL_ROLE,
        plan_sha256=plan_sha,complete=True,heldout_data_accessed=True,
        model_retraining_after_selection=False,
        training_validation_seed_leakage_detected=False,
        selected_checkpoint_sha256=selection['selected']['checkpoint_sha256'],
        preserved_hybrid_checkpoint_sha256=selection['proposed_full_hybrid']['checkpoint_sha256'],
        topology_clusters=len(TOPOLOGIES),streams_per_cluster=len(STREAMS),
        cells=len(CELLS),matched_cohorts=len(CELLS)*len(TOPOLOGIES)*len(STREAMS),
        policies=len(POLICIES),policy_rows=len(all_rows),
        primary_result=next(r for r in primary if r['evidence_family']=='primary'),
        evidence_sha256={f:sha256(root/f) for f in (
            'heldout_runs.csv','heldout_cell_descriptives.csv','heldout_primary_inference.csv',
            'heldout_factorial_effects.csv','heldout_topology_paired_deltas.csv')},
        software=environment(),
        qualification='Synthetic national-scale network, link-disjoint end-to-end single-fault surrogate; NOT MSRP or actual Irish Rail infrastructure.',
        do_not_retrain_or_tune_after_seeing_final=True)
    _json_write_atomic(root/'stage5_final_manifest.json',manifest)
    return manifest


def final_run(out,*,resume=False,confirm_final=False):
    if not confirm_final:
        raise PermissionError('Final heldout data requires explicit --confirm-final flag')
    root=Path(out)
    plan=immutable_plan();plan_sha=hash_json(plan)
    if (root/'stage5_final_manifest.json').exists():
        raise FileExistsError('already finalised: never overwrite/re-run heldout experiment')
    if resume:
        if not root.exists() or not (root/'locked_final_plan.json').is_file():
            raise FileNotFoundError('cannot resume without prior locked final plan')
        prior=json.loads((root/'locked_final_plan.json').read_text(encoding='utf8'))
        if prior.get('plan_sha256')!=plan_sha or prior.get('plan')!=plan:
            raise ValueError('resume protocol, source, checkpoint or library mismatch')
    else:
        if root.exists():raise FileExistsError('refusing to overwrite output folder '+str(root))
        root.mkdir(parents=True,exist_ok=False)
        _json_write_atomic(root/'locked_final_plan.json',dict(plan=plan,plan_sha256=plan_sha))
    models=load_models()
    complete=set()
    for path,expected in _expected_files(root):
        if path.is_file():
            _cohort_validate(_gzip_read(path),plan_sha,expected)
            complete.add(expected)
    for t in TOPOLOGIES:
        for s in STREAMS:
            for n in (500,60):
                required=[(n,k,mode,load,t,s) for cn,k,mode,load in CELLS if cn==n]
                missing=[item for item in required if item not in complete]
                if not missing:continue
                graph=build_network(NetworkSpec(nodes=n,seed=t))
                static=generate_demands(graph,round(n*4.0),s,False)
                dynamic=generate_demands(graph,round(n*4.0),s,True)
                if [(d.request_id,d.source,d.destination,d.service_class,d.bandwidth_mbps,d.arrival,d.priority_weight)
                    for d in static] != [(d.request_id,d.source,d.destination,d.service_class,d.bandwidth_mbps,d.arrival,d.priority_weight)
                    for d in dynamic]:
                    raise AssertionError('unmatched static/dynamic arrivals')
                begin=time.perf_counter()
                inventory=tuple(generate_pairs(graph,d.source,d.destination,6,40) for d in static)
                gen_seconds=time.perf_counter()-begin
                candidate_digests={k:candidates_digest(tuple(tuple(p[:k]) for p in inventory)) for k in (2,6)}
                for nn,k,mode,load,tt,ss in missing:
                    ds=static if mode=='static' else dynamic
                    rows,traces,oracles=_evaluate_cell(graph,ds,inventory,n=n,k=k,mode=mode,
                        topo=t,stream=s,candidates_sha=candidate_digests[k],
                        gen_seconds=gen_seconds,models=models,
                        keep_trace=(n,k,mode,load)==PRIMARY)
                    cell=(n,k,mode,load,t,s)
                    content=dict(version=VERSION,role=FINAL_ROLE,
                        plan_sha256=plan_sha,cell=cell,rows=rows,
                        oracles=oracles,primary_RL_decision_traces=traces)
                    filename=root/'cohorts'/_cohort_filename(n,k,mode,t,s)
                    if filename.exists():raise RuntimeError('refuse to replace cohort '+str(filename))
                    _gzip_write_atomic(filename,content)
                    print(f'HELDOUT_COHORT_OK nodes={n} k={k} mode={mode} topology={t} stream={s} '
                          f'policies={len(rows)} C={rows[3]["weighted_admission"]:.6f} '
                          f'greedy={rows[0]["weighted_admission"]:.6f}',flush=True)
    manifest=_aggregate(root,plan_sha)
    print(f'STAGE5_FINAL_OK rows={manifest["policy_rows"]} cohorts={manifest["matched_cohorts"]} '
          f'topologies={manifest["topology_clusters"]}',flush=True)
    return manifest


def preflight():
    """Frozen artifact proof only: deliberately does not touch reserved seeds."""
    plan=immutable_plan();models=load_models()
    if list(models)!=['C_enhanced','D_full_hybrid']:
        raise AssertionError('unexpected frozen models')
    return dict(result='STAGE5_PREFLIGHT_OK',
                locked_plan_sha256=hash_json(plan),
                C_checkpoint_sha256=plan['selected_C_sha256'],
                D_checkpoint_sha256=plan['nominated_D_sha256'],
                cells=len(CELLS),cohorts=len(CELLS)*len(TOPOLOGIES)*len(STREAMS),
                policy_evaluations=len(CELLS)*len(TOPOLOGIES)*len(STREAMS)*len(POLICIES),
                software=environment(),
                heldout_data_accessed=False)


def devcheck(out):
    """Engineering check on an ALREADY USED VALIDATION seed, never final ones."""
    root=Path(out)
    if root.exists():raise FileExistsError('refuse to overwrite dev check '+str(root))
    models=load_models()
    n=60;t=301;s=72000;load=2.5;k=6;mode='dynamic'
    graph=build_network(NetworkSpec(nodes=n,seed=t))
    ds=generate_demands(graph,round(n*load),s,True)
    begin=time.perf_counter()
    inv=tuple(generate_pairs(graph,d.source,d.destination,6,40) for d in ds)
    gen_s=time.perf_counter()-begin
    rows=[]
    for p in POLICIES:
        s0=time.perf_counter()
        stats,_,decisions=run_episode(graph,ds,inv,policy='hybrid' if p in models else p,
            model=models.get(p),k=k,headroom=10,mode=mode,stochastic=False,
            gate_threshold=0.1 if p=='D_full_hybrid' else 0.0,trace=True,seed=s)
        checked=verify_trace(graph,ds,inv,decisions,k,headroom=10)
        if checked['accepted']!=stats['admitted']:raise AssertionError('devcheck independent oracle mismatch')
        rows.append(dict(stats,policy=p,trace_count=len(decisions),
                         scenario_oracle_sha256=checked['oracle_state_sha256'],
                         simulation_seconds=round(time.perf_counter()-s0,6)))
    root.mkdir(parents=True,exist_ok=False)
    _csv_atomic(root/'stage5_devcheck_results.csv',rows)
    _json_write_atomic(root/'stage5_devcheck_manifest.json',
        dict(version=VERSION,role='DEVELOPMENT_CHECK_VALIDATION_SEEDS_ONLY',
             heldout_data_accessed=False,nodes=n,topology_seed=t,stream_seed=s,
             offered_load=load,mode=mode,k=k,requests=len(ds),generation_seconds=gen_s,
             network_sha256=fingerprint(graph),requests_sha256=stream_digest(ds),
             candidates_sha256=candidates_digest(inv),
             frozen_selection_sha256=sha256(FROZEN/'selection_manifest.json'),
             checkpoint_C_sha256=sha256(FROZEN/'C_enhanced_seed27.pt'),
             checkpoint_D_sha256=sha256(FROZEN/'D_full_hybrid_seed7.pt'),
             software=environment()))
    return dict(result='STAGE5_DEVCHECK_OK',rows=len(rows),
                C_weighted=next(r['weighted_admission'] for r in rows if r['policy']=='C_enhanced'),
                greedy_weighted=next(r['weighted_admission'] for r in rows if r['policy']=='greedy'),
                heldout_data_accessed=False)


def analyze_existing(root):
    root=Path(root)
    if (root/'stage5_final_manifest.json').exists():
        raise FileExistsError('already finalised; no automatic analysis overwrite')
    stored=json.loads((root/'locked_final_plan.json').read_text())
    plan=immutable_plan();plan_sha=hash_json(plan)
    if stored['plan_sha256']!=plan_sha or stored['plan']!=plan:
        raise ValueError('source/checkpoint/plan mismatch: cannot finalize')
    return _aggregate(root,plan_sha)


def verify_results(root):
    """Read-only verification; does not generate any graph, stream, or actions."""
    root=Path(root)
    final=root/'stage5_final_manifest.json'
    if not final.is_file():raise FileNotFoundError('missing completed final manifest')
    manifest=json.loads(final.read_text(encoding='utf8'))
    plan=immutable_plan();plan_sha=hash_json(plan)
    if manifest['role']!=FINAL_ROLE or not manifest['complete'] or manifest['plan_sha256']!=plan_sha:
        raise ValueError('plan hash or completion status invalid')
    locked=json.loads((root/'locked_final_plan.json').read_text(encoding='utf8'))
    if locked.get('plan')!=plan or locked.get('plan_sha256')!=plan_sha:
        raise ValueError('locked plan/source/checkpoint mismatch')
    all_rows=[]
    for path,expected in _expected_files(root):
        if not path.is_file():raise FileNotFoundError('missing matched cohort '+str(path))
        co=_gzip_read(path)
        all_rows.extend(_cohort_validate(co,plan_sha,expected))
        if tuple(expected[:4])==PRIMARY:
            for policy in ('C_enhanced','D_full_hybrid'):
                trace=co.get('primary_RL_decision_traces',{}).get(policy)
                if trace is None or len(trace)!=round(PRIMARY[0]*PRIMARY[3]):
                    raise ValueError('primary RL decision trace missing '+str(path))
                row=next(r for r in co['rows'] if r['policy']==policy)
                if sum(int(x['admitted']) for x in trace)!=int(row['admitted']) or \
                    sum(int(x['overridden']) for x in trace)!=int(row['overrides']):
                    raise ValueError('primary RL trace does not reconcile '+str(path))
    validate_full_grid(all_rows)
    for name,expected_sha in manifest['evidence_sha256'].items():
        if sha256(root/name)!=expected_sha:
            raise ValueError('corrupt or modified final table '+name)
    if len(all_rows)!=manifest['policy_rows']:
        raise ValueError('manifest/row count mismatch')
    # Read-only: stats are recomputed in memory and primary estimate crosschecked.
    _,primary,_,_=analyze(all_rows)
    estimate=next(r['effect_pp'] for r in primary if r['evidence_family']=='primary')
    if abs(estimate-float(manifest['primary_result']['effect_pp']))>1e-10:
        raise ValueError('primary paired estimate does not reconcile')
    return dict(result='STAGE5_VERIFY_OK',verified_policy_rows=len(all_rows),
                matched_cohorts=len(all_rows)//len(POLICIES),topology_clusters=len(TOPOLOGIES),
                primary_delta_pp=estimate,plan_sha256=plan_sha,
                all_five_tables_sha256_verified=True,
                primary_RL_traces_verified=True)
