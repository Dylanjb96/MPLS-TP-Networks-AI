"""Independent *offline* forensic audit of Phase 5G final held-out ZIPs.

Uses archived per-request traces and preregistered comparisons; DOES NOT generate
held-out networks, train agents, or call the original Phase 5G analysis engine.
Unlike Phase 5G Stage 4 runtime oracle, cannot independently rerun single fault
reservations without regenerating the exact graph and candidate pool.
"""
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, stdev
from scipy.stats import t as student_t
import math,csv,io,zipfile,gzip,hashlib,json,time,re,sys

BASE=Path('/mnt/data')
CELLS=['H_BASE','L_BASE','H_BW','L_BW','H_HOLD','L_HOLD','H_K2','L_K2']
POLICIES=['gnn_PPO','mlp_PPO','C_frozen','scenario_minimax','footprint','greedy']
PRIMARY=[('H_BASE','gnn_PPO','mlp_PPO'),('H_BASE','gnn_PPO','scenario_minimax')]
SECONDARY=[('L_BASE','gnn_PPO','mlp_PPO'),('L_BASE','gnn_PPO','scenario_minimax'),('H_BASE','C_frozen','scenario_minimax'),('L_BASE','C_frozen','scenario_minimax'),('H_BASE','gnn_PPO','C_frozen'),('L_BASE','gnn_PPO','C_frozen'),('H_BASE','gnn_PPO','greedy'),('L_BASE','gnn_PPO','greedy')]
FLOATS={'load','weighted_admission','ordinary_admission','exact_trajectory_peak_scenario_utilization','end_of_episode_peak_scenario_utilization','policy_seconds','oracle_seconds','candidate_generation_seconds'}
META={'cell','family','stress','mode','policy','network_sha256','request_sha256','candidates_sha256','oracle_sha256'}
INTS={'k','nodes','topology_seed','stream_seed','requests','admitted','rejected','discretionary','overrides','departures','active_end','structural_candidates_total','feasible_candidates_total','never_feasible_requests','decision_records','feasibility_violations'}
FZ={'gnn_PPO':'7db8b92ab9bc1648a9224eea727efe035848543fe514c7d7c14be3ee584658c9','mlp_PPO':'942c8edd516ad28b4012733bfd8504af01fde50f47444be7d15c8e02995e3363','C_frozen':'e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9'}
errors=[];count=Counter();warnings=[]
def check(ok,k,context=''):
    count[k]+=1
    if not ok:
        msg=f'{k}: {context}'
        if len(errors)<150:errors.append(msg)
def eqnum(a,b):
    try:return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12)
    except (TypeError,ValueError):return False

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def sha(v):return hashlib.sha256(v).hexdigest()
def readcsv(b):return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def sniff_sha(v):return isinstance(v,str) and bool(re.fullmatch('[a-f0-9]{64}',v))

with zipfile.ZipFile(BASE/'Phase5G_Stage4_Final_Reports.zip') as rep:
    check(rep.testzip() is None,'reports_zip_crc')
    submitted_run=json.loads(rep.read('run_manifest.json'))
    submitted_report=json.loads(rep.read('report_manifest.json'))
    release=json.loads(rep.read('stage4_AUDIT_RELEASE.json'))
    policy_rows=readcsv(rep.read('complete_policy_rows.csv'))
    official_primary=readcsv(rep.read('primary_cluster_inference.csv'))
    official_secondary=readcsv(rep.read('secondary_cluster_inference.csv'))
    official_exploratory=readcsv(rep.read('exploratory_robustness.csv'))
    official_summary=readcsv(rep.read('policy_summary.csv'))
    check(len(policy_rows)==2880,'complete_csv_rows',len(policy_rows))
    check(submitted_report['n_policy_rows']==2880 and submitted_report['n_topology_clusters']==20,'report_metadata_dimensions')
    check(submitted_report['source_manifest']==submitted_run,'report_run_manifest_equality')
    check(submitted_run['role']=='final','run_final_role')
    check(submitted_run['checkpoints_sha256']==FZ,'frozen_checkpoint_ids')
    check(submitted_run['cell_ids']==CELLS,'cell_order')
    check(submitted_run['stream_seeds']==[77000,77001,77002],'reserved_streams')
    check(submitted_run['topology_seeds']==list(range(801,821)),'reserved_topologies')
    check(release['passed_development_audit'] is True and release['protocol_sha256']==submitted_run['protocol_sha256'] and release['semantic_lock_sha256']==submitted_run['semantic_lock_sha256'] and release['stage3_freeze_manifest_sha256']==submitted_run['stage3_freeze_sha256'] and release['stage4_source_sha256']==submitted_run['stage4_source_sha256'],'release_matches_run')
    with zipfile.ZipFile(BASE/'Phase5G_Stage4_Independent_Audit_Release_Package.zip') as rz:
        check(rz.read('stage4_AUDIT_RELEASE.json')==rep.read('stage4_AUDIT_RELEASE.json'),'release_exact_original_bytes')
    with zipfile.ZipFile(BASE/'Phase5G_Stage3_Freeze_Evidence.zip') as frozen:
        if 'stage3_FROZEN\\freeze_manifest.json' in frozen.namelist():
            fb=frozen.read('stage3_FROZEN\\freeze_manifest.json')
            check(sha(fb)==submitted_run['stage3_freeze_sha256'],'stage3_original_freeze_sha')
            f=json.loads(fb)
            check(f['protocol_sha256']==submitted_run['protocol_sha256'] and f['selected']['gnn']['sha256']==FZ['gnn_PPO'] and f['selected']['mlp']['sha256']==FZ['mlp_PPO'] and f['old_C_sha256']==FZ['C_frozen'],'stage3_accepted_models_and_protocol')
    with zipfile.ZipFile(BASE/'Phase5G_Stage4_Development_Evaluator_AddOn.zip') as impl:
        for file, recorded in submitted_run['stage4_source_sha256'].items():
            check(sha(impl.read(file))==recorded,'stage4_code_fingerprint',file)
        with zipfile.ZipFile(BASE/'Phase5G_Stage4_Source_Handoff.zip') as prev:
            for file, recorded in submitted_run['original_source_sha256'].items():
                check(sha(prev.read(file))==recorded,'original_code_fingerprint',file)
    # Index report CSV rows; compare each archived policy result to its reported row.
    csv_index={}
    for r in policy_rows:
        key=(r['cell'],int(r['topology_seed']),int(r['stream_seed']),r['policy'])
        check(key not in csv_index,'csv_unique_policy_key',key)
        csv_index[key]=r
    expected={(c,t,s,p) for c in CELLS for t in range(801,821) for s in (77000,77001,77002) for p in POLICIES}
    check(set(csv_index)==expected,'full_preregistered_grid',f'got {len(csv_index)}')
    root_signature=submitted_run['run_signature']
    source_net={} # (family,topology) -> graph digest
    source_input={} # (cell,topo,stream)-> hashes
    summary_rows=[]
    trace_rows_total=0;bad_k_count=0;mlp_greedy_identical=0
    t0=time.perf_counter()
    for cell in CELLS:
        fn=BASE/f'Phase5G_Stage4_{cell}_Cohorts.zip'
        with zipfile.ZipFile(fn) as z:
            check(z.testzip() is None,'cell_archive_crc',cell)
            names=z.namelist()
            expectednames={f'{cell}__{t}__{s}.json.gz' for t in range(801,821) for s in (77000,77001,77002)}
            check(set(names)==expectednames,'cell_has_exact_60_cohorts',cell)
            for t in range(801,821):
                for s in (77000,77001,77002):
                    name=f'{cell}__{t}__{s}.json.gz'
                    body=json.loads(gzip.decompress(z.read(name)))
                    doc=body['body']
                    check(sha(canonical(doc))==body['body_sha256'],'body_sha256',name)
                    check(doc['cohort_key']==name.removesuffix('.json.gz') and doc['role']=='final' and doc['locked_signature']==root_signature,'cohort_fixed_provenance',name)
                    check(len(doc['results'])==6 and set(doc['traces'])==set(POLICIES),'six_policies_per_cohort',name)
                    check([v['policy'] for v in doc['results']]==POLICIES,'policy_order',name)
                    family='hierarchical' if cell[0]=='H' else 'ladder'
                    expected_stress='bandwidth_150' if cell.endswith('BW') else ('short_hold' if cell.endswith('HOLD') else 'base')
                    k=2 if cell.endswith('K2') else 6
                    source_input[cell,t,s]=doc['input_hashes']
                    netkey=(family,t)
                    if netkey not in source_net:source_net[netkey]=doc['input_hashes']['network_sha256']
                    check(source_net[netkey]==doc['input_hashes']['network_sha256'],'per_family_topology_hash_across_cells',f'{cell} {t}')
                    reference_trace=doc['traces']['greedy']
                    demand_signature=tuple((d['request_id'],d['service'],d['weight']) for d in reference_trace)
                    total_weight=sum(int(d['weight']) for d in reference_trace)
                    for row in doc['results']:
                        policy=row['policy'];key=(cell,t,s,policy)
                        csvrow=csv_index.get(key)
                        check(csvrow is not None,'cohort_exists_in_reports_csv',key)
                        if csvrow is None:continue
                        check(set(row)==set(csvrow)==META|INTS|FLOATS,'matching_row_columns',key)
                        for kfield in META:
                            check(row[kfield]==csvrow[kfield],'row_meta_reconcile',f'{key}.{kfield}')
                        for kfield in INTS:
                            check(int(row[kfield])==int(csvrow[kfield]),'row_int_reconcile',f'{key}.{kfield}')
                        for kfield in FLOATS:
                            check(eqnum(row[kfield],csvrow[kfield]),'row_float_reconcile',f'{key}.{kfield}')
                        check(int(row['requests'])==2000 and int(row['nodes'])==500 and int(row['topology_seed'])==t and int(row['stream_seed'])==s and row['family']==family and row['stress']==expected_stress and int(row['k'])==k and row['mode']=='dynamic' and float(row['load'])==4.0,'frozen_condition_semantics',key)
                        check(all(row[field]==doc['input_hashes'][field] for field in doc['input_hashes']),'paired_network_demand_candidate_hashes',key)
                        check(0<=row['admitted']<=2000 and row['admitted']+row['rejected']==2000 and row['decision_records']==2000 and row['feasibility_violations']==0,'policy_record_counts_and_feasibility_logs',key)
                        check(row['never_feasible_requests']==row['rejected'],'forced_rejection_accounting',key)
                        check(all(0 <=float(row[q]) <=1.000000001 for q in ['ordinary_admission','weighted_admission','exact_trajectory_peak_scenario_utilization','end_of_episode_peak_scenario_utilization']),'admission_peak_bounds',key)
                        check(float(row['exact_trajectory_peak_scenario_utilization'])+1e-12>=float(row['end_of_episode_peak_scenario_utilization']),'true_peak_ge_terminal_peak',key)
                        check(all(math.isfinite(float(row[q])) and float(row[q])>=0 for q in ('oracle_seconds','policy_seconds','candidate_generation_seconds')),'valid_runtime_metrics',key)
                        trace=doc['traces'][policy]
                        check(len(trace)==2000,'exact_request_trace_count',key)
                        trace_rows_total+=len(trace)
                        observed_admitted=0;observed_weighted=0;observed_overrides=0;observed_disc=0;observed_feasible=0;obs_structural=0;obs_rejected=0;obs_total_weight=0;prev_active=0
                        for i,rec in enumerate(trace):
                            # Up to 5.76m request decisions; avoid tool output per record.
                            if int(rec['request_index'])!=i:
                                if len(errors)<150:errors.append('trace request order '+str(key)+' index '+str(i))
                            admitted=int(rec['admitted']);chosen=int(rec['chosen_index']);greedy=int(rec['greedy_index']);feasible=int(rec['feasible_candidates']);struct=int(rec['structural_candidates']);weight=int(rec['weight']);active=int(rec['active_after']);ov=int(rec['overridden'])
                            if (not 0<=feasible<=struct<=k or admitted not in (0,1) or ov not in (0,1) or weight<=0 or not 0<=active<=2000 or (feasible==0)!=(admitted==0) or (admitted and (not 0<=chosen<struct or not 0<=greedy<struct)) or (not admitted and (chosen!=-1 or greedy!=-1)) or ov!=int(chosen>=0 and chosen!=greedy) or (not admitted and (rec['chosen_candidate_id'] not in (None,''))) or (chosen>=0 and not rec['chosen_candidate_id']) or (greedy>=0 and not rec['greedy_candidate_id'])):
                                if len(errors)<150:errors.append('invalid trace item '+str(key)+' i='+str(i))
                            if active>prev_active+admitted:
                                if len(errors)<150:errors.append('invalid active transition '+str(key)+' i='+str(i))
                            prev_active=active
                            observed_admitted+=admitted
                            observed_weighted+=weight*admitted
                            observed_overrides+=ov
                            observed_disc+=feasible>1
                            observed_feasible+=feasible
                            obs_structural+=struct
                            obs_total_weight+=weight
                            obs_rejected+=1-admitted
                        check(tuple((d['request_id'],d['service'],d['weight']) for d in trace)==demand_signature,'same_request_identity_weights_all_policies',key)
                        check(observed_admitted==row['admitted'] and observed_admitted==int(trace[-1]['cumulative_admitted']) and prev_active==row['active_end'],'admissions_active_endpoint',key)
                        check(observed_weighted/obs_total_weight==row['weighted_admission'] or eqnum(observed_weighted/obs_total_weight,row['weighted_admission']),'recompute_priority_weighted_admission',key)
                        check(eqnum(observed_admitted/2000,row['ordinary_admission']),'recompute_ordinary_admission',key)
                        check(observed_overrides==row['overrides'],'recompute_overrides',key)
                        check(observed_disc==row['discretionary'],'recompute_discretionary',key)
                        check(observed_feasible==row['feasible_candidates_total'] and obs_structural==row['structural_candidates_total'],'recompute_candidate_inventory_totals',key)
                        check(obs_rejected==row['rejected'] and row['departures']==row['admitted']-row['active_end'],'recompute_departures_conservation',key)
                        check(obs_total_weight==total_weight,'equal_offered_priority_mass',key)
                        if k==2:
                            check(all(int(d['structural_candidates'])<=2 and int(d['feasible_candidates'])<=2 for d in trace),'k2_candidate_masks',key)
                        if policy=='mlp_PPO' and trace==reference_trace: mlp_greedy_identical+=1
                        if policy=='mlp_PPO' and row['overrides']>=5: bad_k_count+=1
                    # end policies
                    if (s==77002 and t%5==0):print('COHORTS_AUDITED',cell,t,'elapsed_sec',int(time.perf_counter()-t0),flush=True)
    # Across conditions same topology/stream has invariant graph, baseline request under K2 and candidate pool under BW/HOLD.
    for family in ('H','L'):
        for t in range(801,821):
            for s in (77000,77001,77002):
                a=source_input[family+'_BASE',t,s]
                for condition in ('BW','HOLD'):
                    b=source_input[family+'_'+condition,t,s]
                    check(b['network_sha256']==a['network_sha256'],'stress_no_graph_rewrite',f'{family} {t} {s} {condition}')
                    check(b['candidates_sha256']==a['candidates_sha256'],'stress_same_candidate_paths',f'{family} {t} {s} {condition}')
                k2=source_input[family+'_K2',t,s]
                check(k2['request_sha256']==a['request_sha256'] and k2['network_sha256']==a['network_sha256'],'k2_same_network_and_traffic',f'{family} {t} {s}')
    # Official statistics independently recomputed from CSV; does not use phase5g_stage4/statistics.py.
    def per_cluster(cell,first,second):
        return [mean(100*(float(csv_index[cell,t,s,first]['weighted_admission'])-float(csv_index[cell,t,s,second]['weighted_admission'])) for s in (77000,77001,77002)) for t in range(801,821)]
    def infer(x):
        effect=mean(x);sd=stdev(x);se=sd/(20**0.5)
        if sd==0:
            p=1.0 if effect==0 else None
            ci=(effect,effect)
        else:
            crit=student_t.ppf(0.975,19)
            ci=(effect-crit*se,effect+crit*se)
            p=2*student_t.sf(abs(effect/se),19)
        return dict(n_topologies=20,streams_per_topology=3,effect_pp=effect,sd_pp=sd,ci95_low_pp=ci[0],ci95_high_pp=ci[1],p_two_sided=p,positive=sum(v>0 for v in x),negative=sum(v<0 for v in x),ties=sum(v==0 for v in x))
    def holm(ps):
        adj=[None]*len(ps);prev=0.0
        for k,i in enumerate(sorted(range(len(ps)),key=lambda i:(ps[i],i))):
            prev=max(prev,min(1.0,(len(ps)-k)*ps[i]));adj[i]=prev
        return adj
    independent={}
    for family,spec,official in [('primary',PRIMARY,official_primary),('secondary',SECONDARY,official_secondary)]:
        items=[infer(per_cluster(*v)) for v in spec]
        adj=holm([v['p_two_sided'] for v in items])
        independent[family]=[]
        for (c,a,b),d,padj in zip(spec,items,adj):
            x=dict(family='co_primary' if family=='primary' else 'secondary',cell=c,first=a,second=b,**d,p_holm_adjusted=padj,holm_significant=bool(padj<0.05))
            r=next((v for v in official if (v['cell'],v['first'],v['second'])==(c,a,b)),None)
            check(r is not None,'formal_hypothesis_grid',str((c,a,b)))
            if r:
                for k,v in x.items():
                    if isinstance(v,float):check(eqnum(v,r[k]),'inference_recalculated_float',str((c,a,b,k,v,r[k])))
                    else:check(str(v)==str(r[k]),'inference_recalculated_other',str((c,a,b,k,v,r[k])))
            independent[family].append(x)
    exploratory=[]
    for fam,ab in (('hierarchical','H'),('ladder','L')):
        for cond in ('BW','HOLD','K2'):
            for a,b in (('gnn_PPO','mlp_PPO'),('gnn_PPO','scenario_minimax')):
                vals=[x-y for x,y in zip(per_cluster(ab+'_'+cond,a,b),per_cluster(ab+'_BASE',a,b))]
                x=dict(family='exploratory_interaction_unadjusted',cell=ab+'_'+cond+'_minus_'+ab+'_BASE',first=a,second=b,**infer(vals))
                r=next((v for v in official_exploratory if (v['cell'],v['first'],v['second'])==(x['cell'],a,b)),None)
                check(r is not None,'exploratory_hypothesis_grid',x['cell'])
                if r:
                    for k,v in x.items():
                        if isinstance(v,float):check(eqnum(v,r[k]),'exploratory_recalculated_float',str((x['cell'],a,b,k)))
                        else:check(str(v)==str(r[k]),'exploratory_recalculated_other',str((x['cell'],a,b,k)))
                exploratory.append(x)
    independent['exploratory']=exploratory
    policy_summary=[]
    for cell in CELLS:
        for p in POLICIES:
            match=[csv_index[cell,t,s,p] for t in range(801,821) for s in (77000,77001,77002)]
            score=mean(float(v['weighted_admission']) for v in match)
            total=sum(int(v['admitted']) for v in match)
            times=sum(float(v['policy_seconds']) for v in match)
            disc=sum(int(v['discretionary']) for v in match)
            overrides=sum(int(v['overrides']) for v in match)
            r=next(v for v in official_summary if v['cell']==cell and v['policy']==p)
            check(eqnum(score,r['mean_weighted_admission']),'summary_weighted_reproduced',f'{cell} {p}')
            check(total==int(r['total_admitted']),'summary_admissions_reproduced',f'{cell} {p}')
            check(eqnum(times,r['total_policy_seconds']),'summary_time_reproduced',f'{cell} {p}')
            check(disc==int(r['discretionary']) and overrides==int(r['overrides']),'summary_route_overrides_reproduced',f'{cell} {p}')
            policy_summary.append(dict(cell=cell,policy=p,mean_weighted_admission=score,total_admitted=total,total_policy_seconds=times,discretionary=disc,overrides=overrides))
    # Summary reconciliation to report JSON source numbers (all official numbers remain unchanged).
    for family in ('primary','secondary','exploratory'):
        official_m=submitted_report[family]
        check(len(official_m)==len(independent[family]),'report_manifest_inference_size',family)
        for row in independent[family]:
            v=next(r for r in official_m if (r['cell'],r['first'],r['second'])==(row['cell'],row['first'],row['second']))
            check(eqnum(row['effect_pp'],v['effect_pp']),'report_manifest_inference_effect',f"{family}:{row['cell']} {row['first']} {row['second']}")
    # Write output.
    out=dict(archive_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [BASE/'Phase5G_Stage4_Final_Reports.zip']+[BASE/f'Phase5G_Stage4_{c}_Cohorts.zip' for c in CELLS]},
             checks_total=sum(count.values()),checks_failed=len(errors),checks_by_category=dict(count),
             complete_cohorts=480,policy_runs=2880,request_level_trace_rows=trace_rows_total,
             all_8_cells=True,mlp_full_trace_equal_to_greedy_cohorts=mlp_greedy_identical,
             mlp_cohorts_with_5plus_overrides=bad_k_count,primary=independent['primary'],
             secondary=independent['secondary'],exploratory=independent['exploratory'],
             policy_summary=policy_summary,errors=errors,warnings=warnings,
             limitations=['Payload and trace integrity are verified independent of the reporter, but scenario oracle feasibility is not independently resimulated for all final cohorts.','Checkpoint fingerprints are verified against frozen manifest and preregistered source; inference decisions are not newly run here.','Only synthetic topologies and end-to-end single-link failure surrogate; not real railway, MSRP or restoration latency.','The 20 topology seeds may share a topology generator and cannot represent infrastructure-independent generalization.'])
    (BASE/'Phase5G_Stage4_Final_Independent_Audit_Results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    with (BASE/'Phase5G_Stage4_Final_Independent_Policy_Summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=policy_summary[0].keys());w.writeheader();w.writerows(policy_summary)
    with (BASE/'Phase5G_Stage4_Final_Independent_Statistical_Comparisons.csv').open('w',newline='',encoding='utf8') as f:
        rows=independent['primary']+independent['secondary']+independent['exploratory'];w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    print('AUDIT_END', 'checks',out['checks_total'],'failures',out['checks_failed'],'traces',trace_rows_total,'elapsed',round(time.perf_counter()-t0,1),flush=True)
    print('PRIMARY',independent['primary']);print('REPORT SUMMARY FIRST',policy_summary[:7])
    for e in errors[:40]:print('ERROR',e)
    if errors:sys.exit(1)
