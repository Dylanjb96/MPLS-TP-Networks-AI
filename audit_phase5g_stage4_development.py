"""Independent Stage 4 development-evidence inspection; absolutely no final seed generation.

Reads user ZIP unchanged. Independently reconstructs dev candidate cohorts and
replays archived actions with the preexisting Phase5F independent single-fault
oracle. Optional full 60/500 replay only on topology 501 / stream 74000.
"""
from __future__ import annotations
import csv,gzip,hashlib,io,json,math,sys,zipfile,collections,statistics
from pathlib import Path
ROOT=Path('/mnt/data/p5g4_audit');EVIDENCE=Path('/mnt/data/Phase5G_Stage4_Development_Evidence.zip')
sys.path.insert(0,str(ROOT))
from phase5g_stage4.protocol import CELLS,BY_CELL,POLICIES,PROTOCOL_SHA256,STAGE3_FREEZE_MANIFEST_SHA256,semantic_lock,STRESS_SEMANTICS,canonical
from phase5g_stage4.evaluation import settings_for,_seal,inspect_payload,source_hashes,cohort,sha_file,fast_oracle_trajectory
from phase5f_stage5.oracle import verify_trace,SingleFaultOracle
from phase5f_stage2.topology import fingerprint
from phase5f_stage2.candidates import candidates_digest
from phase5f.traffic import stream_digest
import numpy as np

counters=collections.Counter();errors=[];cohorts={};results={};manifest={};traces={}

def check(cond, key, why=''):
    counters[key]+=1
    if not cond:
        errors.append(key+' '+str(why))
        if len(errors)<20:print('ERROR',key,why,flush=True)

def same(a,b,epsilon=1e-10):
    return math.isclose(float(a),float(b),abs_tol=epsilon,rel_tol=0)


def load():
  with zipfile.ZipFile(EVIDENCE) as z:
    check(z.testzip() is None,'zip_integrity')
    names=set(z.namelist());check(len(names)==14,'zip_file_count',len(names))
    for exp,scale in [('stage4_dev_smoke',60),('stage4_scale_dev',500)]:
        prefix=exp+'\\';m=json.loads(z.read(prefix+'run_manifest.json'));manifest[scale]=m
        check(m==_seal({k:v for k,v in m.items() if k!='run_signature'}),'manifest_lock',exp)
        check(m['role']=='dev' and m['nodes']==scale,'manifest_dev_only',exp)
        check(m['topology_seeds']==[501] and m['stream_seeds']==[74000],'dev_seedwall',exp)
        check(m['stage3_freeze_sha256']==STAGE3_FREEZE_MANIFEST_SHA256,'stage3_manifest_sha',exp)
        check(m['protocol_sha256']==PROTOCOL_SHA256 and m['semantic_lock_sha256']==semantic_lock(),'protocol_lock',exp)
        check(m['stage4_source_sha256']==source_hashes(),'stage4_source_lock',exp)
        check(m['original_source_sha256']==json.loads((ROOT/'outputs/phase5g/stage3_FROZEN/freeze_manifest.json').read_text())['source_sha256'],'original_source_lock',exp)
        for p,expected in {'gnn_PPO':'7db8b92ab9bc1648a9224eea727efe035848543fe514c7d7c14be3ee584658c9','mlp_PPO':'942c8edd516ad28b4012733bfd8504af01fde50f47444be7d15c8e02995e3363','C_frozen':'e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9'}.items():
            check(m['checkpoints_sha256'][p]==expected,'frozen_model_sha',f'{exp} {p}')
        check(m['cell_ids']==([c.id for c in CELLS] if scale==60 else ['H_BASE']),'expected_cells',exp)
        text=z.read(prefix+'development_summary.csv').decode('utf8')
        summary=list(csv.DictReader(io.StringIO(text)))
        check(len(summary)==(48 if scale==60 else 6),'summary_row_count',exp)
        check(len({(r['cell'],r['policy']) for r in summary})==len(summary),'summary_no_duplicate',exp)
        expected_names={prefix+'cohorts\\'+cell+'__501__74000.json.gz' for cell in m['cell_ids']}
        found={name for name in names if name.startswith(prefix+'cohorts\\')}
        check(found==expected_names,'cohort_inventory',exp)
        for cellid in m['cell_ids']:
            fname=prefix+f'cohorts\\{cellid}__501__74000.json.gz'
            body=inspect_payload(z.read(fname),m,f'{cellid}__501__74000')
            check(body['input_hashes']=={col:body['results'][0][col] for col in ('network_sha256','request_sha256','candidates_sha256')},'input_hash_signature',f'{exp} {cellid}')
            check(body['role']=='dev' and body['cohort_key']==f'{cellid}__501__74000','cohort_role',f'{exp} {cellid}')
            by={(r['cell'],r['policy']):r for r in summary}
            for row in body['results']:
                pol=row['policy'];trace=body['traces'][pol]
                check(all(str(row[col])==str(by[(cellid,pol)][col]) for col in row),'csv_body_identical',f'{exp} {cellid} {pol}')
                check(len(trace)==scale*4 and row['requests']==scale*4,'complete_request_trace',f'{exp} {cellid} {pol}')
                check(all(str(t['request_id']).startswith('F') and int(t['request_index'])==i for i,t in enumerate(trace)),'trace_request_identity',f'{exp} {cellid} {pol}')
                check(all(0<=int(t['chosen_index'])<cellid_to_k(cellid) if int(t['admitted']) else int(t['chosen_index'])==-1 for t in trace),'chosen_action_k_mask',f'{exp} {cellid} {pol}')
                check(all(int(t['overridden'])==int(int(t['chosen_index'])>=0 and int(t['chosen_index'])!=int(t['greedy_index'])) for t in trace),'overrides_consistent',f'{exp} {cellid} {pol}')
                check(all(int(t['admitted'])+int(t['request_index'])>=0 for t in trace),'admitted_nonnegative',f'{exp} {cellid} {pol}')
                ad=sum(int(t['admitted']) for t in trace)
                wsum=sum(int(t['weight']) for t in trace)
                acceptedweight=sum(int(t['weight'])*int(t['admitted']) for t in trace)
                check(ad==row['admitted'] and row['rejected']==scale*4-ad,'admission_conservation',f'{exp} {cellid} {pol}')
                check(same(acceptedweight/wsum,row['weighted_admission']),'weighted_reconstructed',f'{exp} {cellid} {pol}')
                check(same(ad/(scale*4),row['ordinary_admission']),'ordinary_reconstructed',f'{exp} {cellid} {pol}')
                check(int(row['overrides'])==sum(int(t['overridden']) for t in trace),'trace_overrides_reconstructed',f'{exp} {cellid} {pol}')
                check(int(row['discretionary'])==sum(int(t['feasible_candidates'])>1 for t in trace),'trace_discretionary_reconstructed',f'{exp} {cellid} {pol}')
                check(int(row['never_feasible_requests'])==sum(int(t['chosen_index'])<0 for t in trace),'trace_rejections_reconstructed',f'{exp} {cellid} {pol}')
                check(int(row['feasible_candidates_total'])==sum(int(t['feasible_candidates']) for t in trace),'trace_feasible_count_reconstructed',f'{exp} {cellid} {pol}')
                check(int(trace[-1]['cumulative_admitted'])==ad and int(trace[-1]['active_after'])==int(row['active_end']),'trace_end_counters',f'{exp} {cellid} {pol}')
                check(int(row['feasibility_violations'])==0,'logged_feasibility_zero',f'{exp} {cellid} {pol}')
                check(float(row['exact_trajectory_peak_scenario_utilization'])>=float(row['end_of_episode_peak_scenario_utilization'])-1e-10,'trajectory_not_below_endstate',f'{exp} {cellid} {pol}')
                results[(scale,cellid,pol)]=row
                traces[(scale,cellid,pol)]=trace
            cohorts[(scale,cellid)]=body
            print('P5G4_AUDIT_ARCHIVED_OK',scale,cellid,'policies=6',flush=True)
        check([r['cell'] for r in summary]==[c for c in m['cell_ids'] for p in POLICIES],'summary_canonical_cell_order',exp)
        check([r['policy'] for r in summary]==[p for c in m['cell_ids'] for p in POLICIES],'summary_canonical_policy_order',exp)
    replay=json.loads(z.read('stage4_validation_replay_dev\\validation_replay_manifest.json'))
    check(replay['role']=='PHASE5G_STAGE4_STAGE2_VALIDATION_REPLAY_ONLY' and replay['final_seeds_used'] is False,'replay_role_nonfinal')
    check(len(replay['selected_models'])==2 and set(x['policy'] for x in replay['selected_models'])=={'gnn_PPO','mlp_PPO'},'replay_both_models')
    for row in replay['selected_models']:
        pol=row['policy']; check((row['topology_seed'],row['stream_seed'],row['requests'])==(701,76000,2000),'replay_correct_cohort',pol)
        check(row['source_sha256']==source_hashes(),'replay_source_hashes',pol)
        check(row['validation_trace_reproduced'] is True,'replay_full_match_claim',pol)
        path=ROOT/'outputs/phase5g/stage3_FROZEN/traces'/((('gnn' if pol=='gnn_PPO' else 'mlp')+'_selected_validation_trace.csv.gz'))
        with gzip.open(path,'rt',encoding='utf8',newline='') as f:
            filt=[r for r in csv.DictReader(f) if int(r['topology_seed'])==701 and int(r['stream_seed'])==76000]
        check(len(filt)==2000,'frozen_replay_trace_rows',pol)
        actual=sum(int(r['admitted']) for r in filt)
        weighted=sum(int(r['weight'])*int(r['admitted']) for r in filt)/sum(int(r['weight']) for r in filt)
        check(row['admitted']==actual,'validation_replay_admitted',pol)
        check(same(row['weighted_admission'],weighted),'validation_replay_weighted',pol)
        print('P5G4_AUDIT_VALIDATION_MANIFEST_OK',pol,row['admitted'],row['weighted_admission'],flush=True)
    check(set((i['topology_seed'],i['stream_seed']) for i in replay['selected_models'])=={(701,76000)},'validation_only_no_final')
    for scale in (60,500):
        m=manifest[scale];check(set(m['topology_seeds']).isdisjoint(range(801,821)) and set(m['stream_seeds']).isdisjoint((77000,77001,77002)),'dev_manifest_no_final',scale)
    print('P5G4_ARCHIVE_AUDIT_FINISHED',sum(counters.values()),'checks errors=',len(errors),flush=True)


def cellid_to_k(cellid):return 2 if cellid.endswith('K2') else 6


def oracle_replay(*,scale=60,run_selected_models=False):
  from phase5g_stage4.evaluation import models_load,_summary
  model=models_load() if run_selected_models else None
  for cid in manifest[scale]['cell_ids']:
    c=BY_CELL[cid]
    g,d,inv=cohort(c,nodes=scale,topology_seed=501,stream_seed=74000,role='dev')
    expected=cohorts[(scale,cid)]['input_hashes']
    got={'network_sha256':fingerprint(g),'request_sha256':stream_digest(d),'candidates_sha256':candidates_digest(inv)}
    check(got==expected,'regenerated_cohort_sha',f'{scale} {cid}\n{got}\n{expected}')
    # Full independent protection oracle for EVERY recorded policy/arrival.
    for pol in POLICIES:
      row=results[(scale,cid,pol)];trace=traces[(scale,cid,pol)]
      try:
        verdict=verify_trace(g,d,inv,trace,c.k,headroom=10,verify_rejections=True)
      except Exception as e:
        check(False,'independent_oracle_replay',f'{scale} {cid} {pol}: {e}')
        continue
      check(verdict['accepted']==row['admitted'] and verdict['departures']==row['departures'] and verdict['active_end']==row['active_end'] and verdict['overrides']==row['overrides'] and verdict['discretionary']==row['discretionary'] and verdict['oracle_state_sha256']==row['oracle_sha256'],'oracle_complete_trace_match',f'{scale} {cid} {pol} {verdict}')
      if run_selected_models:
        # Re-evaluate the *actual* selected policies using exact saved source
        # and frozen checkpoints; compare every action and state fingerprint.
        t=row['policy'];newstat,newtrace,neworacle,_,_=_summary(t,g,d,inv,c.k,model)
        common=('request_id','request_index','chosen_index','greedy_index','admitted','state_digest',
                'feasible_candidates','structural_candidates','overridden','active_after','cumulative_admitted')
        for i,(current,saved) in enumerate(zip(newtrace,trace)):
          check(all(str(current[k])==str(saved[k]) for k in common),'deterministic_action_replay',f'{scale} {cid} {pol} idx={i}')
        check(neworacle['oracle_sha256']==row['oracle_sha256'] and newstat['admitted']==row['admitted'] and same(newstat['weighted_admission'],row['weighted_admission']),'policy_summary_replay',f'{scale} {cid} {pol}')
    if c.id.endswith('_BW'):
        b=cohorts[(scale,cid)]['input_hashes']
        base='H_BASE' if c.family=='hierarchical' else 'L_BASE'
        check(b['network_sha256']==cohorts[(scale,base)]['input_hashes']['network_sha256'] and b['candidates_sha256']==cohorts[(scale,base)]['input_hashes']['candidates_sha256'],'stress_same_paths',f'{scale} {cid}')
    if c.id.endswith('_K2'):
        check(all(len(x)<=2 for x in inv),'k2_candidate_prefix',f'{scale} {cid}')
    print('P5G4_INDEPENDENT_ORACLE_OK',scale,cid,'policies=',len(POLICIES),flush=True)
  print('P5G4_ORACLE_AUDIT_FINISHED',scale,sum(counters.values()),'checks','errors',len(errors),flush=True)


def summary():
 print('P5G4_FINAL_AUDIT_COUNTS',sum(counters.values()),'checks',len(errors),'errors')
 for key,val in sorted(counters.items()):print('CHECK',key,val)
 for err in errors:print('FAIL',err)
 out=Path('/mnt/data/Phase5G_Stage4_Development_Independent_Audit_Results.json')
 out.write_text(json.dumps({'checks_total':sum(counters.values()),'checks_failed':len(errors),
                            'check_categories':dict(counters),'errors':errors,'input_archive_sha256':sha_file(EVIDENCE),
                            'stage3_freeze_sha256':sha_file(ROOT/'outputs/phase5g/stage3_FROZEN/freeze_manifest.json'),
                            'stage4_source_sha256':source_hashes()},indent=2,sort_keys=True)+'\n')
 if errors:raise SystemExit(1)


if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--oracle-60',action='store_true');ap.add_argument('--oracle-500',action='store_true');ap.add_argument('--replay-60',action='store_true');a=ap.parse_args()
 load()
 if a.oracle_60:oracle_replay(scale=60,run_selected_models=a.replay_60)
 if a.oracle_500:oracle_replay(scale=500)
 summary()
