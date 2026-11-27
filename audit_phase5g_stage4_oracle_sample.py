"""Independent per-request Phase 5F single-fault oracle replays on 8 final cohorts.

One *preselected* (801,77000) topology+stream for each of the 8 frozen cells;
never retrains or tweaks models. Uses Stage 4 candidate/traffic generation but older
separate Phase 5F oracle to check every admission, rejection, and terminal digest.
"""
from pathlib import Path
from dataclasses import replace
import json,gzip,zipfile,time,sys,hashlib
src=Path('/mnt/data/p5g4_final_oracle_runtime')
sys.path.insert(0,str(src))
from phase5g.topology import build_network
from phase5f.traffic import generate_demands,stream_digest
from phase5f_stage2.topology import fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage5.oracle import verify_trace
CELLS=['H_BASE','L_BASE','H_BW','L_BW','H_HOLD','L_HOLD','H_K2','L_K2']
POLICIES=['gnn_PPO','mlp_PPO','C_frozen','scenario_minimax','footprint','greedy']


def transform(demands,cell):
 if cell.endswith('BW'):return tuple(replace(d,bandwidth_mbps=(3*d.bandwidth_mbps+1)//2) for d in demands)
 if cell.endswith('HOLD'):return tuple(replace(d,expiry=d.arrival+max(1,(d.expiry-d.arrival)//2)) for d in demands)
 return tuple(demands)

results=[];start=time.perf_counter();errors=[]
for family in ('H','L'):
 graph=build_network(500,801,'hierarchical' if family=='H' else 'ladder')
 demands=generate_demands(graph,2000,77000,True)
 print('GENERATING 6-PATH INVENTORY',family,flush=True)
 inv6=tuple(generate_pairs(graph,d.source,d.destination,6,40) for d in demands)
 print('CANDIDATES READY',family,'elapsed_sec',int(time.perf_counter()-start),flush=True)
 inv2=tuple(tuple(plans[:2]) for plans in inv6)
 for condition in ('BASE','BW','HOLD','K2'):
  cell=family+'_'+condition
  k=2 if condition=='K2' else 6
  inv=inv2 if k==2 else inv6
  d=transform(demands,cell)
  with zipfile.ZipFile('/mnt/data/Phase5G_Stage4_'+cell+'_Cohorts.zip') as archive:
   data=json.loads(gzip.decompress(archive.read(cell+'__801__77000.json.gz')))['body']
  # This is NOT a source-code hash replay; it recreates the input scenario exactly.
  for label,actual,expected in (
      ('network',fingerprint(graph),data['input_hashes']['network_sha256']),
      ('requests',stream_digest(d),data['input_hashes']['request_sha256']),
      ('candidate inventory',candidates_digest(inv),data['input_hashes']['candidates_sha256'])):
   if actual!=expected:
    errors.append(f'{cell} {label} fingerprint mismatch: {actual} vs {expected}')
    raise RuntimeError(errors[-1])
  for row in data['results']:
   pol=row['policy'];trace=data['traces'][pol]
   r=verify_trace(graph,d,inv,trace,k,headroom=10,verify_rejections=True)
   assert r['accepted']==row['admitted'],(cell,pol,'admissions')
   assert r['departures']==row['departures'],(cell,pol,'departures')
   assert r['overrides']==row['overrides'],(cell,pol,'overrides')
   assert r['discretionary']==row['discretionary'],(cell,pol,'discretionary')
   assert r['active_end']==row['active_end'],(cell,pol,'active_end')
   assert r['oracle_state_sha256']==row['oracle_sha256'],(cell,pol,'oracle_hash')
   results.append(dict(cell=cell,topology_seed=801,stream_seed=77000,policy=pol,accepted=r['accepted'],
                       oracle_sha256=r['oracle_state_sha256'],status='REPRODUCED_INDEPENDENT_ORACLE',
                       requests=len(trace)))
   print('FINAL_ORACLE_REPLAY_OK',cell,pol,r['accepted'],'elapsed_sec',int(time.perf_counter()-start),flush=True)
with Path('/mnt/data/Phase5G_Stage4_Final_Oracle_Sample_Attestation.json').open('w',encoding='utf8') as f:
 json.dump(dict(role='INDEPENDENT_REPLAY_FINAL_RESULTS_AUDIT_NO_RETUNING',scope='one preregistered topology801 stream77000 in each of 8 cells',
  sample_cohorts=8,sample_policy_runs=48,sample_decisions=48*2000,all_correct=True,results=results,
  caveat='Independent protection oracle replay on 8/480 final cohorts (48/2880 policy runs); remaining cohort checks are data-integrity/trace reconstructions.'),f,indent=2)
print('FINAL_SAMPLE_ORACLE_AUDIT_COMPLETE',len(results),'runs',48*2000,'decisions','elapsed_sec',int(time.perf_counter()-start),flush=True)
