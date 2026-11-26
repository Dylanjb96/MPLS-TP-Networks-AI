"""READ-ONLY forensic source drift audit for Phase 5G Stage 2 (Python stdlib only).

Usage from MPLS-AI project root, with this script and original_reference/ extracted:
    python audit_phase5g_stage2_source_drift.py

Does not change training, models, provenance, source code, validation or held-out data.
Creates one text report and one small archive containing the exact source snapshots
and run provenance metadata needed for a reproducibility review.
"""
from __future__ import annotations
import ast, difflib, hashlib, io, json, subprocess, sys, zipfile
from pathlib import Path
from collections import defaultdict

ROOT=Path.cwd()
KIT=Path(__file__).resolve().parent
FILES=('phase5g_stage2/__init__.py','phase5g_stage2/workflow.py')
RUN_ROOT=ROOT/'outputs'/'phase5g'/'stage2_train'
REF_DIR=KIT/'original_reference'
TARGETS=[(arch,seed) for arch in ('mlp','gnn') for seed in (7,17,27)]
REPORT=ROOT/'phase5g_stage2_source_drift_report.txt'
ARCHIVE=ROOT/'Phase5G_Stage2_Source_Drift_Evidence.zip'
ORIGINAL_HASH={
 'phase5g_stage2/__init__.py':'8c31570c5588b32a3b22d0bcd23df60d417b778028ba2a197dd2de1f57bfea68',
 'phase5g_stage2/workflow.py':'d3d2568e0a556104765db2651c849f909af01652517484a3f725851a0ed3690e',
}
def sha(b):return hashlib.sha256(b).hexdigest()
def as_ast(b):
 try:
  return ast.dump(ast.parse(b.decode('utf-8-sig')),include_attributes=False)
 except (UnicodeDecodeError,SyntaxError) as e:return 'PARSE_ERROR: '+str(e)
def git(args):
 try:
  p=subprocess.run(['git',*args],cwd=ROOT,stdout=subprocess.PIPE,
     stderr=subprocess.PIPE,timeout=25,check=False)
  return p.stdout if p.returncode==0 else None
 except (OSError,subprocess.TimeoutExpired):return None

def main():
 problems=[];logs=[];saved={};git_snapshots={};snapshots={};outfiles={}
 def say(*parts):logs.append(' '.join(str(x) for x in parts))
 say('PHASE5G_STAGE2_READ_ONLY_SOURCE_DRIFT_AUDIT_V1')
 say('Note: this is an examination of provenance; it does NOT authorize model selection.')
 for arch,seed in TARGETS:
  key=f'{arch}_seed{seed}';p=RUN_ROOT/key/'training_provenance.json'
  if not p.is_file():
   problems.append('Missing '+str(p));continue
  try:saved[key]=json.loads(p.read_text(encoding='utf-8'))
  except Exception as exc:problems.append(f'{key}: provenance parse error {exc}')
 say('Run provenance present:',len(saved),'/6')
 say('')
 say('=== TRAINING SNAPSHOT MATRIX ===')
 for name in FILES:
  targets=defaultdict(list)
  for key,p in saved.items():targets[p.get('source_sha256',{}).get(name,'MISSING')].append(key)
  say('SOURCE:',name)
  for h,keys in sorted(targets.items()):say('  ',h,':',','.join(keys))
  current=ROOT/name
  if not current.is_file():problems.append('Current source missing: '+name);continue
  b=current.read_bytes();snapshots[f'current/{name}']=b
  say('  Current source SHA256:',sha(b))
  ref=REF_DIR/name
  if not ref.is_file():problems.append('Original reference missing: '+str(ref));continue
  r=ref.read_bytes();snapshots[f'original_reference/{name}']=r
  if sha(r)!=ORIGINAL_HASH[name]:problems.append('Reference file changed/tampered: '+name)
  say('  Original package SHA256:',sha(r))
  say('  Original/current AST-equivalent:',as_ast(r)==as_ast(b))
  a=r.decode('utf-8-sig',errors='replace').splitlines(keepends=True)
  c=b.decode('utf-8-sig',errors='replace').splitlines(keepends=True)
  diff=''.join(difflib.unified_diff(a,c,fromfile='original package/'+name,
                          tofile='current project/'+name,n=4))
  if diff:
   say('  ORIGINAL versus CURRENT unified diff lines:',len(diff.splitlines()))
   outfiles[f'diffs/{Path(name).name}.diff']=diff.encode('utf-8')
   say(diff[:28000])
   if len(diff)>28000:say('  (diff truncated in text report; full diff in evidence ZIP)')
  else:say('  Original matches current byte-for-byte')
  say('')
 say('=== GIT SOURCE HISTORY (read only) ===')
 for name in FILES:
  raw=git(['rev-list','--all','--',name])
  if raw is None:
   say('Git history unavailable for',name,'(Git missing/not a repo/no commits).')
   continue
  revs=raw.decode('ascii',errors='replace').split()
  seen={}
  for rev in revs[:250]:
   b=git(['show',f'{rev}:{name}'])
   if b is None:continue
   h=sha(b)
   if h not in seen:
    seen[h]=rev
    snapname=f'git_source_snapshots/{Path(name).stem}_{h[:12]}.py'
    snapshots[snapname]=b
  git_snapshots[name]=seen
  say(name,':',len(seen),'distinct committed blob versions; history commits scanned',min(len(revs),250))
  for h,rev in seen.items():
   in_runs=','.join(key for key,p in saved.items() if p.get('source_sha256',{}).get(name)==h)
   say('  ',h,'commit',rev[:12],('training_runs='+in_runs if in_runs else '(not a training fingerprint)'))
  trained=set(p.get('source_sha256',{}).get(name) for p in saved.values())
  for h in sorted(trained):
   loc=[]
   if h==sha((ROOT/name).read_bytes()):loc.append('current')
   if h==sha((REF_DIR/name).read_bytes()):loc.append('original package')
   if h in seen:loc.append('Git '+seen[h][:12])
   say('  Trained fingerprint',h[:16], 'recovered from:', ', '.join(loc) or 'NOT AVAILABLE FROM PROJECT/GIT/REFERENCE')
  say('')
 say('=== NON-CHANGED SOURCE FINGERPRINTS, PROGRAM AND FROZEN MODEL ===')
 all_keys=set().union(*(set(p.get('source_sha256',{})) for p in saved.values()))
 for name in sorted(all_keys-set(FILES)):
  vals={p.get('source_sha256',{}).get(name) for p in saved.values()}
  if len(vals)!=1:problems.append('Other source drift: '+name+' '+str(vals))
 say('Other tracked source files identical across six runs:',not any('Other source drift:' in p for p in problems))
 plans=[{k:v for k,v in p.get('plan',{}).items() if k not in ('architecture','agent_seed')} for p in saved.values()]
 say('Shared non-architecture schedule and hyperparameter plan:',len(set(json.dumps(p,sort_keys=True) for p in plans))==1)
 say('Frozen C checkpoint fingerprints identical:',len({p.get('frozen_C_sha256') for p in saved.values()})==1)
 say('')
 say('=== RESEARCH INTEGRITY CONCLUSION ===')
 if len(saved)==6:
  groups=defaultdict(list)
  for key,p in saved.items():groups[tuple((n,p['source_sha256'].get(n)) for n in FILES)].append(key)
  say('Unique two-file source snapshots used for training:',len(groups))
  for i,group in enumerate(groups.values(),1):say('  snapshot',i,':',', '.join(group))
  if len(groups)>1:
   say('STOP: six runs used different source snapshots. Do not rewrite provenance,')
   say('copy old sources over current, rerun final held-out, or call this selection confirmed.')
   say('Review the original-vs-current and Git-history diffs for behavioral impact.')
  else:say('All six runs have the same two-file provenance (verify other checks separately).')
 for issue in problems:say('ISSUE:',issue)
 say('Evidence archive contains ONLY code snapshots, source diffs and six provenance JSON files.')
 say('No checkpoints, final evaluation files or training histories are copied.')
 report='\n'.join(logs)+'\n'
 REPORT.write_text(report,encoding='utf-8')
 with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('phase5g_stage2_source_drift_report.txt',report)
  for k,v in sorted(snapshots.items()):z.writestr(k,v)
  for k,v in sorted(outfiles.items()):z.writestr(k,v)
  for key in sorted(saved):
   p=RUN_ROOT/key/'training_provenance.json'
   z.writestr(f'run_provenance/{key}/training_provenance.json',p.read_bytes())
 print('PHASE5G_STAGE2_SOURCE_DRIFT_AUDIT_OK')
 print('Training run source snapshots:',len({tuple(p.get('source_sha256',{}).get(n) for n in FILES) for p in saved.values()}))
 print('Report:',REPORT)
 print('Source-only evidence ZIP:',ARCHIVE)
 if problems:
  print('AUDIT_WARNINGS:',*problems,sep='\n')
 return 0 if len(saved)==6 and not problems else 2
if __name__=='__main__':sys.exit(main())
