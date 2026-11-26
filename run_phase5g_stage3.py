"""Phase 5G Stage 3 — offline, read-only validation and one-time model/protocol freeze.

No network generation, no training and no reserved Phase 5G final data access.
Never weakens the prior documented docstring-only source-drift exception.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import mean

VERSION = 'PHASE5G_STAGE3_FROZEN_CONFIRMATORY_PLAN_V1'
PROTOCOL_SHA256 = '6edad86fb2f49328784c5e80ef60d967413f6bf4f30f64debf57ad3381ce1d43'
PROJECT = Path(__file__).resolve().parent
ARCHS = ('mlp', 'gnn')
SEEDS = (7, 17, 27)
FINAL_TOPOLOGIES = tuple(range(801, 821))
FINAL_STREAMS = (77000, 77001, 77002)
REQUIRED_CHECKPOINTS = {
    'gnn': {'seed': 27, 'episode': 48, 'sha': '7db8b92ab9bc1648a9224eea727efe035848543fe514c7d7c14be3ee584658c9', 'params': 15778},
    'mlp': {'seed': 17, 'episode': 24, 'sha': '942c8edd516ad28b4012733bfd8504af01fde50f47444be7d15c8e02995e3363', 'params': 15758},
}
FROZEN_F = {
    'C_frozen': ('C_enhanced_seed27.pt', 'e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9'),
    'D_prior_archive': ('D_full_hybrid_seed7.pt', 'e55b7b551b09bffe036d5bb75a471d87bb75568b48fa64aac483f947454e8e44'),
}
DOC_ONLY = {
    'phase5g_stage2/__init__.py': ('original___init__.py','8c31570c5588b32a3b22d0bcd23df60d417b778028ba2a197dd2de1f57bfea68'),
    'phase5g_stage2/workflow.py': ('original_workflow.py','d3d2568e0a556104765db2651c849f909af01652517484a3f725851a0ed3690e'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def equal(a, b, message):
    if a != b:
        raise ValueError(f'{message}: expected={b!r}; observed={a!r}')


def normalized_ast(path):
    """Erase *only* module/class/function initial string docstrings, not other strings."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    class Strip(ast.NodeTransformer):
        def visit_Module(self, node):
            self.generic_visit(node)
            self._strip(node)
            return node
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            self._strip(node)
            return node
        def visit_AsyncFunctionDef(self, node):
            self.generic_visit(node)
            self._strip(node)
            return node
        def visit_ClassDef(self, node):
            self.generic_visit(node)
            self._strip(node)
            return node
        @staticmethod
        def _strip(node):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                del node.body[0]
    return ast.dump(Strip().visit(tree), annotate_fields=True, include_attributes=False)


def source_lock(reference_sources):
    """Reject every source modification other than the exact previously audited docstrings."""
    current = {}
    for rel, expected in reference_sources.items():
        f = PROJECT / rel
        if not f.is_file():
            raise FileNotFoundError('Frozen evaluation source missing: '+str(f))
        current[rel] = digest(f)
        equal(current[rel], expected, 'Current source fingerprint mismatch: '+rel)
    for rel, (snapshot_name, original_sha) in DOC_ONLY.items():
        original = PROJECT / 'stage3_reference' / snapshot_name
        current_f = PROJECT / rel
        equal(digest(original), original_sha, 'Original audited source reference mismatch: '+rel)
        equal(normalized_ast(original), normalized_ast(current_f), 'Algorithm AST differs: '+rel)
    return current


def verify_checkpoint(path, *, arch, seed, episode, score, expected_sha, expected_params, source_fingerprints):
    import torch  # Only safe weights-only deserialization; prior SHA check always first.
    equal(digest(path), expected_sha, 'Checkpoint SHA256 mismatch: '+str(path))
    data = torch.load(path, map_location='cpu', weights_only=True)
    equal(data['role'], 'PHASE5G_STAGE2_VALIDATION_SELECTED_NOT_HELDOUT', 'Wrong checkpoint role')
    equal(data['architecture'], arch, 'Wrong checkpoint architecture')
    equal(int(data['agent_seed']), seed, 'Wrong checkpoint seed')
    equal(int(data['episode']), episode, 'Wrong checkpoint episode')
    if abs(float(data['validation_mean']) - score) > 1e-10:
        raise ValueError('Checkpoint score differs from saved validation score: '+str(path))
    equal(data['sources'], source_fingerprints, 'Checkpoint training source fingerprints differ')
    equal(int(data['model_parameters']), expected_params, 'Wrong checkpoint parameter count')
    model_state = data['model_state']
    counted = sum(int(t.numel()) for t in model_state.values())
    equal(counted, expected_params, 'Unexpected number of model weights')
    if not all(torch.isfinite(t).all().item() for t in model_state.values()):
        raise ValueError('Non-finite model weights: '+str(path))


def _get_selection(selection_root):
    sroot = Path(selection_root)
    manifest_path = sroot / 'selection_manifest.json'
    manifest = read_json(manifest_path)
    equal(manifest['role'], 'VALIDATION_ONLY_MODEL_SELECTION', 'Selection role')
    equal(manifest['selection_method'], 'Exact original Stage2 3-seed mean and tie preferences', 'Selection rule')
    equal(manifest.get('winning_architecture'), 'gnn', 'Expected selected overall architecture')
    equal(manifest.get('agent_seeds'), [7,17,27], 'Selection seed roster')
    equal(manifest.get('trained_architectures'), ['mlp','gnn'], 'Selection architecture roster')
    equal(manifest.get('no_final_data_access'), True, 'Selection final-access status')
    equal(manifest.get('protection_model_approved'), False, 'Protection abstraction approval flag')
    report = read_json(sroot / 'source_equivalence_report.json')
    equal(report.get('altered_training_or_provenance_files'), False, 'Training file integrity exception')
    equal(report.get('protocol'), 'PHASE5G_STAGE2_EXACT_DOCSTRING_COMMENT_EQUIVALENCE_V1', 'Docstring exception protocol')
    for rel in DOC_ONLY:
        equal(report['code_file_differences'][rel]['ast_equal_after_removing_docstrings'], True,
              'Recorded AST source equivalence')
    return manifest, report


def evaluate_existing(runs_root, selection_root, *, echo=True):
    """Read and validate six completed runs and provenance-aware selection, no writes."""
    runs_root = Path(runs_root)
    selection, report = _get_selection(selection_root)
    provs = {}
    plans = []
    exact_accepted_sources = None
    rows = []
    per_source_snapshots = {row['run']:row for row in selection['original_source_snapshots_per_run']}
    equal(set(per_source_snapshots), {f'{a}_seed{s}' for a in ARCHS for s in SEEDS},
          'Six-run source snapshot roster')
    for arch in ARCHS:
        for seed in SEEDS:
            name = f'{arch}_seed{seed}'
            folder = runs_root / name
            p = read_json(folder / 'training_provenance.json')
            equal(p['architecture'], arch, name+' architecture')
            equal(p['agent_seed'], seed, name+' seed')
            equal(p['role'], 'TRAINING_AND_VALIDATION_ONLY', name+' training role')
            equal(p['episodes'], 48, name+' completed episodes')
            equal(p['final_seeds_accessed'], False, name+' no-final-access flag')
            equal(p['msrp_compliant'], False, name+' MSRP flag')
            equal(p['frozen_C_sha256'], FROZEN_F['C_frozen'][1], name+' prior C SHA')
            equal(p['plan']['family'], 'hierarchical', name+' validation topology family')
            equal(p['plan']['val_topologies'], [701,702,703], name+' validation topologies')
            equal(p['plan']['val_streams'], [76000,76001], name+' validation streams')
            equal(p['plan']['train_topologies'], list(range(601,631)), name+' training topologies')
            equal(p['plan']['train_streams'], list(range(75000,75020)), name+' training streams')
            equal(p['plan']['headroom'], 10, name+' headroom')
            equal(p['plan']['k'], 6, name+' routing candidates')
            equal(p['plan']['max_working_paths'], 40, name+' route search bound')
            plans.append({k:v for k,v in p['plan'].items() if k not in ('architecture','agent_seed')})
            equal(p['source_sha256'], per_source_snapshots[name]['original_source_sha256'],
                  name+' preserved original run-level provenance')
            if exact_accepted_sources is None and name == 'gnn_seed7':
                exact_accepted_sources = p['source_sha256']
            for rel, got in p['source_sha256'].items():
                if rel in DOC_ONLY:
                    equal(got in {DOC_ONLY[rel][1], report['code_file_differences'][rel]['current_sha256']}, True,
                          'Unrecognized audited docstring version: '+name+'/'+rel)
                else:
                    equal(got, report['run_audits'][0]['original_source_sha256'][rel],
                          'Unaudited source variation: '+name+'/'+rel)
            ck = folder / p['best_checkpoint']
            trace = folder / p['best_trace']
            equal(digest(ck), p['best_sha256'], name+' best checkpoint bytes')
            equal(digest(trace), p['best_trace_sha256'], name+' best trace bytes')
            valrows = list(csv.DictReader((folder/'validation_summary.csv').open('r',encoding='utf-8',newline='')))
            target = [r for r in valrows if int(r['episode'])==p['best_episode'] and r['policy']==arch+'_PPO']
            equal(len(target), 6, name+' selected validation cohort count')
            seen = {(int(r['topology_seed']),int(r['stream_seed'])) for r in target}
            equal(seen, {(t,s) for t in [701,702,703] for s in [76000,76001]}, name+' validation coverage')
            if any(int(r['feasibility_violations']) != 0 for r in valrows):
                raise ValueError(name+' validation contains recorded feasibility violations')
            score = mean(float(r['weighted_admission']) for r in target)
            if abs(score-p['best_score']) > 1e-10:
                raise ValueError(name+' validation score recomputation failed')
            verify_checkpoint(ck, arch=arch,seed=seed,episode=p['best_episode'],score=score,
                              expected_sha=p['best_sha256'],expected_params=p['param_count'],
                              source_fingerprints=p['source_sha256'])
            snapshot = per_source_snapshots[name]
            equal(snapshot['checkpoint_sha256'],p['best_sha256'],name+' selection-provenance checkpoint')
            equal(snapshot['best_trace_sha256'],p['best_trace_sha256'],name+' selection-provenance trace')
            provs[name] = p
            rows.append({'architecture':arch,'agent_seed':seed,'best_episode':p['best_episode'],
                         'best_weighted':score,'checkpoint_sha256':p['best_sha256']})
            if echo:print(f'PHASE5G_STAGE3_RUN_OK arch={arch} seed={seed} episode={p["best_episode"]} score={score:.8f}')
    if not all(plan==plans[0] for plan in plans):
        raise ValueError('Stage 2 six-run training settings are not matched')
    base_sources = provs['gnn_seed27']['source_sha256']
    source_lock(base_sources)
    for arch in ARCHS:
        available = sorted((r for r in rows if r['architecture']==arch), key=lambda r:(-r['best_weighted'],r['agent_seed']))
        selected = available[0]
        registered = selection['selected_models'][arch]
        for field in ('agent_seed','best_episode','checkpoint_sha256'):
            key = {'agent_seed':'agent_seed','best_episode':'best_episode','checkpoint_sha256':'checkpoint_sha256'}[field]
            equal(selected[field],registered[key],arch+' selected '+field)
        if abs(registered['best_weighted'] - selected['best_weighted'])>1e-10:
            raise ValueError(arch+' selected score mismatch')
        required = REQUIRED_CHECKPOINTS[arch]
        for key,x in [('agent_seed','seed'),('best_episode','episode'),('checkpoint_sha256','sha')]:
            equal(registered[key],required[x],arch+' previously independently audited winner')
        scores = [r['best_weighted'] for r in rows if r['architecture']==arch]
        if echo:print(f'PHASE5G_STAGE3_ARCH_OK arch={arch} mean_best={mean(scores):.8f}')
    if mean(r['best_weighted'] for r in rows if r['architecture']=='gnn') <= mean(r['best_weighted'] for r in rows if r['architecture']=='mlp'):
        raise ValueError('Expected validation-only architecture winner is inconsistent')
    equal(selection['selected_overall'],selection['selected_models']['gnn'],'Overall winner identity')
    frozen_dir=PROJECT/'stage5_frozen'
    frozen_old=read_json(frozen_dir/'selection_manifest.json')
    equal(frozen_old['selected']['checkpoint_sha256'], FROZEN_F['C_frozen'][1], 'Old C original selection')
    equal(frozen_old['proposed_full_hybrid']['checkpoint_sha256'], FROZEN_F['D_prior_archive'][1], 'Old D original selection')
    for label,(name,sha) in FROZEN_F.items():
        equal(digest(frozen_dir/name),sha,'Prior frozen model bytes: '+label)
    protocol=PROJECT/'PHASE5G_STAGE3_PROTOCOL_LOCK.md'
    if VERSION not in protocol.read_text(encoding='utf-8') or digest(protocol) != PROTOCOL_SHA256:
        raise ValueError('Stage 3 protocol file differs from preregistered package bytes; do not silently amend')
    return {'selection':selection,'provenances':provs,'rows':rows,'sources':base_sources,
            'protocol_sha256':digest(protocol)}


def freeze(runs_root,selection_root,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Refusing to overwrite an existing frozen result: '+str(out))
    data=evaluate_existing(runs_root,selection_root)
    out.mkdir(parents=True,exist_ok=False)
    try:
        file_hashes={}
        def save(source, rel):
            dest=out/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
            file_hashes[rel]=digest(dest)
        for arch in ARCHS:
            key=f'{arch}_seed{REQUIRED_CHECKPOINTS[arch]["seed"]}'
            p=data['provenances'][key]
            save(Path(runs_root)/key/p['best_checkpoint'],f'models/{arch}_PPO.pt')
            save(Path(runs_root)/key/p['best_trace'],f'traces/{arch}_selected_validation_trace.csv.gz')
        for label,(name,_) in FROZEN_F.items():
            save(PROJECT/'stage5_frozen'/name,'models/'+label+'.pt')
        for key in data['provenances']:
            save(Path(runs_root)/key/'training_provenance.json',f'stage2_provenance/{key}.json')
        for f in ('selection_manifest.json','selection_architectures.csv','selection_trials.csv','source_equivalence_report.json'):
            save(Path(selection_root)/f,'stage2_selection/'+f)
        for f in ('PHASE5G_STAGE3_PROTOCOL_LOCK.md','run_phase5g_stage3.py'):
            save(PROJECT/f,'freeze_sources/'+f)
        manifest={
            'version':VERSION,'role':'FROZEN_PRE_HELDOUT_NO_EVALUATION',
            'heldout_accessed_by_stage3':False,'stage3_network_generation':False,
            'selected':{
                arch:{'agent_seed':REQUIRED_CHECKPOINTS[arch]['seed'],
                      'episode':REQUIRED_CHECKPOINTS[arch]['episode'],
                      'sha256':REQUIRED_CHECKPOINTS[arch]['sha'],
                      'parameters':REQUIRED_CHECKPOINTS[arch]['params'],
                      'source_run':f'{arch}_seed{REQUIRED_CHECKPOINTS[arch]["seed"]}'}
                for arch in ARCHS},
            'old_C_sha256':FROZEN_F['C_frozen'][1],
            'older_D_archived_not_tested_sha256':FROZEN_F['D_prior_archive'][1],
            'protection_model_approved':False,'msrp_compliant':False,
            'real_railway_topology':False,
            'network_families':['hierarchical','ladder'],
            'stress_arms':['base','bandwidth_150','short_hold'],
            'final_topology_seeds':list(FINAL_TOPOLOGIES),
            'final_stream_seeds':list(FINAL_STREAMS),
            'locked_cell_count':8,'locked_matched_cohorts':480,
            'locked_policy_runs':2880,
            'policies':['gnn_PPO','mlp_PPO','C_frozen','scenario_minimax','footprint','greedy'],
            'source_sha256':data['sources'],
            'protocol_sha256':data['protocol_sha256'],
            'files_sha256':file_hashes,
        }
        (out/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
        verify(out)
    except Exception:
        # Do NOT erase evidence from a partially failed freeze: preserve it for inspection.
        raise
    print('PHASE5G_STAGE3_FREEZE_OK selected_gnn=seed27_ep48 selected_mlp=seed17_ep24 heldout_accessed=False')
    return manifest


def verify(out):
    root=Path(out)
    manifest=read_json(root/'freeze_manifest.json')
    equal(manifest['version'],VERSION,'Stage 3 freeze version')
    equal(manifest['role'],'FROZEN_PRE_HELDOUT_NO_EVALUATION','Stage 3 freeze role')
    equal(manifest['heldout_accessed_by_stage3'],False,'Stage 3 final-data access')
    equal(manifest['stage3_network_generation'],False,'Stage 3 network generation')
    equal(manifest['final_topology_seeds'],list(FINAL_TOPOLOGIES),'Final topology reservation')
    equal(manifest['final_stream_seeds'],list(FINAL_STREAMS),'Final stream reservation')
    equal(manifest['locked_cell_count'],8,'Locked cell count')
    equal(manifest['locked_matched_cohorts'],480,'Locked cohort count')
    equal(manifest['locked_policy_runs'],2880,'Locked policy run count')
    equal(manifest['protection_model_approved'],False,'Protection model approval')
    equal(manifest['old_C_sha256'],FROZEN_F['C_frozen'][1],'Old C lock')
    equal(manifest['protocol_sha256'],PROTOCOL_SHA256,'Frozen Stage 3 protocol differs from preregistered bytes')
    equal(manifest['protocol_sha256'],digest(PROJECT/'PHASE5G_STAGE3_PROTOCOL_LOCK.md'),'Protocol changed since freeze')
    equal(manifest['files_sha256']['freeze_sources/PHASE5G_STAGE3_PROTOCOL_LOCK.md'],
          manifest['protocol_sha256'],'Frozen protocol snapshot not equal to locked protocol')
    equal(manifest['files_sha256']['freeze_sources/run_phase5g_stage3.py'],
          digest(PROJECT/'run_phase5g_stage3.py'),'Stage3 freeze builder source changed')
    for rel, sha in manifest['files_sha256'].items():
        equal(digest(root/rel),sha,'Frozen artifact altered: '+rel)
    source_lock(manifest['source_sha256'])
    for arch in ARCHS:
        row=manifest['selected'][arch]
        required=REQUIRED_CHECKPOINTS[arch]
        for field, expected in [('agent_seed',required['seed']),('episode',required['episode']),
                                ('sha256',required['sha']),('parameters',required['params'])]:
            equal(row[field],expected,'Frozen '+arch+' '+field)
        p=read_json(root/f'stage2_provenance/{arch}_seed{required["seed"]}.json')
        equal(row['sha256'],p['best_sha256'],arch+' provenance SHA')
        equal(digest(root/f'models/{arch}_PPO.pt'),row['sha256'],arch+' frozen model SHA')
        equal(digest(root/f'traces/{arch}_selected_validation_trace.csv.gz'),
              p['best_trace_sha256'],arch+' frozen decision trace SHA')
    for label,(_,sha) in FROZEN_F.items():
        equal(digest(root/f'models/{label}.pt'),sha,label+' prior model SHA')
    print('PHASE5G_STAGE3_VERIFY_OK models=gnn,mlp,C_frozen prior_D_archived=True final_seeds_not_accessed=True')
    return manifest


def main():
    parser=argparse.ArgumentParser(description='Phase 5G Stage3: no training, no heldout generation')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','freeze'):
        s=sub.add_parser(name)
        s.add_argument('--runs-root',default='outputs/phase5g/stage2_train')
        s.add_argument('--selection-root',default='outputs/phase5g/stage2_selection_PROVENANCE_AUDITED')
        if name=='freeze':s.add_argument('--out',default='outputs/phase5g/stage3_FROZEN')
    v=sub.add_parser('verify');v.add_argument('--root',default='outputs/phase5g/stage3_FROZEN')
    args=parser.parse_args()
    if args.command=='preflight':
        evaluate_existing(args.runs_root,args.selection_root)
        print('PHASE5G_STAGE3_PREFLIGHT_OK six_runs=6 models=2 prior_C_verified=True heldout_accessed=False')
    elif args.command=='freeze':freeze(args.runs_root,args.selection_root,args.out)
    elif args.command=='verify':verify(args.root)


if __name__=='__main__':main()
