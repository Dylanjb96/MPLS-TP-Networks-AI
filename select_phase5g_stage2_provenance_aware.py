#!/usr/bin/env python3
"""Read-only validation of 3 audited Stage-2 source snapshots; emit separate model selection.

Does not modify training outputs or source code. Only source-drift exception is the
exact, already audited docstring-to-comment substitution in 2 known files.
All training file provenance and checkpoint-embedded source fingerprints remain
unaltered and must agree with the originating run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
ARCHS = ('mlp', 'gnn')
SEEDS = (7, 17, 27)
FROZEN_C = 'e33f7837022671f21673f90b9000f2fdbb9abe0b7c2ae951f6f081e2925586c9'
SOURCE_RELS = ('phase5g_stage2/__init__.py', 'phase5g_stage2/workflow.py')
HASHES = {
    'phase5g_stage2/__init__.py': {
        'original': '8c31570c5588b32a3b22d0bcd23df60d417b778028ba2a197dd2de1f57bfea68',
        'audited_current': 'aaa0c17a961624123af9a49ba280da4a4afe4472d043541e6c909d07fc22d58f',
    },
    'phase5g_stage2/workflow.py': {
        'original': 'd3d2568e0a556104765db2651c849f909af01652517484a3f725851a0ed3690e',
        'audited_current': '5c6ca89a1459c6403bccda5aaa0535797a8559e3290895936d943f6e41cd4533',
    },
}
# Frozen from the submitted 6 training_provenance.json files. NEVER infer a version.
RUN_VARIANTS = {
    'mlp_seed7': ('original', 'original'),
    'mlp_seed17': ('original', 'audited_current'),
    'mlp_seed27': ('original', 'audited_current'),
    'gnn_seed7': ('audited_current', 'audited_current'),
    'gnn_seed17': ('audited_current', 'audited_current'),
    'gnn_seed27': ('audited_current', 'audited_current'),
}


def require(condition, message):
    if not condition:
        raise RuntimeError('SOURCE_AWARE_SELECTION_BLOCKED: ' + message)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_docstrings(source):
    root = ast.parse(source)
    for node in ast.walk(root):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                expr = node.body[0].value
                if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                    node.body = node.body[1:]
    return ast.dump(root, annotate_fields=True, include_attributes=False)


def audited_equivalence(project_root):
    result = {}
    for rel in SOURCE_RELS:
        fn = rel.rsplit('/', 1)[-1]
        ref = ROOT / 'phase5g_stage2_source_reference'
        orig = ref / ('original_' + fn)
        changed = ref / ('audited_current_' + fn)
        require(orig.is_file() and changed.is_file(), 'missing audited reference snapshots: ' + rel)
        require(digest_file(orig) == HASHES[rel]['original'], 'original reference hash differs: ' + rel)
        require(digest_file(changed) == HASHES[rel]['audited_current'], 'current reference hash differs: ' + rel)
        current = Path(project_root) / rel
        require(current.is_file(), 'missing installed source: ' + rel)
        require(digest_file(current) == HASHES[rel]['audited_current'],
                'installed file differs from audited current source: ' + rel)
        oa = normalize_docstrings(orig.read_text(encoding='utf8'))
        ca = normalize_docstrings(changed.read_text(encoding='utf8'))
        require(oa == ca, 'executable AST changed beyond docstrings: ' + rel)
        require(normalize_docstrings(current.read_text(encoding='utf8')) == oa,
                'current source executable AST differs: ' + rel)
        tree = ast.parse(current.read_text(encoding='utf8'))
        require(not any(isinstance(n, ast.Attribute) and n.attr == '__doc__'
                        for n in ast.walk(tree)), '__doc__ attribute accessed in: ' + rel)
        result[rel] = dict(original_sha256=HASHES[rel]['original'],
                           current_sha256=HASHES[rel]['audited_current'],
                           ast_equal_after_removing_docstrings=True,
                           docstrings_can_differ=True)
    # The other 13 source paths must match training provenance byte-for-byte;
    # no general-purpose 'ignore source mismatch' option exists in this script.
    return result


def rows_csv(path):
    require(Path(path).is_file(), 'missing CSV: ' + str(path))
    with Path(path).open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def load_json(path):
    require(Path(path).is_file(), 'missing JSON: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf8'))


def close(a, b, tol=1e-10):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def checkpoint_read(path):
    # weights_only prevents untrusted arbitrary pickle objects executing on load.
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def run_checks(runs_root, project_root):
    from phase5g_stage2.workflow import sources, schedule, Plan, verify_freeze, SOURCE_FILES
    audit = audited_equivalence(project_root)
    current_sources = sources()
    require(len(current_sources) == 15 and set(current_sources) == set(SOURCE_FILES),
            'unrecognized installed Stage-2 source-file manifest')
    require(all(current_sources[r] == HASHES[r]['audited_current'] for r in SOURCE_RELS),
            'installed known source versions do not match reference')
    frozen = verify_freeze()
    require(frozen['selected']['checkpoint_sha256'] == FROZEN_C, 'frozen Phase 5F C altered')
    root = Path(runs_root)
    selection_rows, run_audits = [], []
    validation_fingerprints = {}
    shared_protocol = None
    for arch in ARCHS:
        for seed in SEEDS:
            run_id = f'{arch}_seed{seed}'
            folder = root / run_id
            p = load_json(folder / 'training_provenance.json')
            plan = p.get('plan')
            require(isinstance(plan, dict), run_id + ': missing plan')
            require(p.get('version') == 'PHASE5G_STAGE2_MATCHED_PPO_DEV_V1'
                    and p.get('role') == 'TRAINING_AND_VALIDATION_ONLY', run_id + ': unexpected role/version')
            require(p.get('architecture') == arch and p.get('agent_seed') == seed
                    and p.get('episodes') == 48 and plan.get('episodes') == 48,
                    run_id + ': architecture, seed or full training episode count differs')
            require(p.get('final_seeds_accessed') is False, run_id + ': final-access provenance flag')
            require(p.get('frozen_C_sha256') == FROZEN_C, run_id + ': frozen C fingerprint differs')
            require(p.get('param_count') == (15758 if arch == 'mlp' else 15778),
                    run_id + ': neural model parameter count differs')
            expected = dict(current_sources)
            v_init, v_workflow = RUN_VARIANTS[run_id]
            expected[SOURCE_RELS[0]] = HASHES[SOURCE_RELS[0]][v_init]
            expected[SOURCE_RELS[1]] = HASHES[SOURCE_RELS[1]][v_workflow]
            recorded = p.get('source_sha256')
            require(recorded == expected, run_id + ': source fingerprints differ from known audited snapshot')
            common = {k: v for k, v in plan.items() if k not in ('architecture', 'agent_seed')}
            if shared_protocol is None:
                shared_protocol = common
            else:
                require(common == shared_protocol, run_id + ': six-run training protocol differs')
            settings = Plan(**plan)
            require(settings.description() == plan, run_id + ': plan validation/deserialization differs')
            tplan = load_json(folder / 'training_plan.json')
            require(tplan['plan'] == plan and tplan['sources'] == recorded,
                    run_id + ': original training_plan.json disagrees')
            require(tplan['selected_frozen_C_sha256'] == FROZEN_C,
                    run_id + ': training_plan frozen checkpoint differs')
            history = rows_csv(folder / 'training_history.csv')
            require(len(history) == 48, run_id + ': incomplete training history')
            for i, (row, sch) in enumerate(zip(history, schedule(settings)), 1):
                require(int(row['episode']) == i and row['architecture'] == arch and int(row['agent_seed']) == seed,
                        run_id + f': training history episode {i} identity differs')
                for key in ('nodes', 'topology_seed', 'stream_seed'):
                    require(int(row[key]) == sch[key], run_id + f': episode {i} {key} differs')
                require(close(row['load'], sch['load']) and row['mode'] == sch['mode']
                        and row['family'] == 'hierarchical', run_id + f': schedule episode {i} differs')
                require(int(row['admitted']) <= int(row['requests']) and int(row['requests']) > 0,
                        run_id + f': invalid request counts at episode {i}')
            interactions = sum(int(r['train_interactions']) for r in history)
            updates = sum(int(r['update_samples']) for r in history)
            require(interactions == p['total_interactions'] and updates == p['total_update_samples'],
                    run_id + ': training counters do not match history')
            state = checkpoint_read(folder / 'latest_state.pt')
            require(state.get('sources') == recorded and state.get('plan') == plan and
                    state.get('episode') == 48 and state.get('version') == p['version'],
                    run_id + ': final state journal differs from original run provenance')
            require(state.get('best_episode') == p['best_episode'] and
                    close(state.get('best_score'), p['best_score']) and
                    state.get('best_hash') == p['best_sha256'] and
                    state.get('best_path') == p['best_checkpoint'] and
                    state.get('best_trace_sha256') == p['best_trace_sha256'] and
                    state.get('best_trace_path') == p['best_trace'],
                    run_id + ': completed state journal does not agree with best checkpoint')
            ck = folder / p['best_checkpoint']
            require(ck.is_file() and digest_file(ck) == p['best_sha256'],
                    run_id + ': best checkpoint SHA mismatch')
            checkpoint = checkpoint_read(ck)
            require(checkpoint.get('role') == 'PHASE5G_STAGE2_VALIDATION_SELECTED_NOT_HELDOUT'
                    and checkpoint.get('version') == p['version']
                    and checkpoint.get('architecture') == arch
                    and checkpoint.get('agent_seed') == seed
                    and checkpoint.get('episode') == p['best_episode']
                    and close(checkpoint.get('validation_mean'), p['best_score'])
                    and checkpoint.get('sources') == recorded  # IMPORTANT: trained snapshot, not current
                    and checkpoint.get('plan') == plan
                    and checkpoint.get('model_parameters') == p['param_count']
                    and checkpoint.get('frozen_C_sha256') == FROZEN_C,
                    run_id + ': checkpoint identity/embedded source provenance mismatch')
            trace_file = folder / p['best_trace']
            require(trace_file.is_file() and digest_file(trace_file) == p['best_trace_sha256'],
                    run_id + ': decision-trace SHA mismatch')
            with gzip.open(trace_file, 'rt', newline='', encoding='utf8') as f:
                trace = list(csv.DictReader(f))
            require(len(trace) == len(plan['val_topologies'])*len(plan['val_streams'])*round(plan['val_size']*plan['val_load']),
                    run_id + ': best decision trace has wrong number of rows')
            require(all(int(r['episode']) == p['best_episode'] and
                        r['architecture'] == arch and int(r['agent_seed']) == seed for r in trace),
                    run_id + ': best trace metadata differs')
            checks = rows_csv(folder / 'validation_checkpoints.csv')
            validation = rows_csv(folder / 'validation_summary.csv')
            expected_episodes = (12, 24, 36, 48)
            require(len(checks) == 4 and [int(r['episode']) for r in checks] == list(expected_episodes),
                    run_id + ': validation checkpoint schedule incomplete')
            expected_n = len(plan['val_topologies'])*len(plan['val_streams'])
            require(len(validation) == 4*5*expected_n, run_id + ': validation summary incomplete')
            improvement = -float('inf')
            winner_ep, winner_score = None, None
            for row in checks:
                ep = int(row['episode'])
                vr = [v for v in validation if int(v['episode']) == ep]
                require(len(vr) == 5*expected_n, run_id + f': validation ep{ep} row count differs')
                policy_counts = Counter(v['policy'] for v in vr)
                require(policy_counts == {x: expected_n for x in
                        ('greedy','footprint','scenario_minimax','C_frozen',arch+'_PPO')},
                        run_id + f': validation ep{ep} policy balance differs')
                expected_cohorts = {(topo, stream) for topo in plan['val_topologies'] for stream in plan['val_streams']}
                actual_cohorts = {(int(v['topology_seed']), int(v['stream_seed'])) for v in vr}
                require(expected_cohorts == actual_cohorts, run_id + f': validation ep{ep} cohorts differ')
                for topo, stream in expected_cohorts:
                    cr = [v for v in vr if int(v['topology_seed']) == topo and int(v['stream_seed']) == stream]
                    require(len(cr) == 5 and len({v['policy'] for v in cr}) == 5 and
                            all(int(v['feasibility_violations']) == 0 for v in cr) and
                            len({v['network_sha256'] for v in cr}) == 1 and
                            len({v['request_sha256'] for v in cr}) == 1 and
                            len({v['candidates_sha256'] for v in cr}) == 1,
                            run_id + f': unmatched/infeasible validation ({topo},{stream}) ep{ep}')
                    signature = tuple(cr[0][q] for q in
                        ('network_sha256', 'request_sha256', 'candidates_sha256'))
                    key = (topo, stream)
                    old = validation_fingerprints.setdefault(key, signature)
                    require(old == signature, run_id + ': cross-run candidate cohort fingerprint differs')
                averages = {policy: mean(float(v['weighted_admission']) for v in vr if v['policy'] == policy)
                            for policy in policy_counts}
                require(close(averages[arch+'_PPO'], row['score']) and
                        all(close(averages[key], row[key+'_score']) for key in
                            ('greedy', 'footprint', 'C_frozen') if key != 'C_frozen') and
                        close(averages['scenario_minimax'], row['minimax_score']) and
                        close(averages['C_frozen'], row['frozen_C_score']),
                        run_id + f': validation summary/checkpoint means differ ep{ep}')
                improved = averages[arch+'_PPO'] > improvement + 1e-12
                require(int(row['improved']) == int(improved),
                        run_id + f': best episode selection rule differs ep{ep}')
                if improved:
                    improvement = averages[arch+'_PPO']
                    winner_ep, winner_score = ep, improvement
                    ck_for_ep = folder / f'checkpoints/best_episode_{ep:04d}.pt'
                    require(ck_for_ep.is_file() and digest_file(ck_for_ep) == row['checkpoint_sha256'],
                            run_id + f': checkpoint hash differs at improvement episode {ep}')
            require(winner_ep == p['best_episode'] and close(winner_score, p['best_score']),
                    run_id + ': best validation episode/score incorrect')
            selection_rows.append(dict(architecture=arch, agent_seed=seed,
                best_episode=p['best_episode'], best_weighted=p['best_score'],
                checkpoint=str(ck), checkpoint_sha256=p['best_sha256'],
                training_interactions=interactions, update_samples=updates))
            run_audits.append(dict(run=run_id, code_variant_init=v_init,
                code_variant_workflow=v_workflow,
                original_source_sha256=recorded,
                checkpoint_sha256=p['best_sha256'], best_trace_sha256=p['best_trace_sha256'],
                validation_best_weighted=p['best_score'], validation_best_episode=p['best_episode'],
                training_episodes=len(history), validation_rows=len(validation),
                source_accepted_under_exact_audited_docstring_exception=True))
            print('SOURCE_AWARE_RUN_OK run=' + run_id + ' best_episode=' + str(p['best_episode']) +
                  ' validation_weighted=' + f'{p["best_score"]:.8f}', flush=True)
    summary = []
    for arch in ARCHS:
        subset = [r for r in selection_rows if r['architecture'] == arch]
        vals = [r['best_weighted'] for r in subset]
        summary.append(dict(architecture=arch, mean_best_weighted=mean(vals),
                            sd_best_weighted=stdev(vals), runs=3))
    winner = max(summary, key=lambda a: (a['mean_best_weighted'], -ARCHS.index(a['architecture'])))
    winners = {arch: sorted([x for x in selection_rows if x['architecture'] == arch],
                           key=lambda x: (-x['best_weighted'], x['agent_seed']))[0]
               for arch in ARCHS}
    return dict(selection_rows=selection_rows, summary=summary, winner=winner,
                chosen=winners, run_audits=run_audits, source_audit=audit)


def write_outputs(root, result):
    root = Path(root)
    require(not root.exists(), 'out directory already exists; refusing to overwrite: ' + str(root))
    root.mkdir(parents=True, exist_ok=False)
    def write_csv(name, records):
        with (root/name).open('w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader(); w.writerows(records)
    def write_json(name, data):
        (root/name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf8')
    write_csv('selection_trials.csv', result['selection_rows'])
    write_csv('selection_architectures.csv', result['summary'])
    write_json('source_equivalence_report.json', dict(
        protocol='PHASE5G_STAGE2_EXACT_DOCSTRING_COMMENT_EQUIVALENCE_V1',
        explanation='The two affected files changed only module/class/function docstrings to comments. ',
        code_file_differences=result['source_audit'], run_audits=result['run_audits'],
        altered_training_or_provenance_files=False,
        no_final_heldout_data_used=True,
        original_selector_remains_unchanged=True))
    write_json('selection_manifest.json', dict(
        version='PHASE5G_STAGE2_PROVENANCE_AWARE_VALIDATION_SELECTION_V1',
        role='VALIDATION_ONLY_MODEL_SELECTION',
        selection_method='Exact original Stage2 3-seed mean and tie preferences',
        source_exception='Exact proven docstring-to-comment source differences, no executable AST differences',
        source_equivalence_report='source_equivalence_report.json',
        original_source_snapshots_per_run=result['run_audits'],
        trained_architectures=list(ARCHS),agent_seeds=list(SEEDS),
        winning_architecture=result['winner']['architecture'],
        selected_models=result['chosen'],
        selected_overall=result['chosen'][result['winner']['architecture']],
        no_final_data_access=True,synthetic_not_real_railway=True,
        protection_model_approved=False,
        selection_rule='highest 3-seed mean of validation-best scores, tie prefer mlp; per model seed highest validation score then lowest seed',
        caution='This is an independently provenance-reconciled selection, not an execution of the unchanged strict selector. Preserve full run-level source SHA256.'))
    print('PHASE5G_STAGE2_SOURCE_AWARE_SELECTION_OK winner=' + result['winner']['architecture'] +
          ' mean=' + f'{result["winner"]["mean_best_weighted"]:.8f}' +
          ' output=' + str(root), flush=True)


def selftest():
    # Independent code-version proof; deliberately does not open training or validation data.
    result = audited_equivalence(ROOT)
    # Selftest on package can run outside MPLS-AI project if the two known current
    # source snapshots are in the bundled reference directory.
    print('SOURCE_AWARE_SEMANTIC_EQUIVALENCE_OK affected_files=' + str(len(result)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', default='outputs/phase5g/stage2_train')
    parser.add_argument('--out', default='outputs/phase5g/stage2_selection_PROVENANCE_AUDITED')
    parser.add_argument('--selftest', action='store_true', help='check audited source equivalence; no run data')
    args = parser.parse_args()
    if args.selftest:
        # For a selftest allow both refs to be used in place of project installed source.
        for rel in SOURCE_RELS:
            fn=rel.rsplit('/',1)[-1]
            orig=ROOT/'phase5g_stage2_source_reference'/('original_'+fn)
            changed=ROOT/'phase5g_stage2_source_reference'/('audited_current_'+fn)
            require(digest_file(orig) == HASHES[rel]['original'] and
                    digest_file(changed) == HASHES[rel]['audited_current'],
                    'selftest reference hash differs: '+rel)
            require(normalize_docstrings(orig.read_text(encoding='utf8')) ==
                    normalize_docstrings(changed.read_text(encoding='utf8')),
                    'selftest executable AST differs: '+rel)
        require(len(RUN_VARIANTS) == 6 and len(HASHES) == 2, 'run snapshot matrix incomplete')
        print('PHASE5G_STAGE2_SOURCE_EQUIVALENCE_SELFTEST_OK affected_files=2 known_runs=6')
        return
    result = run_checks(args.runs_root, ROOT)
    write_outputs(args.out, result)


if __name__ == '__main__':
    main()
