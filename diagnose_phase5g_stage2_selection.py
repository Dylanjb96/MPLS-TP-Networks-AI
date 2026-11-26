"""READ-ONLY Phase 5G Stage 2 selector failure diagnosis.

Run in the MPLS-AI project directory:
    python diagnose_phase5g_stage2_selection.py

No training, validation, held-out data generation, model loading, or edits.
Prints the exact metadata checks corresponding to the selector's error
"invalid run metadata" and file-integrity checks for the six runs.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'outputs' / 'phase5g' / 'stage2_train'
ARCHS = ('mlp', 'gnn')
SEEDS = (7, 17, 27)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def crlf_compatibility(path: Path, wanted: str) -> str:
    """Diagnose pure text newline conversion, without modifying files."""
    b = path.read_bytes()
    if hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest() == wanted:
        return 'YES: converting CRLF to LF alone matches saved hash'
    if hashlib.sha256(b.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest() == wanted:
        return 'YES: converting to CRLF alone matches saved hash'
    return 'no'


def main():
    print('PHASE5G_STAGE2_SELECTION_DIAGNOSIS_READ_ONLY')
    print('PROJECT:', ROOT)
    print('RUNS:', RUNS)
    if not RUNS.is_dir():
        print('ERROR: training directory missing. Run from the project root; place this script there.')
        return 1

    # Find the exact frozen checkpoint fingerprint expected by the current selector.
    # The source selector obtains this from phase5f_stage5.freeze.verify_freeze().
    # Avoid assuming a particular freeze-manifest filename; inspect likely files.
    freeze_candidates = [
        ROOT / 'outputs' / 'phase5f' / 'stage4_selection_FINAL' / 'selection_manifest.json',
        ROOT / 'stage5_frozen' / 'selection_manifest.json',
        ROOT / 'stage5_frozen' / 'stage4_selection_manifest.json',
    ]
    frozen_sha = None
    for file in freeze_candidates:
        if file.is_file():
            try:
                manifest = json.loads(file.read_text(encoding='utf-8'))
                frozen_sha = manifest.get('selected', {}).get('checkpoint_sha256') or frozen_sha
            except (OSError, ValueError, TypeError):
                pass
            if frozen_sha:
                print('FROZEN_MANIFEST:', file.relative_to(ROOT))
                break
    if not frozen_sha:
        print('FROZEN_MANIFEST: not found in conventional locations; will only compare run hashes across all runs')

    saved_sources_by_run = {}
    run_plans = {}
    frozen_by_run = {}
    errors = 0
    for arch in ARCHS:
        for seed in SEEDS:
            name = f'{arch}_seed{seed}'
            folder = RUNS / name
            file = folder / 'training_provenance.json'
            print('\nRUN:', name)
            if not file.is_file():
                print('  FAIL: missing training_provenance.json')
                errors += 1
                continue
            try:
                p = json.loads(file.read_text(encoding='utf-8'))
                plan = p['plan']
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print('  FAIL: malformed provenance:', type(exc).__name__, str(exc))
                errors += 1
                continue
            saved_sources_by_run[name] = p.get('source_sha256', {})
            run_plans[name] = {k: v for k, v in plan.items() if k not in ('architecture', 'agent_seed')}
            frozen_by_run[name] = p.get('frozen_C_sha256')
            fields = {
                'architecture': (p.get('architecture'), arch),
                'agent_seed': (p.get('agent_seed'), seed),
                'episodes': (p.get('episodes'), plan.get('episodes')),
            }
            for key, (got, want) in fields.items():
                ok = got == want
                print(f'  {"OK" if ok else "FAIL"} {key}: provenance={got!r}, expected={want!r}')
                if not ok:
                    errors += 1
            if frozen_sha is not None:
                ok = p.get('frozen_C_sha256') == frozen_sha
                print(f'  {"OK" if ok else "FAIL"} frozen_C_sha256: saved={str(p.get("frozen_C_sha256"))[:16]}, expected={frozen_sha[:16]}')
                if not ok:
                    errors += 1
            else:
                print('  frozen_C_sha256:', p.get('frozen_C_sha256'))

            sources_saved = p.get('source_sha256')
            if not isinstance(sources_saved, dict):
                print('  FAIL source_sha256: missing/not a dict')
                errors += 1
                continue
            expected_files = {
                'phase5f/protection.py', 'phase5f/traffic.py',
                'phase5f_stage2/candidates.py', 'phase5f_stage2/topology.py',
                'phase5f_stage3/agent.py', 'phase5f_stage3/features.py',
                'phase5f_stage3/workflow.py', 'phase5g/__init__.py',
                'phase5g/gnn.py', 'phase5g/workflow.py',
                'phase5g/topology.py', 'phase5g/protocol.py',
                'phase5g_stage2/__init__.py', 'phase5g_stage2/workflow.py',
                'run_phase5g_stage2.py',
            }
            if set(sources_saved) != expected_files:
                print('  FAIL source file list differs from Stage 2 package')
                print('    missing:', sorted(expected_files - set(sources_saved)))
                print('    extra:', sorted(set(sources_saved) - expected_files))
                errors += 1
            mismatched = []
            for rel, wanted in sources_saved.items():
                path = ROOT / rel
                if not path.is_file():
                    mismatched.append((rel, 'MISSING', None, 'n/a'))
                else:
                    got = sha256(path)
                    if got != wanted:
                        if len(mismatched) < 30:
                            compat = crlf_compatibility(path, wanted) if path.suffix in ('.py', '.md', '.txt') else 'n/a'
                            mismatched.append((rel, got[:16], str(wanted)[:16], compat))
            print(f'  {"OK" if not mismatched else "FAIL"} source_sha256: {len(mismatched)} of {len(sources_saved)} files mismatch/missing')
            errors += len(mismatched)
            for rel, got, wanted, compat in mismatched[:12]:
                print(f'    {rel}: current={got}, trained={wanted}, newline_only={compat}')
            if len(mismatched) > 12:
                print(f'    ... plus {len(mismatched) - 12} more file mismatches')
            best = p.get('best_checkpoint')
            if best:
                ck = folder / best
                ok = ck.is_file() and sha256(ck) == p.get('best_sha256')
                print(f'  {"OK" if ok else "FAIL"} best_checkpoint: {best}')
                if not ok:
                    errors += 1
            else:
                print('  FAIL best_checkpoint missing')
                errors += 1
            trace = p.get('best_trace')
            if trace:
                t = folder / trace
                ok = t.is_file() and sha256(t) == p.get('best_trace_sha256')
                print(f'  {"OK" if ok else "FAIL"} best_trace: {trace}')
                if not ok:
                    errors += 1
            else:
                print('  WARNING best_trace missing')
    if run_plans:
        first = next(iter(run_plans))
        bad = [name for name, plan in run_plans.items() if plan != run_plans[first]]
        print('\nSIX_RUN_PROTOCOL_CONSISTENCY:', 'OK' if not bad else 'FAIL ' + ', '.join(bad))
        errors += len(bad)
    if saved_sources_by_run:
        first = next(iter(saved_sources_by_run))
        bad = [name for name, sources in saved_sources_by_run.items() if sources != saved_sources_by_run[first]]
        print('SIX_RUN_SOURCE_PROVENANCE_CONSISTENCY:', 'OK' if not bad else 'DIFFERENT ' + ', '.join(bad))
        errors += len(bad)
    if frozen_by_run:
        unique = set(frozen_by_run.values())
        print('SIX_RUN_FROZEN_C_CONSISTENCY:', 'OK' if len(unique) == 1 else 'FAIL different frozen checkpoints')
        errors += int(len(unique) != 1)
    print('\nDIAGNOSTIC_ISSUES_COUNT:', errors)
    print('NO TRAINING OR RESULT FILES MODIFIED')
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
