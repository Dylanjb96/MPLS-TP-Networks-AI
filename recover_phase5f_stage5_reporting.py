"""Recover Phase 5F Stage 5 REPORTING ONLY after the CSV-schema bug.

This file is intentionally OUTSIDE phase5f_stage5/ and the immutable-runner
source-hash inventory. It does not execute the held-out simulator, regenerate
network structures/traffic, train agents, select models, or modify any frozen
source, model, or saved cohort evidence. It reads all 480 saved cohorts and
finalizes the original preregistered analyses with the original locked plan.

Specific fix: the primary contrast has no Holm-adjusted p-value because the
adjustment applies only to the three secondary contrasts. The unchanged CSV
writer correctly enforces a consistent column set; supply an empty CSV value
for the primary contrast's inapplicable Holm field at serialization time.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing Stage 5 final output directory')
    args = parser.parse_args()
    root = Path(args.root)

    # All imports are of the existing, frozen project implementation.
    from phase5f_stage5 import workflow
    from phase5f_stage5.statistics import POLICIES, PRIMARY, TOPOLOGIES, STREAMS

    if not root.is_dir():
        parser.error('final output directory not found: ' + str(root))
    locked = root / 'locked_final_plan.json'
    if not locked.is_file():
        parser.error('locked_final_plan.json missing: do not run the final experiment again')
    if (root / 'stage5_final_manifest.json').exists():
        parser.error('final manifest already exists; use the read-only Stage 5 verifier instead')

    # Prove all 480 frozen source-backed cohort files exist before finalization.
    expected = {path.name: path for path, _ in workflow._expected_files(root)}
    cohort_dir = root / 'cohorts'
    actual = {p.name: p for p in cohort_dir.glob('*.json.gz')} if cohort_dir.is_dir() else {}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        parser.error(f'expected 480 cohort files; found {len(actual)}; '
                     f'missing={missing[:5]} extra={extra[:5]}')
    if len(expected) != 480:
        raise AssertionError(f'unexpected locked design: {len(expected)} cohorts')
    before = {name: _sha256(path) for name, path in actual.items()}

    # The frozen plan includes all original Stage 5 source hashes, Stage 1-4
    # hashes, model SHA256s, package versions and preregistered cells/seeds.
    plan = workflow.immutable_plan()
    plan_sha = workflow.hash_json(plan)
    saved = json.loads(locked.read_text(encoding='utf8'))
    if saved.get('plan_sha256') != plan_sha or saved.get('plan') != plan:
        raise RuntimeError('locked plan/source/checkpoint/runtime mismatch. '
                           'Do not change or re-install any frozen files.')
    print(f'STAGE5_RECOVERY_LOCK_OK cohorts={len(expected)} '
          f'policies_per_cohort={len(POLICIES)} plan_sha256={plan_sha}', flush=True)

    original_csv_atomic = workflow._csv_atomic
    fixed_tables = []

    def serialize_with_primary_schema_repair(path, rows):
        if Path(path).name != 'heldout_primary_inference.csv':
            return original_csv_atomic(path, rows)
        if len(rows) != 4:
            raise ValueError(f'expected one primary and three secondary contrasts, got {len(rows)}')
        primary = [r for r in rows if r.get('evidence_family') == 'primary']
        secondary = [r for r in rows if r.get('evidence_family') == 'secondary']
        if len(primary) != 1 or len(secondary) != 3:
            raise ValueError('unexpected inference families; refusing to change analysis')
        expected_columns = set(secondary[0])
        if any(set(r) != expected_columns for r in secondary):
            raise ValueError('unexpected inconsistent secondary contrast schema')
        if 'p_holm_adjusted' not in expected_columns or \
                set(primary[0]) != expected_columns - {'p_holm_adjusted'}:
            raise ValueError('not the known missing-Holm-column error; refusing recovery')
        # Empty CSV cell = not applicable, NOT an adjusted p-value of zero.
        patched = [dict(r) for r in rows]
        next(r for r in patched if r['evidence_family'] == 'primary')['p_holm_adjusted'] = ''
        fixed_tables.append(Path(path).name)
        return original_csv_atomic(path, patched)

    workflow._csv_atomic = serialize_with_primary_schema_repair
    try:
        # Existing original analysis: reads the 480 saved cohorts, verifies
        # their identity and complete matched grid, performs topology-clustered
        # inference and writes tables and the official completion manifest.
        manifest = workflow.analyze_existing(root)
    finally:
        workflow._csv_atomic = original_csv_atomic

    if fixed_tables != ['heldout_primary_inference.csv']:
        raise AssertionError('expected exactly one CSV schema repair')
    after = {name: _sha256(path) for name, path in actual.items()}
    if before != after:
        raise AssertionError('COHORT DATA CHANGED; preserve evidence and investigate')

    # Original unmodified verifier recomputes primary effect and SHA256-checks
    # all five published tables and preserved decision traces.
    verified = workflow.verify_results(root)
    if verified['result'] != 'STAGE5_VERIFY_OK':
        raise AssertionError('original Stage 5 verifier did not pass')
    print(f'STAGE5_POSTPROCESS_RECOVERY_OK rows={manifest["policy_rows"]} '
          f'cohorts={manifest["matched_cohorts"]} topologies={manifest["topology_clusters"]} '
          f'cohort_sha256_unchanged=True', flush=True)
    print(f'STAGE5_VERIFY_OK primary_delta_pp={verified["primary_delta_pp"]:.10f} '
          f'all_five_tables_sha256_verified={verified["all_five_tables_sha256_verified"]} '
          f'primary_RL_traces_verified={verified["primary_RL_traces_verified"]}', flush=True)
    print('No held-out cohorts rerun; frozen source and checkpoints were not edited.', flush=True)


if __name__ == '__main__':
    main()
