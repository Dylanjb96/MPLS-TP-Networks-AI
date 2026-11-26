#!/usr/bin/env python3
"""Phase 5G Stage 4 source-only handoff collector (read only).

Run from the existing MPLS-AI project root:
    python collect_phase5g_stage4_source.py

Reads a *fixed allowlist* of Python source and test files plus the exact Stage 3
freeze protocol/manifest. It never loads trained models, runs simulation, reads
traffic streams, changes training outputs, or accesses final test seeds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / 'outputs' / 'phase5g' / 'stage3_FROZEN'
MANIFEST = FREEZE / 'freeze_manifest.json'
OUTPUT = ROOT / 'Phase5G_Stage4_Source_Handoff.zip'

# Intentionally source-only. No checkpoints, traces, output histories, traffic
# snapshots, candidate cache, notebooks, Git internals or files outside ROOT.
PACKAGE_FOLDERS = (
    'phase5f',
    'phase5f_stage2',
    'phase5f_stage3',
    'phase5f_stage4',
    'phase5f_stage5',
    'phase5g',
    'phase5g_stage2',
)
TEST_FILES = (
    'tests/test_phase5g.py',
    'tests/test_phase5g_stage2.py',
    'tests/test_phase5g_stage3.py',
    'tests/test_phase5f.py',
    'tests/test_phase5f_stage2.py',
    'tests/test_phase5f_stage3.py',
    'tests/test_phase5f_stage4.py',
    'tests/test_phase5f_stage5.py',
)
ROOT_FILES = (
    'run_phase5f.py',
    'run_phase5f_stage2.py',
    'run_phase5f_stage3.py',
    'run_phase5f_stage4.py',
    'run_phase5f_stage5.py',
    'run_phase5g.py',
    'run_phase5g_stage2.py',
    'run_phase5g_stage3.py',
    'PHASE5G_STAGE3_PROTOCOL_LOCK.md',
)


def checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_source(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Unsafe/missing source file: {path}')
    if ROOT not in path.resolve().parents:
        raise ValueError(f'Outside project root: {path}')
    if path.stat().st_size > 5_000_000:
        raise ValueError(f'Unexpectedly large source file: {path}')
    return path.read_bytes()


def main() -> int:
    if not MANIFEST.is_file():
        print('ERROR: Stage 3 freeze manifest not found:', MANIFEST)
        return 2
    freeze_bytes = safe_source(MANIFEST)
    freeze = json.loads(freeze_bytes)
    if freeze.get('role') != 'FROZEN_PRE_HELDOUT_NO_EVALUATION':
        print('ERROR: Unrecognized Stage 3 freeze role.')
        return 2
    if freeze.get('heldout_accessed_by_stage3') is not False:
        print('ERROR: Stage 3 held-out-access flag is not false.')
        return 2
    originals = freeze.get('source_sha256')
    if not isinstance(originals, dict) or len(originals) != 15:
        print('ERROR: Expected 15 frozen Stage 2 source fingerprints.')
        return 2

    entries: dict[str, bytes] = {}
    mismatches = []
    for rel, expected in originals.items():
        path = ROOT / rel
        try:
            blob = safe_source(path)
        except (ValueError, OSError) as exc:
            mismatches.append(str(exc))
            continue
        actual = checksum(blob)
        if actual != expected:
            mismatches.append(f'{rel}: expected {expected}, found {actual}')
        entries[rel.replace('\\', '/')] = blob
    if mismatches:
        print('SOURCE_LOCK_MISMATCH: Do not edit frozen source files.')
        for issue in mismatches:
            print('  ', issue)
        print('No handoff ZIP created.')
        return 3

    for folder in PACKAGE_FOLDERS:
        directory = ROOT / folder
        if directory.is_dir():
            for path in sorted(directory.glob('*.py')):
                entries.setdefault(path.relative_to(ROOT).as_posix(), safe_source(path))
    for name in (*ROOT_FILES, *TEST_FILES):
        path = ROOT / name
        if path.is_file():
            entries.setdefault(name, safe_source(path))
    protocol_in_freeze = FREEZE / 'freeze_sources' / 'PHASE5G_STAGE3_PROTOCOL_LOCK.md'
    expected_protocol_hash = freeze.get('protocol_sha256')
    if not protocol_in_freeze.is_file():
        print('ERROR: protocol snapshot missing from frozen evidence.')
        return 3
    protocol_blob = safe_source(protocol_in_freeze)
    if checksum(protocol_blob) != expected_protocol_hash:
        print('ERROR: Stage 3 protocol snapshot hash differs from frozen manifest.')
        return 3
    entries['stage3_reference/PHASE5G_STAGE3_PROTOCOL_LOCK.md'] = protocol_blob
    entries['stage3_reference/freeze_manifest.json'] = freeze_bytes

    # Fail closed if there is another unexpected modification to the active
    # locked protocol, rather than silently preparing a changed Stage 4 design.
    active_protocol = ROOT / 'PHASE5G_STAGE3_PROTOCOL_LOCK.md'
    if active_protocol.is_file() and checksum(safe_source(active_protocol)) != expected_protocol_hash:
        print('ERROR: Active Stage 3 protocol differs from frozen copy.')
        return 3

    if OUTPUT.exists():
        print('ERROR: Refusing to overwrite existing ZIP:', OUTPUT)
        print('Move/archive it manually before running this collector again.')
        return 4
    summary = {
        'role': 'SOURCE_ONLY_STAGE4_IMPLEMENTATION_HANDOFF',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'files_sha256': {k: checksum(v) for k, v in sorted(entries.items())},
        'stage3_freeze_manifest_sha256': checksum(freeze_bytes),
        'stage3_protocol_sha256': expected_protocol_hash,
        'locked_stage2_sources_verified': 15,
        'model_weights_included': False,
        'training_records_included': False,
        'heldout_data_included': False,
        'heldout_data_generated': False,
        'purpose': 'Build and test Phase 5G Stage 4 evaluator on development seeds only',
    }
    entries['stage4_source_collection_manifest.json'] = (
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    ).encode('utf-8')
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for name, blob in sorted(entries.items()):
            z.writestr(name, blob)
    with zipfile.ZipFile(OUTPUT) as z:
        if z.testzip() is not None:
            print('ERROR: ZIP verification failed')
            return 5
    print('PHASE5G_STAGE4_SOURCE_HANDOFF_OK')
    print('Frozen source fingerprints verified: 15/15')
    print('Files included:', len(entries))
    print('No model weights or experiment outputs included.')
    print('No topology/traffic generation or final evaluation executed.')
    print('Upload:', OUTPUT.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
