"""Purely offline Phase 5G Stage 3 protocol and source-fingerprint guards."""
from pathlib import Path
import pytest
from run_phase5g_stage3 import (VERSION,ARCHS,SEEDS,REQUIRED_CHECKPOINTS,
  FINAL_TOPOLOGIES,FINAL_STREAMS,DOC_ONLY,normalized_ast,digest,equal,
  PROJECT,PROTOCOL_SHA256)


def test_stage3_version_fixed():
    assert VERSION=='PHASE5G_STAGE3_FROZEN_CONFIRMATORY_PLAN_V1'


def test_stage3_selected_checkpoints_are_fixed():
    assert REQUIRED_CHECKPOINTS['gnn']['seed']==27
    assert REQUIRED_CHECKPOINTS['gnn']['episode']==48
    assert REQUIRED_CHECKPOINTS['mlp']['seed']==17
    assert REQUIRED_CHECKPOINTS['mlp']['episode']==24


def test_stage3_model_hash_lengths():
    assert all(len(x['sha'])==64 for x in REQUIRED_CHECKPOINTS.values())
    assert all(int(x['sha'],16)>=0 for x in REQUIRED_CHECKPOINTS.values())


def test_stage3_training_factorial():
    assert len([(a,s) for a in ARCHS for s in SEEDS])==6


def test_stage3_holdout_nonoverlap():
    assert set(FINAL_TOPOLOGIES).isdisjoint(range(401,421))
    assert set(FINAL_TOPOLOGIES).isdisjoint(range(601,631))
    assert set(FINAL_TOPOLOGIES).isdisjoint(range(701,711))
    assert set(FINAL_STREAMS).isdisjoint(range(73000,73003))
    assert set(FINAL_STREAMS).isdisjoint(range(75000,75020))
    assert set(FINAL_STREAMS).isdisjoint(range(76000,76010))


def test_stage3_docstring_exception_is_two_files_only():
    assert set(DOC_ONLY)=={'phase5g_stage2/__init__.py','phase5g_stage2/workflow.py'}


def test_stage3_original_snapshots_match_sha():
    for rel,(filename,sha) in DOC_ONLY.items():
        assert digest(PROJECT/'stage3_reference'/filename)==sha


def test_stage3_ast_matches_original_training_sources():
    for rel,(filename,sha) in DOC_ONLY.items():
        assert normalized_ast(PROJECT/'stage3_reference'/filename)==normalized_ast(PROJECT/rel)


def test_stage3_edited_code_ast_rejected(tmp_path):
    p=tmp_path/'wrong.py'
    p.write_text('def f():\n    """x"""\n    return 2\n')
    q=tmp_path/'correct.py'
    q.write_text('def f():\n    #x\n    return 1\n')
    assert normalized_ast(p)!=normalized_ast(q)


def test_stage3_only_docstring_removed(tmp_path):
    p=tmp_path/'a.py';p.write_text('"""A"""\ndef f():\n    """B"""\n    return 3\n')
    q=tmp_path/'b.py';q.write_text('# A\ndef f():\n    # B\n    return 3\n')
    assert normalized_ast(p)==normalized_ast(q)


def test_stage3_sha_changes_if_mutated(tmp_path):
    p=tmp_path/'t.txt';p.write_bytes(b'yes'); a=digest(p)
    p.write_bytes(b'no')
    assert digest(p)!=a


def test_stage3_assertion_fails_closed():
    with pytest.raises(ValueError):
        equal('bad','good','test')


def test_stage3_protocol_exists_and_declares_no_holdout():
    content=(PROJECT/'PHASE5G_STAGE3_PROTOCOL_LOCK.md').read_text(encoding='utf-8')
    assert VERSION in content
    assert digest(PROJECT/'PHASE5G_STAGE3_PROTOCOL_LOCK.md')==PROTOCOL_SHA256
    assert 'Stage 3 does **not** evaluate held-out networks' in content
    assert '2,880' in content
    assert 'Holm' in content


def test_stage3_no_final_generation_entry_point():
    code=(PROJECT/'run_phase5g_stage3.py').read_text(encoding='utf-8')
    assert "('preflight','freeze')" in code
    assert "sub.add_parser('verify')" in code
    assert 'make_cohort' not in code
    assert 'build_network' not in code
