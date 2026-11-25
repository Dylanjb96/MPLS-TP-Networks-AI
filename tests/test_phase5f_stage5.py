"""Strict Stage 5 tests. NO test creates a topology in range 401..420."""
from __future__ import annotations
import copy,hashlib,json
import numpy as np
import pytest
from phase5f.traffic import generate_demands
from phase5f_stage2.topology import NetworkSpec,build_network
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage3.workflow import run_episode
from phase5f_stage5.freeze import verify_freeze,load_models,FROZEN,sha256
from phase5f_stage5.oracle import SingleFaultOracle,verify_trace
from phase5f_stage5.statistics import (POLICIES,PRIMARY,CELLS,TOPOLOGIES,STREAMS,
    validate_full_grid,cluster_contrasts,paired_stats,holm,factorial,analyze)
from phase5f_stage5.workflow import (preflight,devcheck,final_run,immutable_plan,
    _cohort_validate,hash_json,_evaluate_cell)


@pytest.fixture(scope='module')
def development_cohort():
    # Explicitly an ALREADY USED Stage 4 VALIDATION topology/stream.
    n=60;g=build_network(NetworkSpec(nodes=n,seed=301))
    ds=generate_demands(g,96,72000,True)
    inv=tuple(generate_pairs(g,d.source,d.destination,6,40) for d in ds)
    return g,ds,inv


@pytest.fixture(scope='module')
def synthetic_full_grid():
    """Fake arithmetic only: 401..420 are labels, NOT generated networks."""
    rows=[]
    for n,k,mode,load in CELLS:
        for t in TOPOLOGIES:
            for s in STREAMS:
                for p in POLICIES:
                    base=.72+(t-410)*.0001+(s-73001)*.0001
                    delta={'greedy':0.,'footprint':.005,'random_feasible':-.025,
                           'C_enhanced':.02,'D_full_hybrid':.01}[p]
                    rows.append(dict(nodes=n,k=k,mode=mode,load=load,
                         topology_seed=t,stream_seed=s,policy=p,
                         requests=round(n*load),admitted=100,rejected=round(n*load)-100,
                         weighted_admission=base+delta,ordinary_admission=.5,
                         feasibility_violations=0,network_sha256=f'net{n}-{t}',
                         request_sha256=f'req{n}-{mode}-{s}',
                         candidates_sha256=f'cand{n}-{k}-{s}-{t}',
                         overrides=0,discretionary=100,simulation_seconds=.5))
    return rows


def test_freeze_exact_checkpoint_hashes():
    m=verify_freeze()
    assert sha256(FROZEN/'C_enhanced_seed27.pt')==m['selected']['checkpoint_sha256']
    assert sha256(FROZEN/'D_full_hybrid_seed7.pt')==m['proposed_full_hybrid']['checkpoint_sha256']


def test_freeze_fixed_identities():
    m=verify_freeze()
    assert (m['selected']['config'],m['selected']['agent_seed'],m['selected']['best_episode'])==('C_enhanced',27,60)
    assert (m['proposed_full_hybrid']['config'],m['proposed_full_hybrid']['agent_seed'],m['proposed_full_hybrid']['best_episode'])==('D_full_hybrid',7,84)


def test_freeze_two_models_load_with_strict_architecture():
    models=load_models()
    assert set(models)=={'C_enhanced','D_full_hybrid'}
    assert models['C_enhanced'].config.guidance is False
    assert models['D_full_hybrid'].config.guidance is True


def test_preflight_never_accesses_heldout():
    r=preflight()
    assert r['result']=='STAGE5_PREFLIGHT_OK'
    assert r['heldout_data_accessed'] is False
    assert r['cohorts']==480 and r['policy_evaluations']==2400


def test_protocol_full_2x2x2():
    assert len(CELLS)==8 and CELLS.count(PRIMARY)==1
    assert {(n,k,mode) for n,k,mode,load in CELLS}=={
        (n,k,mode) for n in (60,500) for k in (2,6) for mode in ('static','dynamic')}
    assert all(load==4.0 for n,k,mode,load in CELLS)
    assert len(TOPOLOGIES)==20 and len(STREAMS)==3


def test_final_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError):final_run(tmp_path/'must_not_run',confirm_final=False)
    assert not (tmp_path/'must_not_run').exists()


def test_resume_without_locked_plan_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        final_run(tmp_path/'no_previous_run',resume=True,confirm_final=True)


def test_refuses_to_overwrite_complete_final(tmp_path):
    root=tmp_path/'finished';root.mkdir()
    (root/'stage5_final_manifest.json').write_text('{}')
    with pytest.raises(FileExistsError):final_run(root,confirm_final=True)


def test_development_only_oracle_agrees_with_stage3(development_cohort):
    g,ds,inv=development_cohort
    summary,_,decisions=run_episode(g,ds,inv,policy='greedy',trace=True,k=6,mode='dynamic')
    verified=verify_trace(g,ds,inv,decisions,k=6)
    assert (verified['accepted'],verified['departures'],verified['active_end'])==(
        summary['admitted'],summary['departures'],summary['active_end'])


def test_oracle_rejects_corrupt_decision(development_cohort):
    g,ds,inv=development_cohort
    summary,_,decisions=run_episode(g,ds,inv,policy='greedy',trace=True,k=6,mode='dynamic')
    copied=[dict(d) for d in decisions]
    copied[0]['request_id']='FAKE'
    with pytest.raises(AssertionError,match='mismatch'):verify_trace(g,ds,inv,copied,6)


def test_oracle_rejects_infeasible_double_reservation(development_cohort):
    g,ds,inv=development_cohort
    o=SingleFaultOracle(g)
    d=ds[0];p=inv[0][0]
    assert o.feasible(d,p)
    o.admit(d,p)
    assert not o.feasible(d,p)
    with pytest.raises(AssertionError):o.admit(d,p)


def test_oracle_direct_network_scenario_bound(development_cohort):
    g,ds,inv=development_cohort
    o=SingleFaultOracle(g)
    for d,plans in zip(ds,inv):
        o.release(d.arrival)
        options=[p for p in plans if o.feasible(d,p)]
        if options:o.admit(d,options[0])
    assert np.all(o.loads>=0)
    assert np.all(o.loads<=o.cap[None,:])


def test_evaluation_five_policies_matched_and_verified(development_cohort):
    g,ds,inv=development_cohort
    models=load_models()
    rows,traces,oracles=_evaluate_cell(g,ds,inv,n=60,k=6,mode='dynamic',
        topo=301,stream=72000,candidates_sha=candidates_digest(inv),
        gen_seconds=0.0,models=models,keep_trace=True)
    assert [r['policy'] for r in rows]==list(POLICIES)
    assert set(traces)=={'C_enhanced','D_full_hybrid'}
    assert all(len(t)==len(ds) for t in traces.values())
    assert all(r['feasibility_violations']==0 for r in rows)
    assert len(oracles)==5


def test_locked_plan_has_never_tune_fields():
    d=immutable_plan()
    assert d['checkpoint_training_permitted'] is False
    assert d['validation_based_selection_permitted'] is False
    assert d['guidance_gate_D']==0.1 and d['guidance_gate_C']==0.0
    assert d['topology_cluster_count']==20


def test_matched_grid_accepts_complete_synthetic(synthetic_full_grid):
    d=validate_full_grid(synthetic_full_grid)
    assert len(d)==2400


def test_matched_grid_rejects_missing_or_duplicate(synthetic_full_grid):
    with pytest.raises(ValueError,match='missing/extra'):
        validate_full_grid(synthetic_full_grid[:-1])
    with pytest.raises(ValueError,match='duplicate'):
        validate_full_grid(synthetic_full_grid+[synthetic_full_grid[0]])


def test_matched_grid_rejects_mismatched_inputs(synthetic_full_grid):
    rows=[dict(r) for r in synthetic_full_grid]
    rows[1]['request_sha256']='CHANGED'
    with pytest.raises(ValueError,match='unmatched'):
        validate_full_grid(rows)


def test_topology_cluster_effect_and_uncertainty(synthetic_full_grid):
    samples=cluster_contrasts(synthetic_full_grid,PRIMARY,'C_enhanced','greedy')
    assert len(samples)==20 and np.allclose(samples,2.0)
    r=paired_stats(samples)
    assert r['topology_clusters']==20
    assert r['effect_pp']==pytest.approx(2.0)
    assert r['bootstrap95_low_pp']==pytest.approx(2.0)


def test_holm_stepdown_is_monotone():
    p=holm([{'p_two_sided':.05},{'p_two_sided':.001},{'p_two_sided':.03}])
    assert p[1]['p_holm_adjusted']==pytest.approx(.003)
    assert p[2]['p_holm_adjusted']==pytest.approx(.06)
    assert p[0]['p_holm_adjusted']==pytest.approx(.06)


def test_full_factorial_arithmetic_on_synthetic(synthetic_full_grid):
    res=factorial(synthetic_full_grid)
    assert len(res)==7
    assert all(abs(r['effect_pp'])<1e-8 for r in res)
    summary,primary,factorials,paired=analyze(synthetic_full_grid)
    assert len(summary)==40 and len(primary)==4 and len(factorials)==7
    assert len(paired)==8*20*3


def test_cohort_hash_guard_rejects_unmatched_results():
    r=[dict(nodes=60,policy=p,network_sha256='same',request_sha256='same',candidates_sha256='same',
            requests=240,admitted=220,rejected=20,feasibility_violations=0,
            decision_records=240) for p in POLICIES]
    content=dict(role='HELDOUT_EVALUATION_ONCE_ONLY_NO_TRAINING',plan_sha256='test',
                 cell=[60,6,'dynamic',4.0,401,73000],rows=r)
    assert len(_cohort_validate(content,'test',(60,6,'dynamic',4.0,401,73000)))==5
    r[1]['network_sha256']='BAD'
    with pytest.raises(RuntimeError,match='different'):_cohort_validate(content,'test')
