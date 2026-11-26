"""Phase 5G Stage 4 development-only protection, reproducibility and stats tests."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import copy,gzip,json,math,pytest
import torch
import numpy as np
from phase5f.traffic import Demand,generate_demands
from phase5f_stage5.oracle import verify_trace
from phase5g.topology import build_network
from phase5g_stage4.protocol import (CELLS,BY_CELL,POLICIES,PRIMARY,SECONDARY,
    semantic_lock,validate_seeds,STRESS_SEMANTICS,FINAL_TOPOLOGIES,FINAL_STREAMS)
from phase5g_stage4.evaluation import (freeze_lock,models_load,stresses,cohort,
    fast_oracle_trajectory, evaluate_cohort,pack_payload,inspect_payload,
    settings_for,_seal,release_check,summary_rows,source_hashes,run_batches)
from phase5g_stage4.statistics import grid_validate,differences,cluster_stats,holm,analyze


def test_frozen_sources_and_models_verified():
    m=freeze_lock()
    assert len(m['source_sha256'])==15
    assert set(models_load())=={'gnn_PPO','mlp_PPO','C_frozen'}


def test_eight_cells_and_two_hypothesis_families():
    assert len(CELLS)==8 and len(POLICIES)==6 and len(PRIMARY)==2 and len(SECONDARY)==8
    assert len({c.id for c in CELLS})==8


def test_development_seed_role_rejects_reserved_final():
    validate_seeds('dev',501,74000)
    with pytest.raises(ValueError):validate_seeds('dev',801,77000)
    with pytest.raises(ValueError):validate_seeds('dev',401,73000)
    with pytest.raises(ValueError):validate_seeds('final',501,74000)
    with pytest.raises(ValueError):validate_seeds('test',801,77000)


def test_semantic_fingerprint_is_fixed():
    assert len(semantic_lock())==64
    assert len(STRESS_SEMANTICS)==3


def test_bandwidth_stress_is_exact_integer_ceil():
    d=Demand('T',3,20,1,2,5,'signalling_control',5)
    x=stresses((d,),'bandwidth_150')[0]
    assert x.bandwidth_mbps==8 and x.expiry==d.expiry
    assert x.priority_weight==5 and x.source==1 and x.destination==2


def test_short_hold_is_floor_half_clamped():
    d=Demand('T',3,20,1,2,5,'signalling_control',5)
    x=stresses((d,),'short_hold')[0]
    assert x.expiry==11 and x.bandwidth_mbps==5
    assert stresses((replace(d,expiry=4),),'short_hold')[0].expiry==4


def test_unknown_stress_rejected():
    with pytest.raises(ValueError):stresses((),'other')


def test_cohort_stress_compatibility_and_k2_prefix():
    base=cohort(BY_CELL['H_BASE'],nodes=60,topology_seed=501,stream_seed=74000,role='dev')
    bw=cohort(BY_CELL['H_BW'],nodes=60,topology_seed=501,stream_seed=74000,role='dev')
    hold=cohort(BY_CELL['H_HOLD'],nodes=60,topology_seed=501,stream_seed=74000,role='dev')
    k2=cohort(BY_CELL['H_K2'],nodes=60,topology_seed=501,stream_seed=74000,role='dev')
    assert len(base[1])==240
    assert all(x.bandwidth_mbps==(3*y.bandwidth_mbps+1)//2 for x,y in zip(bw[1],base[1]))
    assert all(x.expiry==x.arrival+max(1,(y.expiry-y.arrival)//2) for x,y in zip(hold[1],base[1]))
    assert all(k==full[:2] for full,k in zip(base[2],k2[2]))


def test_cannot_invoke_reserved_seed_through_dev_cohort():
    with pytest.raises(ValueError):
        cohort(BY_CELL['H_BASE'],nodes=60,topology_seed=801,stream_seed=77000,role='dev')


@pytest.fixture(scope='module')
def complete_dev_cohort():
    models=models_load()
    graph,d,inv=cohort(BY_CELL['H_BASE'],nodes=60,topology_seed=501,stream_seed=74000,role='dev')
    rows,traces,hashes=evaluate_cohort(BY_CELL['H_BASE'],nodes=60,topology_seed=501,
                                          stream_seed=74000,role='dev',models=models)
    return graph,d,inv,rows,traces,hashes


def test_six_policy_dispatch_and_masked_actions(complete_dev_cohort):
    g,d,inv,rows,traces,hashes=complete_dev_cohort
    assert [r['policy'] for r in rows]==list(POLICIES)
    assert len(set(r['request_sha256'] for r in rows))==1
    assert all(r['feasibility_violations']==0 and r['decision_records']==240 for r in rows)
    assert all(sum(int(t['admitted']) for t in traces[p])==r['admitted']
               for p,r in zip(POLICIES,rows))


def test_independent_oracle_replay_matches_original_oracle(complete_dev_cohort):
    g,d,inv,rows,traces,hashes=complete_dev_cohort
    for p,r in zip(POLICIES,rows):
        verdict=verify_trace(g,d,inv,traces[p],6,headroom=10)
        assert verdict['accepted']==r['admitted']
        assert verdict['oracle_state_sha256']==r['oracle_sha256']
        assert r['end_of_episode_peak_scenario_utilization']<=r['exact_trajectory_peak_scenario_utilization']+1e-10


def test_k2_neural_fixed_six_slot_actions():
    models=models_load()
    r,t,h=evaluate_cohort(BY_CELL['H_K2'],nodes=60,topology_seed=501,
                             stream_seed=74000,role='dev',models=models)
    assert len(r)==6
    assert all(all(int(x['chosen_index'])<2 for x in t[p]) for p in POLICIES)
    assert all(row['k']==2 and row['feasibility_violations']==0 for row in r)


def test_payload_roundtrip_and_tamper_detect(complete_dev_cohort):
    g,d,inv,rows,traces,hashes=complete_dev_cohort
    lock=_seal({'role':'dev','test':42})
    key='H_BASE__501__74000'
    data=pack_payload(lock,rows,traces,hashes,key)
    assert inspect_payload(data,lock,key)['results']==rows
    obj=json.loads(gzip.decompress(data))
    obj['body']['results'][0]['admitted']+=1
    with pytest.raises(RuntimeError):inspect_payload(gzip.compress(json.dumps(obj).encode()),lock,key)
    with pytest.raises(RuntimeError):inspect_payload(data,_seal({'role':'dev','test':43}),key)


def test_dev_resume_preserves_original_cohort_file(tmp_path):
    o=tmp_path/'dev'
    kw=dict(role='dev',nodes=60,cell_ids=('H_BASE',),topologies=(501,),streams=(74000,))
    rows=run_batches(o,**kw)
    f=o/'cohorts'/'H_BASE__501__74000.json.gz'
    before=f.read_bytes()
    with pytest.raises(FileExistsError):run_batches(o,**kw)
    again=run_batches(o,**kw,resume=True)
    assert before==f.read_bytes()
    assert rows==again
    assert len(summary_rows(o))==6


def test_resume_fails_if_manifest_changed(tmp_path):
    o=tmp_path/'dev'
    kw=dict(role='dev',nodes=60,cell_ids=('H_BASE',),topologies=(501,),streams=(74000,))
    run_batches(o,**kw)
    m=o/'run_manifest.json';v=json.loads(m.read_text())
    v['semantic_lock_sha256']='bad';m.write_text(json.dumps(v))
    with pytest.raises(RuntimeError):run_batches(o,**kw,resume=True)


def test_missing_release_rejects_final_before_network():
    with pytest.raises(RuntimeError,match='STAGE4_FINAL_LOCKED'):
        release_check('/certainly/not/a/stage4_AUDIT_RELEASE.json',role='final')


def test_full_grid_locked_no_subsampling():
    with pytest.raises(ValueError):
        settings_for(role='final',nodes=500,cell_ids=('H_BASE',),
              topologies=FINAL_TOPOLOGIES,streams=FINAL_STREAMS)
    with pytest.raises(ValueError):
        settings_for(role='dev',nodes=60,cell_ids=('H_BASE',),
              topologies=(801,),streams=(77000,))


def synthetic_final_records():
    """Only symbolic seed labels; never generate any final network or traffic."""
    rows=[]
    for c in CELLS:
      for t in FINAL_TOPOLOGIES:
       for s in FINAL_STREAMS:
        for p in POLICIES:
            # a known positive primary GNN improvement with topology variation
            weight={'gnn_PPO':.89+(t-810)*.0001,'mlp_PPO':.87,
                    'C_frozen':.875,'scenario_minimax':.885,'footprint':.86,'greedy':.865}[p]
            rows.append(dict(cell=c.id,family=c.family,stress=c.stress,k=c.k,
                nodes=500,mode='dynamic',load=4.0,topology_seed=t,stream_seed=s,
                policy=p,requests=2000,admitted=1750,rejected=250,
                weighted_admission=weight,overrides=0,discretionary=1,
                decision_records=2000,feasibility_violations=0,policy_seconds=1.0,
                network_sha256='network_'+str(t)+'_'+c.family,
                request_sha256='request_'+str(t)+'_'+str(s)+'_'+c.stress,
                candidates_sha256='candidate_'+str(t)+'_'+str(s)+'_'+c.id))
    return rows


def test_statistical_full_fixture_and_holm_family_sizes():
    rows=synthetic_final_records()
    idx=grid_validate(rows)
    assert len(idx)==2880
    stats=analyze(rows)
    assert len(stats['primary'])==2 and len(stats['secondary'])==8
    assert len(stats['exploratory'])==12 and len(stats['summary'])==48
    assert all(r['n_topologies']==20 for r in stats['primary'])
    assert stats['primary'][0]['effect_pp']>1.0


def test_stats_reject_duplicate_and_missing():
    rows=synthetic_final_records()
    with pytest.raises(ValueError,match='duplicate'):
        grid_validate(rows+[rows[0]])
    with pytest.raises(ValueError,match='incomplete'):
        grid_validate(rows[:-1])


def test_stats_reject_unmatched_inputs():
    rows=synthetic_final_records()
    rows[0]['request_sha256']='different'
    with pytest.raises(ValueError,match='unmatched'):
        grid_validate(rows)


def test_paired_t_interval_and_numerical_holm():
    d=np.arange(20,dtype=float)*.01
    r=cluster_stats(d)
    assert r['n_topologies']==20 and r['ci95_low_pp']<r['effect_pp']<r['ci95_high_pp']
    sample=[dict(p_two_sided=p) for p in (.015,.001,.04)]
    a=holm(sample)
    assert abs(a[1]['p_holm_adjusted']-.003)<1e-12
    assert abs(a[0]['p_holm_adjusted']-.03)<1e-12
    assert abs(a[2]['p_holm_adjusted']-.04)<1e-12


def test_zero_variance_statistical_guard():
    r=cluster_stats(np.zeros(20))
    assert r['p_two_sided']==1 and r['ci95_low_pp']==0
    r=cluster_stats(np.ones(20))
    assert r['p_two_sided'] is None
    with pytest.raises(ValueError):holm([r])
