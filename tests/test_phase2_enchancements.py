#Supplementary Phase 2 structural and synthetic workload tests.
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from railway_tp.topology import TopologyConfig, make_topology
from railway_tp.routes import classify_request
from enhancements.six_improvements import (
    graph_fingerprint, _disconnect_pairs, endpoint_pools, balanced_requests,
    sweep_topologies, gateway_comparison, route_analysis, validate_assumptions,
    demand_study, evidence_audit,
)

ROOT=Path(__file__).resolve().parents[1]


def test_unordered_disconnected_pair_count():
    assert _disconnect_pairs([6,5])==30
    assert _disconnect_pairs([11])==0
    assert _disconnect_pairs([3,2,1])==11


def test_seed_reproduces_same_labelled_graph():
    cfg=TopologyConfig(nodes=60,seed=42,main_nodes=12,subring_count=4)
    a,_=make_topology(cfg)
    b,_=make_topology(cfg)
    assert graph_fingerprint(a)==graph_fingerprint(b)
    c,_=make_topology(replace(cfg,seed=43))
    assert graph_fingerprint(a)!=graph_fingerprint(c)


def test_sweep_is_both_seed_and_shape_sensitive(tmp_path):
    rows=sweep_topologies(ROOT,tmp_path,seeds=(42,43))
    assert len(rows)==6
    assert set(r['shape'] for r in rows)=={'primary','fewer_rings','wider_main'}
    assert all(r['all_cycles'] and r['single_link_connected'] for r in rows)
    assert (tmp_path/'01_sweep_summary.csv').exists()


def test_gateway_counterexample_and_dual_protection_is_graph_only(tmp_path):
    rows=gateway_comparison(ROOT,tmp_path)
    a=next(r for r in rows if r['case_nodes']==12 and r['gateway_mode']=='single')
    b=next(r for r in rows if r['case_nodes']==12 and r['gateway_mode']=='dual')
    assert a['articulation_point_count']==1
    assert a['node_failures_causing_disconnection']==1
    assert b['articulation_point_count']==0
    assert b['node_failures_causing_disconnection']==0
    with (tmp_path/'02_node_failure_details.csv').open(newline='') as f:
        details=list(csv.DictReader(f))
    assert any(int(d['disconnected_surviving_pairs'])>0 for d in details if d['gateway_mode']=='single')


def test_endpoint_pair_pools_label_consistently():
    G,rings=make_topology(TopologyConfig(nodes=60,main_nodes=12,subring_count=4))
    pools=endpoint_pools(G,rings)
    assert all(pools.values())
    for category,pairs in pools.items():
        assert all(classify_request(G,u,v)==category and u!=v for u,v in pairs)


def test_balanced_demand_is_reproducible_and_per_category():
    G,rings=make_topology(TopologyConfig(nodes=60,main_nodes=12,subring_count=4))
    rows=balanced_requests(G,rings,40,seed=19,split='test')
    assert rows==balanced_requests(G,rings,40,seed=19,split='test')
    assert all(sum(r['endpoint_category']==c for r in rows)==10 for c in (
        'main-to-main','within-subring','main-to-subring','subring-to-subring'))
    assert all(r['is_synthetic'] and r['bandwidth_mbps']>0 for r in rows)
    heavy=balanced_requests(G,rings,100,seed=19,split='test',interring_heavy=True)
    assert sum(r['endpoint_category']=='subring-to-subring' for r in heavy)==45


def test_route_categories_are_analysed(tmp_path):
    details,summary=route_analysis(ROOT,tmp_path,samples_per_category=3)
    assert len(summary)==4
    assert len(details)==12
    assert all(r['local_edge_connectivity']>=2 for r in details)
    assert all(r['shortest_hops']>=1 for r in details)


def test_assumptions_explicitly_exclude_phase3_claims(tmp_path):
    result=validate_assumptions(ROOT,tmp_path)
    assert result['example_unique_physical_link_count']==1
    assert result['phase3_protection_scheme_frozen'] is False
    assert (tmp_path/'04_assumptions_validation.json').exists()


def test_demand_splits_are_written_and_unadmitted(tmp_path):
    dataset,summary=demand_study(ROOT,tmp_path,requests_per_split=40)
    assert len(dataset)==9*40
    assert len(summary)==9
    assert all(r['is_synthetic'] for r in dataset)
    assert {r['split'] for r in dataset}=={'train','test','validation'}
    assert (tmp_path/'05_demand_balanced_train.csv').exists()


def test_evidence_audit_checks_original_outputs(tmp_path):
    rows=evidence_audit(ROOT,tmp_path,create_missing=False)
    assert len(rows)==5
    assert all(r['fully_validated'] for r in rows)
    report=json.loads((tmp_path/'06_environment_and_provenance.json').read_text())
    assert 'networkx' in report


@pytest.mark.parametrize('wrong_count',[-10,0,1,2,3])
def test_too_small_balanced_requests_rejected(wrong_count):
    G,rings=make_topology(TopologyConfig(nodes=60,main_nodes=12,subring_count=4))
    with pytest.raises(ValueError):
        balanced_requests(G,rings,wrong_count,seed=1,split='test')
