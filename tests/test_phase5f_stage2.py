"""Prospective Stage 2 diagnostics; no Stage 1 results are changed."""
from pathlib import Path
import csv
import json
import pytest
import numpy as np
import networkx as nx
from phase5f.traffic import generate_demands
from phase5f.protection import ScenarioLedger,verify_pair
from phase5f_stage2.topology import NetworkSpec,build_network,validate_network,fingerprint,edge_set
from phase5f_stage2.candidates import generate_pairs,choice,exact_footprint,candidates_digest
from phase5f_stage2.probe import evaluate,run_probe


@pytest.mark.parametrize('n',[60,150,250,500])
def test_distinct_connected_redundant_networks(n):
    graphs=[build_network(NetworkSpec(nodes=n,seed=s)) for s in (101,102,103)]
    assert all(validate_network(g) for g in graphs)
    assert len({fingerprint(g,structural_only=True) for g in graphs})==3
    assert len({edge_set(g) for g in graphs})==3
    assert len({fingerprint(g) for g in graphs})==3


def test_same_seed_repeat_deterministic():
    a=build_network(NetworkSpec(nodes=60,seed=101))
    b=build_network(NetworkSpec(nodes=60,seed=101))
    assert fingerprint(a)==fingerprint(b)
    assert all(a[u][v]==b[u][v] for u,v in a.edges)


def test_length_part_of_fingerprint():
    a=build_network(NetworkSpec(nodes=60,seed=101))
    b=a.copy()
    u,v=next(iter(b.edges))
    b[u][v]['length_km']+=1
    assert fingerprint(a,structural_only=True)==fingerprint(b,structural_only=True)
    assert fingerprint(a)!=fingerprint(b)


def test_candidates_reproducible_and_prefix():
    g=build_network(NetworkSpec(nodes=60,seed=101))
    d=generate_demands(g,1,70000,False)[0]
    a=generate_pairs(g,d.source,d.destination,6,40)
    b=generate_pairs(g,d.source,d.destination,6,40)
    prefix=generate_pairs(g,d.source,d.destination,2,40)
    assert a==b and a[:2]==prefix and len(a)>=2
    assert len({x.candidate_id for x in a})==len(a)
    assert all(verify_pair(g,d.source,d.destination,p) for p in a)
    assert candidates_digest((a,))==candidates_digest((b,))


def test_footprint_tie_resolved_by_paths_not_input_order():
    g=build_network(NetworkSpec(nodes=60,seed=101))
    d=generate_demands(g,1,70000,False)[0]
    a=generate_pairs(g,d.source,d.destination,6)
    assert len(a)>=2
    # Route-pair ordering must not alter minimum-footprint route selection.
    orig=choice('footprint',list(range(len(a))),a,g,None)
    reversed_=tuple(reversed(a))
    new=choice('footprint',list(range(len(a))),reversed_,g,None)
    assert a[orig].candidate_id==reversed_[new].candidate_id


def test_feasibility_bins_sum_and_no_violations():
    g=build_network(NetworkSpec(nodes=60,seed=102))
    d=generate_demands(g,24,70001,True)
    inventories=[generate_pairs(g,x.source,x.destination,6) for x in d]
    r=evaluate(g,d,inventories,k=6,policy='footprint',seed=70001,headroom=10)
    assert sum(r['feasible_'+str(i)] for i in range(7))+r['feasible_more_than_6']==24
    assert r['admitted']+r['rejected']==24
    assert r['no_feasibility_violations']


def test_probe_smoke_outputs_and_guard(tmp_path):
    out=tmp_path/'stage2_dev'
    result=run_probe(out,sizes=(60,),topology_seeds=(101,102),
                     stream_seeds=(70000,),loads=(1.6,),
                     traffic_modes=('static',),k_values=(2,),
                     policies=('greedy','footprint'))
    assert result['rows']==4
    with (out/'stage2_probe_runs.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==4
    assert rows[0]['network_sha256']!=rows[2]['network_sha256']
    assert (out/'stage2_topology_diversity.csv').is_file()
    with pytest.raises(FileExistsError):
        run_probe(out)


def test_rejects_heldout_and_undefined_load(tmp_path):
    kw=dict(sizes=(60,),topology_seeds=(101,),stream_seeds=(70000,),
            loads=(1.6,),policies=('greedy',),k_values=(2,),traffic_modes=('static',))
    with pytest.raises(ValueError):
        run_probe(tmp_path/'bad',**(kw|dict(stream_seeds=(61000,))))
    with pytest.raises(ValueError):
        run_probe(tmp_path/'bad',**(kw|dict(loads=(2.123,))))


def test_ledger_recompute_explicit_fault_oracle():
    from phase5f.protection import edge as physical_edge
    g=build_network(NetworkSpec(nodes=60,seed=103))
    demands=generate_demands(g,30,70002,True)
    ledger=ScenarioLedger(g,10)
    for req in demands:
        ledger.release_expired(req.arrival)
        for pair in generate_pairs(g,req.source,req.destination,2):
            if ledger.admit(req,pair):break
    arcs=sorted(ledger.edge_to_i,key=ledger.edge_to_i.get)
    brute=np.zeros_like(ledger.loads)
    for scenario in range(len(arcs)+1):
        failed=None if scenario==0 else arcs[scenario-1]
        for req,pair in ledger.active.values():
            working={physical_edge(u,v) for u,v in zip(pair.working,pair.working[1:])}
            path=pair.backup if failed in working else pair.working
            for u,v in zip(path,path[1:]):
                idx=ledger.edge_to_i[physical_edge(u,v)]
                brute[scenario,idx*2:idx*2+2]+=req.bandwidth_mbps
    assert np.array_equal(brute,ledger.loads)
    assert ledger.verify_full_scenario_recompute()
