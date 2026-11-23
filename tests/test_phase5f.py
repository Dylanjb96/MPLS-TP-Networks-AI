import csv
import numpy as np
import pytest
from phase5f.topology import NetworkSpec,build_network,validate_network
from phase5f.protection import generate_pairs,verify_pair,ScenarioLedger
from phase5f.traffic import generate_demands,stream_digest
from phase5f.probe import probe

@pytest.mark.parametrize('n',[60,150,250,500])
def test_topology_shape_and_no_bridges(n):
    g=build_network(NetworkSpec(nodes=n,seed=101))
    assert len(g)==n
    assert validate_network(g)


def test_network_deterministic():
    a=build_network(NetworkSpec(nodes=60,seed=111))
    b=build_network(NetworkSpec(nodes=60,seed=111))
    assert set(a.edges)==set(b.edges)
    assert all(a.edges[e]==b.edges[e] for e in a.edges)


def test_dynamic_arrival_common_random_numbers():
    g=build_network(NetworkSpec(nodes=60,seed=101))
    s=generate_demands(g,40,70000,False)
    d=generate_demands(g,40,70000,True)
    assert stream_digest(s,False)==stream_digest(d,False)
    assert all(x.expiry>40 for x in s)
    assert any(x.expiry<=40 for x in d)


def test_protected_pair_links_truly_disjoint():
    g=build_network(NetworkSpec(nodes=60,seed=101))
    req=generate_demands(g,4,70000,False)[0]
    p=generate_pairs(g,req.source,req.destination,6)
    assert len(p)>=1
    assert all(verify_pair(g,req.source,req.destination,x) for x in p)
    assert len(set(x.candidate_id for x in p))==len(p)


def test_ledger_matches_brute_force_and_release():
    g=build_network(NetworkSpec(nodes=60,seed=101))
    demands=generate_demands(g,30,70003,True)
    led=ScenarioLedger(g,10)
    accepted=[]
    for req in demands:
        led.release_expired(req.arrival)
        plans=generate_pairs(g,req.source,req.destination,2)
        for p in plans:
            if led.admit(req,p):
                accepted.append((req,p)); break
        assert led.verify_full_scenario_recompute()
    for scenario_row in led.loads:
        assert (scenario_row>=0).all()


def test_prefix_candidate_invariance():
    g=build_network(NetworkSpec(nodes=60,seed=101))
    d=generate_demands(g,1,70000,False)[0]
    two=generate_pairs(g,d.source,d.destination,2)
    six=generate_pairs(g,d.source,d.destination,6)
    assert two == six[:2]


def test_rejects_heldout_seeds(tmp_path):
    with pytest.raises(ValueError):
        probe(tmp_path/'x',stream_seeds=(73000,))


def test_stage1_probe_smoke(tmp_path):
    out,_=probe(tmp_path/'out',sizes=(60,),topology_seeds=(101,),
                stream_seeds=(70000,),requests_per_node=.15,
                policies=('greedy','random_feasible'),k_values=(2,6),
                traffic_modes=('static','dynamic'))
    with (out/'probe_runs.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert all(int(x['admitted'])+int(x['rejected'])==9 for x in rows)
    assert len(list(csv.DictReader((out/'candidate_inventory.csv').open())))==9
    with pytest.raises(FileExistsError):
        probe(out,sizes=(60,),requests_per_node=.15)


def test_ledger_against_independent_explicit_fault_loop():
    """Independent scenario/path choice oracle, not ScenarioLedger._partition()."""
    from phase5f.protection import edge as canon
    g=build_network(NetworkSpec(nodes=60,seed=107))
    demands=generate_demands(g,24,70003,True)
    ledger=ScenarioLedger(g,10)
    for d in demands:
        ledger.release_expired(d.arrival)
        for pair in generate_pairs(g,d.source,d.destination,2):
            if ledger.admit(d,pair):break
    arcs=sorted(ledger.edge_to_i,key=lambda x: ledger.edge_to_i[x])
    expected=np.zeros_like(ledger.loads)
    for scenario in range(len(arcs)+1):
        failure = None if scenario==0 else arcs[scenario-1]
        for request,pair in ledger.active.values():
            working_set={canon(a,b) for a,b in zip(pair.working,pair.working[1:])}
            path = pair.backup if failure in working_set else pair.working
            for a,b in zip(path,path[1:]):
                e=canon(a,b)
                j=ledger.edge_to_i[e]
                expected[scenario,2*j] += request.bandwidth_mbps
                expected[scenario,2*j+1] += request.bandwidth_mbps
    assert np.array_equal(expected,ledger.loads)


def test_no_working_path_link_survives_its_own_fault():
    g=build_network(NetworkSpec(nodes=60,seed=108))
    requests=generate_demands(g,10,70000,False)
    for d in requests:
        for p in generate_pairs(g,d.source,d.destination,6):
            gfailed=g.copy()
            for a,b in zip(p.working,p.working[1:]):
                gfailed.remove_edge(a,b)
            assert all(gfailed.has_edge(a,b) for a,b in zip(p.backup,p.backup[1:]))
