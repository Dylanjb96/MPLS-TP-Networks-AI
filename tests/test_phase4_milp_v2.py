#MILP V2 exact lexicographic tie-breaking regression tests.

#The V1 MILP remains selectable as `milp` and is not modified in meaning.

from __future__ import annotations
from dataclasses import replace
from fractions import Fraction

import pytest

from railway_tp.topology import TopologyConfig, make_topology
from railway_tp.requests import generate_requests
from phase3.model import ProtectionSimulator
from phase4.policies import (ALL_POLICIES, POLICIES, choose, choose_greedy,
                              choose_milp, choose_milp_v2,
                              _exact_peak_utilization)
from phase4.cli import benchmark


@pytest.fixture(scope='module')
def net60():
    cfg = TopologyConfig(nodes=60, seed=42, gateway_mode='dual', main_nodes=12,
                         subring_count=4, main_capacity_mbps=10000,
                         sub_capacity_mbps=1000, length_km=2.0)
    G, rings = make_topology(cfg)
    return cfg, G, rings


@pytest.fixture(scope='module')
def net150():
    cfg = TopologyConfig(nodes=150, seed=42, gateway_mode='dual', main_nodes=30,
                         subring_count=6, main_capacity_mbps=10000,
                         sub_capacity_mbps=1000, length_km=2.0)
    G, rings = make_topology(cfg)
    return cfg, G, rings


def test_v1_public_policy_names_unchanged():
    assert POLICIES == ('greedy', 'cspf', 'milp')
    assert ALL_POLICIES == ('greedy', 'cspf', 'milp', 'milp_v2')


def _key(sim, req, plan, base):
    return (_exact_peak_utilization(sim, req, plan, base),
            len(plan.normal_path)-1, plan.candidate_id)


def test_v2_matches_exhaustive_lexicographic_oracle(net60):
    _, G, rings = net60
    sim = ProtectionSimulator(G, rings, 10)
    for req in generate_requests(G, 6, seed=2027):
        candidates = sim.candidates(req, 2)
        base = sim._ledger(sim.accepted())
        feas = [plan for plan in candidates if sim.audit(req, plan).feasible]
        expected = min(feas, key=lambda p: _key(sim, req, p, base)) if feas else None
        result = choose_milp_v2(sim, req, candidates)
        if expected is None:
            assert not result.result.feasible
            assert result.solver_status in ('INFEASIBLE', 'NO_CANDIDATES')
        else:
            assert result.result.feasible
            assert result.plan.candidate_id == expected.candidate_id
            assert result.solver_status == 'OPTIMAL_ONLINE_MINIMAX_LEXICOGRAPHIC_V2'
            assert isinstance(_key(sim, req, expected, base)[0], Fraction)


def _initial_150_greedy_state(net150):
    _, G, rings = net150
    requests = generate_requests(G, 5, seed=2029)
    sim = ProtectionSimulator(G, rings, 10)
    for req in requests[:4]:
        assert choose_greedy(sim, req, sim.candidates(req, 2)).result.feasible
    return sim, requests[4]


def test_150_seed2029_r0005_shorter_equal_primary_candidate(net150):
    sim, req = _initial_150_greedy_state(net150)
    choices = sim.candidates(req, 2)
    assert len(choices) == 2
    assert sorted(len(p.normal_path)-1 for p in choices) == [11, 59]
    base = sim._ledger(sim.accepted())
    scores = [_exact_peak_utilization(sim, req, p, base) for p in choices]
    assert scores == [Fraction(4,9), Fraction(4,9)]
    outcome = choose_milp_v2(sim, req, choices)
    assert outcome.result.feasible
    assert len(outcome.plan.normal_path)-1 == 11
    assert outcome.score == pytest.approx(4/9)


def test_150_seed2029_v1_regression_remains_available(net150):
    sim, req = _initial_150_greedy_state(net150)
    candidates = sim.candidates(req, 2)
    outcome = choose_milp(sim, req, candidates)
    assert outcome.result.feasible
    assert outcome.solver_status == 'OPTIMAL_ONLINE_MINIMAX'
    # The original published V1 formulation retains its tiny epsilon term.
    assert outcome.score is not None


def test_v2_candidate_order_independent(net150):
    sim, req = _initial_150_greedy_state(net150)
    pool = sim.candidates(req, 2)
    _, G, rings = net150
    duplicate = ProtectionSimulator(G, rings, 10)
    for oldreq, oldplan in sim.accepted():
        assert duplicate.admit(oldreq, oldplan).feasible
    forward = choose_milp_v2(sim, req, pool)
    backward = choose_milp_v2(duplicate, req, list(reversed(pool)))
    assert forward.plan.candidate_id == backward.plan.candidate_id
    assert sim.state_digest() == duplicate.state_digest()


def test_v2_impossible_service_is_atomic(net60):
    _, G, rings = net60
    sim = ProtectionSimulator(G, rings, 10)
    req = replace(generate_requests(G, 1, seed=2027)[0], bandwidth_mbps=20000)
    prior = sim.state_digest()
    result = choose_milp_v2(sim, req, sim.candidates(req, 2))
    assert not result.result.feasible
    assert sim.state_digest() == prior


def test_v2_timeout_validation(net60):
    _, G, rings = net60
    sim = ProtectionSimulator(G, rings, 10)
    req = generate_requests(G, 1, seed=2027)[0]
    with pytest.raises(ValueError, match='timeout'):
        choose_milp_v2(sim, req, sim.candidates(req, 2), timeout_sec=0)


def test_v2_nonoptimal_fallback_label(net60, monkeypatch):
    import phase4.policies as mod
    _, G, rings = net60
    sim = ProtectionSimulator(G, rings, 10)
    req = generate_requests(G, 1, seed=2027)[0]
    monkeypatch.setattr(mod, '_milp_select',
                        lambda sim, req, candidates, timeout_sec, **kwargs:
                            (None, 'SOLVER_NONOPTIMAL_STATUS_1', None))
    result = choose_milp_v2(sim, req, sim.candidates(req, 2))
    assert result.result.feasible
    assert result.solver_status == 'FALLBACK_GREEDY_V2_SOLVER_NONOPTIMAL_STATUS_1'


def test_v2_benchmark_paired_hashes_and_label(net60, tmp_path):
    cfg, _, _ = net60
    out = tmp_path/'pilot'
    result = benchmark(cfg, 3, [2027], [10], ['greedy', 'milp', 'milp_v2'], out, 2)
    assert result['runs'] == 3
    import csv, json
    rows = list(csv.DictReader((out/'phase4_runs.csv').open()))
    assert len({r['request_sha256'] for r in rows}) == 1
    assert {r['policy'] for r in rows} == {'greedy', 'milp', 'milp_v2'}
    manifest=json.loads((out/'runs'/'h10_d2027_milp_v2'/'manifest.json').read_text())
    assert manifest['algorithm_version'] == 'V2_PRIMARY_MILP_EXACT_LEXICOGRAPHIC'

