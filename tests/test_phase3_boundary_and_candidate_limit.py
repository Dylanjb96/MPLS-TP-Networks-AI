#Additional Phase 3 negative/boundary tests; no MSRP or operational claims.

#Copy to tests/test_phase3_boundary_and_candidate_limit.py and run:
  #python -m pytest -q tests/test_phase3_boundary_and_candidate_limit.py

from __future__ import annotations

from dataclasses import replace
import copy

import pytest

from phase3.model import ProtectionSimulator, generate_candidates, PathSegment, edge_key
from railway_tp.requests import ServiceRequest
from railway_tp.topology import TopologyConfig, make_topology


def fixture_graph(main=1000, sub=1000, n=12, mode="dual"):
    cfg = TopologyConfig(nodes=n, seed=42, gateway_mode=mode,
                         main_nodes=6 if n == 12 else 12,
                         subring_count=1 if n == 12 else 4,
                         main_capacity_mbps=main, sub_capacity_mbps=sub)
    return make_topology(cfg)


def req(code, source, destination, bandwidth):
    return ServiceRequest(code, source, destination, bandwidth, "operational", 3, "boundary_test")


def first(sim, q):
    plans = sim.candidates(q, max_candidates=64)
    assert plans, f"Fixture error: no plans for {q}"
    return plans[0]


@pytest.mark.parametrize("headroom,bandwidth,expected", [
    (0, 1000, True), (0, 1001, False),
    (10, 900, True), (10, 901, False),
    (20, 800, True), (20, 801, False),
    (99, 10, True), (99, 11, False),
])
def test_exact_single_service_capacity_boundary(headroom, bandwidth, expected):
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings, headroom_percent=headroom)
    q = req(f"B{bandwidth}", 0, 1, bandwidth)
    assert sim.audit(q, first(sim, q)).feasible is expected


def test_headroom_floor_rounding_and_rejection_atomicity():
    G, rings = fixture_graph(main=1001, sub=1001)
    sim = ProtectionSimulator(G, rings, headroom_percent=10)
    assert sim.available_capacity(0, 1) == 900  # floor(1001 * 0.90)
    q_ok = req("S1", 0, 1, 900)
    assert sim.admit(q_ok, first(sim, q_ok)).feasible
    before = sim.state_digest()
    q_bad = req("S2", 0, 1, 1)
    assert not sim.admit(q_bad, first(sim, q_bad)).feasible
    assert sim.state_digest() == before


@pytest.mark.parametrize("headroom", [-1, 100, 10.5, "10"])
def test_invalid_headroom_rejected(headroom):
    G, rings = fixture_graph()
    with pytest.raises(ValueError):
        ProtectionSimulator(G, rings, headroom_percent=headroom)


@pytest.mark.parametrize("source,destination,bandwidth", [
    (0, 1, 0), (0, 1, -1), (0, 1, 0.5), (0, 1, "200"),
    (0, 0, 50), (999, 1, 50), (0, 999, 50),
])
def test_invalid_requests_fail_without_committing(source, destination, bandwidth):
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings)
    q = req("INVALID", source, destination, bandwidth)
    before = sim.state_digest()
    result = sim.admit(q, None)
    assert not result.feasible and "INVALID_REQUEST" in result.reason_codes
    assert sim.state_digest() == before


def test_duplicate_id_rejected_without_mutating_accepted_service():
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings)
    q = req("A", 6, 7, 100)
    assert sim.admit(q, first(sim, q)).feasible
    before = sim.state_digest()
    result = sim.admit(q, first(sim, q))
    assert result.reason_codes == ("DUPLICATE_SERVICE_ID",)
    assert sim.state_digest() == before


def test_malformed_candidate_is_rejected_without_committing():
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings)
    q = req("A", 6, 7, 100)
    good = first(sim, q)
    bad = replace(good, candidate_id="CORRUPTED")
    assert not sim.admit(q, bad).feasible
    assert len(sim.accepted()) == 0


def test_corrupted_fault_path_is_rejected():
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings)
    q = req("A", 6, 7, 100)
    p = first(sim, q)
    failure_id = G.edges[6, 7]["physical_link_id"]
    wrong = dict(p.fault_paths)
    wrong[failure_id] = p.normal_path
    result = sim.admit(q, replace(p, fault_paths=wrong))
    assert not result.feasible
    assert result.reason_codes == ("FAULT_PLAN_NOT_PRECOMPUTED_AS_DECLARED",)
    assert not sim.accepted()


def test_postfault_exact_900_and_901_with_rollback():
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings, headroom_percent=10)
    first_request = req("A", 6, 7, 500)
    assert sim.admit(first_request, first(sim, first_request)).feasible
    second_request = req("B", 8, 9, 400)
    assert sim.admit(second_request, first(sim, second_request)).feasible
    assert len(sim.accepted()) == 2

    sim2 = ProtectionSimulator(G, rings, headroom_percent=10)
    assert sim2.admit(first_request, first(sim2, first_request)).feasible
    prev = sim2.state_digest()
    prev_graph = copy.deepcopy(list(G.edges(data=True)))
    too_much = req("C", 8, 9, 401)
    fail = sim2.admit(too_much, first(sim2, too_much))
    assert not fail.feasible and fail.reason_codes == ("POSTFAULT_CAPACITY_EXCEEDED",)
    assert sim2.state_digest() == prev
    assert list(G.edges(data=True)) == prev_graph


def test_asymmetric_reverse_capacity_is_checked():
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings, headroom_percent=0,
                              arc_capacity_overrides={(1, 0): 800})
    q = req("ASYM", 0, 1, 801)
    decision = sim.audit(q, first(sim, q))
    assert not decision.feasible
    assert decision.reason_codes == ("NORMAL_CAPACITY_EXCEEDED",)


@pytest.mark.parametrize("limit", [0, -1])
def test_candidate_limit_must_be_positive(limit):
    G, rings = fixture_graph()
    q = req("L", 6, 7, 50)
    with pytest.raises(ValueError):
        generate_candidates(G, rings, q, max_candidates=limit)


def test_candidate_cap_outputs_are_deterministic_prefixes():
    G, rings = fixture_graph(n=60)
    q = req("L", 12, 25, 50)
    all_plans = generate_candidates(G, rings, q, max_candidates=10_000)
    assert len(all_plans) >= 2
    for limit in (1, 2, 4, 12, 24):
        limited = generate_candidates(G, rings, q, max_candidates=limit)
        assert [p.candidate_id for p in limited] == [p.candidate_id for p in all_plans[:limit]]


def test_candidate_limit_does_not_override_independent_validator():
    G, rings = fixture_graph(n=60)
    q = req("L", 12, 25, 100)
    sim = ProtectionSimulator(G, rings, headroom_percent=10)
    for candidate in sim.candidates(q, max_candidates=24):
        assert sim.audit(q, candidate).feasible
    assert sim.state_digest() == ProtectionSimulator(G, rings, 10).state_digest()


def test_physical_edge_fault_not_traversed_in_either_direction():
    G, rings = fixture_graph()
    sim = ProtectionSimulator(G, rings)
    q = req("A", 6, 7, 100)
    plan = first(sim, q)
    failed_id = G.edges[6, 7]["physical_link_id"]
    failed = sim.by_id[failed_id]
    for path in (plan.fault_paths[failed_id], tuple(reversed(plan.fault_paths[failed_id]))):
        assert all(edge_key(u, v) != failed for u, v in zip(path, path[1:]))