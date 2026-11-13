#Online Phase-4 provisioning policies using the Phase-3 feasibility oracle.

#Each policy receives only the request arriving now, the previously committed
#services, and the SAME precomputed policy-valid candidate pool. No policy sees
#future requests. The independent ProtectionSimulator.audit/admit is the sole
#source of truth for all acceptance decisions.

#Note: the CSPF-style policy ranks ring-policy plans by normal working hops
#then projected normal-state congestion. It is not unconstrained graph CSPF.
#The online MILP minimizes the worst directed utilization across normal and
#all single-link failure scenarios for ONE new service, given fixed prior
#acceptances. It is NOT a horizon-wide/offline multicommodity optimum.

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Any

from phase3.model import (NORMAL, ProtectionSimulator, ProtectedPlan,
                          AdmissionResult, _path_edges)
from railway_tp.requests import ServiceRequest

POLICIES = ('greedy', 'cspf', 'milp')


@dataclass(frozen=True)
class Choice:
    result: AdmissionResult
    plan: ProtectedPlan | None
    evaluated: int
    solver_status: str
    score: float | None = None


def _commit(sim: ProtectionSimulator, req: ServiceRequest,
            plan: ProtectedPlan, evaluated: int, status: str,
            score: float | None = None) -> Choice:
    # Never take an optimizer's word for feasibility. Atomic Phase-3 commit.
    result = sim.admit(req, plan)
    if not result.feasible:
        raise AssertionError('The independent Phase 3 validator rejected a chosen plan: ' + str(result))
    return Choice(result, plan, evaluated, status, score)


def _reject(sim: ProtectionSimulator, req: ServiceRequest,
            candidates: list[ProtectedPlan], evaluated: int,
            status: str) -> Choice:
    # Same deterministic first-candidate rejection attribution as Phase 3.
    audit = sim.audit(req, candidates[0]) if candidates else sim.audit(req, None)
    if audit.feasible:
        raise AssertionError('Unable to reject a feasible first candidate')
    return Choice(audit, None, evaluated, status)


def _normal_congestion_on_path(sim: ProtectionSimulator, req: ServiceRequest,
                               plan: ProtectedPlan, base_normal: dict) -> float:
    #Highest prospective utilization among directed arcs in the working path.
    bw = req.bandwidth_mbps
    ratios = []
    for a, b in _path_edges(plan.normal_path):
        for u, v in ((a, b), (b, a)):
            ratios.append((base_normal.get((u, v), 0) + bw) /
                          sim.available_capacity(u, v))
    return max(ratios, default=0.0)


def choose_greedy(sim: ProtectionSimulator, req: ServiceRequest,
                  candidates: list[ProtectedPlan]) -> Choice:
    #Exact first-feasible ordering used by Phase 3: shortest hop first.
    checked = 0
    for plan in candidates:
        checked += 1
        if sim.audit(req, plan).feasible:
            return _commit(sim, req, plan, checked, 'FIRST_FEASIBLE')
    return _reject(sim, req, candidates, checked, 'NO_FEASIBLE_CANDIDATE')


def choose_cspf(sim: ProtectionSimulator, req: ServiceRequest,
                candidates: list[ProtectedPlan]) -> Choice:
    #Capacity-constrained shortest working path across identical policy plans.

    #Strictly validate protection under every fault before scoring, then rank by
    #working hop count; projected normal-state arc utilization breaks ties.
    
    base = sim._ledger(sim.accepted())[NORMAL]
    rank: list[tuple[tuple[int, float, str], ProtectedPlan]] = []
    for plan in candidates:
        if not sim.audit(req, plan).feasible:
            continue
        key = (len(plan.normal_path) - 1,
               _normal_congestion_on_path(sim, req, plan, base),
               plan.candidate_id)
        rank.append((key, plan))
    if not rank:
        return _reject(sim, req, candidates, len(candidates), 'NO_FEASIBLE_CANDIDATE')
    key, chosen = min(rank, key=lambda item: item[0])
    return _commit(sim, req, chosen, len(candidates), 'CSPF_STYLE_FEASIBLE_SHORTEST',
                   float(key[0]))


def _scipy():
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:
        raise RuntimeError('The Phase 4 MILP policy requires scipy. Install with: '
                           'python -m pip install "scipy>=1.11"') from exc
    return np, Bounds, LinearConstraint, milp, coo_matrix, vstack


def _milp_select(sim: ProtectionSimulator, req: ServiceRequest,
                 candidates: list[ProtectedPlan], timeout_sec: float
                 ) -> tuple[int | None, str, float | None]:
    #Robust min-max scenario MILP for one online request.

    #Binary x_j chooses exactly one candidate. Continuous z bounds worst
    #utilization. For every directed arc and fault state:
    #   L_existing(s,a) + sum_j delta(j,s,a) x_j <= C_usable(a) * z
    #with z <= 1. Previous choices cannot be rerouted and future requests are
    #unknown. Objective z + tiny deterministic hop tie-breaker.
    
    if not candidates:
        return None, 'NO_CANDIDATES', None
    np, Bounds, LinearConstraint, milp, coo_matrix, vstack = _scipy()
    n = len(candidates)
    base = sim._ledger(sim.accepted())
    # Sparse keys: include only directed arcs with a positive existing load or
    # that a candidate would use in one scenario. Unused arcs are irrelevant.
    deltas: list[dict[tuple[str, int, int], int]] = []
    active: set[tuple[str, int, int]] = set()
    for scenario, arcs in base.items():
        active.update((scenario, u, v) for (u, v), load in arcs.items() if load)
    for plan in candidates:
        inc: dict[tuple[str, int, int], int] = defaultdict(int)
        for scenario in (NORMAL, *sim.failures):
            path = plan.normal_path if scenario == NORMAL else plan.fault_paths[scenario]
            for a, b in _path_edges(path):
                inc[(scenario, a, b)] += req.bandwidth_mbps
                inc[(scenario, b, a)] += req.bandwidth_mbps
        deltas.append(inc)
        active.update(inc)

    rows=[]; cols=[]; vals=[]; ub=[]
    for i, (scenario, u, v) in enumerate(sorted(active)):
        cap = sim.available_capacity(u, v)
        load = base[scenario].get((u,v), 0)
        if cap <= 0:
            raise AssertionError('Nonpositive usable capacity')
        for j, incr in enumerate(deltas):
            delta = incr.get((scenario, u, v), 0)
            if delta:
                rows.append(i); cols.append(j); vals.append(float(delta))
        rows.append(i); cols.append(n); vals.append(-float(cap))
        ub.append(-float(load))
    m = len(ub)
    A = coo_matrix((vals, (rows, cols)), shape=(m, n+1)).tocsr()
    choice_row = coo_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(n))),
                            shape=(1, n+1)).tocsr()
    A = vstack((A, choice_row), format='csr')
    constraint = LinearConstraint(A, np.r_[np.full(m,-np.inf),1.0],
                                  np.r_[np.array(ub), 1.0])
    tie = np.array([len(p.normal_path)-1 for p in candidates], dtype=float)
    tie /= max(1.0, float(tie.max()))
    c = np.r_[1e-7 * tie, 1.0]
    bounds = Bounds(np.zeros(n+1), np.ones(n+1))
    integrality = np.r_[np.ones(n, dtype=int), 0]
    sol = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraint,
               options={'time_limit': float(timeout_sec), 'mip_rel_gap': 0.0})
    if sol.status == 0 and sol.x is not None:
        x = sol.x[:n]
        selected = np.flatnonzero(x > 0.5)
        if len(selected) != 1:
            raise AssertionError('MILP returned non-singleton binary selection')
        return int(selected[0]), 'OPTIMAL_ONLINE_MINIMAX', float(sol.fun)
    if sol.status == 2:
        return None, 'INFEASIBLE', None
    return None, 'SOLVER_NONOPTIMAL_STATUS_' + str(sol.status), None


def choose_milp(sim: ProtectionSimulator, req: ServiceRequest,
                candidates: list[ProtectedPlan],
                timeout_sec: float = 5.0) -> Choice:
    if timeout_sec <= 0:
        raise ValueError('timeout_sec must be positive')
    if not candidates:
        return _reject(sim, req, candidates, 0, 'NO_CANDIDATES')
    idx, status, obj = _milp_select(sim, req, candidates, timeout_sec)
    if idx is not None:
        return _commit(sim, req, candidates[idx], len(candidates), status, obj)
    if status == 'INFEASIBLE':
        # Cross-check robust infeasibility against the independent checker.
        if any(sim.audit(req, p).feasible for p in candidates):
            raise AssertionError('MILP incorrectly declared feasible candidates infeasible')
        return _reject(sim, req, candidates, len(candidates), status)
    # Fail-closed on solver timeout/errors: use explicit, recorded greedy
    # fallback rather than fabricating an optimal MILP result.
    fallback = choose_greedy(sim, req, candidates)
    return Choice(fallback.result, fallback.plan, len(candidates),
                  'FALLBACK_GREEDY_' + status, fallback.score)


def choose(sim: ProtectionSimulator, req: ServiceRequest,
           candidates: list[ProtectedPlan], policy: str,
           milp_timeout_sec: float = 5.0) -> Choice:
    if policy == 'greedy':
        return choose_greedy(sim, req, candidates)
    if policy == 'cspf':
        return choose_cspf(sim, req, candidates)
    if policy == 'milp':
        return choose_milp(sim, req, candidates, milp_timeout_sec)
    raise ValueError(f'Unknown Phase 4 policy {policy!r}; select {POLICIES}')
