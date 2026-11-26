"""Causal, size-normalized features for up to six feasible protected candidates.

Only current request, topology and *current* reservation ledger are consulted.
No lookahead, request-sequence suffix, outcome labels or future demands appear.
"""
from __future__ import annotations
import numpy as np
from phase5f.protection import ScenarioLedger, ProtectedPair
from phase5f.traffic import Demand
from phase5f_stage2.candidates import exact_footprint

MAX_ACTIONS = 6
CAND_DIM = 17
GLOBAL_DIM = 8


def _safe_min_fraction(ledger: ScenarioLedger, scenarios, arcs, bandwidth: int) -> float:
    if len(arcs) == 0 or len(scenarios) == 0:
        return 1.0
    # All quantities are measured at decision time; scenario-dependent reserve.
    cap = ledger.cap[arcs].astype(np.float64)
    available = ledger.cap[arcs][None, :] - ledger.loads[np.ix_(scenarios, arcs)] - bandwidth
    return float(np.clip(np.min(available / cap[None, :]), -10, 1))


def _worst_prior_util(ledger, scenarios, arcs) -> float:
    if len(arcs) == 0 or len(scenarios) == 0:
        return 0.0
    return float(np.max(ledger.loads[np.ix_(scenarios, arcs)] / ledger.cap[arcs][None, :]))


def encode(graph, ledger: ScenarioLedger, demand: Demand,
           plans: tuple[ProtectedPair, ...], feasible: list[int],
           *, k: int = MAX_ACTIONS, dynamic: bool = False):
    """Returns candidate_features, global_features, action_mask.

    All candidate slots outside the current feasible set are masked.  Each
    active slot is indexed by the original candidate ID (no reordering).
    """
    if not 1 <= k <= MAX_ACTIONS or len(plans) > k:
        raise ValueError('candidate shape exceeds the frozen six-action bound')
    if not feasible or len(set(feasible)) != len(feasible):
        raise ValueError('encode requires a nonempty, distinct feasible set')
    if any(i < 0 or i >= len(plans) for i in feasible):
        raise ValueError('invalid feasible candidate index')
    c = np.zeros((MAX_ACTIONS, CAND_DIM), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    first = feasible[0]
    for i in feasible:
        pair = plans[i]
        other, changed, wi, bi = ledger._partition(pair)
        ework = list(zip(pair.working, pair.working[1:]))
        ebackup = list(zip(pair.backup, pair.backup[1:]))
        wlen = sum(graph[u][v]['length_km'] for u,v in ework)
        blen = sum(graph[u][v]['length_km'] for u,v in ebackup)
        footprint = float(exact_footprint(graph,pair))*demand.bandwidth_mbps
        residual_w = _safe_min_fraction(ledger,other,wi,demand.bandwidth_mbps)
        residual_b = _safe_min_fraction(ledger,changed,bi,demand.bandwidth_mbps)
        utilization_w = _worst_prior_util(ledger,other,wi)
        utilization_b = _worst_prior_util(ledger,changed,bi)
        capacities = [graph[u][v]['capacity_mbps'] for u,v in ework+ebackup]
        count = len(ework)+len(ebackup)
        # Fixed scales are declared feature engineering choices, not learned
        # statistics fitted on held-out data. No future arrivals are used.
        c[i]=[
            min(wlen/200.0,10), min(blen/200.0,10),
            min(len(ework)/80.0,10), min(len(ebackup)/80.0,10),
            min(footprint/3.0,10),
            residual_w, residual_b, utilization_w,utilization_b,
            min(min(capacities)/10000.0,5),
            min(sum(capacities)/count/10000.0,5),
            min((len(ework)+len(ebackup))/100.0,10),
            1.0 if i==first else 0.0,
            i/float(max(1,k-1)),
            max(0.0,min(residual_w,residual_b)),
            min(wlen/max(blen,1e-6),10),
            min(blen/max(wlen,1e-6),10),
        ]
        mask[i]=True
    n = graph.number_of_nodes()
    active=len(ledger.active)
    # No future traffic or expiry distribution is passed into the encoder.
    gl=np.asarray([
        min(n/500.0,2.0),
        min(active/max(n,1),10.0),
        min(demand.bandwidth_mbps/200.0,10.0),
        min(demand.priority_weight/5.0,1.0),
        len(feasible)/float(k),
        1.0 if dynamic else 0.0,
        min(demand.arrival/2000.0,10.0),
        min(ledger.peak_utilization(),2.0),
    ],dtype=np.float32)
    if not np.isfinite(c).all() or not np.isfinite(gl).all():
        raise AssertionError('nonfinite state representation')
    return c, gl, mask
