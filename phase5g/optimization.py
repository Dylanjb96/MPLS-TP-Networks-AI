"""Exact online enumeration of a scenario-aware minimax candidate objective.

Only current ledger state and current request are used. With <=6 candidates,
explicit enumeration solves the one-decision discrete optimization exactly.
This is NOT a MILP, not an offline optimum, and has no future information.
"""
from __future__ import annotations
import numpy as np
from phase5f.protection import ScenarioLedger
from phase5f_stage2.candidates import exact_footprint


def objective(ledger:ScenarioLedger,demand,plan,*,existing_peak=None):
    if not ledger.feasible(demand,plan):
        raise ValueError('optimization candidate must be feasible')
    if existing_peak is None: existing_peak=ledger.peak_utilization()
    other,changed,wi,bi=ledger._partition(plan)
    b=int(demand.bandwidth_mbps)
    peak=float(existing_peak)
    energy=0.0
    for scenarios,arcs in ((other,wi),(changed,bi)):
        if len(scenarios)==0 or len(arcs)==0: continue
        old=ledger.loads[np.ix_(scenarios,arcs)].astype(np.float64)
        fraction=b/ledger.cap[arcs].astype(np.float64)
        util=old/ledger.cap[arcs][None,:]
        peak=max(peak,float(np.max(util+fraction[None,:])))
        # Explicit scenario-normalized convex load-balancing energy.
        energy+=float(np.sum(2*util*fraction[None,:]+fraction[None,:]**2))
    # Normalize per fault scenario (unrelated graph edge count otherwise
    # dominates between network sizes, while choices within one state agree).
    energy/=ledger.m+1
    return (round(peak,12),round(energy,12),exact_footprint(ledger.graph,plan),
            plan.working,plan.backup)


def scenario_minimax_choice(ledger:ScenarioLedger,demand,plans,feasible):
    if not feasible:raise ValueError('no feasible alternatives')
    if len(feasible)!=len(set(feasible)):raise ValueError('duplicate feasible indices')
    existing_peak=ledger.peak_utilization()
    return min(feasible,key=lambda i:objective(ledger,demand,plans[i],existing_peak=existing_peak))
