"""Online RL environment. Phase-3 audit/admit is the sole feasibility authority.

Candidate actions 0..K-1 choose a policy-valid, capacity-feasible plan. Action
K is a FORCED rejection only when no candidate can be admitted. Deliberate
rejection when a service is feasible is DISABLED to match Phase-4 baselines.
The network state and rewards never expose future requests to the agent.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence
import numpy as np

from phase3.model import NORMAL, ProtectionSimulator, SCHEME, edge_key, _path_edges, ProtectedPlan
from railway_tp.requests import ServiceRequest, generate_requests
from railway_tp.topology import TopologyConfig, make_topology

FEATURE_NAMES = (
    'bandwidth_400', 'priority_5', 'source_fraction', 'destination_fraction',
    'cat_within_subring', 'cat_main_to_main', 'cat_main_to_subring', 'cat_subring_to_subring',
    'request_progress', 'admission_ratio_so_far', 'admitted_fraction_of_horizon',
    'current_peak_scenario_utilization', 'headroom_fraction', 'candidate_count_fraction',
    'feasible_candidate_fraction', 'candidate_rank', 'working_hops_normalized',
    'mean_fault_hops_normalized', 'max_fault_hops_normalized', 'fraction_changed_fault_paths',
    'prospective_peak_scenario_utilization', 'increase_in_peak_utilization',
    'prospective_working_arc_peak', 'working_edge_fraction', 'forced_reject_flag',
    'capacity_feasible_flag'
)
FEATURE_VERSION = 'PHASE5_STATE_CANDIDATE_FEATURES_V1'


def stream_sha256(requests: Sequence[ServiceRequest]) -> str:
    return hashlib.sha256(json.dumps([asdict(r) for r in requests],
                       sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class PreparedStream:
    requests: tuple[ServiceRequest, ...]
    candidate_sets: tuple[tuple[ProtectedPlan, ...], ...]
    request_sha256: str
    generation_seconds: float


def prepare_stream(cfg: TopologyConfig, count: int, seed: int, k: int) -> PreparedStream:
    if count < 1 or k < 1:
        raise ValueError('count and max_candidates must both be positive')
    graph, rings = make_topology(cfg)
    requests = tuple(generate_requests(graph, count, seed=seed))
    generator = ProtectionSimulator(graph, rings, headroom_percent=10)
    t0 = time.perf_counter()
    plans = tuple(tuple(generator.candidates(r, k)) for r in requests)
    elapsed = time.perf_counter() - t0
    return PreparedStream(requests, plans, stream_sha256(requests), elapsed)


@dataclass(frozen=True)
class Observation:
    rows: np.ndarray              # [max_candidates+1, feature_count]
    mask: np.ndarray              # [max_candidates+1], boolean
    request_id: str


class OnlineEpisode:
    def __init__(self, cfg: TopologyConfig, prepared: PreparedStream,
                 headroom_percent: int = 10, max_candidates: int = 2):
        self.cfg = cfg
        self.prepared = prepared
        self.k = max_candidates
        if not 1 <= self.k:
            raise ValueError('max_candidates must be >=1')
        if any(len(x) > self.k for x in prepared.candidate_sets):
            raise ValueError('prepared candidates exceed max_candidates')
        graph, rings = make_topology(cfg)
        self.sim = ProtectionSimulator(graph, rings, headroom_percent)
        self.pos = 0
        self.records: list[dict] = []
        self._current: Observation | None = None
        self._current_audits = ()
        self._total_weight = sum(r.priority_weight for r in prepared.requests)
        if not prepared.requests or self._total_weight <= 0:
            raise ValueError('episode requires positive-weight requests')

    @property
    def done(self) -> bool:
        return self.pos >= len(self.prepared.requests)

    def observe(self) -> Observation:
        if self.done:
            raise RuntimeError('cannot observe past episode end')
        if self._current is not None:
            return self._current
        req = self.prepared.requests[self.pos]
        plans = self.prepared.candidate_sets[self.pos]
        audits = tuple(self.sim.audit(req, p) for p in plans)
        self._current_audits = audits
        feasible = [bool(a.feasible) for a in audits]
        mask = np.zeros(self.k + 1, dtype=np.bool_)
        mask[:len(plans)] = feasible
        mask[self.k] = not any(feasible)
        assert int(mask.sum()) >= 1
        rows = np.zeros((self.k + 1, len(FEATURE_NAMES)), dtype=np.float32)
        base = self.sim._ledger(self.sim.accepted())
        base_peak = max((load / self.sim.available_capacity(u, v)
                         for scenario, arcs in base.items()
                         for (u, v), load in arcs.items()
                         if edge_key(u, v) != self.sim.by_id.get(scenario)), default=0.0)
        n = len(self.sim.G)
        count = len(self.prepared.requests)
        admitted_so_far = len(self.sim.accepted())
        categories = ('within-subring', 'main-to-main', 'main-to-subring', 'subring-to-subring')
        shared = [
            min(2.0, req.bandwidth_mbps / 400.0),
            min(1.0, req.priority_weight / 5.0),
            req.source / max(1, n-1), req.destination / max(1, n-1),
            *(float(req.endpoint_category == cat) for cat in categories),
            self.pos / count, admitted_so_far / max(1, self.pos),
            admitted_so_far / count, min(2.0, base_peak),
            self.sim.headroom_percent / 100.0,
            len(plans) / self.k, sum(feasible) / self.k,
        ]
        assert len(shared) == 15
        for i, (p, audit) in enumerate(zip(plans, audits)):
            fault_lengths = [len(path)-1 for path in p.fault_paths.values()]
            changed = sum(path != p.normal_path for path in p.fault_paths.values())
            peak = audit.worst_utilization if audit.worst_utilization is not None else 2.0
            peak = float(min(2.0, peak))
            normal = base[NORMAL]
            arcs = [(u, v) for a,b in _path_edges(p.normal_path) for u,v in ((a,b),(b,a))]
            normal_peak = max(((normal.get((u,v),0) + req.bandwidth_mbps)
                               / self.sim.available_capacity(u,v) for u,v in arcs), default=0)
            details = [
                i / max(1, self.k-1),
                (len(p.normal_path)-1) / n,
                (sum(fault_lengths)/len(fault_lengths)/n if fault_lengths else 0.0),
                (max(fault_lengths)/n if fault_lengths else 0.0),
                changed/max(1, len(p.fault_paths)),
                peak, max(0.0, peak-base_peak),
                float(min(2.0, normal_peak)),
                (len(set(edge_key(u,v) for u,v in _path_edges(p.normal_path)))
                 / max(1, self.sim.G.number_of_edges())),
                0.0, float(audit.feasible)
            ]
            rows[i, :] = np.asarray(shared+details, dtype=np.float32)
        # A no-feasible-candidate rejection remains an explicit transition in
        # replay, so the preceding step can bootstrap across this zero-reward step.
        rows[self.k, :] = np.asarray(shared + [0.0]*9 + [1.0, float(mask[self.k])],
                                          dtype=np.float32)
        self._current = Observation(rows, mask, req.request_id)
        return self._current

    def step(self, action: int) -> tuple[float, bool, dict]:
        if self.done:
            raise RuntimeError('episode already complete')
        obs = self.observe()
        if not isinstance(action, (int, np.integer)) or action < 0 or action > self.k or not obs.mask[action]:
            raise ValueError('invalid or capacity-infeasible action (rejection is forced only)')
        req = self.prepared.requests[self.pos]
        plans = self.prepared.candidate_sets[self.pos]
        if action == self.k:
            result = self.sim.audit(req, plans[0] if plans else None)
            if result.feasible:
                raise AssertionError('forced reject must be infeasible')
            plan = None
        else:
            plan = plans[int(action)]
            result = self.sim.admit(req, plan)
            if not result.feasible:
                raise AssertionError('Phase 3 rejected the masked-feasible DDQN action')
        reward = req.priority_weight / 5.0 if result.feasible else 0.0
        record = {
            'request_id': req.request_id, 'source': req.source, 'destination': req.destination,
            'endpoint_category': req.endpoint_category, 'service_class': req.service_class,
            'priority_weight': req.priority_weight, 'bandwidth_mbps': req.bandwidth_mbps,
            'admitted': int(result.feasible), 'reward': round(reward, 6),
            'action': int(action), 'feasible_actions': int(obs.mask.sum()),
            'candidate_count': len(plans),
            'candidate_id': plan.candidate_id if plan else '',
            'working_hops': len(plan.normal_path)-1 if plan else '',
            'reason': ';'.join(result.reason_codes),
            'worst_utilization': result.worst_utilization if result.worst_utilization is not None else '',
            'solver_status': 'MASKED_DDQN_ACTION' if result.feasible else 'FORCED_REJECT',
        }
        self.records.append(record)
        self.pos += 1
        self._current = None
        self._current_audits = ()
        return reward, self.done, record

    def summary(self) -> dict:
        admitted = self.sim.accepted()
        total = len(self.prepared.requests)
        return {
            'requests':total, 'admitted':len(admitted), 'rejected':total-len(admitted),
            'admission_ratio':len(admitted)/total,
            'weighted_admission_ratio':sum(r.priority_weight for r, _ in admitted)/self._total_weight,
            'total_reward':sum(r.priority_weight / 5.0 for r, _ in admitted),
            'state_digest':self.sim.state_digest(),
            'peak_scenario_utilization':self.sim.ledger_summary()['peak_utilization'],
        }
