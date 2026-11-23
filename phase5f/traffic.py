"""Synthetic railway-inspired traffic; all bandwidths/priorities are assumptions.

Static and dynamic arms share arrival, endpoints, bandwidth and priorities.
Only the service-holding model differs; the agent sees the current state only.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import networkx as nx

# name: (weight, selection_weight, allowed synthetic Mbps)
CLASSES = {
    'signalling_control': (5, 19, (5, 10, 20)),
    'operational_voice': (5, 16, (2, 5, 10)),
    'level_crossing': (5, 16, (5, 10, 20)),
    'ocs_control': (3, 14, (10, 25, 50)),
    'cctv_video': (2, 23, (50, 100, 200)),
    'passenger_information': (1, 12, (25, 50, 100)),
}

@dataclass(frozen=True)
class Demand:
    request_id: str
    arrival: int
    expiry: int
    source: int
    destination: int
    bandwidth_mbps: int
    service_class: str
    priority_weight: int


def generate_demands(graph: nx.Graph, count: int, seed: int,
                     dynamic: bool) -> tuple[Demand, ...]:
    if count < 1:
        raise ValueError('count must be positive')
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    bb = list(graph.graph['backbone_nodes'])
    regions = [list(r) for r in graph.graph['regions']]
    result = []
    klasses = list(CLASSES)
    for i in range(count):
        c = rng.choices(klasses, weights=[CLASSES[k][1] for k in klasses])[0]
        priority, _, bw_choices = CLASSES[c]
        bandwidth = rng.choice(bw_choices)
        mode = rng.random()
        if mode < .42:
            r = rng.choice(regions)
            s, t = rng.sample(r, 2)
        elif mode < .76:
            r = rng.choice(regions)
            s, t = rng.choice(r), rng.choice(bb)
        else:
            r1, r2 = rng.sample(regions, 2)
            s, t = rng.choice(r1), rng.choice(r2)
        # Draw duration even in the static arm to ensure parity of subsequent RNG
        # draws across arms; static simply suppresses release before the horizon.
        duration = max(1, round(count * rng.choice((.15, .35, .60, 1.0, 1.5))))
        expiry = i + duration if dynamic else count + 1
        result.append(Demand(f'F{i+1:05d}', i, expiry, s, t,
                             bandwidth, c, priority))
    return tuple(result)


def stream_digest(demands: tuple[Demand, ...], include_expiry=True) -> str:
    rows = [asdict(d) for d in demands]
    if not include_expiry:
        for row in rows: row.pop('expiry')
    return hashlib.sha256(json.dumps(rows, sort_keys=True,
                         separators=(',', ':')).encode()).hexdigest()
