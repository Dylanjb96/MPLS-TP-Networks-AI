"""Immutable Phase 5G Stage 4 prospective specification, derived from Stage 3."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json

PROTOCOL_SHA256='6edad86fb2f49328784c5e80ef60d967413f6bf4f30f64debf57ad3381ce1d43'
STAGE3_FREEZE_MANIFEST_SHA256='cde024cf52dbb56fc55c6380d5d0dd9c00fc731a2eac627276b8b88c03b8f04b'
SOURCE_LOCK_VERSION='PHASE5G_STAGE3_FROZEN_CONFIRMATORY_PLAN_V1'
FINAL_TOPOLOGIES=tuple(range(801,821))
FINAL_STREAMS=(77000,77001,77002)
DEV_TOPOLOGIES=tuple(range(501,521))
DEV_STREAMS=tuple(range(74000,74020))
PREVIOUS_FINAL_TOPOLOGIES=tuple(range(401,421))
PREVIOUS_FINAL_STREAMS=(73000,73001,73002)
POLICIES=('gnn_PPO','mlp_PPO','C_frozen','scenario_minimax','footprint','greedy')
PRIMARY=(('H_BASE','gnn_PPO','mlp_PPO'),('H_BASE','gnn_PPO','scenario_minimax'))
SECONDARY=(
    ('L_BASE','gnn_PPO','mlp_PPO'),('L_BASE','gnn_PPO','scenario_minimax'),
    ('H_BASE','C_frozen','scenario_minimax'),('L_BASE','C_frozen','scenario_minimax'),
    ('H_BASE','gnn_PPO','C_frozen'),('L_BASE','gnn_PPO','C_frozen'),
    ('H_BASE','gnn_PPO','greedy'),('L_BASE','gnn_PPO','greedy'),
)

@dataclass(frozen=True)
class Cell:
    id:str
    family:str
    stress:str
    k:int

CELLS=(Cell('H_BASE','hierarchical','base',6),
       Cell('L_BASE','ladder','base',6),
       Cell('H_BW','hierarchical','bandwidth_150',6),
       Cell('L_BW','ladder','bandwidth_150',6),
       Cell('H_HOLD','hierarchical','short_hold',6),
       Cell('L_HOLD','ladder','short_hold',6),
       Cell('H_K2','hierarchical','base',2),
       Cell('L_K2','ladder','base',2))
BY_CELL={c.id:c for c in CELLS}

# Prospectively documented operationalization of named Stage-3 stress arms.
# Identical to Stage-1 / phase5g.workflow.make_cohort transformations.
STRESS_SEMANTICS={
 'base':'Do not modify offered requests, physical capacities, arrival or expiry times.',
 'bandwidth_150':'Replace each integer bandwidth B (Mbps) by (3*B+1)//2, i.e. ceil(1.5*B); retain graph edge capacities, endpoints, priorities, expiry and arrival unchanged.',
 'short_hold':'For each dynamic request, expiry=arrival+max(1,(original_expiry-arrival)//2); retain bandwidth, priorities, endpoints, capacity and arrival unchanged.',
}
CONDITIONS=dict(nodes=500,load=4.0,mode='dynamic',headroom=10,max_working_paths=40,cohorts=480,policy_runs=2880)


def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def semantic_lock():
    doc=dict(protocol_sha256=PROTOCOL_SHA256,conditions=CONDITIONS,
             cells=[vars(c) for c in CELLS],policies=POLICIES,
             primary=PRIMARY,secondary=SECONDARY,
             final_topologies=FINAL_TOPOLOGIES,final_streams=FINAL_STREAMS,
             stress_semantics=STRESS_SEMANTICS,
             neural_k2_policy='truncate inventory to first two candidates, pad neural action encoding to 6 slots, oracle verifies k=2',
             paired_unit='topology cluster: mean of three within-topology stream differences in percentage points')
    return hashlib.sha256(canonical(doc)).hexdigest()


def validate_seeds(role,t,s):
    if role=='dev': allowed_t,allowed_s=DEV_TOPOLOGIES,DEV_STREAMS
    elif role=='final':allowed_t,allowed_s=FINAL_TOPOLOGIES,FINAL_STREAMS
    else:raise ValueError('unsupported access role; training/validation reserved for original Stage 2')
    if int(t) not in allowed_t or int(s) not in allowed_s:
        raise ValueError(f'RESERVED_SEED_GUARD role={role} topology={t} stream={s}')
    if int(t) in PREVIOUS_FINAL_TOPOLOGIES or int(s) in PREVIOUS_FINAL_STREAMS:
        raise ValueError('previous Phase 5F final seeds are forbidden')


def validate_protocol(root):
    p=Path(root)/'PHASE5G_STAGE3_PROTOCOL_LOCK.md'
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=PROTOCOL_SHA256:
        raise RuntimeError('Stage 3 protocol SHA256 changed; do not generate or evaluate final data')
    return p
