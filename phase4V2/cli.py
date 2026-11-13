"""Reproducible Phase 4 paired online-policy benchmark.

  python -m phase4.cli --config config/primary_60_dual.json \
    --requests 50 --seeds 2027 2028 2029 2030 2031 --headrooms 10 \
    --out outputs/phase4/pilot
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from railway_tp.topology import TopologyConfig, make_topology
from railway_tp.requests import ServiceRequest, generate_requests
from phase3.model import ProtectionSimulator, SCHEME
from .policies import POLICIES, ALL_POLICIES, choose


def _json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _csv(path: Path, rows: list[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def _hash(items: list[ServiceRequest]) -> str:
    return hashlib.sha256(json.dumps([asdict(r) for r in items], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()


def run_one(cfg: TopologyConfig, incoming: list[ServiceRequest], policy: str,
            headroom: int, max_candidates: int, output: Path,
            milp_timeout: float = 5.0) -> dict:
    """One complete sequence on fresh, independent simulator state."""
    if policy not in ALL_POLICIES:
        raise ValueError('unknown policy')
    if not incoming or max_candidates < 1:
        raise ValueError('nonempty request sequence and positive candidate limit required')
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing Phase4 run: ' + str(output))
    G, rings = make_topology(cfg)
    sim = ProtectionSimulator(G, rings, headroom_percent=headroom)
    decisions = []
    decision_s = 0.0; candidate_s = 0.0
    statuses = Counter()
    for req in incoming:
        t0 = time.perf_counter()
        plans = sim.candidates(req, max_candidates=max_candidates)
        candidate_elapsed = time.perf_counter() - t0
        t1 = time.perf_counter()
        choice = choose(sim, req, plans, policy, milp_timeout_sec=milp_timeout)
        selection_elapsed = time.perf_counter() - t1
        decision_s += selection_elapsed
        candidate_s += candidate_elapsed
        statuses[choice.solver_status] += 1
        decisions.append({
            'request_id': req.request_id, 'source': req.source, 'destination': req.destination,
            'category': req.endpoint_category, 'bandwidth_mbps': req.bandwidth_mbps,
            'service_class': req.service_class, 'priority_weight': req.priority_weight,
            'candidate_count': len(plans), 'candidates_evaluated': choice.evaluated,
            'admitted': int(choice.result.feasible),
            'reason': ';'.join(choice.result.reason_codes),
            'candidate_id': choice.result.candidate_id or '',
            'working_hops': len(choice.plan.normal_path) - 1 if choice.plan else '',
            'worst_scenario': choice.result.worst_fault_id or '',
            'worst_utilization': choice.result.worst_utilization
                  if choice.result.worst_utilization is not None else '',
            'solver_status': choice.solver_status,
            'selection_ms': round(selection_elapsed * 1000, 6),
            'candidate_generation_ms': round(candidate_elapsed * 1000, 6),
            'objective_or_score': choice.score if choice.score is not None else '',
        })
    admitted = sim.accepted()
    total_weight = sum(r.priority_weight for r in incoming)
    admitted_weight = sum(r.priority_weight for r, _ in admitted)
    accepted_ids = set(r.request_id for r, _ in admitted)
    reasons = Counter(row['reason'] for row in decisions)
    row = {
        'policy': policy, 'nodes': cfg.nodes, 'topology_seed': cfg.seed,
        'headroom_percent': headroom, 'max_candidates': max_candidates,
        'requests': len(incoming), 'admitted': len(admitted),
        'rejected': len(incoming) - len(admitted),
        'admission_ratio': round(len(admitted) / len(incoming), 6),
        'weighted_admission_ratio': round(admitted_weight / total_weight, 6),
        'normal_capacity_rejections': reasons.get('NORMAL_CAPACITY_EXCEEDED', 0),
        'postfault_capacity_rejections': reasons.get('POSTFAULT_CAPACITY_EXCEEDED', 0),
        'no_plan_rejections': reasons.get('NO_POLICY_VALID_PROTECTED_PLAN', 0),
        'fallback_count': sum(v for k,v in statuses.items() if k.startswith('FALLBACK')),
        'solver_nonoptimal_count': sum(v for k,v in statuses.items() if 'NONOPTIMAL' in k),
        'selection_seconds': round(decision_s, 6),
        'candidate_generation_seconds': round(candidate_s, 6),
        'total_policy_seconds': round(decision_s + candidate_s, 6),
        'avg_selection_ms': round(decision_s / len(incoming) * 1000, 6),
        'state_digest': sim.state_digest(), 'request_sha256': _hash(incoming),
        'peak_scenario_utilization': sim.ledger_summary()['peak_utilization'],
    }
    output.mkdir(parents=True, exist_ok=False)
    _csv(output / 'requests.csv', [asdict(r) for r in incoming], list(asdict(incoming[0])))
    _csv(output / 'decisions.csv', decisions, list(decisions[0]))
    plan_exports = []
    for req, plan in admitted:
        plan_exports.append({
            'request': asdict(req), 'candidate_id': plan.candidate_id,
            'working_forward': list(plan.normal_path),
            'failure_state_forward_paths': {key: list(nodes) for key,nodes in plan.fault_paths.items()},
            'reverse_rule': 'exact reverse of the corresponding forward path',
        })
    _json(output / 'accepted_protection_plans.json', {'model': SCHEME, 'admitted': plan_exports})
    _json(output / 'summary.json', {**row, 'solver_status_counts': dict(statuses),
             'scenarios_checked': len(sim.failures)+1,
             'scenario_ledger': sim.ledger_summary(),
             'protection_model_approved': False,
             'scope': 'Online Phase 4 comparison of Phase 3 single-link scenario surrogate; NOT RFC8227 MSRP'})
    category_rows = []
    for category in sorted(set(r.endpoint_category for r in incoming)):
        relevant = [r for r in incoming if r.endpoint_category == category]
        a = [r for r in relevant if r.request_id in accepted_ids]
        category_rows.append({'category': category, 'offered': len(relevant),
                              'admitted': len(a), 'rate': round(len(a)/len(relevant), 6),
                              'weighted_rate': round(sum(x.priority_weight for x in a) /
                                                     sum(x.priority_weight for x in relevant), 6)})
    _csv(output / 'category_results.csv', category_rows, list(category_rows[0]))
    digests = {}
    for name in ('requests.csv','decisions.csv','accepted_protection_plans.json',
                 'summary.json','category_results.csv'):
        digests[name] = hashlib.sha256((output/name).read_bytes()).hexdigest()
    _json(output / 'manifest.json', {'model': SCHEME, 'sha256': digests,
          'policy': policy, 'request_sha256': row['request_sha256'],
          'algorithm_version': ('V2_PRIMARY_MILP_EXACT_LEXICOGRAPHIC' if policy == 'milp_v2'
                                else 'V1_ORIGINAL'),
          'notes': 'All accepted candidates committed through phase3.model.ProtectionSimulator.admit'})
    return {**row, 'output_dir': str(output)}


def _stats(values: list[float]) -> dict:
    return {'mean': round(statistics.mean(values), 6),
            'sd': round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            'min': round(min(values), 6), 'max': round(max(values), 6)}


def benchmark(cfg: TopologyConfig, requests: int, seeds: list[int],
              headrooms: list[int], policies: list[str], output: Path,
              max_candidates: int = 2, milp_timeout: float = 5.0) -> dict:
    if requests < 1 or not seeds or not headrooms or not policies:
        raise ValueError('requests/seeds/headrooms/policies must be nonempty')
    if len(set(seeds)) != len(seeds) or len(set(headrooms)) != len(headrooms) or len(set(policies)) != len(policies):
        raise ValueError('duplicate seeds, headrooms, or policies')
    if any(p not in ALL_POLICIES for p in policies):
        raise ValueError('unknown Phase4 policy')
    if max_candidates < 1 or any(not 0 <= h < 100 for h in headrooms) or milp_timeout <= 0:
        raise ValueError('invalid candidate/headroom/time limit')
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing benchmark: ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    # Request streams derived identically from topology seed and demand seed.
    # No policy receives subsequent requests before its next online decision.
    G, _ = make_topology(cfg)
    streams = {seed: generate_requests(G, requests, seed=seed) for seed in seeds}
    runrows=[]
    for headroom in headrooms:
        for seed in seeds:
            stream=streams[seed]
            for policy in policies:
                directory = output / 'runs' / f'h{headroom}_d{seed}_{policy}'
                row=run_one(cfg, stream, policy, headroom, max_candidates,
                            directory, milp_timeout)
                row['demand_seed'] = seed
                runrows.append(row)
                print(f'{policy:6s} headroom={headroom:2d}% seed={seed} '
                      f'admitted={row["admitted"]}/{requests} '
                      f'weighted={row["weighted_admission_ratio"]:.3f} '
                      f'fall-backs={row["fallback_count"]}')
    _csv(output / 'phase4_runs.csv', runrows, list(runrows[0]))
    group_rows=[]
    for h in headrooms:
        for p in policies:
            subset=[r for r in runrows if r['headroom_percent']==h and r['policy']==p]
            ratio = _stats([r['admission_ratio'] for r in subset])
            w = _stats([r['weighted_admission_ratio'] for r in subset])
            cost = _stats([r['avg_selection_ms'] for r in subset])
            group_rows.append({
                'headroom_percent': h, 'policy': p, 'seeds': len(subset),
                'requests_per_seed': requests, 'mean_admission_ratio': ratio['mean'],
                'sd_admission_ratio': ratio['sd'],
                'min_admission_ratio': ratio['min'], 'max_admission_ratio': ratio['max'],
                'mean_weighted_ratio': w['mean'], 'sd_weighted_ratio': w['sd'],
                'mean_selection_ms_per_request': cost['mean'],
                'fallbacks_across_seeds': sum(r['fallback_count'] for r in subset)
            })
    _csv(output / 'phase4_summary.csv', group_rows, list(group_rows[0]))
    index={(r['headroom_percent'],r['demand_seed'],r['policy']):r for r in runrows}
    deltas=[]
    if 'greedy' in policies:
        for h in headrooms:
            for seed in seeds:
                base=index[h,seed,'greedy']
                for p in policies:
                    if p=='greedy':continue
                    other=index[h,seed,p]
                    if other['request_sha256'] != base['request_sha256']:
                        raise AssertionError('Unpaired offered request stream!')
                    deltas.append({'headroom_percent':h,'demand_seed':seed,
                                   'policy':p,'baseline':'greedy',
                                   'delta_admitted':other['admitted']-base['admitted'],
                                   'delta_admission_ratio':round(other['admission_ratio']-
                                                                   base['admission_ratio'],6),
                                   'delta_weighted_ratio':round(other['weighted_admission_ratio']-
                                                                  base['weighted_admission_ratio'],6)})
    _csv(output / 'phase4_paired_deltas.csv', deltas,
         ['headroom_percent','demand_seed','policy','baseline','delta_admitted',
          'delta_admission_ratio','delta_weighted_ratio'])
    import importlib.metadata
    package_versions = {}
    for pkg in ('scipy','numpy','networkx'):
        try: package_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:package_versions[pkg]='unavailable'
    provenance = {
        'model':SCHEME, 'phase4_scope':'ONLINE myopic policies, no future request oracle',
        'config':asdict(cfg), 'seeds':seeds, 'headrooms':headrooms,
        'requests_per_seed':requests, 'candidate_cap':max_candidates,
        'algorithms':policies, 'milp_time_limit_seconds':milp_timeout,
        'milp_v2_definition': ('MILP minimizes max scenario utilization without secondary coefficient; '
            'exact rational candidate scores then break optimum ties by working hops and candidate ID'),
        'num_runs':len(runrows), 'python':platform.python_version(),
        'packages':package_versions,
        'request_stream_sha256':{str(s): _hash(streams[s]) for s in seeds},
        'fairness_rules':[
            'Separate fresh Phase3 simulator state per policy/seed/headroom',
            'Same synthetic ordered request stream per paired comparison',
            'Same sorted policy-valid candidate generation and truncation',
            'Same ring-segment single-link failure semantics and headroom',
            'Every admitted plan independently checked by Phase3 audit/admit',
            'No future requests visible to the online policies',
            'MILP is one-request robust minmax utilization, NOT offline optimal admission',
            'CSPF is candidate-constrained shortest working path, NOT unconstrained graph routing',
            'Latency includes policy decision but not candidate generation; see per-run totals',
        ],
        'limitations':[
            'Single physical-link faults only, no node faults or RPS/OAM switching',
            'No RFC8227 MSRP emulation or safety/production claims',
            'Fixed accepted services; no release, reroute, or preemption',
            'The observed 60-node configuration has two policy-valid candidates per request',
            'Running standard deviation across independent demand seeds is descriptive only',
        ]
    }
    _json(output / 'phase4_provenance.json', provenance)
    return {'runs':len(runrows),'output':str(output),'summary':group_rows,
            'paired_deltas':len(deltas)}


def main() -> None:
    ap=argparse.ArgumentParser(description='Phase4 online greedy/CSPF/MILP paired benchmark; not MSRP')
    ap.add_argument('--config',default='config/primary_60_dual.json')
    ap.add_argument('--requests',type=int,default=50)
    ap.add_argument('--seeds',nargs='+',type=int,default=[2027,2028,2029,2030,2031])
    ap.add_argument('--headrooms',nargs='+',type=int,default=[10])
    ap.add_argument('--algorithms',nargs='+',choices=ALL_POLICIES,default=list(POLICIES))
    ap.add_argument('--max-candidates',type=int,default=2)
    ap.add_argument('--milp-time-limit',type=float,default=5.0)
    ap.add_argument('--out',type=Path,default=Path('outputs/phase4/pilot_60'))
    args=ap.parse_args()
    cfg=TopologyConfig(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    result=benchmark(cfg, args.requests,args.seeds,args.headrooms,args.algorithms,
                     args.out,args.max_candidates,args.milp_time_limit)
    print('\nBENCHMARK FINISHED')
    print(json.dumps(result, indent=2))

if __name__=='__main__':
    main()
