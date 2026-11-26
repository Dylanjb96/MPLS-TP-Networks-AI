"""Development-only matched A/B/C opportunity probe; no RL training or test data."""
from __future__ import annotations
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from .topology import NetworkSpec, build_network
from .traffic import generate_demands, stream_digest
from .protection import ScenarioLedger, generate_pairs

VERSION = 'PHASE5F_SYNTHETIC_NATIONAL_STAGE1_V1'
PROTECTION = 'END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP'


def stable_digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True,
                         separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    folder = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob('*.py'))}


def nonlearning_choice(kind, feasible, plans, graph, rng):
    if kind == 'greedy': return feasible[0]
    if kind == 'random_feasible': return rng.choice(feasible)
    if kind == 'footprint':
        def cost(i):
            p = plans[i]
            return sum(1 / graph.edges[u,v]['capacity_mbps']
                       for path in (p.working,p.backup)
                       for u,v in zip(path,path[1:]))
        return min(feasible,key=lambda i:(cost(i),i))
    raise ValueError('unknown baseline '+kind)


def evaluate(graph, demands, inventories, *, k, policy, seed, headroom):
    sim = ScenarioLedger(graph, headroom)
    rng = random.Random(seed + 100_000 * (1+len(policy)))
    admitted = 0
    admitted_weight = 0
    offered_weight = sum(d.priority_weight for d in demands)
    departures=0
    discretionary=0
    overrides=0
    start=time.perf_counter()
    for demand, full_inventory in zip(demands,inventories):
        departures += sim.release_expired(demand.arrival)
        plans = full_inventory[:k]
        feasible = [i for i,p in enumerate(plans) if sim.feasible(demand,p)]
        discretionary += len(feasible) > 1
        if not feasible: continue
        pick = nonlearning_choice(policy,feasible,plans,graph,rng)
        overrides += pick != feasible[0]
        if not sim.admit(demand,plans[pick]):
            raise AssertionError('successful audit followed by failed admission')
        admitted += 1
        admitted_weight += demand.priority_weight
    if not sim.verify_full_scenario_recompute():
        raise AssertionError('ledger did not reproduce from active service set')
    return dict(admitted=admitted, rejected=len(demands)-admitted,
                admission_ratio=admitted/len(demands),
                weighted_admission_ratio=admitted_weight/offered_weight,
                discretionary=discretionary,overrides=overrides,
                departures=departures, peak_utilization=sim.peak_utilization(),
                elapsed_seconds=round(time.perf_counter()-start,5),
                active_at_end=len(sim.active),state_digest=sim.state_digest(),
                no_feasibility_violations=True)


def export_csv(path, rows):
    if not rows: raise ValueError('no rows')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def probe(out:Path, *, sizes=(60,),topology_seeds=(101,),stream_seeds=(70000,),
          k_values=(2,6),traffic_modes=('static','dynamic'),
          policies=('greedy','random_feasible','footprint'),
          requests_per_node=1.6, headroom=10, max_working_paths=40):
    if out.exists():
        raise FileExistsError('refusing to overwrite prior evidence: '+str(out))
    if any(s<70000 or s>70099 for s in stream_seeds):
        raise ValueError('Stage1 allows only development stream seeds 70000..70099')
    if any(s<100 or s>199 for s in topology_seeds):
        raise ValueError('Stage1 allows only development topology seeds 100..199')
    if any(k not in (2,6) for k in k_values):
        raise ValueError('Stage1 controls require candidate limit 2 or 6')
    if not set(traffic_modes) <= {'static','dynamic'}:
        raise ValueError('unknown traffic mode')
    if not set(policies) <= {'greedy','random_feasible','footprint'}:
        raise ValueError('unknown baseline')
    if requests_per_node <= 0 or not 0<=headroom<100:
        raise ValueError('invalid offered load or headroom')
    runs=[]; inventory_rows=[]; cohort_hashes={}; network_hashes={}
    for n in sizes:
        for topology_seed in topology_seeds:
            g=build_network(NetworkSpec(nodes=n,seed=topology_seed))
            network_hashes[f'{n}/{topology_seed}']=stable_digest([
                (min(u,v),max(u,v),d['physical_link_id'],d['capacity_mbps'])
                for u,v,d in sorted(g.edges(data=True))])
            for stream_seed in stream_seeds:
                count=max(1,round(n*requests_per_node))
                base=generate_demands(g,count,stream_seed,dynamic=False)
                dynamic=generate_demands(g,count,stream_seed,dynamic=True)
                assert stream_digest(base,False)==stream_digest(dynamic,False)
                key=f'{n}/{topology_seed}/{stream_seed}'
                cohort_hashes[key]={
                  'common_arrivals_sha256':stream_digest(base,False),
                  'static_sha256':stream_digest(base),
                  'dynamic_sha256':stream_digest(dynamic)}
                t0=time.perf_counter()
                candidate_sets=tuple(generate_pairs(g,d.source,d.destination,
                                     max_candidates=max(k_values),
                                     max_working_paths=max_working_paths) for d in base)
                gen_s=time.perf_counter()-t0
                for idx,plans in enumerate(candidate_sets):
                    inventory_rows.append(dict(nodes=n,topology_seed=topology_seed,
                        stream_seed=stream_seed,request_id=base[idx].request_id,
                        generated=len(plans),k2=min(2,len(plans)),
                        k6=min(6,len(plans)),
                        source=base[idx].source,destination=base[idx].destination))
                for mode in traffic_modes:
                    ds=base if mode=='static' else dynamic
                    for k in k_values:
                        for policy in policies:
                            result=evaluate(g,ds,candidate_sets,k=k,policy=policy,
                                            seed=stream_seed,headroom=headroom)
                            runs.append(dict(nodes=n,topology_seed=topology_seed,
                                demand_seed=stream_seed,mode=mode,k=k,policy=policy,
                                requests=count,physical_links=g.number_of_edges(),
                                generated_gt2=sum(len(x)>2 for x in candidate_sets),
                                candidate_generation_seconds=round(gen_s,5),
                                request_sha256=stream_digest(ds),**result))
    out.mkdir(parents=True)
    export_csv(out/'probe_runs.csv',runs)
    export_csv(out/'candidate_inventory.csv',inventory_rows)
    manifest=dict(version=VERSION,status='DEVELOPMENT_ONLY_NOT_A_FINAL_EXPERIMENT',
      synthetic_research_model=True,protection_model=PROTECTION,
      source_hashes=source_hashes(),network_hashes=network_hashes,
      streams=cohort_hashes, sizes=list(sizes), topology_seeds=list(topology_seeds),
      stream_seeds=list(stream_seeds), k_values=list(k_values),
      traffic_modes=list(traffic_modes), policies=list(policies),
      requests_per_node=requests_per_node,headroom=headroom,
      max_working_paths=max_working_paths,
      total_cell_runs=len(runs),total_requests_inventoried=len(inventory_rows),
      protection_model_approved=False,real_irish_rail_topology=False,
      note='Not IEEE/MPLS-TP MSRP; no restoration times; excludes future oracle and held-out seeds.')
    (out/'probe_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return out,manifest
