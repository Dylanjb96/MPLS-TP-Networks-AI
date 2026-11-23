"""Phase 5F Stage 2 prospective synthetic topology/load/feasibility probe.

No model training, no held-out use, no claim of operational MSRP compliance.
Never overwrite any existing output folder; Stage 1 files remain frozen.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import networkx as nx
import numpy as np
from phase5f.traffic import generate_demands,stream_digest
from phase5f.protection import ScenarioLedger
from . import VERSION
from .topology import NetworkSpec,build_network,fingerprint,edge_set
from .candidates import generate_pairs, candidates_digest,choice


def environment():
    try:
        import torch
        tv=torch.__version__
    except ImportError:
        tv='not_installed'
    return dict(python=platform.python_version(),python_implementation=platform.python_implementation(),
                networkx=nx.__version__,numpy=np.__version__,torch=tv,
                os=platform.platform(),python_executable=sys.executable)


def sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_hashes():
    root=Path(__file__).parent
    files=list(root.glob('*.py'))
    dependencies=[Path(__file__).parent.parent/'phase5f'/p for p in ('traffic.py','protection.py')]
    return {str(p.relative_to(root.parent)).replace('\\','/'):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files+dependencies)}


def csv_out(path,rows):
    if not rows: raise ValueError('empty data output '+str(path))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def memory_rss_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss/1048576,2)
    except ImportError:
        return None


def evaluate(g,demands,inventories,*,k,policy,seed,headroom):
    if len(demands)!=len(inventories):
        raise AssertionError('mismatched request/candidate sequences')
    ledger=ScenarioLedger(g,headroom)
    rng=random.Random(108003+seed+int(sha(policy)[:10],16))
    offered=sum(d.priority_weight for d in demands)
    accepted=0;weighted=0;departures=0;discretionary=0;overrides=0
    counts=Counter(); feasible_sum=0
    begin=time.perf_counter()
    for demand,full in zip(demands,inventories):
        departures+=ledger.release_expired(demand.arrival)
        plans=full[:k]
        indices=[i for i,p in enumerate(plans) if ledger.feasible(demand,p)]
        counts[len(indices)]+=1
        feasible_sum+=len(indices)
        discretionary+=(len(indices)>=2)
        if not indices:
            continue
        chosen=choice(policy,indices,plans,g,rng)
        overrides+=(chosen!=indices[0])
        if not ledger.admit(demand,plans[chosen]):
            raise AssertionError('admission diverges from pre-feasibility check')
        accepted+=1
        weighted+=demand.priority_weight
    if not ledger.verify_full_scenario_recompute():
        raise AssertionError('scenario ledger did not recompute independently')
    duration=time.perf_counter()-begin
    out=dict(admitted=accepted,rejected=len(demands)-accepted,
             admission_ratio=accepted/len(demands),
             weighted_admission_ratio=weighted/offered,
             discretionary=discretionary,overrides=overrides,
             departures=departures,active_at_end=len(ledger.active),
             peak_scenario_utilization=ledger.peak_utilization(),
             simulation_seconds=round(duration,6),
             state_digest=ledger.state_digest(),
             no_feasibility_violations=True)
    for amount in range(7):
        out['feasible_'+str(amount)]=counts[amount]
    out['feasible_more_than_6']=sum(v for n,v in counts.items() if n>6)
    out['avg_feasible_options']=feasible_sum/len(demands)
    if sum(counts.values())!=len(demands):
        raise AssertionError('missing feasibility counts')
    return out


def run_probe(out,*,sizes=(60,),topology_seeds=(101,102,103),stream_seeds=(70000,),
              loads=(1.6,2.5,4.0),k_values=(2,6),
              traffic_modes=('static','dynamic'),policies=('greedy','footprint'),
              headroom=10,max_working_paths=40):
    out=Path(out)
    if out.exists():
        raise FileExistsError('Stage 2 will not overwrite prior evidence: '+str(out))
    if not all(30<=n<=500 for n in sizes):
        raise ValueError('invalid sizes')
    if not all(100<=s<=199 for s in topology_seeds):
        raise ValueError('topology development seeds must be 100..199')
    if not all(70000<=s<=70099 for s in stream_seeds):
        raise ValueError('stream development seeds must be 70000..70099')
    if not all(float(v) in (1.6,2.5,4.0) for v in loads):
        raise ValueError('only predeclared development loads 1.6, 2.5, 4.0')
    if not all(k in (2,6) for k in k_values):
        raise ValueError('candidate limits must be 2,6')
    if not all(t in ('static','dynamic') for t in traffic_modes):
        raise ValueError('invalid traffic mode')
    if not all(p in ('greedy','footprint','random_feasible') for p in policies):
        raise ValueError('invalid policy')
    if not 0<=headroom<100 or max_working_paths<max(k_values):
        raise ValueError('invalid capacity headroom or working-path cap')
    t_all=time.perf_counter()
    networks={};graphs={}; runs=[];inventories_rows=[];cohorts={}
    diversity=[]
    for n in sizes:
        for topo in topology_seeds:
            graph=build_network(NetworkSpec(nodes=n,seed=topo))
            key=f'{n}/{topo}'
            graphs[key]=graph
            networks[key]=dict(full_hash=fingerprint(graph),
                               structural_hash=fingerprint(graph,structural_only=True),
                               nodes=len(graph),edges=graph.number_of_edges(),
                               bridge_count=len(list(nx.bridges(graph))),
                               degree_range=(min(dict(graph.degree()).values()),
                                             max(dict(graph.degree()).values())))
    for n in sizes:
        keys=[f'{n}/{topo}' for topo in topology_seeds]
        for i,k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                a,b=edge_set(graphs[k1]),edge_set(graphs[k2])
                diversity.append(dict(nodes=n,network_1=k1,network_2=k2,
                                      edge_jaccard=len(a&b)/len(a|b),
                                      different_edges=len(a^b),
                                      same_structure=(a==b)))
    for n in sizes:
        for topo in topology_seeds:
            g=graphs[f'{n}/{topo}']
            for stream in stream_seeds:
                for load in loads:
                    count=max(1,round(n*load))
                    static=generate_demands(g,count,stream,False)
                    dynamic=generate_demands(g,count,stream,True)
                    if stream_digest(static,False)!=stream_digest(dynamic,False):
                        raise AssertionError('static/dynamic arrival mismatch')
                    start=time.perf_counter()
                    inv=tuple(generate_pairs(g,d.source,d.destination,
                                             max_candidates=max(k_values),
                                             max_working_paths=max_working_paths)
                              for d in static)
                    gen_time=time.perf_counter()-start
                    cohort=f'{n}/{topo}/{stream}/{load}'
                    cohorts[cohort]=dict(static_requests=stream_digest(static),
                         dynamic_requests=stream_digest(dynamic),
                         common_arrivals=stream_digest(static,False),
                         ordered_candidate_sha256=candidates_digest(inv),
                         request_count=count, candidate_generation_seconds=round(gen_time,6))
                    for d,ps in zip(static,inv):
                        inventories_rows.append(dict(nodes=n,topology_seed=topo,
                            stream_seed=stream,load=load,request_id=d.request_id,
                            requested_bandwidth=d.bandwidth_mbps,
                            structural_candidates=len(ps),
                            candidate_ids='|'.join(p.candidate_id for p in ps)))
                    for mode in traffic_modes:
                        ds=static if mode=='static' else dynamic
                        for k in k_values:
                            for policy in policies:
                                result=evaluate(g,ds,inv,k=k,policy=policy,
                                                seed=stream,headroom=headroom)
                                runs.append(dict(nodes=n,topology_seed=topo,
                                    demand_seed=stream,load=load,mode=mode,k=k,
                                    policy=policy,requests=count,
                                    request_sha256=stream_digest(ds),
                                    network_sha256=networks[f'{n}/{topo}']['full_hash'],
                                    candidates_sha256=cohorts[cohort]['ordered_candidate_sha256'],
                                    generation_seconds=round(gen_time,6),
                                    full_generation_plus_simulation_seconds=
                                        round(gen_time+result['simulation_seconds'],6),
                                    working_set_rss_mb=memory_rss_mb(),**result))
    out.mkdir(parents=True,exist_ok=False)
    csv_out(out/'stage2_probe_runs.csv',runs)
    csv_out(out/'stage2_candidate_inventory.csv',inventories_rows)
    if diversity:
        csv_out(out/'stage2_topology_diversity.csv',diversity)
    info=dict(version=VERSION,role='DEVELOPMENT_ONLY_NOT_HELDOUT',
        synthetic_not_actual_irish_rail=True,
        protection_model='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP',
        protection_model_approved=False,msrp_compliant=False,
        software=environment(),source_sha256=source_hashes(),
        networks=networks,cohorts=cohorts,
        inputs=dict(sizes=list(sizes),topology_seeds=list(topology_seeds),
                    stream_seeds=list(stream_seeds),loads=list(loads),
                    k_values=list(k_values),modes=list(traffic_modes),
                    policies=list(policies),headroom=headroom,
                    max_working_paths=max_working_paths),
        rows=len(runs),inventoried_requests=len(inventories_rows),
        wall_clock_seconds=round(time.perf_counter()-t_all,3),
        warning='Not MSRP compliance; observed feasibility counts are trajectory dependent; separate from Phase 5F Stage1.')
    (out/'stage2_manifest.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return info
