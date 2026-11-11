#Command-line runner for the Phase 3 protection/symmetric capacity surrogate.

#Run from project root, for example:
#python -m phase3.cli --config config/debug_12_dual.json --out outputs/phase3

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from railway_tp.topology import TopologyConfig,make_topology,validate_topology
from railway_tp.requests import generate_requests
from .model import ProtectionSimulator,SCHEME


def dump_json(path:Path, data)->None:
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def write_csv(path:Path,records:list[dict],fields:list[str])->None:
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(records)


def run(cfg:TopologyConfig,out:Path,requests:int=25,seed:int=2026,headroom:int=10,
        max_candidates:int=12)->dict:
    if requests<1:raise ValueError('requests must be >=1')
    out.mkdir(parents=True,exist_ok=True)
    G,rings=make_topology(cfg)
    sim=ProtectionSimulator(G,rings,headroom_percent=headroom)
    incoming=generate_requests(G,requests,seed=seed)
    basename=f'phase3_{cfg.nodes}_{cfg.gateway_mode}_s{cfg.seed}_d{seed}_h{headroom}'
    decisions=[]
    for req in incoming:
        candidate_list=sim.candidates(req,max_candidates=max_candidates)
        decision=sim.select_greedy(req,max_candidates=max_candidates)
        decisions.append({'request_id':req.request_id,'src':req.source,'dst':req.destination,
                          'class':req.service_class,'priority':req.priority_weight,
                          'category':req.endpoint_category,'bandwidth_mbps':req.bandwidth_mbps,
                          'candidate_count':len(candidate_list),
                          'accepted':int(decision.feasible),'reason':';'.join(decision.reason_codes),
                          'candidate_id':decision.candidate_id or '',
                          'worst_fault_id':decision.worst_fault_id or '',
                          'worst_utilization':decision.worst_utilization if decision.worst_utilization is not None else '',
                          'capacity_margin_mbps':decision.capacity_margin_mbps if decision.capacity_margin_mbps is not None else '',
                          'details':decision.details})
    inputs=out/f'{basename}_requests.csv'
    results=out/f'{basename}_decisions.csv'
    accepted_path=out/f'{basename}_accepted_protection_plans.json'
    summary_file=out/f'{basename}_summary.json'
    manifest_file=out/f'{basename}_manifest.json'
    write_csv(inputs,[asdict(r) for r in incoming],
              ['request_id','source','destination','bandwidth_mbps','service_class','priority_weight','endpoint_category'])
    write_csv(results,decisions,['request_id','src','dst','class','priority','category','bandwidth_mbps',
              'candidate_count','accepted','reason','candidate_id','worst_fault_id','worst_utilization',
              'capacity_margin_mbps','details'])
    plan_export=[]
    for req,plan in sim.accepted():
        plan_export.append({'request':asdict(req),'candidate_id':plan.candidate_id,
             'working_forward':list(plan.normal_path),'working_reverse':list(reversed(plan.normal_path)),
             'segments':[{'ring':s.ring_id,'working':list(s.working)} for s in plan.segments],
             'all_single_physical_link_failure_forward_paths':{f:list(p) for f,p in sorted(plan.fault_paths.items())},
             'reverse_fault_rule':'exact reverse of each forward failure-state path'})
    dump_json(accepted_path,{'model':SCHEME,'note':'All paths are preplanned surrogate paths, NOT RFC8227 tunnels',
                             'admitted':plan_export})
    summary={'model':SCHEME,'scope':'Simulation only; no MSRP, no train safety / performance',
             'config':asdict(cfg),'demand_seed':seed,'request_count':len(incoming),
             'admitted_count':len(sim.accepted()),'rejected_count':requests-len(sim.accepted()),
             'protected_admission_ratio':round(len(sim.accepted())/requests,6),
             'weighted_protected_acceptance_ratio':round(sum(r.priority_weight for r,_ in sim.accepted())/
                                                     sum(r.priority_weight for r in incoming),6),
             'reason_counts':dict(Counter(d['reason'] for d in decisions)),
             'candidate_absent_count':sum(d['candidate_count']==0 for d in decisions),
             'fault_scenarios':len(sim.failures),'scenario_ledger':sim.ledger_summary(),
             'state_digest':sim.state_digest(),
             'topology_structural_valid':validate_topology(G,rings)['all_rings_valid_cycles'],
             'protection_model_approved':False,
             'notes':['No node fault, no multiple faults, no release/preemption, no OAM/RPS, no packet timing',
                      'Greedy first-feasible demonstration is NOT a Phase 4/5 comparative result']}
    dump_json(summary_file,summary)
    artifacts=[inputs,results,accepted_path,summary_file]
    digest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    dump_json(manifest_file,{'artifact_sha256':digest,'model':SCHEME,'environment':'Use requirements.txt; Python >= 3.10',
                             'standards_reference':'https://www.rfc-editor.org/info/rfc8227/',
                             'files':[p.name for p in artifacts]})
    print(json.dumps({'nodes':G.number_of_nodes(),'physical_links':G.number_of_edges(),
                      'admitted':len(sim.accepted()),'requests':requests,
                      'single_link_faults_checked':len(sim.failures),
                      'weighted_admission_ratio':summary['weighted_protected_acceptance_ratio'],
                      'outputs':str(out.resolve()),'manifest':str(manifest_file.resolve()),
                      'warning':'Scenario-based ring-segment surrogate; not MPLS-TP MSRP conformance'},indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser(description='Offline Phase 3 scenario-based ring-segment SURROGATE; not MSRP')
    ap.add_argument('--config',type=Path,default=Path('config/debug_12_dual.json'))
    ap.add_argument('--out',type=Path,default=Path('outputs/phase3'))
    ap.add_argument('--requests',type=int,default=25)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--headroom',type=int,choices=range(0,100),metavar='0..99',default=10)
    ap.add_argument('--max-candidates',type=int,default=12)
    args=ap.parse_args()
    cfg=TopologyConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    run(cfg,args.out,args.requests,args.seed,args.headroom,args.max_candidates)

if __name__=='__main__':main()