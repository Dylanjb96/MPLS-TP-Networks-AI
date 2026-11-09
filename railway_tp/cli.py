#End-to-end Phase 2 generation, validation, inventories and evidence files.
from __future__ import annotations
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from .topology import (TopologyConfig,make_topology,validate_topology,
                       link_failure_connectivity,node_failure_connectivity)
from .routes import candidate_paths,ring_arcs
from .requests import generate_requests,requests_to_dict
from .export import export_topology,write_table,checksum,_write_json
from .visualize import save_topology_figure

def examples(G,rings):
    main=rings['main']
    clean_main=[n for n in main if not G.nodes[n]['gateway']]
    main_pair=clean_main[:2] if len(clean_main)>=2 else main[:2]
    subrings=[r for r in rings if r!='main']
    inner={r:[n for n in rings[r] if 'main' not in G.nodes[n]['rings']] for r in subrings}
    cases={'main-to-main':main_pair,
           'main-to-subring':[main_pair[0],inner[subrings[0]][-1]],
           'within-subring':inner[subrings[0]][:2]}
    if len(subrings)>=2:
        cases['subring-to-subring']=[inner[subrings[0]][0],inner[subrings[1]][0]]
    out={}
    for label,(src,dst) in cases.items():
        paths=candidate_paths(G,src,dst,k=5)
        out[label]={'source':src,'target':dst,'candidate_count':len(paths),
                    'candidate_graph_paths':[{'nodes':p,'hops':len(p)-1,
                    'km':sum(G.edges[u,v]['length_km'] for u,v in zip(p,p[1:]))} for p in paths]}
    r=subrings[0]; s,t=inner[r][:2]
    out['two_arcs_within_subring']={'ring':r,'source':s,'target':t,
                                  'physical_arcs':ring_arcs(rings,r,s,t)}
    out['WARNING']='Global graph routes and ring arcs are not yet valid protected MPLS-TP LSP plans.'
    return out

def run(cfg:TopologyConfig,out:Path,request_count:int=50)->dict:
    out.mkdir(parents=True,exist_ok=True)
    G,rings=make_topology(cfg)
    summary=validate_topology(G,rings)
    stem=f'phase2_{cfg.nodes}_{cfg.gateway_mode}_s{cfg.seed}'
    topology_path=export_topology(G,rings,out / f'{stem}_topology.json')
    diagram_path=save_topology_figure(G,out/f'{stem}_diagram.png')
    links=link_failure_connectivity(G)
    nodes=node_failure_connectivity(G)
    requests=generate_requests(G,request_count,seed=cfg.seed+1)
    reqdata=requests_to_dict(requests)
    files=[topology_path,diagram_path]
    files.append(_write_json(out/f'{stem}_validation.json',summary))
    files.append(_write_json(out/f'{stem}_route_examples.json',examples(G,rings)))
    files.append(_write_json(out/f'{stem}_requests.json',reqdata))
    files.append(write_table(out/f'{stem}_nodes.csv',[
       {'node_id':n, 'role':d['role'], 'gateway':d['gateway'], 'rings':';'.join(d['rings'])}
       for n,d in sorted(G.nodes(data=True))],['node_id','role','gateway','rings']))
    files.append(write_table(out/f'{stem}_links.csv',[
       {'link_id':d['physical_link_id'],'source':min(u,v),'target':max(u,v),
        'rings':';'.join(d['rings']),'capacity_mbps':d['capacity_mbps'],
        'reserved_mbps':d['reserved_mbps'],'length_km':d['length_km'],'status':d['status']}
       for u,v,d in sorted(G.edges(data=True))],
       ['link_id','source','target','rings','capacity_mbps','reserved_mbps','length_km','status']))
    files.append(write_table(out/f'{stem}_link_failure_checks.csv',[
       {'failed_link':f"{r['failed_link'][0]}-{r['failed_link'][1]}",
        'connected':r['connected'],'component_count':r['component_count'],
        'component_sizes':';'.join(map(str,r['component_sizes']))} for r in links],
       ['failed_link','connected','component_count','component_sizes']))
    files.append(write_table(out/f'{stem}_node_failure_DIAGNOSTIC.csv',[
       {'failed_node':r['failed_node'],'gateway':r['gateway'],
        'connected':r['connected'],'component_count':r['component_count'],
        'component_sizes':';'.join(map(str,r['component_sizes']))} for r in nodes],
       ['failed_node','gateway','connected','component_count','component_sizes']))
    files.append(write_table(out/f'{stem}_synthetic_demand.csv',reqdata,
       ['request_id','source','destination','bandwidth_mbps','service_class','priority_weight','endpoint_category']))
    files.append(write_table(out/f'{stem}_ring_inventory.csv',[
       {'ring_id':r,'ordered_nodes':';'.join(map(str,rns)),
        'node_count':len(rns),
        'gateway_nodes':';'.join(str(n) for n in rns if G.nodes[n]['gateway'])}
       for r,rns in rings.items()],['ring_id','ordered_nodes','node_count','gateway_nodes']))
    mandatory=['connected','all_rings_valid_cycles','node_ring_membership_consistent',
      'gateway_counts_valid','positive_capacity_and_zero_initial_reservations',
      'unique_physical_link_ids','all_single_link_removals_preserve_graph_connectivity']
    passed=all(summary[k] for k in mandatory)
    manifest={
      'project':'Phase 2 synthetic railway-like interconnected-ring topology',
      'scope':'topology-only, no MPLS-TP shared-ring protection or service admission',
      'config':asdict(cfg),'passed_structural_gate':passed,'validation':summary,
      'request_count':len(requests),'demand_class_counts':dict(Counter(r.service_class for r in requests)),
      'request_endpoint_category_counts':dict(Counter(r.endpoint_category for r in requests)),
      'output_files_sha256':{p.name:checksum(p) for p in files},
      'standards_reference':'https://www.rfc-editor.org/info/rfc8227/',
    }
    manifest_path=_write_json(out/f'{stem}_manifest.json',manifest)
    print(json.dumps({'nodes':summary['node_count'],'links':summary['link_count'],
         'rings':summary['ring_count'],'topology_valid':passed,
         'single_link_connectivity':summary['all_single_link_removals_preserve_graph_connectivity'],
         'node_loss_connected_DIAGNOSTIC':summary['all_single_node_removals_preserve_graph_connectivity_DIAGNOSTIC'],
         'manifest':str(manifest_path),'diagram':str(diagram_path)},indent=2))
    if not passed: raise RuntimeError('Phase 2 structural gate failed — inspect validation JSON')
    return manifest

def main():
    ap=argparse.ArgumentParser(description='Phase 2 physical topology; NOT an MPLS-TP protection simulator')
    ap.add_argument('--config',type=Path,help='JSON configuration for repeatable experiment')
    ap.add_argument('--nodes',type=int,default=12)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--gateway-mode',choices=['single','dual'],default='dual')
    ap.add_argument('--requests',type=int,default=50)
    ap.add_argument('--out',type=Path,default=Path('outputs'))
    args=ap.parse_args()
    if args.config:
        data=json.loads(args.config.read_text(encoding='utf-8'))
        cfg=TopologyConfig(**data)
    else:
        cfg=TopologyConfig(nodes=args.nodes,seed=args.seed,gateway_mode=args.gateway_mode)
    run(cfg,args.out,args.requests)

if __name__=='__main__':
    main()
