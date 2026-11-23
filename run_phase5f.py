"""Stage 1 Phase5F synthetic national-scale opportunity probe (no RL yet)."""
import argparse
from pathlib import Path
from phase5f.probe import probe
from phase5f.topology import NetworkSpec,build_network
from phase5f.protection import generate_pairs
from phase5f.traffic import generate_demands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('preflight')
    f.add_argument('--sizes',type=int,nargs='+',default=[60,150,250,500])
    f.add_argument('--seed',type=int,default=101)
    v=sub.add_parser('probe')
    v.add_argument('--out',type=Path,required=True)
    v.add_argument('--sizes',type=int,nargs='+',default=[60])
    v.add_argument('--topology-seeds',type=int,nargs='+',default=[101])
    v.add_argument('--stream-seeds',type=int,nargs='+',default=[70000])
    v.add_argument('--requests-per-node',type=float,default=1.6)
    v.add_argument('--k',type=int,nargs='+',default=[2,6])
    v.add_argument('--traffic',nargs='+',default=['static','dynamic'])
    v.add_argument('--policies',nargs='+',default=['greedy','random_feasible','footprint'])
    v.add_argument('--headroom',type=int,default=10)
    v.add_argument('--max-working-paths',type=int,default=40)
    a=p.parse_args()
    if a.command=='preflight':
        for size in a.sizes:
            g=build_network(NetworkSpec(nodes=size,seed=a.seed))
            sample=generate_demands(g,1,70000,dynamic=False)[0]
            options=generate_pairs(g,sample.source,sample.destination,
                                   max_candidates=6,max_working_paths=40)
            print(f'OK topology n={size} edges={g.number_of_edges()} '
                  f'regions={len(g.graph["regions"])} '
                  f'sample_link_disjoint_pairs={len(options)}')
    else:
        dest,m=probe(a.out,sizes=a.sizes,topology_seeds=a.topology_seeds,
                     stream_seeds=a.stream_seeds,k_values=a.k,
                     traffic_modes=a.traffic,policies=a.policies,
                     requests_per_node=a.requests_per_node,
                     headroom=a.headroom,max_working_paths=a.max_working_paths)
        print(f'Wrote {m["total_cell_runs"]} development-only cell runs to {dest}')

if __name__=='__main__':main()
