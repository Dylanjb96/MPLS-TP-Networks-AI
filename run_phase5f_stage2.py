"""Phase 5F Stage 2 prospective simulator validation; never accesses held-out seeds."""
from __future__ import annotations
import argparse
from pathlib import Path
from phase5f.traffic import generate_demands
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage2.probe import environment,run_probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    pre=commands.add_parser('preflight')
    pre.add_argument('--sizes',type=int,nargs='+',default=[60,150,250,500])
    pre.add_argument('--seeds',type=int,nargs='+',default=[101,102,103])
    pre.add_argument('--fingerprint',action='store_true')
    p=commands.add_parser('probe')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[60])
    p.add_argument('--topology-seeds',type=int,nargs='+',default=[101,102,103])
    p.add_argument('--stream-seeds',type=int,nargs='+',default=[70000])
    p.add_argument('--loads',type=float,nargs='+',default=[1.6,2.5,4.0])
    p.add_argument('--k',type=int,nargs='+',default=[2,6])
    p.add_argument('--traffic',nargs='+',default=['static','dynamic'])
    p.add_argument('--policies',nargs='+',default=['greedy','footprint'])
    p.add_argument('--headroom',type=int,default=10)
    p.add_argument('--max-working-paths',type=int,default=40)
    args=parser.parse_args()
    if args.command=='preflight':
        print('Software:',environment())
        fingerprints=[]
        for n in args.sizes:
            for s in args.seeds:
                g=build_network(NetworkSpec(nodes=n,seed=s))
                d=generate_demands(g,1,70000,False)[0]
                pairs=generate_pairs(g,d.source,d.destination,6,40)
                print('PASS n={} seed={} links={} structural={} full={} candidates={}'
                    .format(n,s,g.number_of_edges(),fingerprint(g,structural_only=True),
                            fingerprint(g),len(pairs)))
                fingerprints.append((n,s,fingerprint(g),candidates_digest((pairs,))))
        if args.fingerprint:
            import hashlib,json
            report=hashlib.sha256(json.dumps(fingerprints,separators=(',',':')).encode()).hexdigest()
            print('CROSS_MACHINE_FINGERPRINT_SHA256='+report)
    elif args.command=='probe':
        info=run_probe(args.out,sizes=args.sizes,topology_seeds=args.topology_seeds,
                       stream_seeds=args.stream_seeds,loads=args.loads,k_values=args.k,
                       traffic_modes=args.traffic,policies=args.policies,
                       headroom=args.headroom,max_working_paths=args.max_working_paths)
        print('STAGE2_DEV_ONLY runs={} inventories={} wall_s={}'.format(
              info['rows'],info['inventoried_requests'],info['wall_clock_seconds']))

if __name__=='__main__':
    main()
