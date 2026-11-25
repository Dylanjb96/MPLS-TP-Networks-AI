"""Phase 5G DEVELOPMENT ONLY entry point. There is intentionally no final command."""
from __future__ import annotations
import argparse
from phase5g.workflow import preflight,paired_dev_pilot,gnn_smoke


def main():
    p=argparse.ArgumentParser(description='Phase 5G GNN, optimization and topology-transfer development only')
    subs=p.add_subparsers(dest='cmd',required=True)
    subs.add_parser('preflight')
    pilot=subs.add_parser('pilot')
    pilot.add_argument('--out',required=True)
    pilot.add_argument('--sizes',type=int,nargs='+',default=[60])
    pilot.add_argument('--families',nargs='+',default=['hierarchical','ladder'])
    pilot.add_argument('--topology-seeds',type=int,nargs='+',default=[501,502])
    pilot.add_argument('--stream-seeds',type=int,nargs='+',default=[74000])
    pilot.add_argument('--mode',choices=('dynamic','static'),default='dynamic')
    pilot.add_argument('--load',type=float,choices=(1.6,2.5,4.0),default=2.5)
    pilot.add_argument('--stress',choices=('base','bandwidth_150','short_hold'),default='base')
    pilot.add_argument('--k',type=int,choices=(2,6),default=6)
    pilot.add_argument('--policies',nargs='+',default=['greedy','footprint','scenario_minimax','C_frozen'],
                       choices=['greedy','footprint','scenario_minimax','C_frozen'])
    smoke=subs.add_parser('gnn-smoke')
    smoke.add_argument('--out',required=True)
    smoke.add_argument('--episodes',type=int,default=2)
    smoke.add_argument('--agent-seed',type=int,default=7)
    smoke.add_argument('--load',type=float,choices=(1.6,2.5,4.0),default=1.6)
    smoke.add_argument('--mode',choices=('static','dynamic'),default='dynamic')
    args=p.parse_args()
    if args.cmd=='preflight':preflight()
    elif args.cmd=='pilot':
        r=paired_dev_pilot(args.out,sizes=tuple(args.sizes),families=tuple(args.families),
              topology_seeds=tuple(args.topology_seeds),stream_seeds=tuple(args.stream_seeds),
              mode=args.mode,load=args.load,stress=args.stress,k=args.k,policies=tuple(args.policies))
        print(f'PHASE5G_DEV_PILOT_OK policy_rows={len(r)} out={args.out}')
    else:
        hist,val=gnn_smoke(args.out,episodes=args.episodes,agent_seed=args.agent_seed,
                           load=args.load,mode=args.mode)
        print(f'PHASE5G_GNN_SMOKE_OK episodes={len(hist)} validation_policies={len(val)} '
              f'out={args.out}')

if __name__=='__main__':main()
