"""Phase 5F Stage 3: development-only candidate-aware greedy-guided PPO smoke."""
from __future__ import annotations
import argparse
from phase5f.traffic import generate_demands
from phase5f.protection import ScenarioLedger
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs
from phase5f_stage2.probe import environment
from phase5f_stage3.agent import CandidateActorCritic
from phase5f_stage3.features import encode
from phase5f_stage3.workflow import run_smoke


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    subs.add_parser('preflight')
    s=subs.add_parser('smoke',help='Training smoke only: no model selection or held-out evaluation')
    s.add_argument('--out',required=True)
    s.add_argument('--nodes',type=int,default=60)
    s.add_argument('--episodes',type=int,default=4)
    s.add_argument('--train-topologies',type=int,nargs='+',default=[201,202])
    s.add_argument('--train-streams',type=int,nargs='+',default=[71000,71001])
    s.add_argument('--val-topologies',type=int,nargs='+',default=[301])
    s.add_argument('--val-streams',type=int,nargs='+',default=[72000])
    s.add_argument('--load',type=float,default=2.5)
    s.add_argument('--traffic',nargs='+',default=['static','dynamic'])
    s.add_argument('--k',type=int,default=6)
    s.add_argument('--max-working-paths',type=int,default=40)
    s.add_argument('--headroom',type=int,default=10)
    s.add_argument('--agent-seed',type=int,default=7)
    s.add_argument('--learning-rate',type=float,default=0.0003)
    s.add_argument('--bc-coef',type=float,default=0.01)
    s.add_argument('--gate-threshold',type=float,default=0.1)
    a=p.parse_args()
    if a.command=='preflight':
        graph=build_network(NetworkSpec(nodes=60,seed=201))
        req=generate_demands(graph,1,71000,False)[0]
        plans=generate_pairs(graph,req.source,req.destination,max_candidates=6,max_working_paths=40)
        ledger=ScenarioLedger(graph,10)
        feasible=[i for i,plan in enumerate(plans) if ledger.feasible(req,plan)]
        if len(feasible)<2:raise AssertionError('preflight needs discretionary choice')
        model=CandidateActorCritic()
        prediction=model.decide(encode(graph,ledger,req,plans,feasible))
        print('STAGE3_PREFLIGHT_OK nodes=60 routes={} feasible={} selected={} topo_sha={} software={}'.format(
              len(plans),len(feasible),prediction['action'],fingerprint(graph),environment()))
    else:
        result=run_smoke(a.out,nodes=a.nodes,episodes=a.episodes,
            train_topologies=tuple(a.train_topologies),train_streams=tuple(a.train_streams),
            val_topologies=tuple(a.val_topologies),val_streams=tuple(a.val_streams),
            load=a.load,modes=tuple(a.traffic),k=a.k,
            max_working_paths=a.max_working_paths,headroom=a.headroom,
            agent_seed=a.agent_seed,learning_rate=a.learning_rate,
            bc_coef=a.bc_coef,gate_threshold=a.gate_threshold)
        print('STAGE3_SMOKE_OK actions={} validation_runs={} out={} final_data_accessed={}'.format(
              result['training_actions'],result['validation_rows'],a.out,result['final_data_accessed']))


if __name__=='__main__':main()
