"""Phase 5 Windows-compatible command-line entry point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from railway_tp.topology import TopologyConfig
from .agent import AgentConfig
from .experiment import train, evaluate
from .statistics import summarize


def _config(path: Path) -> TopologyConfig:
    return TopologyConfig(**json.loads(path.read_text(encoding='utf-8')))


def build_parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description='Phase 5 masked DDQN (Phase 3 surrogate; NOT MSRP)')
    sub=ap.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train',help='Train on TRAIN seeds; choose checkpoint using VALIDATION seeds')
    tr.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
    tr.add_argument('--out',type=Path,required=True)
    tr.add_argument('--requests',type=int,default=100)
    tr.add_argument('--headroom',type=int,default=10)
    tr.add_argument('--max-candidates',type=int,default=2)
    tr.add_argument('--train-seed-start',type=int,default=10000)
    tr.add_argument('--train-seed-count',type=int,default=80)
    tr.add_argument('--val-seed-start',type=int,default=20000)
    tr.add_argument('--val-seed-count',type=int,default=10)
    tr.add_argument('--episodes',type=int,default=160)
    tr.add_argument('--validate-every',type=int,default=10)
    tr.add_argument('--patience',type=int,default=8)
    tr.add_argument('--agent-seed',type=int,default=7)
    tr.add_argument('--hidden',type=int,default=64)
    tr.add_argument('--gamma',type=float,default=0.99)
    tr.add_argument('--learning-rate',type=float,default=0.0003)
    tr.add_argument('--batch-size',type=int,default=64)
    tr.add_argument('--learning-starts',type=int,default=128)
    tr.add_argument('--replay-capacity',type=int,default=20000)
    tr.add_argument('--train-every',type=int,default=4)
    tr.add_argument('--target-sync',type=int,default=200)
    tr.add_argument('--epsilon-decay-steps',type=int,default=4000)
    ev=sub.add_parser('evaluate',help='One-shot held-out test against Phase4 baselines')
    ev.add_argument('--checkpoint',type=Path,required=True)
    ev.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
    ev.add_argument('--out',type=Path,required=True)
    ev.add_argument('--requests',type=int,default=100)
    ev.add_argument('--headroom',type=int,default=10)
    ev.add_argument('--max-candidates',type=int,default=2)
    ev.add_argument('--test-seed-start',type=int,default=31000)
    ev.add_argument('--test-seed-count',type=int,default=30)
    ev.add_argument('--policies',nargs='+',default=['greedy','cspf','milp_v2','random_feasible','ddqn'])
    ev.add_argument('--milp-time-limit',type=float,default=5.0)
    ev.add_argument('--save-plans',action='store_true')
    ev.add_argument('--allow-shift',action='store_true',
                    help='Explicitly label altered-topology/load transfer evaluation')
    st=sub.add_parser('statistics',help='Summarize held-out paired runs with seed-level t intervals')
    st.add_argument('--eval-dir',type=Path,required=True)
    st.add_argument('--out',type=Path,required=True)
    return ap


def main(argv=None) -> None:
    args=build_parser().parse_args(argv)
    if args.command=='train':
        cfg=AgentConfig(seed=args.agent_seed,hidden=args.hidden,gamma=args.gamma,
                         learning_rate=args.learning_rate,batch_size=args.batch_size,
                         learning_starts=args.learning_starts,replay_capacity=args.replay_capacity,
                         train_every=args.train_every,target_sync=args.target_sync,
                         epsilon_decay_steps=args.epsilon_decay_steps)
        res=train(_config(args.config),args.out,requests=args.requests,headroom=args.headroom,
                  max_candidates=args.max_candidates,train_seed_start=args.train_seed_start,
                  train_seed_count=args.train_seed_count,val_seed_start=args.val_seed_start,
                  val_seed_count=args.val_seed_count,train_episodes=args.episodes,
                  validate_every=args.validate_every,patience=args.patience,agent_config=cfg)
    elif args.command=='evaluate':
        res=evaluate(args.checkpoint,_config(args.config),args.out,requests=args.requests,
                     headroom=args.headroom,max_candidates=args.max_candidates,
                     test_seed_start=args.test_seed_start,test_seed_count=args.test_seed_count,
                     policies=tuple(args.policies),save_plans=args.save_plans,
                     milp_timeout=args.milp_time_limit,allow_shift=args.allow_shift)
    else:
        res=summarize(args.eval_dir,args.out)
    print('\nPHASE 5 COMMAND FINISHED')
    print(json.dumps(res,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
