"""Phase 5E command-line interface. Outputs never overwrite earlier evidence."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from railway_tp.topology import TopologyConfig
from .agent import HybridConfig
from .workflow import probe,train,select,evaluate,statistics_report,plots,preflight


def topo(path):
    return TopologyConfig(**json.loads(Path(path).read_text(encoding='utf-8')))


def parser():
    p=argparse.ArgumentParser(description='Phase 5E: greedy-guided residual PPO, guarded evaluation')
    sub=p.add_subparsers(dest='command',required=True)
    pf=sub.add_parser('preflight')
    pf.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
    pf.add_argument('--phase5d-manifest',type=Path)
    for name in ('probe','train','evaluate'):
        x=sub.add_parser(name)
        x.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
        x.add_argument('--out',type=Path,required=True)
        x.add_argument('--requests',type=int,default=100)
        x.add_argument('--headroom',type=int,default=10)
        x.add_argument('--max-candidates',type=int,default=2)
    pr=sub.choices['probe']
    pr.add_argument('--seed-start',type=int,default=23000)
    pr.add_argument('--seed-count',type=int,default=10)
    pr.add_argument('--capacity-threshold',type=float,default=.05)
    t=sub.choices['train']
    t.add_argument('--features',choices=['base','enhanced'],default='enhanced')
    t.add_argument('--agent-seed',type=int,default=7)
    t.add_argument('--learning-rate',type=float,default=.0003)
    t.add_argument('--hidden',type=int,default=64)
    t.add_argument('--gamma',type=float,default=1.0)
    t.add_argument('--gae-lambda',type=float,default=.95)
    t.add_argument('--ppo-clip',type=float,default=.2)
    t.add_argument('--ppo-epochs',type=int,default=4)
    t.add_argument('--batch',type=int,default=64)
    t.add_argument('--entropy-coef',type=float,default=.01)
    t.add_argument('--greedy-bc-coef',type=float,default=.1)
    t.add_argument('--warmstart-steps',type=int,default=50)
    t.add_argument('--gate-threshold',type=float,default=.15)
    t.add_argument('--train-seed-start',type=int,default=11000)
    t.add_argument('--train-seed-count',type=int,default=80)
    t.add_argument('--val-seed-start',type=int,default=22000)
    t.add_argument('--val-seed-count',type=int,default=20)
    t.add_argument('--episodes',type=int,default=160)
    t.add_argument('--validate-every',type=int,default=10)
    t.add_argument('--patience',type=int,default=8)
    s=sub.add_parser('select')
    s.add_argument('--runs-root',type=Path,required=True)
    s.add_argument('--out',type=Path,required=True)
    s.add_argument('--min-seeds',type=int,default=3)
    e=sub.choices['evaluate']
    e.add_argument('--selection',type=Path,required=True)
    e.add_argument('--role',choices=['primary','development','sensitivity'],default='primary')
    e.add_argument('--seed-start',type=int,default=61000)
    e.add_argument('--seed-count',type=int,default=60)
    e.add_argument('--phase5c-selection',type=Path)
    e.add_argument('--original-ddqn',type=Path)
    e.add_argument('--discrete-sac',type=Path)
    e.add_argument('--save-plans',action='store_true')
    e.add_argument('--milp-timeout',type=float,default=5.0)
    e.add_argument('--capacity-threshold',type=float,default=.05)
    st=sub.add_parser('statistics')
    st.add_argument('--eval-dir',type=Path,required=True)
    st.add_argument('--out',type=Path,required=True)
    pp=sub.add_parser('plots')
    pp.add_argument('--stats',type=Path,required=True)
    pp.add_argument('--out',type=Path,required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=='preflight':
        result=preflight(topo(a.config),a.phase5d_manifest)
    elif a.command=='probe':
        result=probe(topo(a.config),a.out,requests=a.requests,
             headroom=a.headroom,k=a.max_candidates,seed_start=a.seed_start,
             seed_count=a.seed_count,capacity_threshold=a.capacity_threshold)
    elif a.command=='train':
        cfg=HybridConfig(seed=a.agent_seed,enhanced=(a.features=='enhanced'),
            lr=a.learning_rate,hidden=a.hidden,gamma=a.gamma,
            gae_lambda=a.gae_lambda,clip=a.ppo_clip,ppo_epochs=a.ppo_epochs,
            batch=a.batch,entropy_coef=a.entropy_coef,
            greedy_bc_coef=a.greedy_bc_coef,warmstart_steps=a.warmstart_steps,
            gate_threshold=a.gate_threshold)
        result=train(topo(a.config),a.out,cfg,requests=a.requests,
            headroom=a.headroom,k=a.max_candidates,
            train_start=a.train_seed_start,train_count=a.train_seed_count,
            val_start=a.val_seed_start,val_count=a.val_seed_count,
            episodes=a.episodes,validate_every=a.validate_every,patience=a.patience)
    elif a.command=='select':
        result=select(a.out,a.runs_root,min_seeds=a.min_seeds)
    elif a.command=='evaluate':
        result=evaluate(topo(a.config),a.out,a.selection,requests=a.requests,
            headroom=a.headroom,k=a.max_candidates,
            seed_start=a.seed_start,seed_count=a.seed_count,
            role=a.role,phase5c_selection=a.phase5c_selection,
            original_ddqn=a.original_ddqn,discrete_sac=a.discrete_sac,
            save_plans=a.save_plans,milp_timeout=a.milp_timeout,
            capacity_threshold=a.capacity_threshold)
    elif a.command=='statistics':result=statistics_report(a.eval_dir,a.out)
    else:result=plots(a.stats,a.out)
    print(json.dumps(result,sort_keys=True,indent=2))
    return result
