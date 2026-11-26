"""Phase 5C command line. Commands refuse to overwrite completed evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from railway_tp.topology import TopologyConfig
from .common import RLConfig
from .dataset import build_dataset,checksum
from .workflow import train_ppo,train_offline,finetune_awac,evaluate,statistics
from .selection import select_models
from .plots import plot_results


def topo(path):return TopologyConfig(**json.loads(path.read_text(encoding='utf8')))


def parser():
    p=argparse.ArgumentParser(description='Phase5C: masked PPO, discrete BCQ/CQL/AWAC/IQL')
    s=p.add_subparsers(dest='action',required=True)
    for name in ('dataset','train-ppo','train-offline','finetune-awac','evaluate'):
        a=s.add_parser(name)
        a.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
        a.add_argument('--out',type=Path,required=True)
        a.add_argument('--requests',type=int,default=100)
        a.add_argument('--headroom',type=int,default=10)
        a.add_argument('--max-candidates',type=int,default=2)
    d=s.choices['dataset'];d.add_argument('--seed-start',type=int,default=10000)
    d.add_argument('--seed-count',type=int,default=80)
    for name in ('train-ppo','train-offline'):
        a=s.choices[name]
        a.add_argument('--agent-seed',type=int,default=7)
        a.add_argument('--learning-rate',type=float,default=0.0003)
        a.add_argument('--hidden',type=int,default=64)
        a.add_argument('--gamma',type=float,default=0.99)
        a.add_argument('--batch-size',type=int,default=64)
        a.add_argument('--val-seed-start',type=int,default=20000)
        a.add_argument('--val-seed-count',type=int,default=10)
        a.add_argument('--validate-every',type=int,default=10 if name=='train-ppo' else 300)
        a.add_argument('--patience',type=int,default=8)
    pp=s.choices['train-ppo'];pp.add_argument('--train-seed-start',type=int,default=10000)
    pp.add_argument('--train-seed-count',type=int,default=80)
    pp.add_argument('--episodes',type=int,default=160)
    pp.add_argument('--ppo-clip',type=float,default=.2)
    pp.add_argument('--ppo-epochs',type=int,default=4)
    pp.add_argument('--entropy-coef',type=float,default=.01)
    of=s.choices['train-offline'];of.add_argument('--dataset',type=Path,required=True)
    of.add_argument('--algorithm',choices=('bcq','cql','awac','iql'),required=True)
    of.add_argument('--updates',type=int,default=3000)
    of.add_argument('--behavior-threshold',type=float,default=.3)
    of.add_argument('--conservative-alpha',type=float,default=1.)
    of.add_argument('--awac-lambda',type=float,default=1.)
    of.add_argument('--expectile',type=float,default=.7)
    af=s.choices['finetune-awac']
    af.add_argument('--offline-checkpoint',type=Path,required=True)
    af.add_argument('--dataset',type=Path,required=True)
    af.add_argument('--train-seed-start',type=int,default=10000)
    af.add_argument('--train-seed-count',type=int,default=80)
    af.add_argument('--val-seed-start',type=int,default=20000)
    af.add_argument('--val-seed-count',type=int,default=10)
    af.add_argument('--episodes',type=int,default=80)
    af.add_argument('--updates-per-episode',type=int,default=64)
    af.add_argument('--validate-every',type=int,default=10)
    af.add_argument('--exploration-epsilon',type=float,default=0.1)
    af.add_argument('--online-fraction',type=float,default=.5)
    af.add_argument('--patience',type=int,default=8)
    select=s.add_parser('select');select.add_argument('--out',type=Path,required=True)
    select.add_argument('--runs',nargs='+',type=Path)
    select.add_argument('--runs-root',type=Path)
    select.add_argument('--min-seeds',type=int,default=3)
    ev=s.choices['evaluate'];ev.add_argument('--selection',type=Path,required=True)
    ev.add_argument('--original-ddqn',type=Path)
    ev.add_argument('--discrete-sac',type=Path)
    ev.add_argument('--test-seed-start',type=int,default=51000)
    ev.add_argument('--test-seed-count',type=int,default=30)
    ev.add_argument('--save-plans',action='store_true')
    st=s.add_parser('statistics');st.add_argument('--eval-dir',type=Path,required=True)
    st.add_argument('--out',type=Path,required=True)
    plots=s.add_parser('plots');plots.add_argument('--stats',type=Path,required=True)
    plots.add_argument('--out',type=Path,required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.action=='dataset':
        result=build_dataset(topo(a.config),a.out,requests=a.requests,
              headroom=a.headroom,k=a.max_candidates,
              seed_start=a.seed_start,seed_count=a.seed_count)
    elif a.action=='train-ppo':
        config=RLConfig(algorithm='ppo',seed=a.agent_seed,lr=a.learning_rate,
            hidden=a.hidden,gamma=a.gamma,batch=a.batch_size,
            ppo_clip=a.ppo_clip,ppo_epochs=a.ppo_epochs,entropy_coef=a.entropy_coef)
        result=train_ppo(topo(a.config),a.out,config,requests=a.requests,
             headroom=a.headroom,k=a.max_candidates,train_seed_start=a.train_seed_start,
             train_seed_count=a.train_seed_count,val_seed_start=a.val_seed_start,
             val_seed_count=a.val_seed_count,episodes=a.episodes,
             validate_every=a.validate_every,patience=a.patience)
    elif a.action=='train-offline':
        config=RLConfig(algorithm=a.algorithm,seed=a.agent_seed,lr=a.learning_rate,
             hidden=a.hidden,gamma=a.gamma,batch=a.batch_size,
             behavior_threshold=a.behavior_threshold,conservative_alpha=a.conservative_alpha,
             awac_lambda=a.awac_lambda,expectile=a.expectile)
        result=train_offline(topo(a.config),a.out,config,a.dataset,requests=a.requests,
             headroom=a.headroom,k=a.max_candidates,val_seed_start=a.val_seed_start,
             val_seed_count=a.val_seed_count,updates=a.updates,
             validate_every=a.validate_every,patience=a.patience)
    elif a.action=='finetune-awac':
        result=finetune_awac(topo(a.config),a.out,a.offline_checkpoint,a.dataset,
              requests=a.requests,headroom=a.headroom,k=a.max_candidates,
              train_seed_start=a.train_seed_start,train_seed_count=a.train_seed_count,
              val_seed_start=a.val_seed_start,val_seed_count=a.val_seed_count,
              episodes=a.episodes,updates_per_episode=a.updates_per_episode,
              validate_every=a.validate_every,patience=a.patience,
              exploration_epsilon=a.exploration_epsilon,online_fraction=a.online_fraction)
    elif a.action=='select':
        runs=list(a.runs or [])
        if a.runs_root:
            runs += sorted(p.parent for p in a.runs_root.glob('**/training_provenance.json'))
        result=select_models(a.out,runs,a.min_seeds)
    elif a.action=='evaluate':
        manifest=json.loads(a.selection.read_text(encoding='utf8'))
        cp={}
        for winner in manifest['winners']:
            path=Path(winner['selected_checkpoint'])
            if checksum(path)!=winner['checkpoint_sha256']:
                raise ValueError('checkpoint changed after selection '+str(path))
            cp[winner['algorithm']]=path
        if a.original_ddqn:cp['original_ddqn']=a.original_ddqn
        if a.discrete_sac:cp['discrete_sac']=a.discrete_sac
        result=evaluate(topo(a.config),a.out,cp,requests=a.requests,
             headroom=a.headroom,k=a.max_candidates,test_seed_start=a.test_seed_start,
             test_seed_count=a.test_seed_count,save_plans=a.save_plans)
    elif a.action=='statistics':result=statistics(a.eval_dir,a.out)
    else:result={'figures':str(plot_results(a.stats,a.out))}
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
