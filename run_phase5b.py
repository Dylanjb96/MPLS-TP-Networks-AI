"""Phase 5B command line. Separate from frozen Phase 5 sources and results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from railway_tp.topology import TopologyConfig
from phase5.agent import AgentConfig
from phase5.experiment import dump_csv, dump_json, new_output
from phase5.statistics import summarize
from phase5b.sac import SACConfig
from phase5b.workflow import train_model, evaluate_models, guard_splits, diagnose_choice_opportunities


def topology(path):
    return TopologyConfig(**json.loads(path.read_text(encoding='utf-8')))


def tuning_grid(algorithm):
    if algorithm=='ddqn':
        return [
            {'learning_rate':0.0003,'gamma':0.99,'epsilon_decay_steps':4000},
            {'learning_rate':0.0001,'gamma':0.99,'epsilon_decay_steps':8000},
            {'learning_rate':0.0003,'gamma':0.95,'epsilon_decay_steps':8000},
            {'learning_rate':0.0001,'gamma':0.95,'epsilon_decay_steps':4000},
        ]
    if algorithm=='discrete_sac':
        return [{'learning_rate':0.0003,'gamma':0.99,'alpha':x,'target_tau':0.01}
                for x in (0.05,0.1,0.2)]
    raise ValueError('unknown model')


def make_agent_config(algorithm,settings,seed, requests):
    # Short smoke episodes need tiny replay warmup, but full experiments use
    # the exact same warmup/replay convention as Phase5 DDQN.
    kwargs=dict(settings,seed=seed)
    if requests<=10:
        kwargs.update(batch_size=8,learning_starts=8,train_every=1,replay_capacity=128)
    return (AgentConfig(**kwargs) if algorithm=='ddqn' else SACConfig(**kwargs))


def tune(args):
    cfg=topology(args.config)
    training=set(range(args.train_seed_start,args.train_seed_start+args.train_seed_count))
    val=set(range(args.val_seed_start,args.val_seed_start+args.val_seed_count))
    guard_splits(training,val)
    new_output(args.out)
    trials=[]; winners=[]
    def run_one(algo,idx,settings,seed,stage,episodes):
        out=args.out/stage/f'{algo}_cfg{idx}_seed{seed}'
        agent=make_agent_config(algo,settings,seed,args.requests)
        p=train_model(algo,cfg,out,requests=args.requests,headroom=args.headroom,
                      max_candidates=args.max_candidates,
                      train_seed_start=args.train_seed_start,train_seed_count=args.train_seed_count,
                      val_seed_start=args.val_seed_start,val_seed_count=args.val_seed_count,
                      train_episodes=episodes,validate_every=args.validate_every,
                      patience=args.patience,agent_config=agent)
        score=p['best_validation_score']
        row={'stage':stage,'algorithm':algo,'config_index':idx,
             'agent_seed':seed,'hyperparameters_json':json.dumps(settings,sort_keys=True),
             'best_validation_weighted':float(score[0]),
             'best_validation_admission':float(score[1]),
             'best_episode':p['best_episode'],
             'completed_episodes':p['completed_episodes'],
             'gradient_steps':p['gradient_steps'],
             'checkpoint':str(out/'best_model.pt')}
        trials.append(row)
        return row
    for algo in args.algorithms:
        grid=tuning_grid(algo)[:args.max_configs]
        screen=[]
        for i,hp in enumerate(grid):
            screen.append(run_one(algo,i,hp,args.screen_agent_seed,'screen',args.screen_episodes))
        top=sorted(screen,key=lambda r:(r['best_validation_weighted'],
                                         r['best_validation_admission']),reverse=True)[:args.confirm_top]
        scores=[]
        for candidate in top:
            i=candidate['config_index']; hp=grid[i]
            confirmed=[run_one(algo,i,hp,seed,'confirm',args.confirm_episodes)
                       for seed in args.agent_seeds]
            mean_weighted=statistics.mean(x['best_validation_weighted'] for x in confirmed)
            mean_admission=statistics.mean(x['best_validation_admission'] for x in confirmed)
            scores.append((mean_weighted,mean_admission,i,hp,confirmed))
        winner=max(scores,key=lambda t:(t[0],t[1]))
        best_run=max(winner[4],key=lambda x:(x['best_validation_weighted'],
                                              x['best_validation_admission']))
        winners.append({'algorithm':algo,'best_config_index':winner[2],
                        'best_config':winner[3],
                        'mean_validation_weighted_over_initializations':winner[0],
                        'mean_validation_admission_over_initializations':winner[1],
                        'selected_agent_seed':best_run['agent_seed'],
                        'selected_checkpoint':best_run['checkpoint'],
                        'selection_rule':'validation only: config by average weighted over seeds; checkpoint by best weighted'})
    dump_csv(args.out/'tuning_trials.csv',trials)
    dump_json(args.out/'tuning_selection.json',{
        'winners':winners,'training_seeds':[min(training),max(training)],
        'validation_seeds':[min(val),max(val)],
        'historical_31000_31029_NOT_used':True,
        'rule':'prospectively tune on validation only; new test seeds 41000+ evaluated once',
        'hyperparameter_screen_replicates':1,
        'confirm_initialization_seeds':args.agent_seeds,
        'note':'Repeated validation use can overfit. Keep old Phase5 results unchanged.'})
    print('SELECTED (VALIDATION ONLY):')
    print(json.dumps(winners,indent=2))


def parse():
    ap=argparse.ArgumentParser(description='Phase 5B: prospective validation tuning + masked discrete SAC')
    sub=ap.add_subparsers(dest='command',required=True)
    for name in ('diagnose','tune','train-sac','evaluate'):
        p=sub.add_parser(name)
        p.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
        p.add_argument('--out',type=Path,required=True)
        p.add_argument('--requests',type=int,default=100)
        p.add_argument('--headroom',type=int,default=10)
        p.add_argument('--max-candidates',type=int,default=2)
        if name!='evaluate':
            p.add_argument('--train-seed-start',type=int,default=10000)
            p.add_argument('--train-seed-count',type=int,default=80)
            p.add_argument('--val-seed-start',type=int,default=20000)
            p.add_argument('--val-seed-count',type=int,default=10)
            p.add_argument('--validate-every',type=int,default=10)
            p.add_argument('--patience',type=int,default=8)
    diag=sub.choices['diagnose']
    diag.add_argument('--seed-start',type=int,default=20000)
    diag.add_argument('--seed-count',type=int,default=10)
    t=sub.choices['tune']
    t.add_argument('--algorithms',nargs='+',choices=['ddqn','discrete_sac'],default=['ddqn','discrete_sac'])
    t.add_argument('--max-configs',type=int,default=3)
    t.add_argument('--confirm-top',type=int,default=1)
    t.add_argument('--screen-episodes',type=int,default=80)
    t.add_argument('--confirm-episodes',type=int,default=160)
    t.add_argument('--screen-agent-seed',type=int,default=7)
    t.add_argument('--agent-seeds',nargs='+',type=int,default=[7,17,27])
    s=sub.choices['train-sac']
    s.add_argument('--episodes',type=int,default=160)
    s.add_argument('--agent-seed',type=int,default=7)
    s.add_argument('--learning-rate',type=float,default=0.0003)
    s.add_argument('--alpha',type=float,default=0.1)
    s.add_argument('--gamma',type=float,default=0.99)
    s.add_argument('--target-tau',type=float,default=0.01)
    e=sub.choices['evaluate']
    e.add_argument('--original-ddqn',type=Path,required=True)
    e.add_argument('--tuned-ddqn',type=Path)
    e.add_argument('--discrete-sac',type=Path)
    e.add_argument('--test-seed-start',type=int,default=41000)
    e.add_argument('--test-seed-count',type=int,default=30)
    e.add_argument('--save-plans',action='store_true')
    v=sub.add_parser('statistics')
    v.add_argument('--eval-dir',type=Path,required=True)
    v.add_argument('--out',type=Path,required=True)
    return ap.parse_args()


def main():
    args=parse()
    if args.command=='diagnose':
        print(diagnose_choice_opportunities(topology(args.config),args.out,
              requests=args.requests,headroom=args.headroom,max_candidates=args.max_candidates,
              seed_start=args.seed_start,seed_count=args.seed_count))
    elif args.command=='tune':
        if not (args.max_configs>=1 and args.confirm_top>=1 and
                args.screen_episodes>=1 and args.confirm_episodes>=1 and
                args.agent_seeds and len(set(args.agent_seeds))==len(args.agent_seeds)):
            raise ValueError('invalid tuning design')
        tune(args)
    elif args.command=='train-sac':
        cfg=topology(args.config)
        p=train_model('discrete_sac',cfg,args.out,requests=args.requests,
             headroom=args.headroom,max_candidates=args.max_candidates,
             train_seed_start=args.train_seed_start,train_seed_count=args.train_seed_count,
             val_seed_start=args.val_seed_start,val_seed_count=args.val_seed_count,
             train_episodes=args.episodes,validate_every=args.validate_every,
             patience=args.patience,agent_config=SACConfig(seed=args.agent_seed,
                learning_rate=args.learning_rate,alpha=args.alpha,
                gamma=args.gamma,target_tau=args.target_tau,
                batch_size=8 if args.requests<=10 else 64,
                learning_starts=8 if args.requests<=10 else 128,
                replay_capacity=128 if args.requests<=10 else 20000,
                train_every=1 if args.requests<=10 else 4))
        print(json.dumps({'best_episode':p['best_episode'],'best_validation_score':p['best_validation_score']},indent=2))
    elif args.command=='evaluate':
        cp={'original_ddqn':args.original_ddqn}
        if args.tuned_ddqn:cp['tuned_ddqn']=args.tuned_ddqn
        if args.discrete_sac:cp['discrete_sac']=args.discrete_sac
        print(evaluate_models(topology(args.config),args.out,cp,
               requests=args.requests,headroom=args.headroom,max_candidates=args.max_candidates,
               test_seed_start=args.test_seed_start,test_seed_count=args.test_seed_count,
               save_plans=args.save_plans))
    else:
        result=summarize(args.eval_dir,args.out)
        # Avoid presenting the baseline report title as though only DDQN is compared.
        p=args.out/'phase5_statistical_report.md'
        body=p.read_text(encoding='utf-8')
        body=body.replace('# Phase 5 — Held-out masked DDQN versus conventional baselines',
                          '# Phase 5B — New held-out comparison: tuned DDQN, discrete SAC, original DDQN, and baselines')
        p.write_text(body+'\nPhase5B is a post-Phase5 extension evaluated on NEW seeds; historical 31000..31029 excluded.\n',encoding='utf-8')
        print(result)

if __name__=='__main__':main()
