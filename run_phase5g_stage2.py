"""Phase 5G Stage 2 development/training entry point; no held-out command."""
import argparse,json
from phase5g_stage2.workflow import Plan,preflight,run_training,select_models,status


def main():
    p=argparse.ArgumentParser(description='Phase 5G matched-budget GNN vs MLP development')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('preflight')
    a=sub.add_parser('quickcheck')
    a.add_argument('--out',default='outputs/phase5g/stage2_quickcheck_dev')
    a.add_argument('--cache-root',default='outputs/phase5g/stage2_cohort_cache')
    b=sub.add_parser('train')
    b.add_argument('--out',required=True)
    b.add_argument('--model',required=True,choices=('mlp','gnn'))
    b.add_argument('--agent-seed',type=int,required=True,choices=(7,17,27))
    b.add_argument('--episodes',type=int,default=48)
    b.add_argument('--validate-every',type=int,default=12)
    b.add_argument('--cache-root',default='outputs/phase5g/stage2_cohort_cache')
    b.add_argument('--resume',action='store_true')
    s=sub.add_parser('select')
    s.add_argument('--runs-root',default='outputs/phase5g/stage2_train')
    s.add_argument('--out',default='outputs/phase5g/stage2_selection_FINAL')
    q=sub.add_parser('status');q.add_argument('--root',required=True)
    args=p.parse_args()
    if args.command=='preflight':preflight()
    elif args.command=='quickcheck':
        plan=Plan('gnn',7,episodes=2,validate_every=1,train_sizes=(60,),
            train_loads=(1.6,),train_modes=('dynamic',),
            train_topologies=(601,602),train_streams=(75000,75001),
            val_topologies=(701,),val_streams=(76000,),
            val_size=60,val_load=1.6,max_update_samples=8)
        result=run_training(args.out,plan,cache_root=args.cache_root)
        print('PHASE5G_STAGE2_QUICKCHECK_OK '+json.dumps(result,sort_keys=True))
    elif args.command=='train':
        plan=Plan(args.model,args.agent_seed,episodes=args.episodes,
                  validate_every=args.validate_every)
        result=run_training(args.out,plan,cache_root=args.cache_root,resume=args.resume)
        print('PHASE5G_STAGE2_TRAIN_OK '+json.dumps(result,sort_keys=True))
    elif args.command=='select':
        print(json.dumps(select_models(args.runs_root,args.out),sort_keys=True))
    elif args.command=='status':print(json.dumps(status(args.root),sort_keys=True))

if __name__=='__main__':main()
