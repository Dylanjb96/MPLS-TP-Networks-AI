"""Phase 5F Stage 4: development-only full factorial multi-topology PPO training.

Stage 5 heldout topology/traffic seeds are deliberately unsupported.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from phase5f_stage4.ablation import CONFIGS
from phase5f_stage4.workflow import TrainPlan,run_training,ALL_SIZES,ALL_LOADS,validate_role_settings
from phase5f_stage4.selection import select_models
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs
from phase5f.traffic import generate_demands
from phase5f.protection import ScenarioLedger
from phase5f_stage3.features import encode
from phase5f_stage3.agent import CandidateActorCritic
from phase5f_stage4.ablation import transform
from phase5f_stage3.workflow import source_hashes as stage3_source_hashes


def _add_train(sub):
    p=sub.add_parser('train',help='Run one of the 12 Stage 4 development trainings')
    p.add_argument('--out',required=True)
    p.add_argument('--cache-dir',default='outputs/phase5f/stage4_shared_candidate_cache')
    p.add_argument('--config',choices=list(CONFIGS),required=True)
    p.add_argument('--agent-seed',type=int,required=True)
    p.add_argument('--episodes',type=int,default=96)
    p.add_argument('--validate-every',type=int,default=12)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--train-topologies',type=int,nargs='+',default=list(range(201,221)))
    p.add_argument('--train-streams',type=int,nargs='+',default=list(range(71000,71008)))
    p.add_argument('--val-topologies',type=int,nargs='+',default=[301,302,303,304])
    p.add_argument('--val-streams',type=int,nargs='+',default=[72000,72001])
    p.add_argument('--train-sizes',type=int,nargs='+',default=list(ALL_SIZES))
    p.add_argument('--train-loads',type=float,nargs='+',default=list(ALL_LOADS))
    p.add_argument('--train-modes',nargs='+',default=['static','dynamic'])
    p.add_argument('--val-size',type=int,default=500)
    p.add_argument('--val-load',type=float,default=4.0)
    p.add_argument('--val-mode',choices=['static','dynamic'],default='dynamic')
    p.add_argument('--k',type=int,default=6)
    p.add_argument('--headroom',type=int,default=10)
    p.add_argument('--max-working-paths',type=int,default=40)
    p.add_argument('--learning-rate',type=float,default=0.0003)
    p.add_argument('--bc-coef',type=float,default=0.01)
    p.add_argument('--eval-gate-threshold',type=float,default=0.1)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--ppo-epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--entropy-coef',type=float,default=0.01)
    p.add_argument('--resume',action='store_true')


def _preflight(stage3_manifest=None):
    if stage3_manifest is not None:
        data=json.loads(Path(stage3_manifest).read_text(encoding="utf8"))
        observed=stage3_source_hashes()
        for k,v in observed.items():
            if data.get("source_sha256",{}).get(k)!=v:
                raise ValueError("Stage 3 source mismatch: "+k)
    validate_role_settings((201,),(71000,),(301,),(72000,))
    graph=build_network(NetworkSpec(nodes=60,seed=201))
    d=generate_demands(graph,1,71000,False)[0]
    pairs=generate_pairs(graph,d.source,d.destination,6,40)
    led=ScenarioLedger(graph,10)
    feasible=[i for i,p in enumerate(pairs) if led.feasible(d,p)]
    if len(feasible)<2:raise AssertionError('insufficient discretionary routes')
    obs=encode(graph,led,d,pairs,feasible)
    model=CandidateActorCritic()
    import torch
    torch.manual_seed(777)
    for c in CONFIGS.values():
        scores=model.decide(transform(obs,c),greedy_idx=feasible[0])
        if scores['action'] not in feasible:raise AssertionError('infeasible action')
        if (not c.guidance and transform(obs,c)[0][:,12].any()):
            raise AssertionError('unguided arm sees greedy reference')
    print('STAGE4_PREFLIGHT_OK candidate_actions='+str(len(feasible))+' topology_sha='+fingerprint(graph))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    pf=sub.add_parser('preflight')
    pf.add_argument('--stage3-manifest',default=None,help='Verify unchanged Stage 3 and Stage 2 sources')
    sub.add_parser('protocol',help='Print predeclared training and held-out partitions')
    _add_train(sub)
    s=sub.add_parser('select',help='Validation-only select after all 12 complete trainings')
    s.add_argument('--runs-root',required=True)
    s.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='preflight':_preflight(a.stage3_manifest)
    elif a.command=='protocol':
        print(json.dumps(dict(arms=list(CONFIGS),training_topologies='201..220',
            training_streams='71000..71007',validation_topologies='301..310',
            validation_streams='72000..72002',
            reserved_final_topologies='401..420',reserved_final_streams='73000..73002',
            primary_validation='500 nodes, k=6, dynamic, load 4.0; mean per cohort',
            planned_training='4 configurations x 3 initializations, episodes 96',
            protection='not MSRP, not the actual Iarnrod Eireann network'),indent=2))
    elif a.command=='select':
        m=select_models(a.runs_root,a.out)
        print('STAGE4_SELECTION_OK config={} seed={} score={:.6f}'.format(
          m['selected']['config'],m['selected']['agent_seed'],m['selected']['best_weighted']))
    elif a.command=='train':
        plan=TrainPlan(config=a.config,agent_seed=a.agent_seed,episodes=a.episodes,
            validate_every=a.validate_every,patience=a.patience,
            train_topologies=tuple(a.train_topologies),train_streams=tuple(a.train_streams),
            val_topologies=tuple(a.val_topologies),val_streams=tuple(a.val_streams),
            train_sizes=tuple(a.train_sizes),train_loads=tuple(a.train_loads),
            train_modes=tuple(a.train_modes),val_size=a.val_size,
            val_load=a.val_load,val_mode=a.val_mode,
            k=a.k,headroom=a.headroom,max_working_paths=a.max_working_paths,
            lr=a.learning_rate,bc_coef=a.bc_coef,
            eval_gate_threshold=a.eval_gate_threshold,hidden=a.hidden,
            ppo_epochs=a.ppo_epochs,batch_size=a.batch_size,entropy_coef=a.entropy_coef)
        m=run_training(a.out,a.cache_dir,plan,resume=a.resume)
        print('STAGE4_TRAIN_OK config={} seed={} episodes={} best_episode={} best_weighted={:.6f} out={}'.format(
           a.config,a.agent_seed,m['completed_episodes'],m['best_episode'],
           m['best_mean_weighted_validation'],a.out))

if __name__=='__main__':main()
