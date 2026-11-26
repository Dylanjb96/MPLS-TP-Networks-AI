"""Phase5D CLI: read-only analysis; no training, no checkpoint writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from railway_tp.topology import TopologyConfig
from .analysis import (behavior, counterfactual, milp_trace,
                       load_selected_models, development_seed_guard)


def parser():
    p=argparse.ArgumentParser(description='Phase 5D: frozen policy behaviour, '
                        'counterfactual developer-stream branching, MILP divergence tracing')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('behavior','counterfactual','milp-trace'):
        q=sub.add_parser(name)
        q.add_argument('--config',type=Path,default=Path('config/primary_60_dual.json'))
        q.add_argument('--out',type=Path,required=True)
        q.add_argument('--requests',type=int,default=100)
        q.add_argument('--headroom',type=int,default=10)
        q.add_argument('--max-candidates',type=int,default=2)
    b=sub.choices['behavior']
    b.add_argument('--selection',type=Path)
    b.add_argument('--original-ddqn',type=Path)
    b.add_argument('--discrete-sac',type=Path)
    b.add_argument('--baseline-only',action='store_true',help='software smoke test, no models loaded')
    b.add_argument('--seed-start',type=int,default=20000)
    b.add_argument('--seed-count',type=int,default=10)
    b.add_argument('--retrospective',action='store_true',
                   help='explicitly mark inspected historical tests; never for model selection')
    c=sub.choices['counterfactual']
    c.add_argument('--seed-start',type=int,default=21000)
    c.add_argument('--seed-count',type=int,default=10)
    c.add_argument('--max-branches',type=int,default=8)
    c.add_argument('--horizon',type=int,default=100)
    m=sub.choices['milp-trace']
    m.add_argument('--seed',type=int,default=41027)
    m.add_argument('--retrospective',action='store_true',required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    cfg=TopologyConfig(**json.loads(a.config.read_text(encoding='utf8')))
    common={'requests':a.requests,'headroom':a.headroom,'k':a.max_candidates}
    if a.command=='behavior':
        if a.seed_count<=0:raise ValueError('positive seed count required')
        seeds=list(range(a.seed_start,a.seed_start+a.seed_count))
        development_seed_guard(seeds,retrospective=a.retrospective)
        if a.baseline_only and any((a.selection,a.original_ddqn,a.discrete_sac)):
            raise ValueError('--baseline-only cannot be combined with checkpoints')
        if not a.baseline_only and not all((a.selection,a.original_ddqn,a.discrete_sac)):
            raise ValueError('complete behaviour diagnostic requires --selection '
                             '--original-ddqn and --discrete-sac (or use --baseline-only for smoke)')
        models,hashes=load_selected_models(a.selection,a.original_ddqn,a.discrete_sac,
                                           cfg,a.requests,a.headroom,a.max_candidates,seeds)
        result=behavior(cfg,a.out,models,**common,seeds=seeds,
                        model_hashes=hashes,retrospective=a.retrospective)
    elif a.command=='counterfactual':
        if a.seed_count<=0:raise ValueError('positive seed count required')
        seeds=list(range(a.seed_start,a.seed_start+a.seed_count))
        result=counterfactual(cfg,a.out,**common,seeds=seeds,
                              max_branches=a.max_branches,horizon=a.horizon)
    else:
        result=milp_trace(cfg,a.out,**common,seed=a.seed,retrospective=a.retrospective)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
