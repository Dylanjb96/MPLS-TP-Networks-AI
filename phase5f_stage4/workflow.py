"""Stage 4: auditable, resumable, *development-only* 2x2 PPO factorial.

No command in this module can generate or evaluate final-held-out topology or
traffic seeds.  Model checkpoints are selected using predeclared development
validation cohorts, not previous Phase 5C/E held-out results.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json,os,pickle,platform,random,time
import numpy as np
import torch
from phase5f.traffic import Demand,generate_demands,stream_digest
from phase5f.protection import ProtectedPair
from phase5f_stage2.topology import NetworkSpec,build_network,fingerprint
from phase5f_stage2.candidates import generate_pairs,candidates_digest
from phase5f_stage2.probe import environment
from phase5f_stage3.agent import CandidateActorCritic,ppo_update
from phase5f_stage3.workflow import run_episode,source_hashes as stage3_sources
from phase5f_stage3.workflow import _validate_seed_role
from .ablation import CONFIGS, Ablation,MaskedFeaturePolicy,mask_training_observations
from . import VERSION

PROTECTION='END_TO_END_LINK_DISJOINT_PAIR_SINGLE_FAULT_SURROGATE_NOT_MSRP'
ALL_SIZES=(60,150,250,500)
ALL_LOADS=(1.6,2.5,4.0)
ALL_MODES=('static','dynamic')
TRAIN_TOPO=tuple(range(201,221))
TRAIN_STREAM=tuple(range(71000,71008))
VAL_TOPO=tuple(range(301,311))
VAL_STREAM=tuple(range(72000,72003))
RESERVED_TEST_TOPO='401..420 (never evaluated in Stage 4)'
RESERVED_TEST_STREAM='73000..73002 (never evaluated in Stage 4)'


def _digest(blob:bytes):return hashlib.sha256(blob).hexdigest()

def _sha(path):return _digest(Path(path).read_bytes())

def _json_write(path,data):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf8')
    os.replace(tmp,target)

def _csv_write(path,rows):
    if not rows:raise ValueError('refusing to write empty CSV '+str(path))
    keys=list(rows[0]);
    if any(set(r)!=set(keys) for r in rows):raise ValueError('inconsistent CSV row schemas')
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    os.replace(tmp,target)

def _pt_write(path,obj):
    target=Path(path);tmp=target.with_name(target.name+'.tmp')
    torch.save(obj,tmp);os.replace(tmp,target)

def _source_hashes():
    root=Path(__file__).resolve().parent.parent
    ours={str(p.relative_to(root)).replace('\\','/'): _sha(p)
          for p in sorted((root/'phase5f_stage4').glob('*.py'))}
    cli=root/'run_phase5f_stage4.py'
    if cli.exists():ours['run_phase5f_stage4.py']=_sha(cli)
    return {**stage3_sources(),**ours}


def validate_role_settings(train_topos,train_streams,val_topos,val_streams):
    """Fail closed: no passing final-test seeds as any form of development data."""
    _validate_seed_role('train',tuple(train_topos),tuple(train_streams))
    _validate_seed_role('validation',tuple(val_topos),tuple(val_streams))
    if len(set(train_topos))!=len(train_topos) or len(set(train_streams))!=len(train_streams):
        raise ValueError('duplicate training seed')
    if len(set(val_topos))!=len(val_topos) or len(set(val_streams))!=len(val_streams):
        raise ValueError('duplicate validation seed')


@dataclass(frozen=True)
class TrainPlan:
    config:str
    agent_seed:int
    episodes:int=96
    validate_every:int=12
    patience:int=5  # number of non-improving evaluations; zero disables early stop
    train_topologies:tuple[int,...]=TRAIN_TOPO
    train_streams:tuple[int,...]=TRAIN_STREAM
    val_topologies:tuple[int,...]=(301,302,303,304)
    val_streams:tuple[int,...]=(72000,72001)
    train_sizes:tuple[int,...]=ALL_SIZES
    train_loads:tuple[float,...]=ALL_LOADS
    train_modes:tuple[str,...]=ALL_MODES
    # The primary study target is the 500-node / K=6 / dynamic / load=4 cell.
    val_size:int=500
    val_load:float=4.0
    val_mode:str='dynamic'
    k:int=6
    headroom:int=10
    max_working_paths:int=40
    lr:float=0.0003
    gamma:float=1.0  # Stage 3 uses undiscounted suffix return
    bc_coef:float=0.01
    eval_gate_threshold:float=0.1
    hidden:int=64
    ppo_epochs:int=2
    batch_size:int=64
    entropy_coef:float=0.01

    def validate(self):
        if self.config not in CONFIGS:raise ValueError('unknown factorial arm')
        if self.agent_seed<0 or self.agent_seed>2**31:raise ValueError('bad model seed')
        if self.episodes<1 or self.episodes>20000 or self.validate_every<1:
            raise ValueError('bad episode or evaluation interval')
        if self.patience<0:raise ValueError('bad patience')
        validate_role_settings(self.train_topologies,self.train_streams,self.val_topologies,self.val_streams)
        for sizes in (self.train_sizes,(self.val_size,)):
            if not sizes or any(n not in ALL_SIZES for n in sizes):
                raise ValueError('invalid network size')
        if not self.train_loads or any(x not in ALL_LOADS for x in self.train_loads) or self.val_load not in ALL_LOADS:
            raise ValueError('invalid offered load')
        if not self.train_modes or any(x not in ALL_MODES for x in self.train_modes) or self.val_mode not in ALL_MODES:
            raise ValueError('invalid traffic mode')
        if self.k not in (2,6) or self.max_working_paths<self.k or not 0<=self.headroom<100:
            raise ValueError('unsupported Stage 4 routing condition')
        if self.gamma!=1.0 or self.lr<=0 or not 0<=self.eval_gate_threshold<=10 or self.bc_coef<0:
            raise ValueError('unsupported optimization settings')
        if self.hidden<8 or self.ppo_epochs<1 or self.batch_size<1 or self.entropy_coef<0:
            raise ValueError('invalid model settings')
        if not CONFIGS[self.config].guidance and self.bc_coef!=0.01:
            # Under A/C the field is retained as the common nominal setting;
            # the *effective* coefficient is forced to zero during training.
            raise ValueError('A/C must retain common nominal bc coefficient .01')

    def dict(self):
        d=asdict(self)
        d['effective_bc_coef']=self.bc_coef if CONFIGS[self.config].guidance else 0.0
        d['effective_gate_threshold']=self.eval_gate_threshold if CONFIGS[self.config].guidance else 0.0
        d['model_selection_metric']='mean per-validation-cohort priority-weighted admission'
        d['model_selection_cell']={'nodes':self.val_size,'load':self.val_load,'mode':self.val_mode,'k':self.k}
        return d


def _schedule(plan:TrainPlan):
    """Full combination coverage before repeat: shuffled size x load x traffic blocks."""
    combos=[(n,load,mode) for n in plan.train_sizes for load in plan.train_loads for mode in plan.train_modes]
    r=random.Random(plan.agent_seed+0x5F04)
    top=list(plan.train_topologies);stream=list(plan.train_streams)
    schedule=[]
    for ep in range(plan.episodes):
        if ep%len(combos)==0:
            block=combos.copy();r.shuffle(block)
        n,load,mode=block[ep%len(combos)]
        if ep%len(top)==0:
            r.shuffle(top)
        if ep%len(stream)==0:
            r.shuffle(stream)
        schedule.append(dict(nodes=n,load=load,mode=mode,
                             topology_seed=top[ep%len(top)],stream_seed=stream[ep%len(stream)]))
    return schedule


class CohortCache:
    """Content-validated disk cache of precomputed candidate inventories.

    Contains only explicitly permitted training / validation seeds. The cache is
    local JSON/gzip, not unsafe untrusted pickle, and is keyed by generator
    source hashes.  It is safe to share among the 12 training processes.
    """
    def __init__(self,root):
        self.root=Path(root)
        self.gen_sources={k:v for k,v in stage3_sources().items()
                          if k.startswith(('phase5f/','phase5f_stage2/'))}
        self.hits=0;self.misses=0

    def get(self,nodes,topo,stream,load,mode,k,max_working_paths,role):
        if role not in ('train','validation'):raise ValueError('Stage 4 never reads held-out')
        _validate_seed_role(role,(topo,),(stream,))
        if nodes not in ALL_SIZES or load not in ALL_LOADS or mode not in ALL_MODES or k not in (2,6):
            raise ValueError('undeclared cohort')
        spec={'n':nodes,'t':topo,'s':stream,'load':load,'mode':mode,'k':k,
              'max_working_paths':max_working_paths,'sources':self.gen_sources}
        tag=_digest(json.dumps(spec,sort_keys=True).encode())
        file=self.root/f'{tag}.json.gz'
        graph=build_network(NetworkSpec(nodes=nodes,seed=topo))
        demands=generate_demands(graph,round(nodes*load),stream,mode=='dynamic')
        network_sha=fingerprint(graph);request_sha=stream_digest(demands)
        if file.exists():
            stored=json.loads(gzip.decompress(file.read_bytes()))
            if stored.get('spec')!=spec or stored.get('network_sha')!=network_sha or stored.get('request_sha')!=request_sha:
                raise RuntimeError('cohort cache provenance mismatch: '+str(file))
            inv=tuple(tuple(ProtectedPair(tuple(p[0]),tuple(p[1]),p[2]) for p in row)
                      for row in stored['inventory'])
            if candidates_digest(inv)!=stored['candidate_sha']:
                raise RuntimeError('corrupted candidate cache: '+str(file))
            self.hits+=1
        else:
            inv=tuple(generate_pairs(graph,d.source,d.destination,k,max_working_paths)
                      for d in demands)
            content={'spec':spec,'network_sha':network_sha,'request_sha':request_sha,
                 'candidate_sha':candidates_digest(inv),'inventory':[
                  [(list(p.working),list(p.backup),p.candidate_id) for p in row]
                  for row in inv]}
            self.root.mkdir(parents=True,exist_ok=True)
            tmp=file.with_name(file.name+'.tmp')
            tmp.write_bytes(gzip.compress(json.dumps(content,separators=(',',':')).encode()))
            os.replace(tmp,file)
            self.misses+=1
        return graph,demands,inv,dict(network_sha256=network_sha,
                                     request_sha256=request_sha,candidates_sha256=candidates_digest(inv))


def _validation(plan,model,cache):
    """Fixed, predeclared, matched *validation* cells; no update or lookahead."""
    cfg=CONFIGS[plan.config]
    model.eval()
    agent=MaskedFeaturePolicy(model,cfg)
    rows=[];cohort_data={}
    for topo in plan.val_topologies:
        for stream in plan.val_streams:
            g,demands,inv,digests=cache.get(plan.val_size,topo,stream,plan.val_load,
                                             plan.val_mode,plan.k,plan.max_working_paths,'validation')
            key=f'{plan.val_size}/{topo}/{stream}/{plan.val_load}/{plan.val_mode}/{plan.k}'
            cohort_data[key]=digests
            for policy in ('greedy','footprint','hybrid'):
                summary,_,decisions=run_episode(g,demands,inv,policy=policy,
                    model=agent if policy=='hybrid' else None,k=plan.k,
                    headroom=plan.headroom,mode=plan.val_mode,stochastic=False,
                    gate_threshold=plan.eval_gate_threshold if cfg.guidance else 0.0,
                    seed=plan.agent_seed,trace=policy=='hybrid')
                rows.append(dict(config=plan.config,agent_seed=plan.agent_seed,
                    nodes=plan.val_size,topology_seed=topo,stream_seed=stream,
                    load=plan.val_load,mode=plan.val_mode,k=plan.k,policy=policy,
                    **{x:summary[x] for x in (
                      'requests','admitted','rejected','ordinary_admission','weighted_admission',
                      'discretionary','overrides','departures','active_end',
                      'avg_feasible_candidates','peak_scenario_utilization','state_digest',
                      'feasibility_violations')},
                    decision_records=len(decisions),
                    network_sha256=digests['network_sha256'],
                    request_sha256=digests['request_sha256'],
                    candidates_sha256=digests['candidates_sha256']))
    return rows,cohort_data


def _val_means(rows):
    counts=Counter(r['policy'] for r in rows)
    if set(counts)!={'greedy','footprint','hybrid'} or len(set(counts.values()))!=1:
        raise AssertionError('unbalanced validation policies')
    means={p:float(np.mean([r['weighted_admission'] for r in rows if r['policy']==p]))
           for p in counts}
    # Do not confuse metric levels: mean seed-level ratio, not pooled numerator.
    return means


def _checkpoint(model,plan,episode,score,cohort_sha):
    return dict(state_dict=model.state_dict(),version=VERSION,
                train_role='DEVELOPMENT_TRAINING_VALIDATION_ONLY',
                config=plan.config,agent_seed=plan.agent_seed,episode=episode,
                mean_weighted_validation=score,
                hidden=plan.hidden,feature_dims={'candidate':17,'global':8},
                max_actions=6,
                plan=plan.dict(),source_sha256=_source_hashes(),
                validation_cohorts=cohort_sha,
                protection_model=PROTECTION,protection_model_approved=False)


def run_training(out,cache_dir,plan:TrainPlan,*,resume=False):
    """Full Stage 4 training; validation-only best checkpoint selection.

    For long Windows runs --resume restores both model/optimizer and all RNG
    states from ``latest_state.pt``; incomplete evidence never passes selection.
    Each unique output directory represents one *planned* factorial run.
    """
    plan.validate()
    out=Path(out);cache=CohortCache(cache_dir)
    settings=json.loads(json.dumps(plan.dict()));schema=_source_hashes()
    schedule=_schedule(plan)
    if not resume:
        if out.exists():raise FileExistsError('refusing to overwrite prior run: '+str(out))
        out.mkdir(parents=True,exist_ok=False)
        _json_write(out/'training_plan.json',dict(settings=settings,source_sha256=schema,
                        schedule_sha256=_digest(json.dumps(schedule,sort_keys=True).encode()),
                        reserved_test_topologies=RESERVED_TEST_TOPO,
                        reserved_test_streams=RESERVED_TEST_STREAM,
                        role='DEVELOPMENT_TRAINING_NO_HELDOUT'))
        torch.manual_seed(plan.agent_seed)
        np.random.seed(plan.agent_seed)
        torch.set_num_threads(1)  # avoids platform-dependent CPU overthreading
        model=CandidateActorCritic(hidden=plan.hidden)
        optimizer=torch.optim.Adam(model.parameters(),lr=plan.lr)
        completed=0;best=-float('inf');best_episode=None;bad_checks=0
        train_rows=[];validation_rows=[];checkpoints=[];validation_digests={}
    else:
        if not out.exists():raise FileNotFoundError('cannot resume missing run '+str(out))
        stored=json.loads((out/'training_plan.json').read_text())
        if (stored['settings']!=settings or stored['source_sha256']!=schema or
            stored['schedule_sha256']!=_digest(json.dumps(schedule,sort_keys=True).encode())):
            raise ValueError('resume plan/source/schedule mismatch')
        if (out/'training_provenance.json').exists():
            raise ValueError('already completed; refusing to resume a frozen run')
        state=torch.load(out/'latest_state.pt',map_location='cpu',weights_only=False)
        torch.set_num_threads(1)
        model=CandidateActorCritic(hidden=plan.hidden)
        model.load_state_dict(state['model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=plan.lr)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        completed=state['completed'];best=state['best'];best_episode=state['best_episode']
        bad_checks=state['bad_checks'];train_rows=state['train_rows']
        validation_rows=state['validation_rows'];checkpoints=state['checkpoints']
        validation_digests=state['validation_digests']
    start=time.perf_counter()
    cfg=CONFIGS[plan.config];policy=MaskedFeaturePolicy(model,cfg)
    remaining=range(completed,plan.episodes)
    if plan.patience and bad_checks>=plan.patience:
        remaining=range(0)  # resumed run must preserve prior early-stop decision
    for ep in remaining:
        meta=schedule[ep]
        g,ds,inv,_=cache.get(meta['nodes'],meta['topology_seed'],meta['stream_seed'],
                  meta['load'],meta['mode'],plan.k,plan.max_working_paths,'train')
        model.train()
        stats,hist,_=run_episode(g,ds,inv,policy='hybrid',model=policy,
                    headroom=plan.headroom,k=plan.k,mode=meta['mode'],
                    stochastic=True,seed=plan.agent_seed+ep)
        masked=mask_training_observations(hist,cfg)
        update=ppo_update(model,optimizer,masked,
                   epochs=plan.ppo_epochs,batch_size=plan.batch_size,
                   entropy_coef=plan.entropy_coef,
                   bc_coef=plan.bc_coef if cfg.guidance else 0.0,
                   seed=plan.agent_seed+ep)
        if not np.isfinite(update['loss']):raise FloatingPointError('non-finite PPO loss')
        train_rows.append(dict(episode=ep+1,config=plan.config,agent_seed=plan.agent_seed,
            topology_seed=meta['topology_seed'],stream_seed=meta['stream_seed'],
            nodes=meta['nodes'],load=meta['load'],mode=meta['mode'],k=plan.k,
            requests=stats['requests'],admitted=stats['admitted'],
            weighted_admission=stats['weighted_admission'],
            ordinary_admission=stats['ordinary_admission'],
            discretionary=stats['discretionary'],overrides=stats['overrides'],
            optimization_actions=update['n'],ppo_loss=update['loss'],
            feasibility_violations=stats['feasibility_violations']))
        if (ep+1)%plan.validate_every==0 or ep+1==plan.episodes:
            valrows,digests=_validation(plan,model,cache)
            validation_digests.update(digests)
            means=_val_means(valrows);score=means['hybrid']
            valrows=[dict(episode=ep+1,**r) for r in valrows]
            validation_rows.extend(valrows)
            improved=score>best+1e-12
            if improved:
                best=score;best_episode=ep+1;bad_checks=0
                _pt_write(out/'best_model.pt',_checkpoint(model,plan,ep+1,score,digests))
            else:
                bad_checks+=1
            checkpoints.append(dict(episode=ep+1,config=plan.config,agent_seed=plan.agent_seed,
                validation_weighted_hybrid=score,
                validation_weighted_greedy=means['greedy'],
                validation_weighted_footprint=means['footprint'],
                hybrid_minus_greedy_pp=(score-means['greedy'])*100,
                hybrid_minus_footprint_pp=(score-means['footprint'])*100,
                hybrid_overrides=sum(r['overrides'] for r in valrows if r['policy']=='hybrid'),
                hybrid_discretionary=sum(r['discretionary'] for r in valrows if r['policy']=='hybrid'),
                best_so_far=best,improved=int(improved)))
            print(f'VALIDATION {plan.config} seed={plan.agent_seed} ep={ep+1} '
                  f'weighted_hybrid={score:.6f} greedy={means["greedy"]:.6f} '
                  f'footprint={means["footprint"]:.6f} overrides={checkpoints[-1]["hybrid_overrides"]}',
                  flush=True)
        # Persist every completed episode; any run may be continued without
        # contaminating original training/validation/final seed partitions.
        latest=dict(completed=ep+1,best=best,best_episode=best_episode,bad_checks=bad_checks,
            model=model.state_dict(),optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),train_rows=train_rows,
            validation_rows=validation_rows,checkpoints=checkpoints,
            validation_digests=validation_digests)
        _pt_write(out/'latest_state.pt',latest)
        if checkpoints and plan.patience and bad_checks>=plan.patience:break
    if not checkpoints or not (out/'best_model.pt').exists():
        raise AssertionError('training produced no selected validation checkpoint')
    _csv_write(out/'training_history.csv',train_rows)
    _csv_write(out/'validation_summary.csv',validation_rows)
    _csv_write(out/'validation_checkpoints.csv',checkpoints)
    stopped_early=len(train_rows)<plan.episodes
    best_sha=_sha(out/'best_model.pt')
    manifest=dict(version=VERSION,role='DEVELOPMENT_TRAINING_VALIDATION_ONLY',
        config=plan.config,agent_seed=plan.agent_seed,settings=settings,
        config_factors=asdict(cfg),completed_episodes=len(train_rows),
        intended_max_episodes=plan.episodes,stopped_early=stopped_early,
        best_episode=best_episode,best_mean_weighted_validation=best,
        checkpoint_sha256=best_sha,train_rows=len(train_rows),
        validation_rows=len(validation_rows),validation_evaluations=len(checkpoints),
        validation_cohort_hashes=validation_digests,
        source_sha256=schema,software=environment(),
        provenance='no heldout used',final_test_seeds_accessed=False,
        heldout_topology_seeds_reserved=RESERVED_TEST_TOPO,
        heldout_stream_seeds_reserved=RESERVED_TEST_STREAM,
        protection_model=PROTECTION,protection_model_approved=False,
        synthetic_not_actual_irish_rail=True,msrp_compliant=False,
        no_future_oracle=True,training_actions=sum(r['optimization_actions'] for r in train_rows),
        cache_hits=cache.hits,cache_misses=cache.misses,
        wall_s_this_invocation=round(time.perf_counter()-start,3),
        selection_note='Best individual checkpoint by mean validation weighted admission; ties prefer earlier episode. Group configuration selection separate.')
    _json_write(out/'training_provenance.json',manifest)
    return manifest
