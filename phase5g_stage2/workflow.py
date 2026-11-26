#Development-only matched GNN vs MLP PPO with resumable, provenance-locked runs.

#Neither Phase 5F Stage 5 nor reserved Phase 5G final seeds are generated or read.
#Both arms share protected candidates, full episodes, schedules, reward definition,
#PPO objective, update-sample cap, optimizer, gradient clipping and validation.
#They differ in whether graph message passing is added to the same 17+8 features.
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
import csv, gzip, hashlib, json, os, platform, random, time
import numpy as np
import networkx as nx
import torch
from torch.distributions import Categorical
from phase5f.protection import ProtectedPair, ScenarioLedger
from phase5f.traffic import stream_digest
from phase5f_stage2.topology import fingerprint
from phase5f_stage2.candidates import candidates_digest
from phase5f_stage3.features import encode
from phase5f_stage3.agent import CandidateActorCritic
from phase5f_stage3.workflow import run_episode
from phase5f_stage5.oracle import verify_trace
from phase5f_stage5.freeze import verify_freeze, load_models
from phase5g.protocol import check_seed_role
from phase5g.workflow import make_cohort, simulate_minimax, simulate_gnn, _decision_record
from phase5g.gnn import GraphCandidateActorCritic, graph_observation, tensors
from . import VERSION

SIZES=(60,150,250,500)
LOADS=(1.6,2.5,4.0)
MODES=('static','dynamic')
ARCHITECTURES=('mlp','gnn')
MODEL_SEEDS=(7,17,27)
TRAIN_TOPO=tuple(range(601,631))
TRAIN_STREAM=tuple(range(75000,75020))
VAL_TOPO=(701,702,703)
VAL_STREAM=(76000,76001)
EVAL_CELL=dict(nodes=500,load=4.0,mode='dynamic',k=6,family='hierarchical')
SOURCE_FILES=(
 'phase5f/protection.py','phase5f/traffic.py','phase5f_stage2/candidates.py',
 'phase5f_stage2/topology.py','phase5f_stage3/agent.py',
 'phase5f_stage3/features.py','phase5f_stage3/workflow.py',
 'phase5g/__init__.py','phase5g/gnn.py','phase5g/workflow.py',
 'phase5g/topology.py','phase5g/protocol.py',
 'phase5g_stage2/__init__.py','phase5g_stage2/workflow.py',
 'run_phase5g_stage2.py'
)
ROOT=Path(__file__).resolve().parent.parent


def software():
    return dict(python=platform.python_version(),os=platform.platform(),
                networkx=nx.__version__,numpy=np.__version__,torch=torch.__version__,
                cpu_threads=torch.get_num_threads())


def sha256_bytes(data):return hashlib.sha256(data).hexdigest()
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def sources():
    return {rel:sha256_file(ROOT/rel) for rel in SOURCE_FILES}

def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    os.replace(temp,path)

def atomic_torch(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(data,tmp)
    os.replace(tmp,path)

def atomic_csv(path,rows,columns):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(columns),extrasaction='raise')
        writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)

HISTORY_COLUMNS=('episode','architecture','agent_seed','nodes','topology_seed',
    'stream_seed','load','mode','family','requests','admitted','weighted_admission',
    'discretionary','overrides','train_interactions','update_samples','loss',
    'episode_seconds')
VALIDATION_COLUMNS=('architecture','agent_seed','episode','nodes','family',
    'topology_seed','stream_seed','load','mode','k','policy','requests',
    'admitted','rejected','weighted_admission','discretionary','overrides',
    'departures','peak_scenario_utilization','decision_records',
    'network_sha256','request_sha256','candidates_sha256','oracle_sha256',
    'feasibility_violations','simulation_policy_seconds')
CHECK_COLUMNS=('architecture','agent_seed','episode','score','greedy_score',
    'footprint_score','minimax_score','frozen_C_score','rl_overrides',
    'rl_discretionary','improved','checkpoint_sha256')
TRACE_COLUMNS=('architecture','agent_seed','episode','topology_seed','stream_seed',
    'request_index','request_id','service','weight','structural_candidates',
    'feasible_candidates','greedy_index','chosen_index','greedy_candidate_id',
    'chosen_candidate_id','overridden','admitted','active_after',
    'cumulative_admitted','state_digest','greedy_probability')

@dataclass(frozen=True)
class Plan:
    architecture:str
    agent_seed:int
    episodes:int=48
    validate_every:int=12
    train_sizes:tuple[int,...]=SIZES
    train_loads:tuple[float,...]=LOADS
    train_modes:tuple[str,...]=MODES
    train_topologies:tuple[int,...]=TRAIN_TOPO
    train_streams:tuple[int,...]=TRAIN_STREAM
    val_topologies:tuple[int,...]=VAL_TOPO
    val_streams:tuple[int,...]=VAL_STREAM
    val_size:int=500
    val_load:float=4.0
    val_mode:str='dynamic'
    family:str='hierarchical'  # Validation on original family only; reserve ladder transfer.
    k:int=6
    headroom:int=10
    max_working_paths:int=40
    gnn_hidden:int=32
    mlp_hidden:int=101  # 15,758 MLP parameters vs 15,778 GNN (0.13% difference)
    learning_rate:float=0.0003
    ppo_epochs:int=2
    batch_size:int=8
    max_update_samples:int=48
    gamma:float=1.0
    clip:float=0.2
    entropy_coef:float=0.01
    critic_coef:float=0.5
    grad_clip:float=0.5
    bc_coef:float=0.0
    eval_gate_threshold:float=0.0

    def validate(self):
        if self.architecture not in ARCHITECTURES or self.agent_seed not in MODEL_SEEDS:
            raise ValueError('unknown architecture or unregistered initialization seed')
        if self.episodes<1 or self.episodes>10000 or self.validate_every<1:
            raise ValueError('invalid episode or validation schedule')
        if not self.train_sizes or len(set(self.train_sizes))!=len(self.train_sizes) or any(n not in SIZES for n in self.train_sizes):
            raise ValueError('invalid training sizes')
        if not self.train_loads or len(set(self.train_loads))!=len(self.train_loads) or any(x not in LOADS for x in self.train_loads):
            raise ValueError('invalid training loads')
        if not self.train_modes or len(set(self.train_modes))!=len(self.train_modes) or any(x not in MODES for x in self.train_modes):
            raise ValueError('invalid training traffic modes')
        if self.family!='hierarchical' or self.val_size not in SIZES or self.val_load not in LOADS or self.val_mode not in MODES:
            raise ValueError('Stage 2 uses hierarchical-only training and selection; ladder held back for transfer')
        if not self.train_topologies or not self.train_streams or not self.val_topologies or not self.val_streams:
            raise ValueError('empty training or validation partition')
        for role,topos,streams in [('train',self.train_topologies,self.train_streams),
                                   ('validation',self.val_topologies,self.val_streams)]:
            if len(set(topos))!=len(topos) or len(set(streams))!=len(streams):
                raise ValueError('duplicate partition seed')
            for t in topos:
                for s in streams:check_seed_role(role,t,s)
        if self.k!=6 or self.headroom!=10 or self.max_working_paths!=40:
            raise ValueError('Phase 5G Stage 2 frozen protection settings changed')
        if self.gnn_hidden!=32 or self.mlp_hidden!=101 or self.gamma!=1.0 or self.bc_coef!=0.0 or self.eval_gate_threshold!=0.0:
            raise ValueError('matched algorithm settings changed')
        if self.learning_rate!=0.0003 or self.ppo_epochs!=2 or self.clip!=0.2 or self.entropy_coef!=0.01:
            raise ValueError('unregistered PPO hyperparameters')
        if self.batch_size!=8 or not 1<=self.max_update_samples<=192 or self.critic_coef!=0.5 or self.grad_clip!=0.5:
            raise ValueError('unregistered update parameters')

    def description(self):
        self.validate()
        data=asdict(self)
        return {k:(list(v) if isinstance(v,tuple) else v) for k,v in data.items()}


def schedule(plan):
    """Matched per-agent-seed stream of size/load/mode blocks and topo/traffic seeds."""
    choices=[(n,load,mode) for n in plan.train_sizes for load in plan.train_loads for mode in plan.train_modes]
    rng=random.Random(plan.agent_seed+6502)
    tp=list(plan.train_topologies);sp=list(plan.train_streams)
    seq=[]
    for ep in range(plan.episodes):
        if ep%len(choices)==0:
            block=choices.copy();rng.shuffle(block)
        n,l,m=block[ep%len(choices)]
        if ep%len(tp)==0:rng.shuffle(tp)
        if ep%len(sp)==0:rng.shuffle(sp)
        seq.append(dict(nodes=n,load=l,mode=m,
                        topology_seed=tp[ep%len(tp)],stream_seed=sp[ep%len(sp)]))
    return seq


class CohortCache:
    #Safe gzip-JSON protected-candidate cache; separate, source-locked cohort content.

    #Same candidate inventory is reused by both neural arms. No unsafe pickle.
    #For a cache hit generator work for candidates is entirely avoided.
    
    def __init__(self,root):
        self.root=Path(root);self.hits=0;self.misses=0
        gen=('phase5f/protection.py','phase5f/traffic.py',
             'phase5f_stage2/candidates.py','phase5f_stage2/topology.py',
             'phase5g/topology.py','phase5g/workflow.py')
        self.src={k:sha256_file(ROOT/k) for k in gen}

    def get(self,*,nodes,topology_seed,stream_seed,load,mode,role,family='hierarchical'):
        check_seed_role(role,topology_seed,stream_seed)
        if family!='hierarchical':raise ValueError('training/validation on hierarchical only')
        spec=dict(nodes=nodes,topology_seed=topology_seed,stream_seed=stream_seed,
                  load=load,mode=mode,role=role,family=family,k=6,headroom=10,
                  max_working_paths=40,source_sha256=self.src)
        key=sha256_bytes(json.dumps(spec,sort_keys=True).encode())
        fn=self.root/(key+'.json.gz')
        # make_cohort unavoidably constructs requests and candidates. Instead
        # reconstruct graph/requests independently, then read candidate cache.
        from phase5g.topology import build_network
        from phase5f.traffic import generate_demands
        graph=build_network(nodes,topology_seed,family)
        demands=generate_demands(graph,round(nodes*load),stream_seed,mode=='dynamic')
        netsha=fingerprint(graph);reqsha=stream_digest(demands)
        if fn.exists():
            doc=json.loads(gzip.decompress(fn.read_bytes()))
            if doc.get('spec')!=spec or doc.get('network_sha')!=netsha or doc.get('requests_sha')!=reqsha:
                raise RuntimeError('candidate-cache cohort provenance mismatch')
            inv=tuple(tuple(ProtectedPair(tuple(p[0]),tuple(p[1]),p[2]) for p in row)
                      for row in doc['inventory'])
            if candidates_digest(inv)!=doc['candidates_sha']:
                raise RuntimeError('candidate-cache digest mismatch')
            self.hits+=1
        else:
            from phase5f_stage2.candidates import generate_pairs
            inv=tuple(generate_pairs(graph,d.source,d.destination,6,40) for d in demands)
            digest=candidates_digest(inv)
            doc=dict(spec=spec,network_sha=netsha,requests_sha=reqsha,candidates_sha=digest,
                     inventory=[[(list(p.working),list(p.backup),p.candidate_id) for p in plans]
                                for plans in inv])
            self.root.mkdir(parents=True,exist_ok=True)
            tmp=fn.with_name(fn.name+'.tmp')
            tmp.write_bytes(gzip.compress(json.dumps(doc,separators=(',',':')).encode()))
            os.replace(tmp,fn);self.misses+=1
        return graph,demands,inv,dict(network_sha256=netsha,request_sha256=reqsha,
                                     candidates_sha256=candidates_digest(inv))


def simulate_mlp(graph,demands,inventory,model,*,mode,stochastic=False,trace=False):
    if mode not in MODES:raise ValueError('invalid explicit traffic mode')
    led=ScenarioLedger(graph,10)
    records=[];samples=[];weights=np.zeros(len(demands),dtype=np.float64)
    admitted=0;disc=0;over=0;departures=0
    for t,(d,plans) in enumerate(zip(demands,inventory)):
        departures+=led.release_expired(d.arrival)
        feasible=[i for i,p in enumerate(plans) if led.feasible(d,p)]
        disc+=len(feasible)>1
        chosen=feasible[0] if feasible else None
        if len(feasible)>1:
            c,g,mask=encode(graph,led,d,plans,feasible,k=6,dynamic=mode=='dynamic')
            decision=model.decide((c,g,mask),deterministic=not stochastic,greedy_idx=feasible[0])
            chosen=decision['action']
            if stochastic:
                samples.append(dict(t=t,observation=(c,g,mask),action=chosen,
                                    greedy=feasible[0],old_logp=decision['log_prob']))
        if chosen is not None:
            if chosen not in feasible or not led.admit(d,plans[chosen]):
                raise AssertionError('masked MLP acted infeasibly')
            admitted+=1;weights[t]=d.priority_weight
            over+=int(chosen!=feasible[0])
        if trace:records.append(_decision_record(d,t,plans,feasible,chosen,led,admitted))
    if not led.verify_full_scenario_recompute():raise AssertionError('MLP ledger recomputation failed')
    total=sum(d.priority_weight for d in demands)
    if stochastic:
        suffix=np.cumsum(weights[::-1])[::-1]/total
        for s in samples:s['return']=float(suffix[s['t']])
    summary=dict(requests=len(demands),admitted=admitted,rejected=len(demands)-admitted,
        weighted_admission=float(weights.sum()/total),ordinary_admission=admitted/len(demands),
        discretionary=disc,overrides=over,departures=departures,active_end=len(led.active),
        peak_scenario_utilization=led.peak_utilization(),feasibility_violations=0)
    return summary,records,samples


def forward_one(model,observation,architecture):
    if architecture=='gnn':return model(tensors(observation))
    c,g,mask=observation
    ct=torch.as_tensor(c,dtype=torch.float32).unsqueeze(0)
    gt=torch.as_tensor(g,dtype=torch.float32).unsqueeze(0)
    mt=torch.as_tensor(mask,dtype=torch.bool).unsqueeze(0)
    logits,value=model(ct,gt,mt)
    return logits.squeeze(0),value.squeeze(0)


def matched_ppo_update(model,optimizer,samples,*,architecture,seed,epochs=2,batch_size=8):
    #Identical clipped PPO, entropy, value loss and grad clip for both arms.

    #Uses unbiased uniform sub-sampling from the *complete* current episode,
    #with full-trajectory undiscounted reward-to-go already recorded.
    
    if not samples:return dict(samples=0,loss=0.0,updates=0)
    ob=[x['observation'] for x in samples]
    actions=torch.tensor([x['action'] for x in samples],dtype=torch.long)
    old=torch.tensor([x['old_logp'] for x in samples],dtype=torch.float32)
    returns=torch.tensor([x['return'] for x in samples],dtype=torch.float32)
    with torch.no_grad():
        values=torch.stack([forward_one(model,x,architecture)[1] for x in ob])
        advantage=returns-values
        if len(samples)>1:
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
    rng=np.random.default_rng(seed)
    loss_values=[]
    for ep in range(epochs):
        for batch in np.array_split(rng.permutation(len(samples)),max(1,(len(samples)+batch_size-1)//batch_size)):
            logprobs=[];entropies=[];predicted=[]
            for j in batch:
                i=int(j)
                logits,v=forward_one(model,ob[i],architecture)
                dist=Categorical(logits=logits)
                logprobs.append(dist.log_prob(actions[i]))
                entropies.append(dist.entropy());predicted.append(v)
            ids=torch.as_tensor(batch,dtype=torch.long)
            lp=torch.stack(logprobs)
            ratio=torch.exp((lp-old[ids]).clamp(-20,20))
            surrogate=torch.minimum(ratio*advantage[ids],
                                    torch.clamp(ratio,0.8,1.2)*advantage[ids])
            values=torch.stack(predicted)
            loss=-surrogate.mean()+0.5*(values-returns[ids]).square().mean()
            loss=loss-0.01*torch.stack(entropies).mean()
            if not torch.isfinite(loss):raise AssertionError('nonfinite PPO update')
            optimizer.zero_grad(set_to_none=True)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step();loss_values.append(float(loss.detach()))
    return dict(samples=len(samples),loss=float(np.mean(loss_values)),updates=len(loss_values))


def reduce_samples(samples,cap,seed):
    if len(samples)<=cap:return samples
    idx=np.sort(np.random.default_rng(seed).choice(len(samples),size=cap,replace=False))
    return [samples[int(i)] for i in idx]


def build_model(architecture,seed):
    if architecture not in ARCHITECTURES:raise ValueError('unknown architecture')
    torch.manual_seed(seed)
    model=(GraphCandidateActorCritic(hidden=32) if architecture=='gnn'
           else CandidateActorCritic(hidden=101))
    return model


def param_count(model):return sum(x.numel() for x in model.parameters())


def _simulate(architecture,g,d,inv,model,*,mode,stochastic=False,trace=False):
    if architecture=='gnn':
        return simulate_gnn(g,d,inv,model,k=6,stochastic=stochastic,mode=mode,trace=trace)
    return simulate_mlp(g,d,inv,model,mode=mode,stochastic=stochastic,trace=trace)


def _baseline(g,d,inv,policy,mode,seed,models):
    if policy=='scenario_minimax':return simulate_minimax(g,d,inv,k=6,trace=True)
    name='hybrid' if policy=='C_frozen' else policy
    s,_,records=run_episode(g,d,inv,policy=name,
        model=models.get('C_enhanced') if policy=='C_frozen' else None,
        headroom=10,k=6,mode=mode,stochastic=False,trace=True,seed=seed)
    return s,records


def _summary_row(plan,epoch,cohort,policy,summary,trace,digests,oracle,elapsed):
    return dict(architecture=plan.architecture,agent_seed=plan.agent_seed,
        episode=epoch,nodes=plan.val_size,family='hierarchical',
        topology_seed=cohort[0],stream_seed=cohort[1],load=plan.val_load,
        mode=plan.val_mode,k=6,policy=policy,
        requests=int(summary['requests']),admitted=int(summary['admitted']),
        rejected=int(summary['rejected']),weighted_admission=float(summary['weighted_admission']),
        discretionary=int(summary['discretionary']),overrides=int(summary['overrides']),
        departures=int(summary['departures']),peak_scenario_utilization=float(summary['peak_scenario_utilization']),
        decision_records=len(trace),network_sha256=digests['network_sha256'],
        request_sha256=digests['request_sha256'],candidates_sha256=digests['candidates_sha256'],
        oracle_sha256=oracle['oracle_state_sha256'],feasibility_violations=0,
        simulation_policy_seconds=round(float(elapsed),6))


def evaluate(plan,model,cache,epoch,*,frozen_models=None):
    """Never consults ladder results or final seeds when selecting checkpoints."""
    baseline_models=frozen_models or load_models()
    rows=[];best_trace=[]
    for topo in plan.val_topologies:
        for stream in plan.val_streams:
            g,d,inv,h=cache.get(nodes=plan.val_size,topology_seed=topo,
                stream_seed=stream,load=plan.val_load,mode=plan.val_mode,role='validation')
            for policy in ('greedy','footprint','scenario_minimax','C_frozen',plan.architecture+'_PPO'):
                sim_start=time.perf_counter()
                if policy.endswith('_PPO'):
                    summary,trace,_=_simulate(plan.architecture,g,d,inv,model,
                                              mode=plan.val_mode,trace=True)
                else:
                    summary,trace=_baseline(g,d,inv,policy,plan.val_mode,stream,baseline_models)
                simulation_elapsed=time.perf_counter()-sim_start
                verdict=verify_trace(g,d,inv,trace,6,headroom=10)
                if (verdict['accepted']!=summary['admitted'] or
                    verdict['overrides']!=summary['overrides'] or
                    verdict['departures']!=summary['departures']):
                    raise AssertionError('independent single-fault oracle mismatch')
                rows.append(_summary_row(plan,epoch,(topo,stream),policy,summary,trace,h,verdict,
                                         simulation_elapsed))
                if policy.endswith('_PPO'):
                    best_trace.extend(dict(architecture=plan.architecture,agent_seed=plan.agent_seed,
                                           episode=epoch,topology_seed=topo,stream_seed=stream,**r)
                                      for r in trace)
    counts=Counter(r['policy'] for r in rows)
    if len(set(counts.values()))!=1 or len(rows)!=len(plan.val_topologies)*len(plan.val_streams)*5:
        raise AssertionError('unbalanced paired validation results')
    means={p:float(np.mean([r['weighted_admission'] for r in rows if r['policy']==p]))
           for p in counts}
    return rows,means,best_trace


def _write_trace(path,rows):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=TRACE_COLUMNS)
        w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def _read_state(root,plan,src):
    saved=torch.load(root/'latest_state.pt',map_location='cpu',weights_only=True)
    if saved.get('plan')!=plan.description() or saved.get('sources')!=src or saved.get('version')!=VERSION:
        raise RuntimeError('resumption prohibited: plan or installed source changed')
    if len(saved['history'])!=saved['episode']:
        raise RuntimeError('invalid committed episode count')
    return saved


def _reconcile_csv(root,state):
    atomic_csv(root/'training_history.csv',state['history'],HISTORY_COLUMNS)
    atomic_csv(root/'validation_summary.csv',state['validation_rows'],VALIDATION_COLUMNS)
    atomic_csv(root/'validation_checkpoints.csv',state['checks'],CHECK_COLUMNS)


def run_training(out,plan,*,cache_root='outputs/phase5g/stage2_cohort_cache',resume=False):
    """Persistent single-run training with a strict episode-level checkpoint journal."""
    plan.validate()
    if plan.episodes>=len(SIZES)*len(LOADS)*len(MODES) and (set(plan.train_sizes)!=set(SIZES) or
         set(plan.train_loads)!=set(LOADS) or set(plan.train_modes)!=set(MODES)):
        # Full runs must cover all three factors; smaller smoke checks are allowed.
        raise ValueError('full training must cover all predeclared size/load/traffic cells')
    torch.set_num_threads(1)
    root=Path(out);src=sources()
    freeze=verify_freeze()
    if not root.exists():
        root.mkdir(parents=True)
        atomic_json(root/'training_plan.json',dict(version=VERSION,plan=plan.description(),
            sources=src,software=software(),
            selected_frozen_C_sha256=freeze['selected']['checkpoint_sha256'],
            role='TRAIN_AND_VALIDATION_ONLY_NO_FINAL_DATA'))
        model=build_model(plan.architecture,plan.agent_seed)
        opt=torch.optim.Adam(model.parameters(),lr=plan.learning_rate)
        state=dict(version=VERSION,episode=0,plan=plan.description(),sources=src,
            model_state=model.state_dict(),optimizer_state=opt.state_dict(),
            torch_rng=torch.get_rng_state(),history=[],validation_rows=[],checks=[],
            best_score=float('-inf'),best_episode=0,best_path='',best_hash='',
            best_trace_path='',best_trace_sha256='',
            total_interactions=0,total_update_samples=0)
        atomic_torch(root/'latest_state.pt',state)
    else:
        if not resume:raise FileExistsError('Refusing to overwrite Stage 2 training: '+str(root))
        if not (root/'training_plan.json').is_file() or not (root/'latest_state.pt').is_file():
            raise RuntimeError('missing plan/state; cannot safely resume')
        p=json.loads((root/'training_plan.json').read_text(encoding='utf8'))
        if (p.get('plan')!=plan.description() or p.get('sources')!=src or
            p.get('software')!=software() or
            p.get('selected_frozen_C_sha256')!=freeze['selected']['checkpoint_sha256']):
            raise RuntimeError('run provenance changed; resumption forbidden')
        state=_read_state(root,plan,src)
        model=build_model(plan.architecture,plan.agent_seed)
        model.load_state_dict(state['model_state'],strict=True)
        opt=torch.optim.Adam(model.parameters(),lr=plan.learning_rate)
        opt.load_state_dict(state['optimizer_state'])
        torch.set_rng_state(state['torch_rng'])
        if state['best_episode']:
            ck=root/state['best_path']
            if not ck.is_file() or sha256_file(ck)!=state['best_hash']:
                raise RuntimeError('best checkpoint content missing or altered')
            trace_path=root/state['best_trace_path']
            if not trace_path.is_file() or sha256_file(trace_path)!=state['best_trace_sha256']:
                raise RuntimeError('best validation decision trace missing or altered')
    _reconcile_csv(root,state)
    if state['episode']>=plan.episodes:
        # An interruption after the final journal commit but before writing
        # training_provenance.json must not strand an otherwise finished run.
        if not (root/'training_provenance.json').is_file():
            atomic_json(root/'training_provenance.json',dict(
                version=VERSION,role='TRAINING_AND_VALIDATION_ONLY',
                architecture=plan.architecture,agent_seed=plan.agent_seed,
                episodes=state['episode'],plan=plan.description(),
                source_sha256=src,software=software(),
                frozen_C_sha256=freeze['selected']['checkpoint_sha256'],
                param_count=param_count(model),best_episode=state['best_episode'],
                best_score=state['best_score'],best_checkpoint=state['best_path'],
                best_sha256=state['best_hash'],best_trace=state['best_trace_path'],
                best_trace_sha256=state['best_trace_sha256'],
                total_interactions=state['total_interactions'],
                total_update_samples=state['total_update_samples'],
                cohort_cache_hits=0,cohort_cache_misses=0,
                wall_seconds=0.0,
                recovered_provenance_after_completed_journal=True,
                final_seeds_accessed=False,msrp_compliant=False,
                real_railway_topology=False))
        return dict(status='ALREADY_COMPLETE',episode=state['episode'],out=str(root),
                    best_episode=state['best_episode'],best_score=state['best_score'])
    cache=CohortCache(cache_root)
    schedule_data=schedule(plan)
    frozen=load_models()
    start=time.perf_counter()
    for epi in range(state['episode'],plan.episodes):
        episode_start=time.perf_counter()
        item=schedule_data[epi]
        g,d,inv,_=cache.get(**item,role='train')
        model.eval()
        summary,_,samples=_simulate(plan.architecture,g,d,inv,model,
                                    mode=item['mode'],stochastic=True,trace=False)
        update_samples=reduce_samples(samples,plan.max_update_samples,
                                      seed=plan.agent_seed*100000+epi)
        model.train()
        update=matched_ppo_update(model,opt,update_samples,architecture=plan.architecture,
                  seed=plan.agent_seed+epi,epochs=plan.ppo_epochs,batch_size=plan.batch_size)
        state['total_interactions']+=summary['requests']
        state['total_update_samples']+=update['samples']
        state['history'].append(dict(episode=epi+1,architecture=plan.architecture,
           agent_seed=plan.agent_seed,**item,family='hierarchical',requests=summary['requests'],
           admitted=int(summary['admitted']),weighted_admission=float(summary['weighted_admission']),
           discretionary=int(summary['discretionary']),overrides=int(summary['overrides']),
           train_interactions=int(summary['requests']),update_samples=update['samples'],loss=update['loss'],
           episode_seconds=round(time.perf_counter()-episode_start,6)))
        if (epi+1)%plan.validate_every==0 or (epi+1)==plan.episodes:
            model.eval()
            val_rows,means,trace=evaluate(plan,model,cache,epi+1,frozen_models=frozen)
            score=means[plan.architecture+'_PPO']
            better=score>state['best_score']+1e-12
            ck_hash=''
            if better:
                checkpoint_rel=f'checkpoints/best_episode_{epi+1:04d}.pt'
                checkpoint=root/checkpoint_rel
                data=dict(role='PHASE5G_STAGE2_VALIDATION_SELECTED_NOT_HELDOUT',
                    version=VERSION,architecture=plan.architecture,agent_seed=plan.agent_seed,
                    episode=epi+1,validation_mean=score,model_state=model.state_dict(),
                    plan=plan.description(),sources=src,
                    model_parameters=param_count(model),
                    frozen_C_sha256=freeze['selected']['checkpoint_sha256'])
                atomic_torch(checkpoint,data)
                ck_hash=sha256_file(checkpoint)
                trace_rel=f'checkpoints/best_decisions_episode_{epi+1:04d}.csv.gz'
                _write_trace(root/trace_rel,trace)
                state.update(best_score=score,best_episode=epi+1,best_path=checkpoint_rel,
                             best_hash=ck_hash,best_trace_path=trace_rel,
                             best_trace_sha256=sha256_file(root/trace_rel))
            state['validation_rows'].extend(val_rows)
            state['checks'].append(dict(architecture=plan.architecture,agent_seed=plan.agent_seed,
                episode=epi+1,score=score,greedy_score=means['greedy'],
                footprint_score=means['footprint'],minimax_score=means['scenario_minimax'],
                frozen_C_score=means['C_frozen'],
                rl_overrides=sum(r['overrides'] for r in val_rows if r['policy']==plan.architecture+'_PPO'),
                rl_discretionary=sum(r['discretionary'] for r in val_rows if r['policy']==plan.architecture+'_PPO'),
                improved=int(better),checkpoint_sha256=ck_hash))
        state.update(episode=epi+1,model_state=model.state_dict(),
                     optimizer_state=opt.state_dict(),torch_rng=torch.get_rng_state())
        atomic_torch(root/'latest_state.pt',state)
        _reconcile_csv(root,state)
        print(f'PHASE5G_STAGE2_EPISODE_OK architecture={plan.architecture} seed={plan.agent_seed} '
              f'episode={epi+1}/{plan.episodes} size={item["nodes"]} load={item["load"]} '
              f'mode={item["mode"]} weighted={summary["weighted_admission"]:.6f} '
              f'update_samples={update["samples"]}',flush=True)
    atomic_json(root/'training_provenance.json',dict(version=VERSION,role='TRAINING_AND_VALIDATION_ONLY',
        architecture=plan.architecture,agent_seed=plan.agent_seed,episodes=state['episode'],
        plan=plan.description(),source_sha256=src,software=software(),
        frozen_C_sha256=freeze['selected']['checkpoint_sha256'],
        param_count=param_count(model),best_episode=state['best_episode'],
        best_score=state['best_score'],best_checkpoint=state['best_path'],
        best_sha256=state['best_hash'],best_trace=state['best_trace_path'],
        best_trace_sha256=state['best_trace_sha256'],
        total_interactions=state['total_interactions'],
        total_update_samples=state['total_update_samples'],
        cohort_cache_hits=cache.hits,cohort_cache_misses=cache.misses,
        wall_seconds=round(time.perf_counter()-start,2),
        final_seeds_accessed=False,msrp_compliant=False,real_railway_topology=False))
    return dict(status='TRAIN_OK',out=str(root),episodes=state['episode'],
                best_score=state['best_score'],best_episode=state['best_episode'],
                interactions=state['total_interactions'])


def select_models(runs_root,out):
    #Select best architecture by three-seed mean; select best seed within each.

    #The selector never reads model-evaluation data beyond validation scores.
    #It requires a complete six-run factorial with identical training conditions.
    
    root=Path(runs_root);out=Path(out)
    if out.exists():raise FileExistsError('Refusing to overwrite model selection: '+str(out))
    rows=[];shared=None
    for arch in ARCHITECTURES:
        for seed in MODEL_SEEDS:
            folder=root/f'{arch}_seed{seed}'
            provfile=folder/'training_provenance.json'
            if not provfile.is_file():raise ValueError('missing completed run '+str(folder))
            p=json.loads(provfile.read_text(encoding='utf8'))
            plan=p['plan']
            common={k:v for k,v in plan.items() if k not in ('architecture','agent_seed')}
            if shared is None:shared=common
            elif shared!=common:raise RuntimeError('mismatched 6-run training protocol')
            if (p.get('architecture')!=arch or p.get('agent_seed')!=seed or
                p.get('episodes')!=plan['episodes'] or p.get('source_sha256')!=sources() or
                p.get('frozen_C_sha256')!=verify_freeze()['selected']['checkpoint_sha256']):
                raise ValueError('invalid run metadata: '+str(folder))
            ck=folder/p['best_checkpoint']
            if not ck.is_file() or sha256_file(ck)!=p['best_sha256']:
                raise RuntimeError('selected checkpoint mismatch: '+str(folder))
            chks=list(csv.DictReader((folder/'validation_checkpoints.csv').open(encoding='utf8',newline='')))
            improved=[c for c in chks if int(c['improved'])==1]
            if not improved or max(float(c['score']) for c in chks)<p['best_score']-1e-10:
                raise RuntimeError('bad validation checkpoint selection')
            checkpoint=torch.load(ck,map_location='cpu',weights_only=True)
            if (checkpoint.get('architecture')!=arch or checkpoint.get('agent_seed')!=seed or
                checkpoint.get('episode')!=p['best_episode'] or
                abs(checkpoint.get('validation_mean',-1)-p['best_score'])>1e-10 or
                checkpoint.get('sources')!=sources() or checkpoint.get('role')!=
                'PHASE5G_STAGE2_VALIDATION_SELECTED_NOT_HELDOUT'):
                raise RuntimeError('invalid neural checkpoint identity: '+str(ck))
            valrows=list(csv.DictReader((folder/'validation_summary.csv').open(encoding='utf8',newline='')))
            bestrows=[r for r in valrows if int(r['episode'])==p['best_episode']]
            expected=len(plan['val_topologies'])*len(plan['val_streams'])
            grouped=Counter(r['policy'] for r in bestrows)
            if (len(bestrows)!=5*expected or grouped!={x:expected for x in
                ('greedy','footprint','scenario_minimax','C_frozen',arch+'_PPO')}):
                raise RuntimeError('validation pairing incomplete at selected checkpoint')
            rederived=float(np.mean([float(r['weighted_admission']) for r in bestrows
                                     if r['policy']==arch+'_PPO']))
            if abs(rederived-p['best_score'])>1e-10:
                raise RuntimeError('recomputed validation score differs from selected checkpoint')
            for topo in plan['val_topologies']:
                for stream in plan['val_streams']:
                    group=[r for r in bestrows if int(r['topology_seed'])==topo and
                           int(r['stream_seed'])==stream]
                    if (len(group)!=5 or len({r['network_sha256'] for r in group})!=1 or
                        len({r['request_sha256'] for r in group})!=1 or
                        len({r['candidates_sha256'] for r in group})!=1 or
                        any(int(r['feasibility_violations']) for r in group)):
                        raise RuntimeError('unmatched validation scenario or infeasible selection')
            rows.append(dict(architecture=arch,agent_seed=seed,best_episode=p['best_episode'],
               best_weighted=p['best_score'],checkpoint=str(ck),checkpoint_sha256=p['best_sha256'],
               training_interactions=p['total_interactions'],update_samples=p['total_update_samples']))
    results=[]
    for arch in ARCHITECTURES:
        subset=[x for x in rows if x['architecture']==arch]
        results.append(dict(architecture=arch,mean_best_weighted=float(np.mean([r['best_weighted'] for r in subset])),
                            sd_best_weighted=float(np.std([r['best_weighted'] for r in subset],ddof=1)),
                            runs=3))
    # Tie preference goes to simpler non-GNN architecture, a priori.
    winning=max(results,key=lambda r:(r['mean_best_weighted'],-ARCHITECTURES.index(r['architecture'])))
    winners={arch:sorted((r for r in rows if r['architecture']==arch),
                  key=lambda r:(-r['best_weighted'],r['agent_seed']))[0]
             for arch in ARCHITECTURES}
    out.mkdir(parents=True)
    atomic_csv(out/'selection_trials.csv',rows,rows[0].keys())
    atomic_csv(out/'selection_architectures.csv',results,results[0].keys())
    atomic_json(out/'selection_manifest.json',dict(version=VERSION,role='VALIDATION_ONLY_MODEL_SELECTION',
        sources=sources(),trained_architectures=list(ARCHITECTURES),agent_seeds=list(MODEL_SEEDS),
        selection_rule='highest 3-seed mean of validation-best scores, tie prefer mlp; per model seed highest validation score then lowest seed',
        winning_architecture=winning['architecture'],
        selected_models=winners,selected_overall=winners[winning['architecture']],
        no_final_data_access=True,synthetic_not_real_railway=True,
        protection_model_approved=False))
    return dict(status='PHASE5G_STAGE2_SELECTION_OK',winner=winning['architecture'],
                mean=winning['mean_best_weighted'],out=str(out))


def preflight():
    p=Plan('gnn',7,episodes=2,validate_every=1,train_sizes=(60,),
           train_loads=(1.6,),train_modes=('dynamic',),
           train_topologies=(601,),train_streams=(75000,),
           val_topologies=(701,),val_streams=(76000,),val_size=60,val_load=1.6)
    p.validate()
    manifest=verify_freeze()
    for arch in ARCHITECTURES:
        m=build_model(arch,7)
        g,d,inv=make_cohort(nodes=60,topology_seed=501,stream_seed=74000,
            family='hierarchical',load=1.6,mode='dynamic',role='dev')
        led=ScenarioLedger(g,10)
        for req,plans in zip(d,inv):
            feasible=[i for i,z in enumerate(plans) if led.feasible(req,z)]
            if len(feasible)>1:
                o=(graph_observation(g,led,req,plans,feasible,k=6,dynamic=True)
                   if arch=='gnn' else encode(g,led,req,plans,feasible,k=6,dynamic=True))
                logits,value=forward_one(m,o,arch)
                assert logits.shape==(6,) and bool(torch.isfinite(value))
                assert not bool(torch.isfinite(torch.tensor(float('inf'))))
                print(f'PHASE5G_STAGE2_PREFLIGHT_ARM_OK arch={arch} params={param_count(m)} feasible={len(feasible)}')
                break
        else:raise AssertionError('no feasible pairs found')
    print('PHASE5G_STAGE2_PREFLIGHT_OK frozen_C_sha256='+manifest['selected']['checkpoint_sha256'])


def status(root):
    p=Path(root)
    if not (p/'training_plan.json').exists():raise FileNotFoundError('training plan not found')
    data=json.loads((p/'training_plan.json').read_text(encoding='utf8'))
    plan=Plan(**data['plan'])
    state=_read_state(p,plan,sources())
    return dict(episode=state['episode'],planned=plan.episodes,best_episode=state['best_episode'],
                best_score=state['best_score'],status='complete' if state['episode']==plan.episodes else 'incomplete',
                total_interactions=state['total_interactions'])
