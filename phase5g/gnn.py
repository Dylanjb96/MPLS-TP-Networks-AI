"""Pure PyTorch physical-link message-passing candidate scorer.

Unlike Stage 3's graph-aggregate features, this model propagates information
along physical links and pools representations of each candidate's working and
backup links. No PyG dependency and no future requests are supplied.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from phase5f_stage3.features import encode,CAND_DIM,GLOBAL_DIM,MAX_ACTIONS

NODE_DIM=8
EDGE_DIM=7


def graph_observation(graph,ledger,demand,plans,feasible,*,k=6,dynamic=True):
    if not feasible:raise ValueError('graph observation requires feasible choices')
    c,global_features,mask=encode(graph,ledger,demand,plans,feasible,k=k,dynamic=dynamic)
    nodes=sorted(graph.nodes())
    mapping={u:i for i,u in enumerate(nodes)}
    links=sorted((min(u,v),max(u,v)) for u,v in graph.edges())
    emapping={e:i for i,e in enumerate(links)}
    n=len(nodes);m=len(links)
    eidx=np.asarray([(mapping[u],mapping[v]) for u,v in links],dtype=np.int64).T
    edge_feat=np.zeros((m,EDGE_DIM),dtype=np.float32)
    node_feat=np.zeros((n,NODE_DIM),dtype=np.float32)
    normal=ledger.loads[0,:]
    worst=np.max(ledger.loads,axis=0)
    pernode=[[] for _ in nodes]
    for j,(u,v) in enumerate(links):
        i=ledger.edge_to_i[(u,v)]
        usable=float(ledger.cap[2*i]);
        nu=float(max(normal[2*i],normal[2*i+1])/usable)
        wu=float(max(worst[2*i],worst[2*i+1])/usable)
        d=graph[u][v]
        edge_feat[j]=[min(usable/10000.0,10),min(d['length_km']/100.0,10),
                      nu,wu,max(0,1-wu),
                      float(d['layer'] in ('backbone','corridor_rail','ladder_rung','backbone_bypass')),
                      float(d['layer'] in ('regional_crosslink','corridor_crosslink'))]
        pernode[mapping[u]].append((nu,wu,usable))
        pernode[mapping[v]].append((nu,wu,usable))
    for u in nodes:
        j=mapping[u];records=pernode[j]
        node_feat[j]=[
            float(graph.nodes[u]['layer']=='backbone'),
            min(graph.degree(u)/10,5),
            sum(x[0] for x in records)/len(records),
            max(x[1] for x in records),
            min(sum(x[2] for x in records)/len(records)/10000,10),
            float(u==demand.source),float(u==demand.destination),
            min(len(records)/10,5)]
    work=np.zeros((MAX_ACTIONS,m),dtype=np.float32)
    backup=np.zeros((MAX_ACTIONS,m),dtype=np.float32)
    for i in feasible:
        for arr,path in ((work,plans[i].working),(backup,plans[i].backup)):
            for u,v in zip(path,path[1:]):
                arr[i,emapping[(min(u,v),max(u,v))]]=1.0
    if not np.isfinite(node_feat).all() or not np.isfinite(edge_feat).all():
        raise AssertionError('nonfinite GNN state')
    return dict(node=node_feat,edge=edge_feat,idx=eidx,work=work,
                backup=backup,candidate=c,global_=global_features,mask=mask)


def tensors(obs):
    return dict(node=torch.as_tensor(obs['node'],dtype=torch.float32),
                edge=torch.as_tensor(obs['edge'],dtype=torch.float32),
                idx=torch.as_tensor(obs['idx'],dtype=torch.long),
                work=torch.as_tensor(obs['work'],dtype=torch.float32),
                backup=torch.as_tensor(obs['backup'],dtype=torch.float32),
                candidate=torch.as_tensor(obs['candidate'],dtype=torch.float32),
                global_=torch.as_tensor(obs['global_'],dtype=torch.float32),
                mask=torch.as_tensor(obs['mask'],dtype=torch.bool))


class GraphCandidateActorCritic(nn.Module):
    def __init__(self,hidden:int=32):
        super().__init__()
        if hidden<8:raise ValueError('hidden must be >= 8')
        self.hidden=hidden
        self.node_in=nn.Linear(NODE_DIM,hidden)
        self.edge_in=nn.Linear(EDGE_DIM,hidden)
        self.msg1=nn.Linear(hidden*2,hidden)
        self.self1=nn.Linear(hidden,hidden)
        self.msg2=nn.Linear(hidden*2,hidden)
        self.self2=nn.Linear(hidden,hidden)
        self.link=nn.Linear(hidden*2,hidden)
        dim=CAND_DIM+GLOBAL_DIM+hidden*2
        self.actor=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),
                                  nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,1))

    def forward(self,obs):
        n=obs['node'];e=obs['edge'];idx=obs['idx']
        if n.ndim!=2 or n.shape[1]!=NODE_DIM or e.ndim!=2 or e.shape[1]!=EDGE_DIM:
            raise ValueError('wrong graph feature dimensions')
        if idx.shape!=(2,len(e)):raise ValueError('wrong physical-link incidence shape')
        mask=obs['mask']
        if mask.shape!=(MAX_ACTIONS,) or not bool(mask.any()):
            raise ValueError('GNN must have at least one feasible action')
        a,b=idx
        h=torch.tanh(self.node_in(n))
        z=torch.tanh(self.edge_in(e))
        deg=torch.zeros((len(n),1),dtype=h.dtype,device=h.device)
        deg.index_add_(0,a,torch.ones((len(e),1),device=h.device))
        deg.index_add_(0,b,torch.ones((len(e),1),device=h.device))
        for msg,self_fc in ((self.msg1,self.self1),(self.msg2,self.self2)):
            in_a=msg(torch.cat((h[b],z),dim=-1))
            in_b=msg(torch.cat((h[a],z),dim=-1))
            agg=torch.zeros_like(h)
            agg.index_add_(0,a,in_a)
            agg.index_add_(0,b,in_b)
            h=torch.tanh(self_fc(h)+agg/deg.clamp_min(1))
        link=torch.tanh(self.link(torch.cat((0.5*(h[a]+h[b]),z),dim=-1)))
        work=obs['work'];backup=obs['backup']
        w_pool=(work@link)/work.sum(dim=-1,keepdim=True).clamp_min(1)
        b_pool=(backup@link)/backup.sum(dim=-1,keepdim=True).clamp_min(1)
        repeats=obs['global_'].unsqueeze(0).expand(MAX_ACTIONS,-1)
        x=torch.cat((obs['candidate'],repeats,w_pool,b_pool),dim=1)
        logits=self.actor(x).squeeze(-1).masked_fill(~mask,-1e9)
        pooled=x[mask].mean(dim=0,keepdim=True)
        value=self.critic(pooled).squeeze()
        return logits,value

    @torch.no_grad()
    def decide(self,obs,*,deterministic=True,greedy_idx=0,gate_threshold=0.0):
        t=tensors(obs)
        logits,value=self.forward(t)
        dist=Categorical(logits=logits)
        action=int(torch.argmax(logits)) if deterministic else int(dist.sample())
        if not bool(t['mask'][action]) or not bool(t['mask'][greedy_idx]):
            raise AssertionError('selected infeasible action')
        if deterministic and action!=greedy_idx and float(logits[action]-logits[greedy_idx])<gate_threshold:
            action=greedy_idx
        return dict(action=action,log_prob=float(dist.log_prob(torch.tensor(action))),
                    value=float(value),greedy_probability=float(dist.probs[greedy_idx]))


def ppo_update(model,optim,samples,*,epochs=2,batch_size=16,seed=0,bc_coef=0.0):
    if not samples:return dict(samples=0,loss=0.0)
    rng=np.random.default_rng(seed)
    t=[tensors(x['observation']) for x in samples]
    returns=torch.tensor([x['return'] for x in samples],dtype=torch.float32)
    old=torch.tensor([x['old_logp'] for x in samples],dtype=torch.float32)
    actions=torch.tensor([x['action'] for x in samples],dtype=torch.int64)
    greedy=torch.tensor([x['greedy'] for x in samples],dtype=torch.int64)
    with torch.no_grad():
        vals=torch.stack([model(o)[1] for o in t])
    advantage=returns-vals
    if len(samples)>1:
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
    lastloss=0.0
    for epoch in range(epochs):
        for ix in np.array_split(rng.permutation(len(samples)),max(1,(len(samples)+batch_size-1)//batch_size)):
            newlp=[];ent=[];value=[];greedy_lp=[]
            for i in ix:
                logits,v=model(t[int(i)])
                dist=Categorical(logits=logits)
                newlp.append(dist.log_prob(actions[int(i)]))
                ent.append(dist.entropy())
                value.append(v)
                greedy_lp.append(-dist.log_prob(greedy[int(i)]))
            logp=torch.stack(newlp);entropy=torch.stack(ent)
            value=torch.stack(value);greedy_loss=torch.stack(greedy_lp)
            ids=torch.as_tensor(ix,dtype=torch.long)
            ratio=torch.exp((logp-old[ids]).clamp(-20,20))
            surrogate=torch.minimum(ratio*advantage[ids],
                                    torch.clamp(ratio,.8,1.2)*advantage[ids])
            loss=-surrogate.mean()+.5*((value-returns[ids])**2).mean()-0.01*entropy.mean()+bc_coef*greedy_loss.mean()
            if not torch.isfinite(loss):raise AssertionError('GNN PPO loss nonfinite')
            optim.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step();lastloss=float(loss.detach())
    return dict(samples=len(samples),loss=lastloss)
