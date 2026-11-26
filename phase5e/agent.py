"""Conservative, on-policy residual PPO with greedy behavioural regularisation.

This is a research implementation, not a claim of PPO library equivalence or
safe policy improvement. The greedy policy is the mandatory action-0 fallback.
PPO optimizes cumulative weighted admission on full demand sequences.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from phase5.agent import ActionQNetwork
from phase5c.common import StateValue,logits_and_prob,seed_all
from .policy import feature_names,gate_action
from . import VERSION


@dataclass(frozen=True)
class HybridConfig:
    seed:int=7
    enhanced:bool=True
    hidden:int=64
    lr:float=.0003
    gamma:float=1.0
    gae_lambda:float=.95
    clip:float=.2
    ppo_epochs:int=4
    batch:int=64
    entropy_coef:float=.01
    greedy_bc_coef:float=.1
    warmstart_steps:int=50
    gate_threshold:float=.15

    def __post_init__(self):
        if (self.seed<0 or self.hidden<4 or self.lr<=0 or not 0<self.gamma<=1 or
            not 0<self.gae_lambda<=1 or not 0<self.clip<1 or self.ppo_epochs<1 or
            self.batch<1 or self.entropy_coef<0 or self.greedy_bc_coef<0 or
            self.warmstart_steps<0 or not 0<=self.gate_threshold<=1):
            raise ValueError('invalid hybrid hyperparameters')


class ResidualPPO:
    def __init__(self, k:int, cfg:HybridConfig):
        if k<2:raise ValueError('residual learning requires max_candidates>=2')
        self.k,self.config=k,cfg
        self.names=feature_names(cfg.enhanced)
        seed_all(cfg.seed)
        self.rng=random.Random(cfg.seed)
        self.actor=ActionQNetwork(len(self.names),cfg.hidden)
        self.value=StateValue(k,len(self.names),cfg.hidden)
        self.opt=torch.optim.Adam([*self.actor.parameters(),*self.value.parameters()],lr=cfg.lr)
        self.gradient_steps=0
        self.last_loss=None
        self.train_steps=0

    @torch.no_grad()
    def logits(self,rows,mask):
        x=torch.as_tensor(rows[None],dtype=torch.float32)
        return self.actor(x)[0].numpy()

    @torch.no_grad()
    def act(self, rows, mask, *, gate:bool=True,threshold:float|None=None):
        if not mask[0] or not bool(mask.any()):raise ValueError('invalid residual mask')
        if int(mask.sum())==1:return 0
        z=self.logits(rows,mask)
        if not gate:return int(np.argmax(np.where(mask,z,-np.inf)))
        result,_=gate_action(z,mask,self.config.gate_threshold if threshold is None else threshold)
        return result

    def warmstart(self,states:list[tuple[np.ndarray,np.ndarray]]):
        """Greedy imitation only at states with >1 feasible candidate.

        This initializes a conservative residual action preference. No
        validation or held-out state may be passed to this method.
        """
        states=[(r,m) for r,m in states if int(m.sum())>1]
        if not states or self.config.warmstart_steps==0:return 0
        r=torch.as_tensor(np.stack([x[0] for x in states]),dtype=torch.float32)
        m=torch.as_tensor(np.stack([x[1] for x in states]),dtype=torch.bool)
        for step in range(self.config.warmstart_steps):
            ids=[self.rng.randrange(len(states)) for _ in range(min(self.config.batch,len(states)))]
            _,lp=logits_and_prob(self.actor(r[ids]),m[ids])
            loss=-lp[:,0].mean()
            self.opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),5.0)
            self.opt.step();self.gradient_steps+=1
        return len(states)

    def train_episode(self,env):
        trajectory=[]
        while not env.done:
            obs=env.observe()
            rows=obs.rows.copy();mask=obs.mask.copy()
            with torch.no_grad():
                x=torch.as_tensor(rows[None],dtype=torch.float32)
                m=torch.as_tensor(mask[None],dtype=torch.bool)
                probs,logs=logits_and_prob(self.actor(x),m)
                p=probs[0].numpy()
                # Sampling exactly from current categorical policy is required
                # for the PPO importance ratio to be valid.
                action=int(min(len(p)-1,np.searchsorted(np.cumsum(p),
                                self.rng.random(),side='right')))
                lp=float(logs[0,action].item());v=float(self.value(x,m).item())
            r,done,_=env.step(action)
            trajectory.append((rows,mask,action,lp,r,v,done,obs.discretionary))
            self.train_steps+=1
        n=len(trajectory)
        vals=np.asarray([x[5] for x in trajectory],dtype=np.float32)
        rew=np.asarray([x[4] for x in trajectory],dtype=np.float32)
        adv=np.zeros(n,dtype=np.float32)
        gae=0.0
        for t in range(n-1,-1,-1):
            next_v=vals[t+1] if t+1<n else 0.0
            d=rew[t]+self.config.gamma*next_v-vals[t]
            gae=d+self.config.gamma*self.config.gae_lambda*gae
            adv[t]=gae
        target=torch.as_tensor(vals+adv,dtype=torch.float32)
        advantages=torch.as_tensor(adv,dtype=torch.float32)
        if n>1:advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        rows=torch.as_tensor(np.stack([x[0] for x in trajectory]),dtype=torch.float32)
        masks=torch.as_tensor(np.stack([x[1] for x in trajectory]),dtype=torch.bool)
        acts=torch.as_tensor([x[2] for x in trajectory],dtype=torch.long)
        old=torch.as_tensor([x[3] for x in trajectory],dtype=torch.float32)
        discretionary=torch.as_tensor([x[7] for x in trajectory],dtype=torch.bool)
        order=list(range(n))
        for _ in range(self.config.ppo_epochs):
            self.rng.shuffle(order)
            for start in range(0,n,self.config.batch):
                ids=order[start:start+self.config.batch]
                prob,lp=logits_and_prob(self.actor(rows[ids]),masks[ids])
                chosen=lp.gather(1,acts[ids,None]).squeeze(1)
                ratio=(chosen-old[ids]).exp()
                a=advantages[ids]
                policy_loss=-torch.minimum(ratio*a,torch.clamp(
                    ratio,1-self.config.clip,1+self.config.clip)*a).mean()
                value_loss=.5*(self.value(rows[ids],masks[ids])-target[ids]).square().mean()
                entropy=-(prob*lp*masks[ids]).sum(-1).mean()
                # Greedy-guidance penalty ONLY where an override exists; do
                # not let forced actions overwhelm the BC loss.
                valid=discretionary[ids]
                imitation=-lp[valid,0].mean() if bool(valid.any()) else lp.sum()*0
                loss=policy_loss+.5*value_loss-self.config.entropy_coef*entropy + \
                     self.config.greedy_bc_coef*imitation
                self.opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(),*self.value.parameters()],5.0)
                self.opt.step();self.gradient_steps+=1
                self.last_loss=float(loss.detach())
        return env.summary()

    def save(self,path:Path,meta:dict):
        torch.save({'family':VERSION,'config':asdict(self.config),'k':self.k,
                    'features':self.names,'actor':self.actor.state_dict(),
                    'value':self.value.state_dict(),'metadata':meta},path)

    @classmethod
    def load(cls,path:Path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('family')!=VERSION:raise ValueError('not a Phase 5E checkpoint')
        agent=cls(data['k'],HybridConfig(**data['config']))
        if agent.names!=data['features']:raise ValueError('incompatible feature definition')
        agent.actor.load_state_dict(data['actor'])
        agent.value.load_state_dict(data['value'])
        agent.actor.eval();agent.value.eval()
        return agent,data['metadata']
