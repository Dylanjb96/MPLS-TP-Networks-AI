"""On-policy masked categorical PPO with clipped objective and GAE(lambda).

Implementation is a compact research implementation, not a claim of equivalence
to any particular third-party PPO library. Validation/evaluation is greedy and
never updates policy weights.
"""
from __future__ import annotations
from dataclasses import asdict
import random
import numpy as np
import torch
from torch import nn
from phase5.agent import ActionQNetwork
from .common import CheckpointMixin,RLConfig,StateValue,logits_and_prob,seed_all


class MaskedPPO(CheckpointMixin):
    def __init__(self,features:int,actions:int,config:RLConfig):
        if config.algorithm!='ppo':raise ValueError('expected ppo')
        self.features,self.actions,self.config=features,actions,config
        seed_all(config.seed)
        self.rng=random.Random(config.seed)
        self.actor=ActionQNetwork(features,config.hidden)
        self.value=StateValue(actions,features,config.hidden)
        self.opt=torch.optim.Adam(list(self.actor.parameters())+list(self.value.parameters()),lr=config.lr)
        self.gradient_steps=0
        self.env_steps=0
        self.last_loss=None

    @torch.no_grad()
    def act(self,rows:np.ndarray,mask:np.ndarray,*,greedy:bool=False)->int:
        if not mask.any():raise ValueError('empty legal action set')
        x=torch.as_tensor(rows[None],dtype=torch.float32)
        m=torch.as_tensor(mask[None],dtype=torch.bool)
        probs,_=logits_and_prob(self.actor(x),m)
        p=probs[0].numpy()
        return int(p.argmax()) if greedy else int(min(len(p)-1,
                       np.searchsorted(np.cumsum(p),self.rng.random(),side='right')))

    def train_episode(self,env)->dict:
        trajectories=[]
        while not env.done:
            obs=env.observe()
            rows=obs.rows.copy(); mask=obs.mask.copy()
            with torch.no_grad():
                x=torch.as_tensor(rows[None],dtype=torch.float32)
                m=torch.as_tensor(mask[None],dtype=torch.bool)
                probs,lp=logits_and_prob(self.actor(x),m)
                val=float(self.value(x,m).item())
                p=probs[0].numpy()
                action=int(min(len(p)-1,np.searchsorted(np.cumsum(p),
                            self.rng.random(),side='right')))
                old_logp=float(lp[0,action].item())
            r,done,_=env.step(action)
            trajectories.append((rows,mask,action,old_logp,r,val,done))
            self.env_steps+=1
        rewards=np.asarray([x[4] for x in trajectories],np.float32)
        values=np.asarray([x[5] for x in trajectories],np.float32)
        dones=np.asarray([x[6] for x in trajectories],np.float32)
        adv=np.zeros_like(values)
        gae=0.0
        for t in range(len(trajectories)-1,-1,-1):
            next_v=values[t+1] if t+1<len(values) else 0.
            delta=rewards[t]+self.config.gamma*next_v*(1-dones[t])-values[t]
            gae=delta+self.config.gamma*self.config.gae_lambda*(1-dones[t])*gae
            adv[t]=gae
        returns=values+adv
        n=len(values)
        r=torch.as_tensor(np.stack([x[0] for x in trajectories]),dtype=torch.float32)
        m=torch.as_tensor(np.stack([x[1] for x in trajectories]),dtype=torch.bool)
        a=torch.as_tensor([x[2] for x in trajectories],dtype=torch.long)
        old=torch.as_tensor([x[3] for x in trajectories],dtype=torch.float32)
        targets=torch.as_tensor(returns,dtype=torch.float32)
        advantages=torch.as_tensor(adv,dtype=torch.float32)
        if n>1: advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        all_ids=list(range(n))
        for _ in range(self.config.ppo_epochs):
            self.rng.shuffle(all_ids)
            for start in range(0,n,self.config.batch):
                ids=all_ids[start:start+self.config.batch]
                prob,logp=logits_and_prob(self.actor(r[ids]),m[ids])
                selected=logp.gather(1,a[ids,None]).squeeze(1)
                ratio=torch.exp(selected-old[ids])
                unclipped=ratio*advantages[ids]
                clipped=torch.clamp(ratio,1-self.config.ppo_clip,
                                    1+self.config.ppo_clip)*advantages[ids]
                policy_loss=-torch.minimum(unclipped,clipped).mean()
                entropy=-(prob*logp*m[ids]).sum(-1).mean()
                value_loss=0.5*(self.value(r[ids],m[ids])-targets[ids]).square().mean()
                loss=policy_loss+0.5*value_loss-self.config.entropy_coef*entropy
                self.opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters())+
                                         list(self.value.parameters()),5.0)
                self.opt.step();self.gradient_steps+=1
                self.last_loss=float(loss.item())
        return env.summary()

    def _weights(self):return {'actor':self.actor.state_dict(),'value':self.value.state_dict()}
    def _restore(self,w):
        self.actor.load_state_dict(w['actor']);self.value.load_state_dict(w['value'])
