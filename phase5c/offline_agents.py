"""Discrete BCQ, CQL, AWAC (offline-only adaptation), and IQL agents.

These are transparent, self-contained research implementations; AWAC here
performs *offline* advantage-weighted updates only, not its original online
fine-tuning stage. Q and policy networks are candidate-row MLPs with strict
capacity-feasible action masking. All policy updates use fixed logged data.
"""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from phase5.agent import ActionQNetwork
from .common import (CheckpointMixin,RLConfig,StateValue,logits_and_prob,
                     get_batch,seed_all)


class OfflineAgent(CheckpointMixin):
    def __init__(self,features:int,actions:int,config:RLConfig):
        if config.algorithm=='ppo':raise ValueError('use MaskedPPO for PPO')
        self.features,self.actions,self.config=features,actions,config
        seed_all(config.seed)
        self.rng=random.Random(config.seed)
        self.q=ActionQNetwork(features,config.hidden)
        self.q_target=ActionQNetwork(features,config.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.q_opt=torch.optim.Adam(self.q.parameters(),lr=config.lr)
        self.policy=ActionQNetwork(features,config.hidden) if config.algorithm!='cql' else None
        self.policy_opt=(torch.optim.Adam(self.policy.parameters(),lr=config.lr)
                         if self.policy is not None else None)
        self.q2=ActionQNetwork(features,config.hidden) if config.algorithm=='iql' else None
        self.q2_opt=(torch.optim.Adam(self.q2.parameters(),lr=config.lr)
                     if self.q2 is not None else None)
        self.value=StateValue(actions,features,config.hidden) if config.algorithm=='iql' else None
        self.value_opt=(torch.optim.Adam(self.value.parameters(),lr=config.lr)
                        if self.value is not None else None)
        self.gradient_steps=0;self.last_loss=None

    def _supported(self,rows,mask):
        if self.config.algorithm!='bcq':return mask
        with torch.no_grad():
            probs,_=logits_and_prob(self.policy(rows),mask)
            keep=probs >= (probs.max(-1,keepdim=True).values*self.config.behavior_threshold)
            return mask & keep

    @torch.no_grad()
    def act(self,rows:np.ndarray,mask:np.ndarray,*,greedy:bool=True)->int:
        x=torch.as_tensor(rows[None],dtype=torch.float32)
        m=torch.as_tensor(mask[None],dtype=torch.bool)
        if not bool(m.any()):raise ValueError('empty legal action set')
        if self.config.algorithm in ('awac','iql'):
            score=self.policy(x)
        else:
            score=self.q(x)
        allowed=self._supported(x,m)
        return int(score.masked_fill(~allowed,-torch.inf).argmax(-1).item())

    def update(self,data:dict,ids:np.ndarray)->float:
        b=get_batch(data,ids)
        rows,mask,a=b['rows'],b['mask'],b['action']
        reward,done=b['reward'],b['done'].float()
        nxt,nmask=b['next_rows'],b['next_mask']
        assert bool(mask.gather(1,a[:,None]).all())
        if self.config.algorithm=='iql':
            return self._iql(rows,mask,a,reward,done,nxt,nmask)
        if self.config.algorithm=='awac':
            self._policy_actor_update(rows,mask,a,mode='awac')
        if self.config.algorithm=='bcq':
            self._policy_actor_update(rows,mask,a,mode='bcq')
        with torch.no_grad():
            if self.config.algorithm=='awac':
                p,_=logits_and_prob(self.policy(nxt),nmask)
                nextval=(p*self.q_target(nxt)).sum(-1)
            else:
                supported=self._supported(nxt,nmask)
                next_a=self.q(nxt).masked_fill(~supported,-torch.inf).argmax(-1)
                nextval=self.q_target(nxt).gather(1,next_a[:,None]).squeeze(1)
            target=reward+self.config.gamma*(1-done)*nextval
        qv=self.q(rows).gather(1,a[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(qv,target)
        if self.config.algorithm=='cql':
            # CQL(H): logsumexp over legal actions - Q(s, logged action)
            legal_q=self.q(rows).masked_fill(~mask,-1e9)
            loss=loss+self.config.conservative_alpha*(torch.logsumexp(legal_q,-1)-qv).mean()
        self.q_opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(),5.0)
        self.q_opt.step()
        self._polyak()
        self.gradient_steps+=1;self.last_loss=float(loss.item())
        return self.last_loss

    def _policy_actor_update(self,rows,mask,a,mode):
        p,logp=logits_and_prob(self.policy(rows),mask)
        if mode=='awac':
            with torch.no_grad():
                qval=self.q(rows)
                advantage=qval.gather(1,a[:,None]).squeeze(1)-(p*qval).sum(-1)
                weight=(advantage/self.config.awac_lambda).clamp(max=np.log(self.config.max_weight)).exp()
            loss=-(weight*logp.gather(1,a[:,None]).squeeze(1)).mean()
        else:
            loss=-logp.gather(1,a[:,None]).mean()
        self.policy_opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),5.0)
        self.policy_opt.step()

    def _iql(self,rows,mask,a,reward,done,nxt,nmask):
        with torch.no_grad():
            q_logged=torch.minimum(self.q(rows),self.q2(rows)).gather(1,a[:,None]).squeeze(1)
        v=self.value(rows,mask)
        diff=q_logged-v
        weight=torch.where(diff>0,self.config.expectile,1-self.config.expectile)
        vloss=(weight*diff.square()).mean()
        self.value_opt.zero_grad(set_to_none=True);vloss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(),5.0);self.value_opt.step()
        with torch.no_grad():
            target=reward+self.config.gamma*(1-done)*self.value(nxt,nmask)
        qloss=F.smooth_l1_loss(self.q(rows).gather(1,a[:,None]).squeeze(1),target)
        q2loss=F.smooth_l1_loss(self.q2(rows).gather(1,a[:,None]).squeeze(1),target)
        self.q_opt.zero_grad(set_to_none=True);qloss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(),5.0);self.q_opt.step()
        self.q2_opt.zero_grad(set_to_none=True);q2loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(),5.0);self.q2_opt.step()
        with torch.no_grad():
            qdata=torch.minimum(self.q(rows),self.q2(rows)).gather(1,a[:,None]).squeeze(1)
            va=self.value(rows,mask)
            aw=( (qdata-va)/self.config.awac_lambda).clamp(max=np.log(self.config.max_weight)).exp()
        _,lp=logits_and_prob(self.policy(rows),mask)
        actor_loss=-(aw*lp.gather(1,a[:,None]).squeeze(1)).mean()
        self.policy_opt.zero_grad(set_to_none=True);actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),5.0);self.policy_opt.step()
        self.gradient_steps+=1;self.last_loss=float((vloss+qloss+q2loss+actor_loss).item())
        return self.last_loss

    def _polyak(self):
        with torch.no_grad():
            for old,new in zip(self.q_target.parameters(),self.q.parameters()):
                old.lerp_(new,self.config.tau)

    def _weights(self):
        weights={'q':self.q.state_dict(),'q_target':self.q_target.state_dict()}
        if self.policy is not None:weights['policy']=self.policy.state_dict()
        if self.q2 is not None:weights['q2']=self.q2.state_dict()
        if self.value is not None:weights['value']=self.value.state_dict()
        return weights

    def _restore(self,w):
        self.q.load_state_dict(w['q']);self.q_target.load_state_dict(w['q_target'])
        if self.policy is not None:self.policy.load_state_dict(w['policy'])
        if self.q2 is not None:self.q2.load_state_dict(w['q2'])
        if self.value is not None:self.value.load_state_dict(w['value'])
