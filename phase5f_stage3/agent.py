"""Candidate-shared PPO actor + state-value critic; masks infeasible actions."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .features import MAX_ACTIONS,CAND_DIM,GLOBAL_DIM


class CandidateActorCritic(nn.Module):
    def __init__(self, hidden: int=64):
        super().__init__()
        if hidden<8: raise ValueError('hidden width too small')
        self.hidden=hidden
        self.actor=nn.Sequential(nn.Linear(CAND_DIM+GLOBAL_DIM,hidden),nn.Tanh(),
                                 nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.value_net=nn.Sequential(nn.Linear(CAND_DIM+GLOBAL_DIM,hidden),nn.Tanh(),
                                     nn.Linear(hidden,1))

    def forward(self, candidate, global_features, mask):
        if candidate.ndim!=3 or global_features.ndim!=2 or mask.ndim!=2:
            raise ValueError('actor expects batched candidate/global/mask arrays')
        if candidate.shape[1:]!=(MAX_ACTIONS,CAND_DIM) or global_features.shape[1]!=GLOBAL_DIM:
            raise ValueError('incorrect encoded feature dimensions')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('no feasible actions')
        global_repeated=global_features.unsqueeze(1).expand(-1,MAX_ACTIONS,-1)
        x=torch.cat((candidate,global_repeated),dim=-1)
        logits=self.actor(x).squeeze(-1)
        logits=logits.masked_fill(~mask,-1e9)
        pooled=(candidate*mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1,keepdim=True)
        value=self.value_net(torch.cat((pooled,global_features),dim=-1)).squeeze(-1)
        return logits,value

    @torch.no_grad()
    def decide(self, encoded, *, deterministic=True, greedy_idx=0,
               gate_threshold=0.0):
        c,g,mask=encoded
        xt=torch.from_numpy(np.ascontiguousarray(c)).unsqueeze(0)
        gt=torch.from_numpy(np.ascontiguousarray(g)).unsqueeze(0)
        mt=torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        logits,value=self.forward(xt,gt,mt)
        dist=Categorical(logits=logits)
        action=int(logits.argmax(dim=-1).item()) if deterministic else int(dist.sample().item())
        if not mask[action] or not mask[greedy_idx]:
            raise AssertionError('masked action or unavailable greedy reference')
        # Gate is applied only to deterministic evaluation; training is on-policy.
        if deterministic and action!=greedy_idx and (logits[0,action]-logits[0,greedy_idx]).item()<gate_threshold:
            action=greedy_idx
        return dict(action=action,log_prob=float(dist.log_prob(torch.tensor([action])).item()),
                    value=float(value.item()),logits=logits.squeeze(0).cpu().numpy().tolist(),
                    greedy_probability=float(dist.probs[0,greedy_idx].item()))


def ppo_update(model:CandidateActorCritic, optimizer, observations, *,
               epochs=2,batch_size=64,clip=0.2,entropy_coef=0.01,
               bc_coef=0.01,seed=0):
    """Proper on-policy clipped objective for a batch of discretionary actions.

    Each sample must contain old_logp and a complete-episode remaining return.
    Forced decisions are accounted in those returns but not parameterized by RL.
    """
    if not observations:
        return dict(n=0,loss=0.0)
    rng=np.random.default_rng(seed)
    C=torch.tensor(np.stack([r['c'] for r in observations]),dtype=torch.float32)
    G=torch.tensor(np.stack([r['g'] for r in observations]),dtype=torch.float32)
    M=torch.tensor(np.stack([r['mask'] for r in observations]),dtype=torch.bool)
    A=torch.tensor([r['action'] for r in observations],dtype=torch.int64)
    OLD=torch.tensor([r['old_logp'] for r in observations],dtype=torch.float32)
    RET=torch.tensor([r['return'] for r in observations],dtype=torch.float32)
    GD=torch.tensor([r['greedy'] for r in observations],dtype=torch.int64)
    with torch.no_grad():
        _,values=model(C,G,M)
        adv=RET-values
        if adv.numel()>1:
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
    losses=[]
    for epoch in range(epochs):
        for inds in np.array_split(rng.permutation(len(observations)),
                                   max(1,int(np.ceil(len(observations)/batch_size)))):
            if not len(inds):continue
            b=torch.as_tensor(inds,dtype=torch.int64)
            logits,value=model(C[b],G[b],M[b])
            dist=Categorical(logits=logits)
            lp=dist.log_prob(A[b])
            ratio=torch.exp(lp-OLD[b])
            surr=torch.minimum(ratio*adv[b],torch.clamp(ratio,1-clip,1+clip)*adv[b])
            # Critic uses the same terminal-normalized return the actor optimizes.
            loss=-surr.mean()+0.5*(value-RET[b]).pow(2).mean()
            loss-=entropy_coef*dist.entropy().mean()
            if bc_coef:
                loss+=bc_coef*(-dist.log_prob(GD[b]).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()
            losses.append(float(loss.detach().item()))
    return dict(n=len(observations),loss=float(np.mean(losses)),updates=len(losses))
