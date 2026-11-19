"""Masked *discrete* Soft Actor-Critic, with twin critics and fixed entropy temperature.

For the finite, Phase3-feasible candidate set. This is not continuous-action SAC.
Uses the same [K+1, F] observations and forced-only reject rule as DDQN.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from phase5.agent import ActionQNetwork, Transition


@dataclass(frozen=True)
class SACConfig:
    hidden: int = 64
    gamma: float = 0.99
    learning_rate: float = 0.0003
    alpha: float = 0.1                 # fixed entropy coefficient, tune by validation
    target_tau: float = 0.01           # Polyak averaging each gradient step
    batch_size: int = 64
    replay_capacity: int = 20000
    learning_starts: int = 128
    train_every: int = 4
    seed: int = 7

    def __post_init__(self):
        if (self.hidden < 4 or not 0 < self.gamma <= 1 or self.learning_rate <= 0 or
            self.alpha < 0 or not 0 < self.target_tau <= 1 or self.batch_size < 1 or
            self.replay_capacity < self.batch_size or self.learning_starts < self.batch_size or
            self.train_every < 1):
            raise ValueError('invalid discrete SAC hyperparameters')


def masked_distribution(logits: torch.Tensor, masks: torch.Tensor):
    """Return legal-only probabilities and log-probabilities; one legal action minimum."""
    if logits.shape != masks.shape or not torch.all(masks.any(dim=-1)):
        raise ValueError('invalid masked categorical distribution')
    legal_logits = logits.masked_fill(~masks, -1e9)
    logp = F.log_softmax(legal_logits, dim=-1)
    probs = F.softmax(legal_logits, dim=-1) * masks.to(logits.dtype)
    return probs, logp


class MaskedDiscreteSAC:
    def __init__(self, feature_count: int, config: SACConfig):
        self.feature_count = feature_count
        self.config = config
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.actor = ActionQNetwork(feature_count, config.hidden)
        self.critic1 = ActionQNetwork(feature_count, config.hidden)
        self.critic2 = ActionQNetwork(feature_count, config.hidden)
        self.target1 = ActionQNetwork(feature_count, config.hidden)
        self.target2 = ActionQNetwork(feature_count, config.hidden)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.eval(); self.target2.eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.q1_opt = torch.optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.q2_opt = torch.optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        self.replay: deque[Transition] = deque(maxlen=config.replay_capacity)
        self.env_steps = 0
        self.gradient_steps = 0
        self.last_loss = None

    @torch.no_grad()
    def act(self, rows: np.ndarray, mask: np.ndarray, *, greedy: bool = False) -> int:
        if not mask.any():
            raise ValueError('no legal action')
        logits = self.actor(torch.from_numpy(rows[None].astype(np.float32)))
        probs, _ = masked_distribution(logits, torch.from_numpy(mask[None].astype(bool)))
        p = probs.squeeze(0).numpy()
        if greedy:
            return int(np.argmax(p))
        # Python RNG for reproducible sampling independent of torch training calls.
        r = self.rng.random()
        return int(min(len(p)-1, np.searchsorted(np.cumsum(p), r, side='right')))

    def remember(self, transition: Transition):
        if not transition.mask[transition.action]:
            raise AssertionError('replay contains infeasible action')
        self.replay.append(transition)
        self.env_steps += 1

    def learn(self):
        if (len(self.replay) < self.config.learning_starts or
            self.env_steps % self.config.train_every != 0):
            return None
        batch = self.rng.sample(self.replay, self.config.batch_size)
        rows = torch.as_tensor(np.stack([t.rows for t in batch]), dtype=torch.float32)
        masks = torch.as_tensor(np.stack([t.mask for t in batch]), dtype=torch.bool)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        nxt = torch.as_tensor(np.stack([t.next_rows for t in batch]), dtype=torch.float32)
        next_masks = torch.as_tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.bool)
        # Terminal transitions have an all-false stored next mask. Create a dummy
        # legal index for softmax; the bootstrap term is then multiplied by zero.
        next_masks = next_masks.clone()
        next_masks[dones, 0] = True
        with torch.no_grad():
            np_, nlog = masked_distribution(self.actor(nxt), next_masks)
            nmin = torch.minimum(self.target1(nxt), self.target2(nxt))
            v_next = (np_ * (nmin - self.config.alpha * nlog)).sum(dim=1)
            target = rewards + self.config.gamma * (~dones).float() * v_next
        q1 = self.critic1(rows).gather(1, actions[:,None]).squeeze(1)
        q2 = self.critic2(rows).gather(1, actions[:,None]).squeeze(1)
        l1 = F.smooth_l1_loss(q1, target)
        l2 = F.smooth_l1_loss(q2, target)
        for loss, optimizer, model in ((l1,self.q1_opt,self.critic1),(l2,self.q2_opt,self.critic2)):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        # Actor loss is an EXACT expectation over the small discrete action set.
        p, logp = masked_distribution(self.actor(rows), masks)
        with torch.no_grad():
            minq = torch.minimum(self.critic1(rows), self.critic2(rows))
        actor_loss = (p * (self.config.alpha*logp - minq)).sum(dim=1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()
        with torch.no_grad():
            tau=self.config.target_tau
            for target_model, model in ((self.target1,self.critic1),(self.target2,self.critic2)):
                for tp,p_ in zip(target_model.parameters(),model.parameters()):
                    tp.mul_(1-tau).add_(p_,alpha=tau)
        self.gradient_steps += 1
        self.last_loss=float(l1.item()+l2.item()+actor_loss.item())
        return self.last_loss

    def save(self, path, metadata:dict):
        torch.save({'model_type':'discrete_sac','feature_count':self.feature_count,
                    'config':asdict(self.config),'actor':self.actor.state_dict(),
                    'critic1':self.critic1.state_dict(),'critic2':self.critic2.state_dict(),
                    'metadata':metadata}, path)

    @classmethod
    def load(cls, path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d.get('model_type')!='discrete_sac':
            raise ValueError('not a Phase5B discrete-SAC checkpoint')
        agent=cls(int(d['feature_count']),SACConfig(**d['config']))
        agent.actor.load_state_dict(d['actor'])
        agent.critic1.load_state_dict(d['critic1'])
        agent.critic2.load_state_dict(d['critic2'])
        agent.target1.load_state_dict(d['critic1'])
        agent.target2.load_state_dict(d['critic2'])
        return agent,d['metadata']
