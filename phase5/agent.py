"""Replay-based masked Double DQN; PyTorch required only for Phase 5."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ActionQNetwork(nn.Module):
    """Shared MLP scores each candidate independently, including forced reject."""
    def __init__(self, feature_count: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_count, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        # [..., action_count, feature_count] -> [..., action_count]
        return self.net(rows).squeeze(-1)


@dataclass(frozen=True)
class AgentConfig:
    hidden: int = 64
    gamma: float = 0.99
    learning_rate: float = 0.0003
    batch_size: int = 64
    replay_capacity: int = 20000
    learning_starts: int = 128
    train_every: int = 4
    target_sync: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 4000
    seed: int = 7

    def __post_init__(self):
        if self.hidden < 4 or not 0 < self.gamma <= 1 or self.learning_rate <= 0:
            raise ValueError('invalid DDQN hyperparameters')
        if (self.batch_size < 1 or self.replay_capacity < self.batch_size or
            self.learning_starts < self.batch_size or self.train_every < 1 or
            self.target_sync < 1 or self.epsilon_decay_steps < 1):
            raise ValueError('invalid replay/schedule configuration')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('invalid epsilon bounds')


@dataclass(frozen=True)
class Transition:
    rows: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_rows: np.ndarray
    next_mask: np.ndarray
    done: bool


class MaskedDDQN:
    def __init__(self, feature_count: int, config: AgentConfig):
        self.feature_count = feature_count
        self.config = config
        self.random = random.Random(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.online = ActionQNetwork(feature_count, config.hidden)
        self.target = ActionQNetwork(feature_count, config.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque[Transition] = deque(maxlen=config.replay_capacity)
        self.env_steps = 0
        self.gradient_steps = 0
        self.last_loss: float | None = None

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + frac*(self.config.epsilon_end-self.config.epsilon_start)

    @torch.no_grad()
    def act(self, rows: np.ndarray, mask: np.ndarray, *, greedy: bool = False) -> int:
        allowed = np.flatnonzero(mask)
        if not len(allowed):
            raise ValueError('empty legal action mask')
        if not greedy and self.random.random() < self.epsilon():
            return int(self.random.choice(allowed.tolist()))
        # Stable first-index tie-breaking when Q-values are equal.
        q = self.online(torch.from_numpy(rows[None].astype(np.float32)))[0].numpy()
        return int(np.argmax(np.where(mask, q, -np.inf)))

    def remember(self, transition: Transition) -> None:
        if not transition.mask[transition.action]:
            raise AssertionError('replay contains an invalid action')
        self.replay.append(transition)
        self.env_steps += 1

    def learn(self) -> float | None:
        if (len(self.replay) < self.config.learning_starts or
            self.env_steps % self.config.train_every != 0):
            return None
        batch = self.random.sample(self.replay, self.config.batch_size)
        rows = torch.as_tensor(np.stack([t.rows for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        nxt = torch.as_tensor(np.stack([t.next_rows for t in batch]), dtype=torch.float32)
        masks = torch.as_tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.bool)
        selected = self.online(rows).gather(1, actions[:,None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: SELECT next action with online weights; EVALUATE with target.
            online_next = self.online(nxt).masked_fill(~masks, -torch.inf)
            # Every nonterminal observation has at least the forced-reject action.
            next_action = online_next.argmax(dim=1)
            target_q = self.target(nxt).gather(1, next_action[:,None]).squeeze(1)
            targets = rewards + self.config.gamma * (~dones).float() * target_q
        loss = F.smooth_l1_loss(selected, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path, metadata: dict) -> None:
        torch.save({'online': self.online.state_dict(), 'feature_count':self.feature_count,
                    'agent_config':asdict(self.config), 'metadata':metadata}, path)

    @classmethod
    def load(cls, path) -> tuple['MaskedDDQN', dict]:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        config = AgentConfig(**ckpt['agent_config'])
        agent = cls(int(ckpt['feature_count']), config)
        agent.online.load_state_dict(ckpt['online'])
        agent.target.load_state_dict(ckpt['online'])
        return agent, ckpt['metadata']
