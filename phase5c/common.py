"""Shared masks, tensor operations, and checkpointing for Phase 5C.

Action dimension = number of policy-valid candidates plus forced reject; there
is no voluntary rejection. Every observation includes at least one legal action.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from phase5.agent import ActionQNetwork
from phase5.environment import FEATURE_NAMES
from . import VERSION


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def logits_and_prob(logits: torch.Tensor, mask: torch.Tensor):
    if logits.shape != mask.shape or not bool(mask.any(dim=-1).all()):
        raise ValueError('invalid action mask or logits dimensions')
    z = logits.masked_fill(~mask, -1e9)
    return torch.softmax(z, -1) * mask.float(), torch.log_softmax(z, -1)


def get_batch(data: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, torch.Tensor]:
    return {k:torch.as_tensor(v[ids],dtype=torch.bool if k in ('mask','next_mask','done')
                              else torch.long if k == 'action' else torch.float32)
            for k,v in data.items() if k in
            ('rows','mask','action','reward','next_rows','next_mask','done')}


class StateValue(nn.Module):
    def __init__(self, actions: int, features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(actions*(features+1),hidden),nn.ReLU(),
                                 nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self, rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Never infer value from illegal candidate feature vectors.
        x = torch.cat((rows * mask.unsqueeze(-1).float(), mask.unsqueeze(-1).float()),dim=-1)
        return self.net(x.flatten(start_dim=-2)).squeeze(-1)


@dataclass(frozen=True)
class RLConfig:
    algorithm: str = 'ppo'
    hidden: int = 64
    lr: float = 0.0003
    gamma: float = 0.99
    seed: int = 7
    batch: int = 64
    tau: float = 0.01
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    behavior_threshold: float = 0.3
    conservative_alpha: float = 1.0
    awac_lambda: float = 1.0
    expectile: float = 0.7
    max_weight: float = 20.0

    def __post_init__(self):
        from . import ALGORITHMS
        if (self.algorithm not in ALGORITHMS or self.hidden < 4 or self.lr <= 0 or
            not 0 < self.gamma <= 1 or self.batch < 1 or not 0 < self.tau <= 1 or
            not 0 < self.ppo_clip < 1 or self.ppo_epochs < 1 or
            not 0 < self.gae_lambda <= 1 or self.entropy_coef < 0 or
            not 0 < self.behavior_threshold <= 1 or self.conservative_alpha < 0 or
            self.awac_lambda <= 0 or not 0 < self.expectile < 1 or self.max_weight < 1):
            raise ValueError('invalid Phase 5C algorithm configuration')


class CheckpointMixin:
    """Store only weights + primitive metadata (weights_only=True safe loading)."""
    def save(self,path:Path,meta:dict) -> None:
        torch.save({'family':self.config.algorithm,'cfg':asdict(self.config),
                    'features':self.features,'actions':self.actions,
                    'weights':self._weights(), 'metadata':meta},path)

    @classmethod
    def load(cls,path:Path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        config=RLConfig(**ck['cfg'])
        target=make_agent(config,ck['features'],ck['actions'])
        target._restore(ck['weights'])
        return target,ck['metadata']


def make_agent(config:RLConfig,features:int=len(FEATURE_NAMES),actions:int=3):
    if config.algorithm=='ppo':
        from .ppo import MaskedPPO
        return MaskedPPO(features,actions,config)
    from .offline_agents import OfflineAgent
    return OfflineAgent(features,actions,config)
