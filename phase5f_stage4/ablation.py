"""Frozen 2x2 factorial ablations: greedy guidance and current-state congestion features.

The network architecture and 25-dimensional tensor shape never change across arms.
Excluded channels are zeroed both for acting and PPO updates.  A/B differ only
in greedy-default observation plus behavior cloning, C/D likewise.  A/C do not
use the greedy-default flag or an evaluation gate; all arms use the identical
feasibility mask, PPO model width and physical resource constraints.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Ablation:
    name:str
    guidance:bool
    enhanced:bool

CONFIGS={
 'A_standard':Ablation('A_standard',False,False),
 'B_guided':Ablation('B_guided',True,False),
 'C_enhanced':Ablation('C_enhanced',False,True),
 'D_full_hybrid':Ablation('D_full_hybrid',True,True),
}

# Among Stage 3's 17 per-candidate fields, these are designated congestive /
# resource-related enhancements.  All other candidate fields remain usable.
ENHANCED_CAND_COLUMNS=(4,5,6,7,8,9,10,14,15,16)
ENHANCED_GLOBAL_COLUMNS=(1,7)
GREEDY_DEFAULT_COLUMN=12


def transform(obs, config:Ablation):
    c,g,mask=obs
    c=np.array(c,dtype=np.float32,copy=True)
    g=np.array(g,dtype=np.float32,copy=True)
    mask=np.asarray(mask,dtype=bool).copy()
    if not config.enhanced:
        c[...,list(ENHANCED_CAND_COLUMNS)]=0.0
        g[...,list(ENHANCED_GLOBAL_COLUMNS)]=0.0
    if not config.guidance:
        c[...,GREEDY_DEFAULT_COLUMN]=0.0
    c[~mask]=0
    return c,g,mask


class MaskedFeaturePolicy:
    """Applies identical feature ablation to every online model decision."""
    def __init__(self,model,config:Ablation):
        self.model=model
        self.config=config

    def decide(self,encoded,**kwargs):
        return self.model.decide(transform(encoded,self.config),**kwargs)


def mask_training_observations(history,config:Ablation):
    result=[]
    for row in history:
        c,g,m=transform((row['c'],row['g'],row['mask']),config)
        result.append({**row,'c':c,'g':g,'mask':m})
    return result
