"""Feasibility-preserving residual action space and state features.

No future demand is visible. Exactly one native OnlineEpisode.step() is used
for each request; no external commits or manual ledger manipulation.
Residual action 0 means the FIRST FEASIBLE candidate in the original ordering
(or the forced reject action if there are no feasible candidates).
Other residual actions are the remaining feasible candidate indices in order.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from phase3.model import NORMAL, _path_edges
from phase5.environment import FEATURE_NAMES, OnlineEpisode

EXTRA_NAMES = (
    'is_greedy_default', 'delta_peak_from_greedy', 'delta_hops_from_greedy',
    'projected_min_scenario_slack', 'delta_slack_from_greedy',
    'mean_projected_scenario_util', 'near_saturated_arc_fraction',
    'reservation_footprint_per_node',
)


def feature_names(enhanced: bool):
    return list(FEATURE_NAMES) + (list(EXTRA_NAMES) if enhanced else ['is_greedy_default'])


@dataclass(frozen=True)
class ResidualObservation:
    rows: np.ndarray
    mask: np.ndarray
    mapping: tuple[int, ...]             # residual index -> original native candidate index
    greedy_native: int
    request_id: str
    discretionary: bool


def scenario_features(env: OnlineEpisode, idx: int):
    """Prospective scenario reservations using ONLY the current ledger and request.

    Phase-3 audit remains authoritative. These are advisory state features,
    NOT independent feasibility predicates or restoration-time estimates.
    """
    p=env.prepared.candidate_sets[env.pos][idx]
    req=env.prepared.requests[env.pos]
    ledger=env.sim._ledger(env.sim.accepted())
    ratios=[];footprint=0
    for scenario in (NORMAL,*env.sim.failures):
        path=p.normal_path if scenario==NORMAL else p.fault_paths[scenario]
        for u,v in _path_edges(path):
            for a,b in ((u,v),(v,u)):
                cap=env.sim.available_capacity(a,b)
                if cap <= 0:raise AssertionError('nonpositive usable capacity')
                projected=ledger[scenario].get((a,b),0)+req.bandwidth_mbps
                ratios.append(projected/cap)
                footprint+=1
    if not ratios:return (1.0,0.0,0.0,0.0)
    return (float(np.clip(1-max(ratios),0,1)),
            float(np.clip(np.mean(ratios),0,2)),
            sum(x>=.85 for x in ratios)/len(ratios),
            footprint/max(1,len(env.sim.G)*len(env.sim.failures)+len(env.sim.G)))


class ResidualEpisode:
    def __init__(self, env:OnlineEpisode, *, enhanced:bool):
        self.env=env; self.enhanced=enhanced; self.k=env.k
        self._obs: ResidualObservation|None=None
        self.override_count=0
        self.discretionary_count=0
        self.greedy_count=0

    @property
    def done(self):return self.env.done

    def observe(self)->ResidualObservation:
        if self.done:raise ValueError('cannot observe completed residual episode')
        if self._obs is not None:return self._obs
        obs=self.env.observe(); feasible=np.flatnonzero(obs.mask[:self.k]).tolist()
        greedy=feasible[0] if feasible else self.k
        mapping=tuple([greedy]+[i for i in feasible if i!=greedy])
        mask=np.zeros(self.k,dtype=np.bool_)
        mask[:len(mapping)]=True
        rows=np.zeros((self.k,len(feature_names(self.enhanced))),dtype=np.float32)
        projected={}
        if self.enhanced:
            for i in feasible:projected[i]=scenario_features(self.env,i)
        for r,i in enumerate(mapping):
            native=obs.rows[i]
            if self.enhanced:
                base=obs.rows[greedy]
                sl,mu,near,foot=projected.get(i,(1.,0.,0.,0.))
                sl_g=projected.get(greedy,(1.,0.,0.,0.))[0]
                extra=[float(r==0),float(native[20]-base[20]),
                       float(native[16]-base[16]),sl,sl-sl_g,mu,near,foot]
            else: extra=[float(r==0)]
            rows[r,:]=np.asarray([*native,*extra],dtype=np.float32)
        self._obs=ResidualObservation(rows,mask,mapping,greedy,obs.request_id,
                                        len(mapping)>1)
        return self._obs

    def step(self, residual_action:int):
        obs=self.observe()
        if not isinstance(residual_action,(int,np.integer)) or residual_action<0 or residual_action>=self.k or not obs.mask[residual_action]:
            raise ValueError('invalid or infeasible residual action')
        native_action=obs.mapping[residual_action]
        self.greedy_count+=int(residual_action==0)
        self.override_count+=int(residual_action!=0)
        self.discretionary_count+=int(obs.discretionary)
        reward,done,rec=self.env.step(native_action)
        rec.update({'residual_action':int(residual_action),'greedy_native_action':obs.greedy_native,
                    'overrode_greedy':int(residual_action!=0),
                    'discretionary':int(obs.discretionary)})
        self._obs=None
        return reward,done,rec

    def summary(self):
        result=self.env.summary()
        result.update(overrides=self.override_count,discretionary=self.discretionary_count,
                      greedy_steps=self.greedy_count)
        return result


def gate_action(logits:np.ndarray,mask:np.ndarray,threshold:float):
    """Greedy by default. Override only for a positive confidence margin.

    The margin compares categorical policy probabilities, NOT calibrated
    probabilities of improved admission. Threshold >=0, max 1.
    """
    if not 0<=threshold<=1:raise ValueError('threshold must be in [0,1]')
    legal=np.flatnonzero(mask)
    if not len(legal) or legal[0]!=0:raise ValueError('expected greedy fallback at action 0')
    if len(legal)==1:return 0,0.0
    z=np.asarray(logits,dtype=float)
    v=z[legal]-np.max(z[legal]);p=np.exp(v);p/=sum(p)
    # Deterministic first-alternative tie break.
    candidates=legal[legal!=0];alt=int(candidates[np.argmax([p[list(legal).index(i)] for i in candidates])])
    margin=float(p[list(legal).index(alt)]-p[0])
    return (alt if margin>threshold else 0),margin


def controls_action(kind:str, observation:ResidualObservation, rng, threshold=0.10):
    """Non-learning controls use the same residual action mapping and mask."""
    if not observation.discretionary:return 0
    if kind=='random_override':return rng.choice(range(len(observation.mapping)))
    if kind=='capacity_override':
        # Strictly compare prospective maximum scenario utilization (feature 20).
        # Never override on ties.
        scores=[observation.rows[i,20] for i in range(len(observation.mapping))]
        best=int(np.argmin(scores))
        return best if best!=0 and scores[0]-scores[best]>threshold else 0
    if kind=='greedy':return 0
    raise ValueError('unknown non-learning control')
