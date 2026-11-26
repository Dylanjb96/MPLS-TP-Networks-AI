"""Topology-clustered inference and complete locked-grid assertions (no model code)."""
from __future__ import annotations
from collections import Counter,defaultdict
import math
import numpy as np
from scipy.stats import t as student_t
from .protocol import CELLS,BY_CELL,POLICIES,PRIMARY,SECONDARY,FINAL_TOPOLOGIES,FINAL_STREAMS


def grid_validate(rows):
    expected={(c.id,t,s,p) for c in CELLS for t in FINAL_TOPOLOGIES for s in FINAL_STREAMS for p in POLICIES}
    found={}
    for r in rows:
        key=(r['cell'],int(r['topology_seed']),int(r['stream_seed']),r['policy'])
        if key in found:raise ValueError('duplicate cell/topology/stream/policy: '+str(key))
        found[key]=r
    if set(found)!=expected:
        raise ValueError(f'full final grid incomplete/extraneous: missing={len(expected-set(found))} extra={len(set(found)-expected)}')
    for c in CELLS:
      for t in FINAL_TOPOLOGIES:
       for s in FINAL_STREAMS:
        grouped=[found[c.id,t,s,p] for p in POLICIES]
        if any(int(r['nodes'])!=500 or int(r['k'])!=c.k or r['family']!=c.family or r['stress']!=c.stress
               or r['mode']!='dynamic' or float(r['load'])!=4.0 for r in grouped):
            raise ValueError('wrong frozen experimental condition: '+str((c.id,t,s)))
        for col in ('network_sha256','request_sha256','candidates_sha256','requests'):
            if len({str(r[col]) for r in grouped})!=1:
                raise ValueError('unmatched paired cohort: '+str((c.id,t,s,col)))
        for r in grouped:
            if (int(r['admitted'])+int(r['rejected'])!=2000 or int(r['requests'])!=2000 or
                int(r['feasibility_violations'])!=0 or int(r['decision_records'])!=2000 or
                not 0<=float(r['weighted_admission'])<=1 or not 0<=int(r['overrides'])<=int(r['discretionary'])):
                raise ValueError('invalid result row: '+str((c.id,t,s,r['policy'])))
    return found


def differences(index, cell, a, b):
    if cell not in BY_CELL or a not in POLICIES or b not in POLICIES:raise ValueError('unknown comparison')
    arr=[]
    for topo in FINAL_TOPOLOGIES:
        arr.append(float(np.mean([100*(float(index[cell,topo,s,a]['weighted_admission'])-
                                  float(index[cell,topo,s,b]['weighted_admission']))
                                  for s in FINAL_STREAMS])))
    return np.asarray(arr,dtype=np.float64)


def cluster_stats(diffs):
    x=np.asarray(diffs,dtype=np.float64)
    if x.shape!=(20,) or not np.isfinite(x).all():raise ValueError('20 finite topology clusters required')
    est=float(x.mean());sd=float(x.std(ddof=1));se=sd/math.sqrt(20)
    if sd==0:
        if est!=0:
            return dict(n_topologies=20,streams_per_topology=3,effect_pp=est,sd_pp=0.0,ci95_low_pp=est,ci95_high_pp=est,
              p_two_sided=None,zero_variance_nonzero=True,positive=int((x>0).sum()),negative=int((x<0).sum()),ties=int((x==0).sum()))
        p=1.0;low=high=0.0
    else:
        cutoff=float(student_t.ppf(.975,19));low=est-cutoff*se;high=est+cutoff*se
        p=float(2*student_t.sf(abs(est/se),19))
    return dict(n_topologies=20,streams_per_topology=3,effect_pp=est,sd_pp=sd,ci95_low_pp=low,ci95_high_pp=high,
                p_two_sided=p,zero_variance_nonzero=False,
                positive=int((x>0).sum()),negative=int((x<0).sum()),ties=int((x==0).sum()))


def holm(rows):
    if any(r['p_two_sided'] is None for r in rows):
        raise ValueError('zero-variance nonzero effect: conventional paired t-test undefined; require separately documented analysis')
    order=sorted(range(len(rows)), key=lambda j:(rows[j]['p_two_sided'],j));bound=0.0
    for pos,j in enumerate(order):
        bound=max(bound,min(1.0,(len(rows)-pos)*rows[j]['p_two_sided']))
        rows[j]['p_holm_adjusted']=min(1.0,bound)
        rows[j]['holm_significant']=bool(rows[j]['p_holm_adjusted']<0.05)
    return rows


def analyze(rows):
    idx=grid_validate(rows)
    co=[dict(family='co_primary',cell=c,first=a,second=b,**cluster_stats(differences(idx,c,a,b)))
        for c,a,b in PRIMARY]
    sec=[dict(family='secondary',cell=c,first=a,second=b,**cluster_stats(differences(idx,c,a,b)))
         for c,a,b in SECONDARY]
    holm(co);holm(sec)
    # Exploratory stress and K2 sensitivity, compared to the base cell in the same family.
    exploratory=[]
    for fam,pre in (('hierarchical','H'),('ladder','L')):
        for condition in ('BW','HOLD','K2'):
            for first,second in (('gnn_PPO','mlp_PPO'),('gnn_PPO','scenario_minimax')):
                diff=differences(idx,f'{pre}_{condition}',first,second)-differences(idx,f'{pre}_BASE',first,second)
                exploratory.append(dict(family='exploratory_interaction_unadjusted',cell=f'{pre}_{condition}_minus_{pre}_BASE',
                                        first=first,second=second,**cluster_stats(diff)))
    means=[]
    for c in CELLS:
        for p in POLICIES:
            matching=[idx[c.id,t,s,p] for t in FINAL_TOPOLOGIES for s in FINAL_STREAMS]
            means.append(dict(cell=c.id,family=c.family,stress=c.stress,k=c.k,policy=p,cohorts=60,
                              mean_weighted_admission=float(np.mean([float(r['weighted_admission']) for r in matching])),
                              total_admitted=sum(int(r['admitted']) for r in matching),
                              offered=sum(int(r['requests']) for r in matching),
                              overrides=sum(int(r['overrides']) for r in matching),
                              discretionary=sum(int(r['discretionary']) for r in matching),
                              total_policy_seconds=sum(float(r['policy_seconds']) for r in matching)))
    return dict(primary=co,secondary=sec,exploratory=exploratory,summary=means)
