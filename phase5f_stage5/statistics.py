#Precommitted topology-clustered paired inference; no test-data selection.

#Units are independent *network-topology seeds* (n=20), not the 60 streams and
#certainly not the 2,000 LSP arrivals per stream. All estimates in percentage
#points. Primary is C_enhanced minus greedy in 500/K6/dynamic/load4.

from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t

POLICIES=('greedy','footprint','random_feasible','C_enhanced','D_full_hybrid')
PRIMARY=(500,6,'dynamic',4.0)
# Preregistered full 2x2x2 A/B/C factorial at load 4.0, primary first.
CELLS=(PRIMARY, (60,2,'static',4.0), (60,2,'dynamic',4.0),
       (60,6,'static',4.0),(60,6,'dynamic',4.0),
       (500,2,'static',4.0),(500,2,'dynamic',4.0),(500,6,'static',4.0))
TOPOLOGIES=tuple(range(401,421))
STREAMS=(73000,73001,73002)
BOOTSTRAP_REPS=10000
BOOTSTRAP_SEED=5192026


def key(row):
    return (int(row['nodes']),int(row['k']),str(row['mode']),float(row['load']),
            int(row['topology_seed']),int(row['stream_seed']))


def _index(rows):
    d={}
    for r in rows:
        p=r['policy']; idx=(*key(r),p)
        if idx in d:raise ValueError('duplicate result: '+str(idx))
        d[idx]=r
    return d


def validate_full_grid(rows):
    d=_index(rows)
    expected={(n,k,mode,load,t,s,p) for (n,k,mode,load) in CELLS
              for t in TOPOLOGIES for s in STREAMS for p in POLICIES}
    if set(d)!=expected:
        raise ValueError(f'missing/extra heldout results: expected {len(expected)}, got {len(d)}, missing={len(expected-set(d))}, extra={len(set(d)-expected)}')
    for n,k,mode,load,t,s,p in expected:
        r=d[n,k,mode,load,t,s,p]
        assert r['policy']==p
        if (int(r['requests'])!=round(n*load) or
            int(r['admitted'])+int(r['rejected'])!=int(r['requests']) or
            int(r['feasibility_violations'])!=0 or
            not 0<=float(r['weighted_admission'])<=1):
            raise ValueError('invalid counts/feasibility '+str((n,k,mode,load,t,s,p)))
    # Same per-cohort arrivals/network/candidates regardless of evaluated policy.
    for n,k,mode,load,t,s in {(v[:6]) for v in expected}:
        rr=[d[n,k,mode,load,t,s,p] for p in POLICIES]
        for attr in ('network_sha256','request_sha256','candidates_sha256','requests'):
            if len({str(row[attr]) for row in rr})!=1:
                raise ValueError('unmatched policy inputs: '+str((n,k,mode,load,t,s,attr)))
        if any(float(row['weighted_admission'])>1.000001 for row in rr):
            raise ValueError('impossible admission')
    return d


def cluster_contrasts(rows,cell,first,second):
    d=_index(rows)
    n,k,mode,load=cell
    vals=[]
    for topo in TOPOLOGIES:
        per_stream=[]
        for stream in STREAMS:
            a=d[n,k,mode,load,topo,stream,first]
            b=d[n,k,mode,load,topo,stream,second]
            per_stream.append((float(a['weighted_admission'])-float(b['weighted_admission']))*100)
        vals.append(float(np.mean(per_stream)))
    return np.array(vals,dtype=np.float64)


def paired_stats(x,boot_reps=BOOTSTRAP_REPS):
    x=np.asarray(x,dtype=np.float64)
    n=len(x)
    if n<3:raise ValueError('at least three distinct topology clusters required')
    estimate=float(x.mean());sd=float(x.std(ddof=1));sem=sd/np.sqrt(n)
    critical=float(student_t.ppf(.975,n-1))
    if sd==0:
        p=1.0 if estimate==0 else 0.0
    else:
        stat=estimate/sem
        p=float(min(1.0,2*student_t.sf(abs(stat),n-1)))
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    sims=np.mean(x[rng.integers(0,n,size=(boot_reps,n))],axis=1)
    low,high=np.percentile(sims,[2.5,97.5])
    return dict(topology_clusters=n,streams_per_topology=len(STREAMS),
        effect_pp=estimate,sample_sd_pp=sd,standard_error_pp=float(sem),
        t95_low_pp=estimate-critical*sem,t95_high_pp=estimate+critical*sem,
        bootstrap95_low_pp=float(low),bootstrap95_high_pp=float(high),
        bootstrap_replicates=boot_reps,p_two_sided=p,
        positive_clusters=int(np.sum(x>0)),negative_clusters=int(np.sum(x<0)),
        tied_clusters=int(np.sum(x==0)))


def holm(rows):
    """Step-down Holm adjustment over one prespecified secondary family."""
    m=len(rows);order=sorted(range(m),key=lambda i:float(rows[i]['p_two_sided']))
    greatest=0.0
    for rank,i in enumerate(order):
        greatest=max(greatest,min(1.0,(m-rank)*float(rows[i]['p_two_sided'])))
        rows[i]['p_holm_adjusted']=min(1.0,greatest)
    return rows


def factorial(rows):
    #Effects of A-size, B-candidates, C-dynamics on C-vs-greedy gain.

    #Effect convention: positive means higher C-minus-greedy delta when that
    #factor is at (+) level. Other factors are averaged with equal weight.
    
    raw={cell:cluster_contrasts(rows,cell,'C_enhanced','greedy') for cell in CELLS}
    # Encoding: A=500 vs60, B=K6 vs K2, C=dynamic vs static.
    factors={cell:(1 if cell[0]==500 else -1,
                   1 if cell[1]==6 else -1,
                   1 if cell[2]=='dynamic' else -1) for cell in CELLS}
    out=[]
    for name,columns in [('A_network_size',(0,)),('B_route_candidates',(1,)),
                         ('C_dynamic_traffic',(2,)),('AB',(0,1)),('AC',(0,2)),
                         ('BC',(1,2)),('ABC',(0,1,2))]:
        effects=[]
        for t in range(len(TOPOLOGIES)):
            effects.append(.25*sum(raw[cell][t]*np.prod([factors[cell][j] for j in columns])
                                     for cell in CELLS))
        out.append(dict(comparison=name,estimate='factorial_C_minus_greedy',
                        **paired_stats(effects)))
    return holm(out)


def analyze(rows):
    validate_full_grid(rows)
    result=[]
    for cell in CELLS:
        for p in POLICIES:
            matched=[r for r in rows if (int(r['nodes']),int(r['k']),r['mode'],float(r['load']))==cell
                     and r['policy']==p]
            result.append(dict(nodes=cell[0],k=cell[1],mode=cell[2],load=cell[3],policy=p,
                cohort_count=len(matched),topology_clusters=len(TOPOLOGIES),
                mean_weighted_admission=float(np.mean([float(r['weighted_admission']) for r in matched])),
                mean_ordinary_admission=float(np.mean([float(r['ordinary_admission']) for r in matched])),
                sum_admitted=sum(int(r['admitted']) for r in matched),
                sum_offered=sum(int(r['requests']) for r in matched),
                total_overrides=sum(int(r['overrides']) for r in matched),
                total_discretionary=sum(int(r['discretionary']) for r in matched),
                avg_simulation_seconds=float(np.mean([float(r['simulation_seconds']) for r in matched]))))
    primary=[]
    contrasts=[('PRIMARY_C_minus_greedy','C_enhanced','greedy','primary'),
               ('SECONDARY_C_minus_footprint','C_enhanced','footprint','secondary'),
               ('SECONDARY_D_minus_greedy','D_full_hybrid','greedy','secondary'),
               ('SECONDARY_C_minus_D','C_enhanced','D_full_hybrid','secondary')]
    for name,a,b,family in contrasts:
        primary.append(dict(comparison=name,first=a,second=b,
                            evidence_family=family,**paired_stats(cluster_contrasts(rows,PRIMARY,a,b))))
    secondary=holm([r for r in primary if r['evidence_family']=='secondary'])
    primary=[r for r in primary if r['evidence_family']=='primary']+secondary
    factorial_rows=factorial(rows)
    # All 8 cells compared descriptively, without presenting unadjusted
    # individual-cell findings as confirmatory positives.
    paired=[]
    for cell in CELLS:
        for topo_index,topo in enumerate(TOPOLOGIES):
            for a,b in [('C_enhanced','greedy'),('C_enhanced','footprint'),
                        ('D_full_hybrid','greedy')]:
                v=cluster_contrasts(rows,cell,a,b)[topo_index]
                paired.append(dict(nodes=cell[0],k=cell[1],mode=cell[2],load=cell[3],
                    topology_seed=topo,policy=a,baseline=b,paired_topology_mean_delta_pp=float(v)))
    return result,primary,factorial_rows,paired
