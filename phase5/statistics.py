"""Seed-level held-out DDQN vs Phase4 baselines statistical summary.

Intervals quantify demand-stream variability for a FIXED synthetic topology.
Zero paired differences do not establish equivalence.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import statistics
from scipy.stats import t
from .experiment import new_output, dump_csv


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _interval(items: list[float]) -> dict:
    n=len(items)
    if n < 1:
        raise ValueError('empty data')
    mean=statistics.mean(items)
    sd=statistics.stdev(items) if n>1 else 0.0
    h=t.ppf(.975,n-1)*sd/n**.5 if n>1 else None
    return {'n':n,'mean':round(mean,8),'sd':round(sd,8),
            'lower_95':round(mean-h,8) if h is not None else '',
            'upper_95':round(mean+h,8) if h is not None else '',
            'zero_variance_warning':int(n>1 and sd == 0)}


def summarize(evaluation_dir: Path, output: Path) -> dict:
    runs=read_csv(evaluation_dir/'phase5_runs.csv')
    paired=read_csv(evaluation_dir/'phase5_paired_deltas.csv')
    if not runs or not paired:
        raise ValueError('not enough matched runs')
    all_seeds=sorted({int(r['demand_seed']) for r in runs})
    by_policy:dict[str,list[dict]]=defaultdict(list)
    matched={}
    for r in runs:
        key=(int(r['demand_seed']),r['policy'])
        if key in matched:raise ValueError(f'duplicate run {key}')
        matched[key]=r
        if int(r['admitted'])+int(r['rejected'])!=int(r['requests']):
            raise AssertionError('acceptance accounting mismatch')
        if abs(int(r['admitted'])/int(r['requests'])-float(r['admission_ratio']))>1e-5:
            raise AssertionError('invalid admission ratio')
        by_policy[r['policy']].append(r)
    policies=set(by_policy)
    for seed in all_seeds:
        if {p for s,p in matched if s==seed}!=policies:
            raise AssertionError('unpaired set of policies')
        hashes={matched[seed,p]['request_sha256'] for p in policies}
        if len(hashes)!=1:raise AssertionError('request hash mismatch')
        specs={(matched[seed,p]['nodes'],matched[seed,p]['headroom_percent'],
                matched[seed,p]['max_candidates'],matched[seed,p]['requests'],
                matched[seed,p]['topology_seed']) for p in policies}
        if len(specs)!=1:raise AssertionError('inconsistent paired configuration')
    new_output(output)
    rows=[]
    for p,items in sorted(by_policy.items()):
        d={'policy':p,'seeds':len(items)}
        for name,field in [('admission','admission_ratio'),('weighted','weighted_admission_ratio'),
                           ('selection_ms','avg_selection_ms')]:
            z=_interval([float(r[field]) for r in items])
            for col in ('mean','sd','lower_95','upper_95'):
                d[f'{name}_{col}']=z[col]
        d['admitted_total']=sum(int(r['admitted']) for r in items)
        d['offered_total']=sum(int(r['requests']) for r in items)
        rows.append(d)
    dump_csv(output/'condition_summary.csv',rows)
    index={(int(p['demand_seed']),p['policy']):p for p in paired}
    paired_rows=[]
    for policy in sorted(policies-{'greedy'}):
        x=[index[seed,policy] for seed in all_seeds]
        out={'policy':policy,'baseline':'greedy','seeds':len(x)}
        for name,field in [('admission','delta_admission_ratio'),
                           ('weighted','delta_weighted_ratio'),
                           ('selection_ms','delta_avg_selection_ms')]:
            z=_interval([float(q[field]) for q in x])
            for col in ('mean','sd','lower_95','upper_95'):
                out[f'delta_{name}_{col}']=z[col]
        out['positive_admission_seeds']=sum(float(q['delta_admission_ratio'])>0 for q in x)
        out['negative_admission_seeds']=sum(float(q['delta_admission_ratio'])<0 for q in x)
        out['zero_admission_seeds']=sum(float(q['delta_admission_ratio'])==0 for q in x)
        paired_rows.append(out)
    dump_csv(output/'paired_summary.csv',paired_rows)
    text=['# Phase 5 — Held-out masked DDQN versus conventional baselines','',
          f'- Matched held-out demand sequences: **{len(all_seeds)}**.',
          '- Main inferential unit: **whole demand seed**, not individual requests.',
          '- Comparisons: identical requests (SHA-256 verified) and topology configuration per seed.',
          '- Confidence intervals: two-sided **95% Student t** intervals over seed-level results.',
          '- This does **not** include uncertainty across topology seeds or railway infrastructure.',
          '- Zero-width intervals from zero sample variance **do not establish equivalence**.',
          '- Timings include policy selection; DDQN additionally includes feature construction / masking.',
          '- Underlying protection model is the Phase-3 single-link scenario surrogate, **not MSRP**.','',
          '## Held-out admission','',
          '| Policy | Admitted / Offered | Mean admission (95% CI) | Weighted admission | Selection ms/request |',
          '|---|---:|---:|---:|---:|']
    def fmt_ci(z,prefix):
        m=z[f'{prefix}_mean']
        lo=z[f'{prefix}_lower_95'];hi=z[f'{prefix}_upper_95']
        if lo=='' or hi=='':return f'{100*m:.2f}% [CI unavailable]'
        return f'{100*m:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%]'
    for r in rows:
        text.append(f'| {r["policy"]} | {r["admitted_total"]}/{r["offered_total"]} | '
                    f'{fmt_ci(r,"admission")} | {100*r["weighted_mean"]:.2f}% | '
                    f'{r["selection_ms_mean"]:.2f} |')
    text+=['','## Paired differences (algorithm minus greedy)','',
           '| Algorithm | Admission difference (95% CI), percentage points | Positive / Negative / Zero seeds |',
           '|---|---:|---:|']
    for r in paired_rows:
        l=r['delta_admission_lower_95'];h=r['delta_admission_upper_95']
        ci=f'[{100*l:+.2f}, {100*h:+.2f}]' if l!='' else 'CI unavailable'
        text.append(f'| {r["policy"]} | {100*r["delta_admission_mean"]:+.2f} {ci} | '
                    f'{r["positive_admission_seeds"]} / {r["negative_admission_seeds"]} / '
                    f'{r["zero_admission_seeds"]} |')
    text+=['','## Reporting checklist','',
           '1. Disclose checkpoint selection by **validation only**; do not retune after observing held-out results.',
           '2. Report both weighted and unweighted admissions, not merely cumulative RL reward.',
           '3. Report learning curves, three random training initializations if possible, and compute budgets.',
           '4. Compare baseline versions **greedy / CSPF-style / MILP V2** from the same test seeds.',
           '5. State that inference-only latency is not restoration/switching time.',
           '6. Report finite two-candidate limitation, fixed topology seed, forced-only rejection, and no MSRP signaling.',
           '7. Do not claim statistical equivalence from zero observed differences.']
    (output/'phase5_statistical_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return {'seeds':len(all_seeds),'policies':sorted(policies),'paired_rows':len(paired_rows),
            'report':str(output/'phase5_statistical_report.md')}
