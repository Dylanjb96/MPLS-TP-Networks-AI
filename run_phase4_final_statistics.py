#!/usr/bin/env python3
#Reproducible Phase 4 seed-level statistical analysis.

#Works with phase4_runs.csv from the Phase 4 project, without changes to the
#simulation code. Uses only Python standard library and SciPy (already required
#by the Phase 4 online MILP).

#Examples (from the MPLS-AI root, Windows cmd):
  #python run_phase4_final_statistics.py --root outputs/phase4 --out outputs/phase4_statistics
  #python run_phase4_final_statistics.py --inputs path/to/phase4_runs.csv --out outputs/phase4_statistics

#Statistical unit: independently seeded full request sequences; NOT individual
#service requests, failed-link scenarios, or measurements from repeated exports.

import argparse
import csv
import math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t as student_t

POLICIES = ('greedy', 'cspf', 'milp')
GROUP_KEYS = ('nodes','topology_seed','headroom_percent','max_candidates','requests')
KEYS = GROUP_KEYS + ('demand_seed','policy')
CHECK_EQ = ('admitted','rejected','admission_ratio','weighted_admission_ratio',
            'normal_capacity_rejections','postfault_capacity_rejections',
            'no_plan_rejections','fallback_count','solver_nonoptimal_count',
            'request_sha256','state_digest')
INT_COLS = {'nodes','topology_seed','headroom_percent','max_candidates','requests',
            'demand_seed','admitted','rejected','normal_capacity_rejections',
            'postfault_capacity_rejections','no_plan_rejections','fallback_count',
            'solver_nonoptimal_count'}
FLOAT_COLS = {'admission_ratio','weighted_admission_ratio','avg_selection_ms',
              'selection_seconds','candidate_generation_seconds',
              'total_policy_seconds','peak_scenario_utilization'}


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = set(KEYS) | {'admitted','rejected','admission_ratio','weighted_admission_ratio',
                                'normal_capacity_rejections','postfault_capacity_rejections',
                                'no_plan_rejections','fallback_count','solver_nonoptimal_count',
                                'request_sha256','avg_selection_ms','state_digest'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing fields {sorted(missing)}')
        for line_no, original in enumerate(reader, 2):
            row = dict(original)
            row['_source_file'] = str(path)
            for name in INT_COLS:
                row[name] = int(row[name])
            for name in FLOAT_COLS:
                if name in row and row[name] not in ('', None):
                    row[name] = float(row[name])
            row['policy'] = row['policy'].strip().lower()
            if row['policy'] not in POLICIES:
                raise ValueError(f'{path}:{line_no}: unknown policy {row["policy"]}')
            if row['admitted'] + row['rejected'] != row['requests']:
                raise ValueError(f'{path}:{line_no}: admissions + rejections != requests')
            if sum(row[n] for n in ('normal_capacity_rejections',
                   'postfault_capacity_rejections','no_plan_rejections')) != row['rejected']:
                raise ValueError(f'{path}:{line_no}: rejection reasons do not sum')
            if not math.isclose(row['admission_ratio'],row['admitted']/row['requests'],abs_tol=5.1e-6):
                raise ValueError(f'{path}:{line_no}: incorrect admission ratio')
            if not (0 <= row['weighted_admission_ratio'] <= 1):
                raise ValueError(f'{path}:{line_no}: weighted admission outside [0,1]')
            yield row


def ci(values, clamp=None):
    """Two-sided 95% Student t CI across independent demand seeds."""
    vals = [float(v) for v in values]
    if not vals:
        return (None,None,None,None)
    m = mean(vals)
    if len(vals) == 1:
        return (m,None,None,None)
    sd = stdev(vals)
    err = student_t.ppf(0.975, df=len(vals)-1) * sd / math.sqrt(len(vals))
    lo, hi = m - err, m + err
    if clamp is not None:
        lo = max(clamp[0], lo)
        hi = min(clamp[1], hi)
    return (m, sd, lo, hi)


def outcsv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fpercent(value, digits=2):
    return 'n/a' if value is None else f'{100 * value:.{digits}f}%'


def fnum(value, digits=2):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='outputs/phase4', help='Search recursively for phase4_runs.csv')
    ap.add_argument('--inputs', nargs='+', type=Path, help='Instead of --root, read the listed phase4_runs CSVs')
    ap.add_argument('--out', default='outputs/phase4_statistics', help='New output directory')
    args = ap.parse_args()

    files = sorted(set(args.inputs or Path(args.root).rglob('phase4_runs.csv')))
    if not files:
        raise SystemExit('No phase4_runs.csv files found. Check --root or provide --inputs.')

    unique = {}
    duplicates = []
    for path in files:
        for row in read_csv(path):
            key = tuple(row[c] for c in KEYS)
            if key in unique:
                first = unique[key]
                conflicting = [c for c in CHECK_EQ if row[c] != first[c]
                               and not (isinstance(row[c],float) and isinstance(first[c],float)
                                        and math.isclose(row[c], first[c], rel_tol=0, abs_tol=1e-6))]
                if conflicting:
                    raise ValueError(f'Conflicting repeat runs for {key}: {conflicting}. '
                                     f'Files: {first["_source_file"]}; {row["_source_file"]}. '
                                     'Separate differing versions before analysis.')
                duplicates.append((key,str(path)))
                continue
            unique[key] = row
    rows = list(unique.values())
    rows.sort(key=lambda r: tuple(r[c] for c in KEYS))
    groups = defaultdict(dict)
    for row in rows:
        gkey = tuple(row[c] for c in GROUP_KEYS) + (row['demand_seed'],)
        groups[gkey][row['policy']] = row
    for gkey, bypolicy in groups.items():
        if set(bypolicy) != set(POLICIES):
            raise ValueError(f'Incomplete paired policies for {gkey}: {list(bypolicy)}')
        hashes = {r['request_sha256'] for r in bypolicy.values()}
        if len(hashes) != 1:
            raise ValueError(f'Unpaired request datasets for {gkey}: {hashes}')

    condition = defaultdict(list)
    for row in rows:
        condition[tuple(row[k] for k in GROUP_KEYS) + (row['policy'],)].append(row)
    cond_out = []
    for groupkey, rr in sorted(condition.items()):
        base = dict(zip(GROUP_KEYS + ('policy',),groupkey))
        ad = ci([r['admission_ratio'] for r in rr],clamp=(0,1))
        wt = ci([r['weighted_admission_ratio'] for r in rr],clamp=(0,1))
        ms = ci([r['avg_selection_ms'] for r in rr])
        cond_out.append({**base,'seeds':len(rr),'seed_ids':','.join(str(r['demand_seed']) for r in rr),
                         'total_offered':sum(r['requests'] for r in rr),
                         'total_admitted':sum(r['admitted'] for r in rr),
                         'total_normal_rejections':sum(r['normal_capacity_rejections'] for r in rr),
                         'total_postfault_rejections':sum(r['postfault_capacity_rejections'] for r in rr),
                         'total_no_plan_rejections':sum(r['no_plan_rejections'] for r in rr),
                         'sum_fallbacks':sum(r['fallback_count'] for r in rr),
                         'sum_solver_nonoptimal':sum(r['solver_nonoptimal_count'] for r in rr),
                         **dict(zip(('mean_admission','sd_admission','ci95_admission_low',
                                     'ci95_admission_high'),ad)),
                         **dict(zip(('mean_weighted','sd_weighted','ci95_weighted_low',
                                     'ci95_weighted_high'),wt)),
                         **dict(zip(('mean_selection_ms','sd_selection_ms','ci95_selection_ms_low',
                                     'ci95_selection_ms_high'),ms))})

    pair_rows=[]
    pair_map=defaultdict(list)
    for key, pp in sorted(groups.items()):
        g=pp['greedy']
        for policy in ('cspf','milp'):
            p=pp[policy]
            result=dict(zip(GROUP_KEYS+('demand_seed',),key))
            result.update(policy=policy,baseline='greedy',
                          delta_admitted=p['admitted']-g['admitted'],
                          delta_admission_ratio=p['admission_ratio']-g['admission_ratio'],
                          delta_weighted_ratio=p['weighted_admission_ratio']-g['weighted_admission_ratio'],
                          delta_selection_ms=p['avg_selection_ms']-g['avg_selection_ms'],
                          ratio_selection_ms=(p['avg_selection_ms']/g['avg_selection_ms']
                                              if g['avg_selection_ms'] > 0 else None),
                          same_final_state=(p['state_digest']==g['state_digest']))
            pair_rows.append(result)
            pair_map[tuple(result[k] for k in GROUP_KEYS)+(policy,)].append(result)
    paired_sum=[]
    for key, rr in sorted(pair_map.items()):
        admission=ci([r['delta_admission_ratio'] for r in rr])
        weighted=ci([r['delta_weighted_ratio'] for r in rr])
        timing=ci([r['delta_selection_ms'] for r in rr])
        same=all(r['delta_admission_ratio'] == 0 for r in rr)
        paired_sum.append({**dict(zip(GROUP_KEYS+('policy',),key)),
                           'baseline':'greedy','seeds':len(rr),
                           'seeds_with_positive_admission_difference':sum(r['delta_admission_ratio']>0 for r in rr),
                           'seeds_with_negative_admission_difference':sum(r['delta_admission_ratio']<0 for r in rr),
                           'seeds_with_same_final_state':sum(r['same_final_state'] for r in rr),
                           'all_observed_admission_differences_zero':same,
                           **dict(zip(('mean_delta_admission','sd_delta_admission',
                                       'ci95_delta_admission_low','ci95_delta_admission_high'),admission)),
                           **dict(zip(('mean_delta_weighted','sd_delta_weighted',
                                       'ci95_delta_weighted_low','ci95_delta_weighted_high'),weighted)),
                           **dict(zip(('mean_delta_selection_ms','sd_delta_selection_ms',
                                       'ci95_delta_selection_ms_low','ci95_delta_selection_ms_high'),timing))})

    destination=Path(args.out)
    destination.mkdir(parents=True,exist_ok=True)
    outcsv(destination/'master_runs.csv',rows, list(rows[0]))
    outcsv(destination/'condition_summary.csv',cond_out,list(cond_out[0]))
    outcsv(destination/'paired_seed_differences.csv',pair_rows,list(pair_rows[0]))
    outcsv(destination/'paired_summary.csv',paired_sum,list(paired_sum[0]))

    look={(x['nodes'],x['requests'],x['headroom_percent'],x['policy']):x
          for x in cond_out if x['topology_seed']==42 and x['max_candidates']==2}

    lines=[
        '# Phase 4 — Final seed-level statistical analysis', '',
        '## Scope', '',
        f'- Input files scanned: **{len(files)}**.',
        f'- Original rows: **{len(rows)+len(duplicates)}**; unique matched policy runs: **{len(rows)}**; '
            f'duplicate exports removed: **{len(duplicates)}**.',
        f'- Unique paired seed/condition groups: **{len(groups)}**.',
        '- All three algorithms were present for every retained condition, and their request hashes matched within each paired seed.',
        '- No repeated-export timings are treated as independent observations.',
        '- Main inference unit: **demand seed / whole sequential request stream**, not the 25–100 individual requests.',
        '- Topology seed is fixed at 42 in the illustrated benchmark: these CIs do **not** include uncertainty across different network topologies.',
        '', '## Primary tables', '',
    ]
    def section(title,settings):
        lines.extend([f'### {title}', '',
                      '| Condition | Policy | Seeds | Admission mean (95% t-CI) | Weighted admission | Selection ms/request |',
                      '|---|---|---:|---|---:|---:|'])
        for label,n,requests,h in settings:
            for policy in POLICIES:
                row=look.get((n,requests,h,policy))
                if row is None: continue
                if row['seeds']<2:
                    ci_str=f'{fpercent(row["mean_admission"])} (CI unavailable; n=1)'
                else:
                    ci_str=(f'{fpercent(row["mean_admission"])} '
                            f'[{fpercent(row["ci95_admission_low"])}, {fpercent(row["ci95_admission_high"])}]')
                lines.append(f'| {label} | {policy} | {row["seeds"]} | {ci_str} | '
                             f'{fpercent(row["mean_weighted"])} | {fnum(row["mean_selection_ms"])} |')
        lines.append('')
    section('60-node offered-load sensitivity, 10% headroom',
            [('25 requests',60,25,10),('50 requests',60,50,10),('100 requests',60,100,10)])
    section('60-node headroom sensitivity, 50 requests',
            [('0% headroom',60,50,0),('10% headroom',60,50,10),('20% headroom',60,50,20)])
    section('Network-size sensitivity, 50 requests, 10% headroom',
            [('60 nodes',60,50,10),('100 nodes',100,50,10),('150 nodes',150,50,10)])

    lines.extend(['## Paired comparisons: algorithm minus greedy', '',
                  '| Network | Requests | Headroom | Policy | Seeds | Admission difference (95% t-CI), pp | Positive / negative seeds |',
                  '|---:|---:|---:|---|---:|---|---:|'])
    for x in paired_sum:
        if x['nodes'] == 12 or x['requests'] == 15:continue
        m=x['mean_delta_admission']
        low=x['ci95_delta_admission_low'];high=x['ci95_delta_admission_high']
        if low is None: interval='n/a'
        else: interval=f'{100*m:+.2f} [{100*low:+.2f}, {100*high:+.2f}]'
        lines.append(f'| {x["nodes"]} | {x["requests"]} | {x["headroom_percent"]}% | '
                     f'{x["policy"]} | {x["seeds"]} | {interval} | '
                     f'{x["seeds_with_positive_admission_difference"]} / '
                     f'{x["seeds_with_negative_admission_difference"]} |')
    lines.extend(['', '## Interpretation rules and limitations', '',
       '1. **Paired comparisons are essential.** Compare each CSPF-style/MILP run to greedy using exactly the same topology, headroom, request count, demand seed and request SHA-256.',
       '2. **Zero observed paired differences are not proof of statistical equivalence.** If every observed difference is zero, the generated t interval is [0,0] mechanically; this is a descriptive property of the sample and not a valid equivalence test.',
       '3. **For fewer than two seeds no confidence interval is reported.** The one-seed 12-node and early 100-node development results are smoke tests, not inferential evidence.',
       '4. **Small n / sparse differences** make t intervals approximate. Do not interpret wide intervals as confirming superiority or infer equivalence from a nonsignificant difference.',
       '5. **Only one topology seed** was used per network size. The comparisons describe a fixed synthetic configuration and do not generalize over all possible physical architectures.',
       '6. **Fixed requests across network sizes** do not represent equal load relative to network capacity. Interpret network-size trends as sensitivity under fixed offered-request counts, not isolated causal effects of scale.',
       '7. **Runtime** reflects this implementation, computer and workload. It is selection time, not deployment switching time or end-to-end protection recovery time.',
       '8. **Request prefix validation** between 25-, 50- and 100-request experiments requires the saved per-run requests.csv files. The aggregate phase4_runs.csv hashes alone cannot prove prefix identity.',
       '9. **Decision-divergence case studies** (60 nodes, headroom 20%, seed 2028; 150 nodes, headroom 10%, seed 2029) require per-request decisions.csv and accepted_protection_plans.json to identify the first divergent routing choice.',
       '10. **Protection scope** remains RING_SEGMENT_SCENARIO_SURROGATE_V1_NOT_MSRP: no true MSRP protocol signalling, node failure, simultaneous failures or restoration timing.',
       '', '## Next actions before Phase 5', '',
       'Archive the source code, configurations, all baseline run outputs, test logs and this analysis. Ask the supervisor to approve the protection-model scope and two-candidate action-space limitation. Define separate DDQN training, validation and held-out test seeds. On the held-out test seeds, re-run all classical baselines alongside the final locked DDQN policy with exactly the same requests and feasibility validator.',
       '' ])
    (destination/'phase4_statistical_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (destination/'audit_log.txt').write_text('\n'.join([
       f'FILES_SCANNED={len(files)}',f'ORIGINAL_RUNS={len(rows)+len(duplicates)}',
       f'UNIQUE_RUNS={len(rows)}',f'DUPLICATE_EXPORTS_IGNORED={len(duplicates)}',
       f'PAIRED_CONDITIONS={len(groups)}','ALL_INTEGRITY_CHECKS=PASS',
       'WARNING=Aggregate CSV cannot verify request prefix equivalence between different offered loads.',
       'WARNING=Confidence intervals use demand seeds only; topology variability is not sampled.',
       'WARNING=Observed all-zero paired difference is not a formal equivalence test.',
       'FILES:']+list(map(str,files))),encoding='utf-8')
    print(f'PASS: {len(files)} CSV files, {len(rows)} unique runs, '
          f'{len(duplicates)} duplicate exports skipped, {len(groups)} paired seed-conditions.')
    print(f'Outputs: {destination.resolve()}')
    for n,req,h in [(60,25,10),(60,50,10),(60,100,10),(100,50,10),(150,50,10)]:
        r=look.get((n,req,h,'greedy'))
        if r:print(f'{n} nodes, {req} requests, headroom {h}%: '
                   f'{r["seeds"]} seeds, greedy admission {fpercent(r["mean_admission"])}')

if __name__=='__main__':
    main()
