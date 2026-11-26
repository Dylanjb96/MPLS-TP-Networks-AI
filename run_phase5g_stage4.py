"""Phase 5G Stage 4: preflight, development smoke/scale, gated final, analysis.

Final is deliberately unavailable in this engineering package until an
independent audit has approved a hash-specific external release manifest.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from phase5g_stage4.protocol import CELLS,FINAL_TOPOLOGIES,FINAL_STREAMS,semantic_lock
from phase5g_stage4.evaluation import (freeze_lock,models_load,source_hashes,run_batches,
    summary_rows,write_dev_summary,release_check,ROW_FIELDS,replay_selected_validation)
from phase5g_stage4.statistics import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='cmd',required=True)
    sp.add_parser('preflight')
    a=sp.add_parser('smoke')
    a.add_argument('--out',default='outputs/phase5g/stage4_dev_smoke')
    a.add_argument('--resume',action='store_true')
    b=sp.add_parser('scale-dev')
    b.add_argument('--out',default='outputs/phase5g/stage4_scale_dev')
    b.add_argument('--resume',action='store_true')
    r=sp.add_parser('validation-replay')
    r.add_argument('--out',default='outputs/phase5g/stage4_validation_replay_dev')
    c=sp.add_parser('final')
    c.add_argument('--out',default='outputs/phase5g/stage4_final')
    c.add_argument('--resume',action='store_true')
    c.add_argument('--confirm-final',action='store_true')
    c.add_argument('--audit-release',default='stage4_AUDIT_RELEASE.json')
    d=sp.add_parser('report-final')
    d.add_argument('--root',default='outputs/phase5g/stage4_final')
    d.add_argument('--out',default='outputs/phase5g/stage4_final_report')
    d.add_argument('--audit-release',default='stage4_AUDIT_RELEASE.json')
    args=p.parse_args()
    if args.cmd=='preflight':
        manifest=freeze_lock()
        models=models_load()
        assert set(models)=={'gnn_PPO','mlp_PPO','C_frozen'}
        print('PHASE5G_STAGE4_PREFLIGHT_OK source_files=15 frozen_models=3 '+
              'semantic_sha256='+semantic_lock()+' final_seeds_accessed=False')
    elif args.cmd in ('smoke','scale-dev'):
        smoke=args.cmd=='smoke'
        rows=run_batches(args.out,role='dev',nodes=60 if smoke else 500,
            cell_ids=tuple(c.id for c in CELLS) if smoke else ('H_BASE',),
            topologies=(501,),streams=(74000,),resume=args.resume)
        folder=Path(args.out)
        if not (folder/'development_summary.csv').is_file():write_dev_summary(folder)
        print('PHASE5G_STAGE4_DEV_SMOKE_OK' if smoke else 'PHASE5G_STAGE4_SCALE_DEV_OK',
              f'rows={len(rows)} final_seeds_accessed=False')
    elif args.cmd=='validation-replay':
        replay_selected_validation(args.out)
    elif args.cmd=='final':
        if not args.confirm_final:raise ValueError('final requires --confirm-final after independent audit')
        release_check(args.audit_release,role='final')
        run_batches(args.out,role='final',nodes=500,
            cell_ids=tuple(c.id for c in CELLS),topologies=FINAL_TOPOLOGIES,
            streams=FINAL_STREAMS,resume=args.resume)
        print('PHASE5G_STAGE4_FINAL_RESULT_READY execute report-final using the same audit release')
    elif args.cmd=='report-final':
        release_check(args.audit_release,role='final')
        rows=summary_rows(args.root,require_complete=True)
        settings=json.loads((Path(args.root)/'run_manifest.json').read_text(encoding='utf8'))
        if settings['role']!='final':raise ValueError('refuses analysis on development data')
        stats=analyze(rows)
        out=Path(args.out)
        if out.exists():raise FileExistsError('Refusing to overwrite final statistical report')
        out.mkdir(parents=True,exist_ok=False)
        def save_csv(name,records,fields=None):
            with (out/name).open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=fields or list(records[0]));w.writeheader();w.writerows(records)
        save_csv('complete_policy_rows.csv',rows,ROW_FIELDS)
        save_csv('primary_cluster_inference.csv',stats['primary'])
        save_csv('secondary_cluster_inference.csv',stats['secondary'])
        save_csv('exploratory_robustness.csv',stats['exploratory'])
        save_csv('policy_summary.csv',stats['summary'])
        (out/'report_manifest.json').write_text(json.dumps({
          'role':'PHASE5G_STAGE4_LOCKED_CONFIRMATORY_ANALYSIS',
          'source_manifest':settings,'n_policy_rows':len(rows),
          'n_topology_clusters':20,'primary':stats['primary'],
          'secondary':stats['secondary'],'exploratory':stats['exploratory']},
           indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
        print('PHASE5G_STAGE4_ANALYSIS_COMPLETE cohorts=480 policy_rows=2880 '+
              'primary_tests=2 secondary_tests=8 statistical_unit=20_topologies')

if __name__=='__main__':main()
