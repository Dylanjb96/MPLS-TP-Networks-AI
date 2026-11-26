"""Strict validation-only Stage 4 four-arm/three-seed model selection.

All twelve completed runs are required. Missing or inconsistent evidence is
an error: never silently select from a favourable subset of configurations.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import csv,hashlib,json
import numpy as np
from . import VERSION
from .ablation import CONFIGS
from .workflow import _csv_write,_json_write,_sha

REQUIRED_AGENT_SEEDS=(7,17,27)

def _readcsv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _group_signature(settings):
    excluded={'config','agent_seed','effective_bc_coef','effective_gate_threshold'}
    return {k:v for k,v in settings.items() if k not in excluded}


def select_models(runs_root,out,required_seeds=REQUIRED_AGENT_SEEDS):
    root=Path(runs_root);out=Path(out)
    if out.exists():raise FileExistsError('refuse to overwrite selection: '+str(out))
    required_seeds=tuple(required_seeds)
    if len(required_seeds)!=3 or len(set(required_seeds))!=3 or any(s<0 for s in required_seeds):
        raise ValueError('selection requires three distinct frozen model initializations')
    rows=[];source_hashes=None;signature=None;common_val_cohorts=None;common_baselines=None
    for config in CONFIGS:
        for seed in required_seeds:
            folder=root/f'{config}_seed{seed}'
            provenance=folder/'training_provenance.json'
            if not provenance.exists():
                raise FileNotFoundError('missing completed training run: '+str(provenance))
            m=json.loads(provenance.read_text())
            if m.get('role')!='DEVELOPMENT_TRAINING_VALIDATION_ONLY' or m.get('final_test_seeds_accessed') is not False:
                raise ValueError('non-development or heldout-contaminated run')
            if m['config']!=config or m['agent_seed']!=seed or m['version']!=VERSION:
                raise ValueError('manifest arm/seed/version mismatch '+str(folder))
            if m['protection_model_approved'] or m['msrp_compliant']:
                raise ValueError('model protection role metadata invalid')
            plan=m['settings'];key=_group_signature(plan)
            if signature is None:signature=key
            elif signature!=key:raise ValueError('training configuration settings not matched')
            if source_hashes is None:source_hashes=m['source_sha256']
            elif source_hashes!=m['source_sha256']:raise ValueError('different source hashes across runs')
            if common_val_cohorts is None:common_val_cohorts=m['validation_cohort_hashes']
            elif common_val_cohorts!=m['validation_cohort_hashes']:
                raise ValueError('validation cohort hashes differ across runs')
            checkpoint=folder/'best_model.pt'
            if not checkpoint.is_file() or _sha(checkpoint)!=m['checkpoint_sha256']:
                raise ValueError('checkpoint hash mismatch: '+str(checkpoint))
            t=_readcsv(folder/'training_history.csv')
            v=_readcsv(folder/'validation_summary.csv')
            checkpoints=_readcsv(folder/'validation_checkpoints.csv')
            if (len(t)!=m['train_rows'] or len(v)!=m['validation_rows'] or
                len(checkpoints)!=m['validation_evaluations'] or not t or not checkpoints):
                raise ValueError('incomplete training evidence '+str(folder))
            if any(float(r['feasibility_violations'])!=0 for r in t+v):
                raise ValueError('feasibility violation in training/validation')
            best=max(float(c['validation_weighted_hybrid']) for c in checkpoints)
            best_ep=min(int(c['episode']) for c in checkpoints
                        if abs(float(c['validation_weighted_hybrid'])-best)<1e-12)
            if abs(best-m['best_mean_weighted_validation'])>1e-9 or best_ep!=m['best_episode']:
                raise ValueError('best-checkpoint selection mismatch '+str(folder))
            bestrows=[r for r in v if int(r['episode'])==best_ep and r['policy']=='hybrid']
            expected=len(plan['val_topologies'])*len(plan['val_streams'])
            if len(bestrows)!=expected:
                raise ValueError('incomplete selected checkpoint validation rows')
            if abs(float(np.mean([float(r['weighted_admission']) for r in bestrows]))-best)>1e-9:
                raise ValueError('best checkpoint validation score does not reconcile')
            if len(set((r['topology_seed'],r['stream_seed']) for r in bestrows))!=expected:
                raise ValueError('duplicate/missing validation cohorts')
            baseline_rows={(r['topology_seed'],r['stream_seed'],r['policy']):
                (r['weighted_admission'],r['admitted'],r['state_digest'])
                for r in v if int(r['episode'])==best_ep and r['policy'] in ('greedy','footprint')}
            if len(baseline_rows)!=2*expected:raise ValueError('missing paired baseline cohorts')
            if common_baselines is None:common_baselines=baseline_rows
            elif common_baselines!=baseline_rows:raise ValueError('baseline reproducibility differs across runs')
            rows.append(dict(config=config,agent_seed=seed,checkpoint=str(checkpoint),
                  checkpoint_sha256=m['checkpoint_sha256'],
                  best_episode=best_ep,best_weighted=best,
                  weighted_greedy=float(np.mean([float(r['weighted_admission']) for r in v
                                                 if int(r['episode'])==best_ep and r['policy']=='greedy'])),
                  weighted_footprint=float(np.mean([float(r['weighted_admission']) for r in v
                                                 if int(r['episode'])==best_ep and r['policy']=='footprint'])),
                  train_episodes=m['completed_episodes'],
                  selected_val_overrides=sum(int(r['overrides']) for r in bestrows),
                  selected_val_discretionary=sum(int(r['discretionary']) for r in bestrows)))
    groups=[]
    for config in CONFIGS:
        rr=[r for r in rows if r['config']==config]
        if len(rr)!=len(required_seeds):raise AssertionError('missing initializations')
        groups.append(dict(config=config,initializations=len(rr),
          mean_best_weighted=float(np.mean([r['best_weighted'] for r in rr])),
          sample_sd_best_weighted=float(np.std([r['best_weighted'] for r in rr],ddof=1)),
          mean_greedy=float(np.mean([r['weighted_greedy'] for r in rr])),
          mean_footprint=float(np.mean([r['weighted_footprint'] for r in rr]))))
    # Strict tie-break: simpler, preregistered A -> B -> C -> D ordering.
    winner=sorted(groups,key=lambda x:(-x['mean_best_weighted'],list(CONFIGS).index(x['config'])))[0]
    chosen=sorted((r for r in rows if r['config']==winner['config']),
                 key=lambda r:(-r['best_weighted'],r['agent_seed']))[0]
    # Also preserve proposed D hybrid as a predetermined *ablation* even if
    # validation selects a different, simpler factorial arm.
    d_hybrid=sorted((r for r in rows if r['config']=='D_full_hybrid'),
                 key=lambda r:(-r['best_weighted'],r['agent_seed']))[0]
    out.mkdir(parents=True,exist_ok=False)
    _csv_write(out/'selection_trials.csv',rows)
    _csv_write(out/'selection_configurations.csv',groups)
    manifest=dict(version=VERSION,role='VALIDATION_ONLY_MODEL_SELECTION',
        complete_factorial=True,config_count=4,initializations_per_config=3,
        training_runs=12,train_root=str(root),
        selection_rule='highest mean of best validation-weighted admission across 3 initializations; tied configurations prefer A,B,C,D; within winning config highest individual score then lowest agent seed',
        heldout_data_accessed=False,no_retraining_after_final=True,
        selected=dict(chosen),proposed_full_hybrid=dict(d_hybrid),
        configs=groups,required_agent_seeds=list(required_seeds),
        model_selection_cell=signature['model_selection_cell'],
        shared_settings_signature=signature,source_sha256=source_hashes,
        qualification='Synthetic end-to-end link-disjoint single-link-failure surrogate, not MSRP or an actual Irish Rail infrastructure map.')
    _json_write(out/'selection_manifest.json',manifest)
    return manifest
