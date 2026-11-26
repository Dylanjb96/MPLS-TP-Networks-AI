"""Select models using validation *only* across configuration x initialization runs."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
from dataclasses import asdict
import statistics
from phase5.experiment import new_output,dump_json,dump_csv
from . import VERSION,ALGORITHMS
from .dataset import checksum


def select_models(out:Path,runs:list[Path],min_seeds=3):
    if len(runs)<len(ALGORITHMS)*min_seeds:
        raise ValueError('need repeated training runs for all 5 algorithms')
    groups=defaultdict(list)
    source_data=None
    for run in runs:
        p=run/'training_provenance.json'; model=run/'best_model.pt'
        if not p.exists() or not model.exists():raise FileNotFoundError(run)
        d=json.loads(p.read_text(encoding='utf8'))
        algo=d['algorithm']
        if algo not in ALGORITHMS:raise ValueError('unexpected model '+algo)
        cfg=dict(d['config']);seed=cfg.pop('seed')
        regime=d.get('training_regime', 'on_policy' if algo=='ppo' else 'offline')
        key=(algo,json.dumps({'hyperparameters':cfg,'training_regime':regime},sort_keys=True))
        validation=tuple(d['validation_seeds'])
        training=tuple(d['training_seeds'])
        if source_data is None:source_data=(validation,training,d['topology'],d['requests_per_seed'],d['headroom'],d['max_candidates'])
        elif source_data!=(validation,training,d['topology'],d['requests_per_seed'],d['headroom'],d['max_candidates']):
            raise ValueError('incompatible validation/training cohorts or environments')
        groups[key].append({'path':str(model),'seed':seed,
            'score':tuple(d['best_validation_score']),'checkpoint_sha256':checksum(model),
            'training_provenance_sha256':checksum(p)})
    winners=[];trials=[]
    for algo in ALGORITHMS:
        options=[]
        for (family,key),v in groups.items():
            if family!=algo:continue
            if len({x['seed'] for x in v})!=len(v):raise ValueError('duplicate agent initialization')
            if len(v)<min_seeds:
                raise ValueError(f'{algo} config has {len(v)} seeds; need >= {min_seeds}')
            mean_w=statistics.mean(x['score'][0] for x in v)
            mean_a=statistics.mean(x['score'][1] for x in v)
            options.append((mean_w,mean_a,key,v))
            trials.append({'algorithm':algo,'config_json':key,'initializations':len(v),
                           'mean_validation_weighted':mean_w,
                           'mean_validation_admission':mean_a})
        if not options:raise ValueError('missing algorithm '+algo)
        _,_,key,records=max(options,key=lambda x:(round(x[0],12),round(x[1],12),x[2]))
        selected=max(records,key=lambda r:(r['score'][0],r['score'][1],-r['seed']))
        winners.append({'algorithm':algo,'config':json.loads(key)['hyperparameters'],
                        'training_regime':json.loads(key)['training_regime'],
                        'mean_validation_weighted':statistics.mean(x['score'][0] for x in records),
                        'mean_validation_admission':statistics.mean(x['score'][1] for x in records),
                        'selected_seed':selected['seed'],
                        'selected_checkpoint':selected['path'],
                        'checkpoint_sha256':selected['checkpoint_sha256'],
                        'selection_rule':'config by mean validation weighted score across seeds; selected checkpoint by validation weighted score'})
    new_output(out)
    dump_csv(out/'selection_trials.csv',trials)
    dump_json(out/'selection_manifest.json',{'version':VERSION,
       'rule':'validation only; test dataset never consulted',
       'training_seeds':list(source_data[1]),'validation_seeds':list(source_data[0]),
       'topology':source_data[2],'requests_per_seed':source_data[3],
       'headroom':source_data[4],'max_candidates':source_data[5],
       'min_confirm_initializations':min_seeds,'winners':winners})
    return {'winners':winners}
