"""Generate auditable *training-only* offline transitions from simulated behavior.

Offline learning NEVER invokes the environment for optimization updates. Each
behavior-policy trajectory begins with an empty reservation state; candidate
sets are generated using the same Phase3 code as all conventional baselines.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from railway_tp.topology import TopologyConfig
from phase3.model import SCHEME
from phase5.environment import FEATURE_NAMES, FEATURE_VERSION, OnlineEpisode,prepare_stream
from phase5.experiment import new_output,dump_json
from . import VERSION

EXCLUDED = set(range(2027,2042)) | set(range(31000,31030)) | set(range(41000,41030)) | set(range(51000,51030))


def forbidden(*groups) -> None:
    all_sets=[set(g) for g in groups]
    for i,a in enumerate(all_sets):
        if a & EXCLUDED:
            raise ValueError('train/validation includes already examined or reserved test seeds')
        if any(a & b for b in all_sets[i+1:]):
            raise ValueError('train/validation overlap')


def checksum(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(cfg:TopologyConfig,out:Path,*,requests=100,headroom=10,k=2,
                  seed_start=10000,seed_count=80,policies=('greedy','random_feasible')):
    seeds=range(seed_start,seed_start+seed_count)
    forbidden(seeds)
    if min(requests,k,seed_count)<1 or set(policies)!={'greedy','random_feasible'}:
        raise ValueError('behavior mix must contain greedy and random_feasible')
    new_output(out)
    entries={name:[] for name in ('rows','mask','action','reward','next_rows','next_mask','done',
                                 'demand_seed','behavior_id')}
    hashes={}
    counts={'greedy':0,'random_feasible':0}; feasible_hist={0:0,1:0,2:0}
    for seed in seeds:
        prep=prepare_stream(cfg,requests,seed,k)
        hashes[str(seed)]=prep.request_sha256
        for policy_index,policy in enumerate(policies):
            env=OnlineEpisode(cfg,prep,headroom,k)
            rng=random.Random(981733+seed*13+policy_index)
            while not env.done:
                obs=env.observe()
                legal=np.flatnonzero(obs.mask).tolist()
                action=int(legal[0] if policy=='greedy' else rng.choice(legal))
                reward,done,_=env.step(action)
                nxt=env.observe() if not done else None
                entries['rows'].append(obs.rows.copy())
                entries['mask'].append(obs.mask.copy())
                entries['action'].append(action)
                entries['reward'].append(float(reward))
                entries['next_rows'].append(nxt.rows.copy() if nxt else np.zeros_like(obs.rows))
                entries['next_mask'].append(nxt.mask.copy() if nxt else np.eye(1,k+1,k,dtype=bool)[0])
                entries['done'].append(done)
                entries['demand_seed'].append(seed)
                entries['behavior_id'].append(policy_index)
                feasible_hist[min(2,int(obs.mask[:k].sum()))]+=1
                counts[policy]+=1
        print(f'phase5c dataset seed={seed}  trajectories={len(policies)}',flush=True)
    arrays={key:np.asarray(val,dtype={'rows':np.float32,'mask':np.bool_,
        'action':np.int64,'reward':np.float32,'next_rows':np.float32,
        'next_mask':np.bool_,'done':np.bool_,'demand_seed':np.int64,
        'behavior_id':np.int64}[key]) for key,val in entries.items()}
    if not np.all(arrays['mask'][np.arange(len(arrays['action'])),arrays['action']]):
        raise AssertionError('logged illegal action')
    np.savez_compressed(out/'offline_transitions.npz',**arrays)
    meta={'phase5c_version':VERSION,'dataset_role':'TRAINING_ONLY',
          'source_model':SCHEME,'protection_model_approved':False,
          'topology':asdict(cfg),'headroom':headroom,'max_candidates':k,
          'requests_per_seed':requests,'seeds':list(seeds),'stream_sha256':hashes,
          'behaviors':list(policies),'behavior_counts':counts,
          'feasible_candidate_histogram':feasible_hist,
          'rows':len(arrays['action']),'feature_names':list(FEATURE_NAMES),
          'feature_version':FEATURE_VERSION}
    meta['npz_sha256']=checksum(out/'offline_transitions.npz')
    dump_json(out/'offline_dataset_manifest.json',meta)
    return meta


def load_dataset(directory:Path):
    manifest=json.loads((directory/'offline_dataset_manifest.json').read_text(encoding='utf8'))
    if checksum(directory/'offline_transitions.npz')!=manifest['npz_sha256']:
        raise ValueError('offline dataset sha256 mismatch')
    if manifest['feature_names']!=list(FEATURE_NAMES) or manifest['feature_version']!=FEATURE_VERSION:
        raise ValueError('offline dataset feature schema mismatch')
    data_file=np.load(directory/'offline_transitions.npz',allow_pickle=False)
    data={k:data_file[k] for k in data_file.files}
    n=len(data['action'])
    if n!=manifest['rows'] or not np.all(data['mask'][np.arange(n),data['action']]):
        raise ValueError('invalid recorded transitions')
    if np.any(~data['mask'].any(axis=1)) or np.any(~data['next_mask'].any(axis=1)):
        raise ValueError('invalid empty next/action masks')
    return data,manifest
