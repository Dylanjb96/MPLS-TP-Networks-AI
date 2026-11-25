"""Stage 5 frozen evidence: validates exact Stage 4 selection and both checkpoints.

The validation process never constructs or accesses a reserved held-out graph.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import torch
from phase5f_stage3.agent import CandidateActorCritic
from phase5f_stage4.ablation import CONFIGS,MaskedFeaturePolicy
from phase5f_stage4.workflow import _source_hashes

FROZEN=Path(__file__).resolve().parent.parent/'stage5_frozen'
SELECTION_NAME='selection_manifest.json'
MODELS={'C_enhanced':('selected','C_enhanced_seed27.pt','C_enhanced',27,60),
        'D_full_hybrid':('proposed_full_hybrid','D_full_hybrid_seed7.pt','D_full_hybrid',7,84)}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def verify_freeze():
    f=FROZEN/SELECTION_NAME
    if not f.is_file():raise FileNotFoundError('Stage 5 frozen selection missing: '+str(f))
    data=json.loads(f.read_text(encoding='utf8'))
    if (data.get('role')!='VALIDATION_ONLY_MODEL_SELECTION' or
        data.get('heldout_data_accessed') is not False or
        data.get('complete_factorial') is not True or
        data.get('training_runs')!=12 or
        data.get('required_agent_seeds')!=[7,17,27] or
        data.get('model_selection_cell')!={'nodes':500,'load':4.0,'mode':'dynamic','k':6}):
        raise ValueError('invalid Stage 4 selection manifest or non-frozen selection')
    current=_source_hashes()
    frozen=data['source_sha256']
    if current!=frozen:
        diff={k:(frozen.get(k),current.get(k)) for k in set(current)|set(frozen)
              if frozen.get(k)!=current.get(k)}
        raise RuntimeError('STAGE5 SOURCE LOCK MISMATCH (do not edit old stages): '+str(diff))
    for label,(manifest_key,filename,config,seed,ep) in MODELS.items():
        row=data[manifest_key]
        if (row['config']!=config or row['agent_seed']!=seed or row['best_episode']!=ep):
            raise ValueError('frozen selection identity mismatch: '+label)
        f=FROZEN/filename
        if not f.is_file() or sha256(f)!=row['checkpoint_sha256']:
            raise RuntimeError('frozen model SHA-256 mismatch: '+label)
    return data


def load_models():
    selection=verify_freeze()
    torch.set_num_threads(1)
    outputs={}
    for label,(key,filename,config,seed,ep) in MODELS.items():
        p=FROZEN/filename
        # Verify hash prior to torch.load. weights_only=True rejects arbitrary
        # pickled Python objects even if a malicious file appears on disk.
        chk=torch.load(p,map_location='cpu',weights_only=True)
        if (chk.get('config')!=config or chk.get('agent_seed')!=seed or
            chk.get('episode')!=ep or chk.get('hidden')!=64 or
            chk.get('max_actions')!=6 or chk.get('feature_dims')!={'candidate':17,'global':8} or
            chk.get('source_sha256')!=selection['source_sha256'] or
            chk.get('train_role')!='DEVELOPMENT_TRAINING_VALIDATION_ONLY' or
            chk.get('protection_model_approved') is not False):
            raise ValueError('frozen checkpoint metadata mismatch: '+label)
        model=CandidateActorCritic(hidden=64)
        model.load_state_dict(chk['state_dict'],strict=True)
        model.eval()
        outputs[label]=MaskedFeaturePolicy(model,CONFIGS[config])
    return outputs
