"""Generate train masks on GT Alive, holdout mask on predicted AD (no holdout GT)."""
from pathlib import Path
import json
import numpy as np
from settings import OUT,MATCHES
from ball_missing.mask_generator import generate_mask


def _mask_one(match,ref,seed=42):
    with np.load(OUT/'base'/f'{match}.npz') as z:
        n=len(z['frame']);nd=np.zeros(n,bool)
        report={};period=z['period']
        for p in (1,2):
            ix=np.flatnonzero(period==p)
            observed=np.isfinite(z['ball'][ix]).all(axis=1)
            syn,details=generate_mask(ref[ix],observed,z['frame'][ix],seed=seed+101*p, trials=12)
            nd[ix]=syn;report[f'p{p}']=details
    (OUT/'masks').mkdir(parents=True,exist_ok=True)
    np.save(OUT/'masks'/f'{match}_nd_mask.npy',nd)
    (OUT/'masks'/f'{match}_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(f'[mask] {match}: synthetic={nd.mean():.2%}; GT reference only for training matches',flush=True)
    return nd


def create(train_matches,validation_match=None):
    for i,m in enumerate(train_matches):
        with np.load(OUT/'base'/f'{m}.npz') as z:ref=z['gt_orig']!=2
        _mask_one(m,ref,seed=42+503*i)
    if validation_match:
        # AD prediction is generated only by models trained WITHOUT this match.
        with np.load(OUT/'ad_probs'/f'{validation_match}.npz') as p:
            ref=p['aux_alive'].astype(bool)
        _mask_one(validation_match,ref,seed=9921)


def assemble_dataset(matches):
    folder=OUT/'dataset';folder.mkdir(exist_ok=True,parents=True)
    for m in matches:
        with np.load(OUT/'base'/f'{m}.npz') as z:d={k:z[k] for k in z.files}
        d['nd_mask']=np.load(OUT/'masks'/f'{m}_nd_mask.npy')
        if len(d['nd_mask'])!=len(d['frame']):raise ValueError(f'Wrong mask length {m}')
        np.savez_compressed(folder/f'{m}.npz',**d)
