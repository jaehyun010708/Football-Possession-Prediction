"""Holdout AD->ball visibility->E4HA->3-state Viterbi; compute diagnostics."""
from __future__ import annotations
import json
import numpy as np
from settings import OUT
from e4ha.model import load_model,inputs,home_prob
from inference.decoders import vit3,spell_proxy


def validate(match,mode='validate'):
    mpath=OUT/'models'/mode/'e4ha_mid.pt'
    model,device=load_model(mpath)
    with np.load(OUT/'dataset'/f'{match}.npz') as z:
        data={k:z[k] for k in z.files}
    with np.load(OUT/'ad_probs'/f'{match}.npz') as p:
        d=p['p_dead'];aux=p['aux_state']
    all_pred=np.zeros(len(d),np.int8)
    home_probs=np.zeros(len(d),np.float32)
    for half in (1,2):
        ix=np.flatnonzero(data['period']==half)
        xx=data['xy'][ix];ball=data['ball'][ix]
        f,pxy,mask,hl=inputs(xx[:,:,0],xx[:,:,1],data['teams'])
        vis=(aux[ix]!=2)&np.isfinite(ball).all(axis=1)&(~data['nd_mask'][ix])
        hp=home_prob(model,f,pxy,mask,hl,ball,vis,device)
        home_probs[ix]=hp
        all_pred[ix]=vit3((1-d[ix])*hp,(1-d[ix])*(1-hp),d[ix])
    gt=data['gt_mid']
    from inference.decoders import spells
    pred_ha=sum(sum(c in (0,1) for _,_,c in spells(all_pred[data['period']==p])) for p in (1,2))
    gt_ha=sum(sum(c in (0,1) for _,_,c in spells(gt[data['period']==p])) for p in (1,2))
    metrics={'match':match,'frame_accuracy':round(float(np.mean(all_pred==gt)),5),
             'pred_ha_spells':pred_ha,'gt_ha_spells':gt_ha,
             **spell_proxy(gt,all_pred,data['period'])}
    (OUT/'predictions').mkdir(exist_ok=True,parents=True)
    np.savez_compressed(OUT/'predictions'/f'{match}_validate.npz',frame=data['frame'],period=data['period'],
                        gt_mid=gt,pred=all_pred,p_dead=d,p_home_cond=home_probs)
    (OUT/'predictions'/f'{match}_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
    print('[validate] '+json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    return metrics
