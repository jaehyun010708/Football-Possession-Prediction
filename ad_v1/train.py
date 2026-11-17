"""Train player-only attention/GRU V1 using the original pa_sm implementation."""
from __future__ import annotations
import sys,shutil,json
from pathlib import Path
import numpy as np
import torch
from settings import OUT,ROOT

sys.path.insert(0,str(ROOT/'third_party'))
from pa_sm.pipeline import fit, predict_arrays, load_checkpoint, seed_everything, _dataloader
from pa_sm.data import save_match
from pa_sm.model import PlayerAttentionGRU
from pa_sm.losses import multitask_loss
from pa_sm.decoders import duration_prior

# Exactly the hyperparameters of train/train_v1.py in the uploaded ZIP.
CFG={'seed':42,'preprocessing':{'fps':5,'source_fps':25,'pitch_x':105.0,'pitch_y':68.0,'max_slots':46,'max_speed_mps':12.0,'pressure_radius_m':5.0},
 'augmentation':{'flip_x':True,'flip_y':True,'shuffle_tokens':True,'position_noise_m':0.0,'dropout_probability':0.0,'duplicate_probability':0.0},
 'model':{'feature_dim':11,'embed_dim':64,'heads':4,'attention_layers':2,'seeds':4,'frame_dim':128,'hidden_size':64,'gru_layers':2,'dropout':0.1},
 'training':{'window':100,'train_stride':50,'inference_stride':50,'batch_size':8,'epochs':30,'lr':0.0003,'weight_decay':0.0001,'boundary_weight':0.2,'max_boundary_pos_weight':20.0,'early_stop_patience':7,'num_workers':0,'grad_clip':1.0},
 'decoder':{'max_segment_seconds':180,'switch_cost_ha':2.0,'switch_cost_ad':2.5,'boundary_weight':0.4,'duration_weight':0.15,'duration_clip':5.0,'viterbi_switch_cost_ha':2.0,'viterbi_switch_cost_ad':2.5}}


def create_input_files(matches):
    paths={}
    for m in matches:
        paths[m]=[]
        with np.load(OUT/'base'/f'{m}.npz') as z:
            for p in (1,2):
                idx=np.flatnonzero(z['period']==p)
                dest=OUT/'cache'/'v1_input'/f'{m}_p{p}.npz'
                save_match(dest,z['xy'][idx],z['teams'],z['frame'][idx],z['gt_mid'][idx].astype(np.int64),z['period'][idx])
                paths[m].append(str(dest))
    return paths


def train(train_matches,validation_match,out,epochs=30,full=False):
    matches=list(train_matches)+([] if full else [validation_match])
    paths=create_input_files(matches)
    files=[p for m in train_matches for p in paths[m]]
    cfg=json.loads(json.dumps(CFG))
    cfg['training']['epochs']=int(epochs)
    if torch.cuda.is_available():cfg['training']['num_workers']=2
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if not full:
        # Validation match cannot overlap training matches (enforced in pa_sm.fit).
        best=fit(cfg,files,paths[validation_match],'barcelona_auto_validate','auto')
        shutil.copy2(best,out)
        ckpt=torch.load(out,map_location='cpu',weights_only=False)
        history=json.loads((Path(best).parent/'history.json').read_text())
        best_epoch=int(min(history,key=lambda x:x['val_loss'])['epoch'])
        return best_epoch
    # Fit all seven with a pre-fixed epoch count; do not re-use the holdout for early stopping.
    seed_everything(int(cfg['seed']))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dl=_dataloader(files,cfg,True)
    model=PlayerAttentionGRU(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    for ep in range(int(epochs)):
        model.train();losses=[]
        for x,mask,y in dl:
            x,mask,y=x.to(device),mask.to(device),y.to(device)
            o=model(x,mask);loss=multitask_loss(o,y,cfg)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']))
            opt.step();losses.append(loss.item())
        print(f'[V1 fit-all] epoch {ep+1}/{epochs}: loss {np.mean(losses):.4f}',flush=True)
    prior=duration_prior(files)
    torch.save({'state_dict':model.state_dict(),'config':cfg,'prior':prior,
                'train_paths':files},out)
    return int(epochs)


def predict(checkpoint,match):
    model,cfg,_,dev=load_checkpoint(str(checkpoint),'auto')
    with np.load(OUT/'base'/f'{match}.npz') as z:
        n=len(z['frame']); pr=np.zeros((n,3),np.float32)
        for p in (1,2):
            ix=np.flatnonzero(z['period']==p)
            probs,_=predict_arrays(model,cfg,z['xy'][ix],z['teams'],dev)
            pr[ix]=probs
    return pr
