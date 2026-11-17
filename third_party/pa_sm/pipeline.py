"""Training and inference entry points. Files always stay beneath this package's outputs/."""
from pathlib import Path
import json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import WindowDataset,load_match,INV
from .features import make_features
from .model import PlayerAttentionGRU,probabilities
from .losses import multitask_loss
from .decoders import duration_prior,viterbi,semi_markov
from .metrics import report

ROOT=Path(__file__).resolve().parent.parent


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def _dataloader(paths,cfg,train):
    ds=WindowDataset(paths,cfg,train)
    return DataLoader(ds,batch_size=int(cfg['training']['batch_size']),shuffle=train,
                      num_workers=int(cfg['training']['num_workers']),pin_memory=torch.cuda.is_available())


def fit(cfg,train_paths,val_paths,run_name,device='auto'):
    if not train_paths or not val_paths:raise ValueError('Use nonempty train and validation file lists')
    def match_id(path):
        return Path(path).stem.rsplit('_p',1)[0]
    if {match_id(p) for p in train_paths} & {match_id(p) for p in val_paths}:
        raise ValueError('Train and validation must be split by MATCH, not by period/window')
    seed_everything(int(cfg['seed']))
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    train_loader=_dataloader(train_paths,cfg,True);val_loader=_dataloader(val_paths,cfg,False)
    model=PlayerAttentionGRU(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['lr']),weight_decay=float(cfg['training']['weight_decay']))
    run_dir=ROOT/'outputs'/run_name;run_dir.mkdir(parents=True,exist_ok=True)
    with open(run_dir/'config.json','w') as f:json.dump(cfg,f,indent=2)
    with open(run_dir/'splits.json','w') as f:json.dump({'train':list(map(str,train_paths)),'val':list(map(str,val_paths))},f,indent=2)
    prior=duration_prior(train_paths)
    with open(run_dir/'duration_prior.json','w') as f:json.dump(prior,f,indent=2)
    history=[];best=float('inf');bad=0
    for epoch in range(int(cfg['training']['epochs'])):
        record={'epoch':epoch+1}
        for mode,loader in (('train',train_loader),('val',val_loader)):
            model.train(mode=='train');losses=[]
            for features,mask,labels in loader:
                features=features.to(device);mask=mask.to(device);labels=labels.to(device)
                with torch.set_grad_enabled(mode=='train'):
                    outputs=model(features,mask);loss=multitask_loss(outputs,labels,cfg)
                    if mode=='train':
                        opt.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']))
                        opt.step()
                losses.append(float(loss.detach()))
            record[mode+'_loss']=float(np.mean(losses))
        history.append(record)
        print(f"epoch {epoch+1:03d} train {record['train_loss']:.4f} val {record['val_loss']:.4f}",flush=True)
        if record['val_loss']<best-1e-4:
            best=record['val_loss'];bad=0
            torch.save({'state_dict':model.state_dict(),'config':cfg,'prior':prior,
                        'train_paths':list(map(str,train_paths))},run_dir/'best.pt')
        else:bad+=1
        with open(run_dir/'history.json','w') as f:json.dump(history,f,indent=2)
        if bad>=int(cfg['training']['early_stop_patience']):break
    print('Best checkpoint:',run_dir/'best.pt')
    return run_dir/'best.pt'


def load_checkpoint(path,device='auto'):
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    ckpt=torch.load(path,map_location=dev,weights_only=False)
    model=PlayerAttentionGRU(ckpt['config']).to(dev)
    model.load_state_dict(ckpt['state_dict']);model.eval()
    return model,ckpt['config'],ckpt['prior'],dev


@torch.inference_mode()
def predict_arrays(model,cfg,xy,teams,device):
    """Full continuous period prediction by overlapped windows; no prediction across halves."""
    feat,mask=make_features(xy,teams,fps=cfg['preprocessing']['fps'],
                    pitch=(cfg['preprocessing']['pitch_x'],cfg['preprocessing']['pitch_y']),
                    vmax=cfg['preprocessing']['max_speed_mps'],radius=cfg['preprocessing']['pressure_radius_m'])
    indices=cfg['model'].get('feature_indices')
    if indices is not None:feat=feat[:,:,indices]
    T=len(feat);W=int(cfg['training']['window']);S=int(cfg['training']['inference_stride'])
    starts=list(range(0,max(T-W+1,1),S))
    starts=sorted(set(starts+[max(0,T-W)]))
    accum=np.zeros((T,4),np.float64);weights=np.zeros(T,np.float64)
    # Average HEAD probabilities, not logits or class labels.
    bs=int(cfg['training']['batch_size'])
    for ix in range(0,len(starts),bs):
        ss=starts[ix:ix+bs]
        ff=np.zeros((len(ss),W,feat.shape[1],feat.shape[2]),np.float32)
        mm=np.zeros((len(ss),W,mask.shape[1]),bool)
        for j,s in enumerate(ss):
            length=min(W,T-s);ff[j,:length]=feat[s:s+length];mm[j,:length]=mask[s:s+length]
        pred=model(torch.from_numpy(ff).to(device),torch.from_numpy(mm).to(device))
        prob=torch.sigmoid(pred).cpu().numpy()
        for j,s in enumerate(ss):
            L=min(W,T-s)
            taper=np.minimum(np.arange(1,L+1),np.arange(L,0,-1)).astype(float)
            taper=np.maximum(taper,1.)
            accum[s:s+L]+=prob[j,:L]*taper[:,None];weights[s:s+L]+=taper
    if np.any(weights==0):raise RuntimeError('Some frames not covered by inference windows')
    p=accum/weights[:,None]
    alive=p[:,0];home=p[:,1]
    state=np.stack((alive*home,alive*(1-home),1-alive),axis=-1)
    return state,p[:,2:4]


def infer_files(checkpoint,files,run_name,decoder='semi',device='auto',submission=False,decoder_override=None):
    model,cfg,prior,dev=load_checkpoint(checkpoint,device)
    if decoder_override is not None:cfg['decoder'].update(decoder_override)
    out_dir=ROOT/'outputs'/run_name;out_dir.mkdir(parents=True,exist_ok=True)
    if decoder not in ('raw','viterbi','semi'):raise ValueError(decoder)
    results=[]
    for f in files:
        m=load_match(f)
        state,bound=predict_arrays(model,cfg,m['xy'],m['teams'],dev)
        if decoder=='raw':out=np.argmax(state,axis=1).astype(np.int8)
        elif decoder=='viterbi':out=viterbi(state,cfg['decoder'])
        else:out=semi_markov(state,bound,cfg['decoder'],prior,int(cfg['preprocessing']['fps']))
        stem=Path(f).stem
        np.savez_compressed(out_dir/f'{stem}_pred.npz',frames=m['frames'],prob=state,boundary=bound,label=out)
        if np.any(m['labels']!=-100):
            scores=report(m['labels'],out,frames=m['frames'],fps=int(cfg['preprocessing']['fps']))
            with open(out_dir/f'{stem}_metrics.json','w') as fp:json.dump(scores,fp,indent=2)
            print(stem,decoder,scores)
        results.append((m['frames'],out))
    if submission:
        # For Barcelona: ATD frames already 0 based. Fill graded halves with 5FPS predictions.
        full=np.array(['X']*199319,dtype='<U1')
        halves=((11185,90651),(110808,190197))
        for frames,labels in results:
            for a,b in halves:
                ins=(frames>=a)&(frames<=b)
                selected=frames[ins];labs=labels[ins]
                if not len(selected):continue
                for j,fr in enumerate(selected):
                    end=int(selected[j+1]) if j+1<len(selected) else b+1
                    full[int(fr):end]=INV[int(labs[j])]
        import pandas as pd
        output=out_dir/'submission.csv'
        pd.DataFrame({'frame':np.arange(len(full)),'label':full}).to_csv(output,index=False)
        print('Wrote',output)
    return results
