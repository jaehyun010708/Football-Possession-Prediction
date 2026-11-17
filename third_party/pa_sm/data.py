"""Common NPZ contract used by Sportec and ATD loaders."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .augment import augment
from .features import make_features


LABELS={'H':0,'A':1,'D':2,'X':-100}
INV={0:'H',1:'A',2:'D',-100:'X'}


def save_match(path,xy,teams,frames,labels=None,period=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arr={'xy':np.asarray(xy,np.float32),'teams':np.asarray(teams,np.int8),
         'frames':np.asarray(frames,np.int64),
         'labels':np.asarray(labels if labels is not None else np.full(len(xy),-100),np.int64),
         'period':np.asarray(period if period is not None else np.zeros(len(xy)),np.int16)}
    assert len(arr['xy'])==len(arr['frames'])==len(arr['labels'])==len(arr['period'])
    np.savez_compressed(path,**arr)


def load_match(path):
    with np.load(path,allow_pickle=False) as d:
        out={k:d[k].copy() for k in ('xy','teams','frames','labels','period')}
    if out['xy'].shape[1]!=len(out['teams']):raise ValueError('slots do not match team IDs')
    return out


class WindowDataset(Dataset):
    def __init__(self,paths,cfg,train):
        self.matches=[load_match(p) for p in paths]
        self.cfg=cfg; self.train=train
        self.window=int(cfg['training']['window']); step=int(cfg['training']['train_stride'])
        self.index=[]
        for i,m in enumerate(self.matches):
            for per in np.unique(m['period']):
                ids=np.flatnonzero(m['period']==per)
                if not len(ids):continue
                # Blocks only: do not cross halftime or gaps
                starts=list(range(int(ids[0]),int(ids[-1])+1,step))
                last=max(int(ids[0]),int(ids[-1])-self.window+1)
                starts=sorted(set(starts+[last]))
                for s in starts:self.index.append((i,s,min(self.window,int(ids[-1])+1-s)))
        if not self.index:raise ValueError('No windows found')
    def __len__(self):return len(self.index)
    def __getitem__(self,k):
        import torch
        i,s,L=self.index[k]; m=self.matches[i]; w=self.window
        xy=m['xy'][s:s+L].copy(); teams=m['teams'].copy()
        if self.train:
            rng=np.random.default_rng(np.random.randint(0,2**31))
            xy,teams=augment(xy,teams,rng,self.cfg['augmentation'],(self.cfg['preprocessing']['pitch_x'],self.cfg['preprocessing']['pitch_y']))
        feat,mask=make_features(xy,teams,fps=self.cfg['preprocessing']['fps'],
                pitch=(self.cfg['preprocessing']['pitch_x'],self.cfg['preprocessing']['pitch_y']),
                vmax=self.cfg['preprocessing']['max_speed_mps'],radius=self.cfg['preprocessing']['pressure_radius_m'])
        indices=self.cfg['model'].get('feature_indices')
        if indices is not None:feat=feat[:,:,indices]
        y=m['labels'][s:s+L]
        if L<w:
            feat=np.pad(feat,((0,w-L),(0,0),(0,0)))
            mask=np.pad(mask,((0,w-L),(0,0)))
            y=np.pad(y,(0,w-L),constant_values=-100)
        return torch.from_numpy(feat),torch.from_numpy(mask),torch.from_numpy(y.copy())
