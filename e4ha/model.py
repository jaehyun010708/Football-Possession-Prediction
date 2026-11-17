"""Original E4HA Net and window inference from the uploaded sub_13 source."""
from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from settings import ROOT
sys.path.insert(0,str(ROOT/'third_party'))
from models_new import FrameEncoder
T=50;NMAX=26


class Net(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.enc=FrameEncoder(10,d,True)
        self.gru=nn.GRU(2*d,d,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
        self.head=nn.Linear(2*d,3)
        self.bhead=nn.Linear(2*d,1)
    def forward(self,f,m,b):
        B,T_,N,F=f.shape
        x,_,_=self.enc(f.reshape(B*T_,N,F),m.reshape(B*T_,N),b.reshape(B*T_,3))
        h,_=self.gru(x.reshape(B,T_,-1))
        return self.head(h),self.bhead(h).squeeze(-1)


def inputs(X,Y,team):
    """X,Y (time,slots) metres; team=-1 home, +1 away, 0 padding."""
    n=len(X)
    hl=np.nanmean(X[:,team==-1])<np.nanmean(X[:,team==1])
    side=np.where(team==-1,-1. if hl else 1.,1. if hl else -1.)
    feat=np.zeros((n,NMAX,7),np.float32)
    pxy=np.zeros((n,NMAX,2),np.float32)
    mask=np.zeros((n,NMAX),bool);cnt=np.zeros(n,int)
    for j in np.flatnonzero(team!=0):
        x,y=X[:,j].astype(float),Y[:,j].astype(float)
        vx=np.r_[0,np.diff(x)]/.2;vy=np.r_[0,np.diff(y)]/.2
        jump=np.hypot(vx,vy)>12;vx[jump]=np.nan;vy[jump]=np.nan
        vk=~np.isnan(vx);vx=np.nan_to_num(vx);vy=np.nan_to_num(vy)
        rows=np.flatnonzero(np.isfinite(x)&np.isfinite(y));rows=rows[cnt[rows]<NMAX];c=cnt[rows]
        feat[rows,c]=np.stack([np.full(len(rows),side[j]),x[rows]/52.5-1,y[rows]/34-1,
                               vx[rows]/8,vy[rows]/8,np.hypot(vx[rows],vy[rows])/8,vk[rows]],axis=1)
        pxy[rows,c]=np.stack([x[rows],y[rows]],axis=1);mask[rows,c]=True;cnt[rows]+=1
    return feat,pxy,mask,hl


@torch.no_grad()
def home_prob(model,feat,pxy,mask,hl,ball,vis,device):
    """Original overlap/taper/log-prob pooling, with native NaN sanitization."""
    vis=np.asarray(vis,bool)&np.isfinite(ball).all(axis=1)
    safe=np.nan_to_num(ball,nan=0,posinf=0,neginf=0)
    dx=np.where(vis[:,None],pxy[...,0]-safe[:,None,0],0)
    dy=np.where(vis[:,None],pxy[...,1]-safe[:,None,1],0)
    F=np.concatenate([feat,np.stack([dx/20,dy/20,np.hypot(dx,dy)/20],-1)*mask[...,None]],-1).astype(np.float32)
    B=np.stack([np.where(vis,safe[:,0]/52.5-1,0),np.where(vis,safe[:,1]/34-1,0),vis.astype(float)],axis=-1).astype(np.float32)
    n=len(F);lp=np.zeros((n,2));weights=np.zeros(n)
    starts=sorted(set(list(range(0,max(1,n-T+1),25))+[max(0,n-T)]))
    taper=np.minimum(np.arange(1,T+1),np.arange(T,0,-1)).astype(float)
    model.eval()
    for k in range(0,len(starts),64):
        ss=starts[k:k+64]
        feats=[];masks=[];balls=[]
        for s in ss:
            length=min(T,n-s)
            f=np.zeros((T,NMAX,10),np.float32);f[:length]=F[s:s+length]
            mk=np.zeros((T,NMAX),bool);mk[:length]=mask[s:s+length]
            b=np.zeros((T,3),np.float32);b[:length]=B[s:s+length]
            feats.append(f);masks.append(mk);balls.append(b)
        out,_=model(torch.from_numpy(np.stack(feats)).to(device),
                    torch.from_numpy(np.stack(masks)).to(device),
                    torch.from_numpy(np.stack(balls)).to(device))
        logp=Fn.log_softmax(out[...,:2],-1).cpu().numpy()
        for j,s in enumerate(ss):
            length=min(T,n-s)
            lp[s:s+length]+=logp[j,:length]*taper[:length,None]
            weights[s:s+length]+=taper[:length]
    if not np.all(weights>0):raise ValueError('Some E4 frames uncovered')
    left=np.exp(lp[:,0]/weights)
    return left if hl else 1-left


def load_model(path,device=None):
    dev=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net=Net().to(dev)
    net.load_state_dict(torch.load(str(path),map_location=dev,weights_only=True))
    net.eval()
    return net,dev
