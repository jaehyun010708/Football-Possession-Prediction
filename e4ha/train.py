"""E4HA: ball-input model that predicts only the team (H/A) of alive frames, trained on the middle label (gt_mid).
Model parts from mtah-e4 (third_party/models_new.py). Ball gaps are simulated with the new_data gap-length
distribution (nd_mask of the training matches). Epoch selection on J03WPY. Same as train_and_eval_combo.py with lab='mid'.
  python train/train_e4ha.py [--epochs N]    -> work/train_out/e4ha_mid.pt  (= models/e4ha_mid.zip)
"""
import argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from e4ha.common_train import D, M, TRAIN, VAL, LAB, OUT, dev, FIT_ALL
from settings import OUT as OUTPUT_ROOT
from models_new import FrameEncoder
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--mask-mode",choices=["fixed","dynamic"],default="fixed"); ap.add_argument("--windows-per-epoch",type=int,default=6000); args = ap.parse_args(); T0 = time.time()
LABS=[LAB]
T=50;NMAX=26
class Net(nn.Module):
    def __init__(s,d=64):
        super().__init__();s.enc=FrameEncoder(10,d,True);s.gru=nn.GRU(2*d,d,num_layers=2,batch_first=True,bidirectional=True,dropout=0.1);s.head=nn.Linear(2*d,3);s.bhead=nn.Linear(2*d,1)
    def forward(s,f,m,b):
        B,T_,N,F=f.shape;x,_,_=s.enc(f.reshape(B*T_,N,F),m.reshape(B*T_,N),b.reshape(B*T_,3));h,_=s.gru(x.reshape(B,T_,-1));return s.head(h),s.bhead(h).squeeze(-1)
def base_feats(z,ix):
    """same player tokens as mtah-e4 build_ds (the 3 ball columns are added later). left team side=-1"""
    X=z['xy'][ix,:,0].astype(float);Y=z['xy'][ix,:,1].astype(float);team=z['teams'];n=len(ix)
    hl=np.nanmean(X[:,team==-1])<np.nanmean(X[:,team==1]);side=np.where(team==-1,-1. if hl else 1.,1. if hl else -1.)
    feat=np.zeros((n,NMAX,7),np.float32);pxy=np.zeros((n,NMAX,2),np.float32);mask=np.zeros((n,NMAX),bool);cnt=np.zeros(n,int)
    for j in np.flatnonzero(team!=0):
        x,y=X[:,j],Y[:,j];vx=np.r_[0,np.diff(x)]/0.2;vy=np.r_[0,np.diff(y)]/0.2;jump=np.hypot(vx,vy)>12;vx[jump]=np.nan;vy[jump]=np.nan
        vk=~np.isnan(vx);vx=np.nan_to_num(vx);vy=np.nan_to_num(vy);rows=np.flatnonzero(~np.isnan(x));rows=rows[cnt[rows]<NMAX];c=cnt[rows]
        feat[rows,c]=np.stack([np.full(len(rows),side[j]),x[rows]/52.5-1,y[rows]/34-1,vx[rows]/8,vy[rows]/8,np.hypot(vx[rows],vy[rows])/8,vk[rows]],1)
        pxy[rows,c]=np.stack([x[rows],y[rows]],1);mask[rows,c]=True;cnt[rows]+=1
    return feat,pxy,mask,hl
def assemble(feat,pxy,mask,ball,vis):
    """add ball features. ball (..,2) metres, vis (..) bool"""
    dx=np.where(vis[...,None],pxy[...,0]-ball[...,None,0],0);dy=np.where(vis[...,None],pxy[...,1]-ball[...,None,1],0)
    F=np.concatenate([feat,np.stack([dx/20,dy/20,np.hypot(dx,dy)/20],-1)*mask[...,None]],-1).astype(np.float32)
    B=np.stack([np.where(vis,ball[...,0]/52.5-1,0),np.where(vis,ball[...,1]/34-1,0),vis.astype(float)],-1).astype(np.float32)
    return F,B
PARTS={}
for m in M:
    z=D[m];PARTS[m]=[]
    for p in (1,2):
        ix=np.flatnonzero(z['period']==p);f,pxy,mk,hl=base_feats(z,ix);e=dict(ix=ix,match=m,feat=f,pxy=pxy,mask=mk,hl=hl,ball=z['ball'][ix].astype(float),alive=z['gt_orig'][ix]!=2)
        for lab in LABS:
            g=z['gt_'+lab][ix];e['lab_'+lab]=np.where(g==2,2,np.where((g==0)==hl,0,1))     # 0 = left team, 1 = right team, 2 = dead
        PARTS[m].append(e)
GAPLEN=[]                                              # new_data gap lengths (training matches)
for m in TRAIN:
    a=D[m]['nd_mask'].astype(int);d=np.diff(np.r_[0,a,0]);GAPLEN+=list(np.flatnonzero(d==-1)-np.flatnonzero(d==1))
GAPLEN=np.array(GAPLEN)
if args.mask_mode=='dynamic' and not len(GAPLEN): raise ValueError('No gap lengths for dynamic mode')
if len(GAPLEN): print('gap length (5 Hz frames) median',np.median(GAPLEN),'max',GAPLEN.max(),flush=True)
def bursts(n,p,sampler,rng,pre=70):
    f=np.zeros(n+pre,bool)
    for s in np.flatnonzero(rng.random(n+pre)<p):f[s:s+sampler()]=True
    return f[pre:]
def change_targets(lab):
    b=np.zeros(len(lab),np.float32);near=np.zeros(len(lab),bool);ch=np.flatnonzero((lab[1:]!=lab[:-1])&(lab[1:]!=2)&(lab[:-1]!=2))+1
    for c in ch:b[max(0,c-1):c+1]=1;near[max(0,c-2):c+2]=True
    return b,near
def short_steps(lab,maxlen=25):
    s=np.zeros(len(lab),bool);i=0
    while i<len(lab):
        j=i
        while j<len(lab) and lab[j]==lab[i]:j+=1
        if lab[i]!=2 and j-i<maxlen:s[i:j]=True
        i=j
    return s
@torch.no_grad()
def e4_predict(model,e,vis):
    model.eval();F,B=assemble(e['feat'],e['pxy'],e['mask'],e['ball'],vis);n=len(F);lp=np.zeros((n,2));bl=np.zeros(n);w=np.zeros(n)
    st=sorted(set(list(range(0,max(1,n-T+1),25))+[max(0,n-T)]));tri=np.minimum(np.arange(1,T+1),np.arange(T,0,-1)).astype(float)
    for k in range(0,len(st),64):
        ss=st[k:k+64];o,b=model(*[torch.from_numpy(np.stack([a[s:s+T] for s in ss])).to(dev) for a in (F,e['mask'],B)])
        o=Fn.log_softmax(o[...,:2],-1).cpu().numpy();b=b.cpu().numpy()
        for j,s in enumerate(ss):lp[s:s+T]+=o[j]*tri[:,None];bl[s:s+T]+=b[j]*tri;w[s:s+T]+=tri
    model.train();return lp/w[:,None],bl/w
def e4_train(lab,tag,EP):
    torch.manual_seed(0);rng=np.random.default_rng(0);model=Net().to(dev);NW=args.windows_per_epoch
    sets=[e for m in TRAIN for e in PARTS[m]];aux=[change_targets(e['lab_'+lab]) for e in sets];idx=[];wts=[]
    for i,e in enumerate(sets):
        cs=np.r_[0,np.cumsum(short_steps(e['lab_'+lab]))];ca=np.r_[0,np.cumsum(e['alive'])]
        for s in range(0,len(e['alive'])-T+1,10):
            if ca[s+T]-ca[s]>=10:idx.append((i,s));wts.append(2.0 if cs[s+T]-cs[s]>0 else 1.0)
    wts=np.array(wts)/np.sum(wts);opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP);best=(1e9,None,-1)
    for ep in range(EP):
        sel_all=rng.choice(len(idx),NW,p=wts);tot=0;nb=0
        for k in range(0,NW,32):
            Fs,Ms,Bs,Ls,Bt,Nr=[],[],[],[],[],[]
            for i,s in (idx[j] for j in sel_all[k:k+32]):
                e=sets[i];sl=slice(s,s+T);mask=e['mask'][sl].copy();lab_=e['lab_'+lab][sl].copy()
                if args.mask_mode=='fixed':
                    gap=D[e['match']]['nd_mask'][e['ix'][sl]]
                else:
                    frac=rng.uniform(0.05,0.20);gap=bursts(T,frac/GAPLEN.mean(),lambda:int(rng.choice(GAPLEN)),rng)
                vis=e['alive'][sl] & np.isfinite(e['ball'][sl]).all(axis=1) & ~gap                                   # ball always hidden in dead frames (Sportec dead ball is static, unlike the competition)
                if rng.random()<0.7:                                       # simulate missing players: ~7% of player-frames, mean 6 s
                    for c in np.flatnonzero(mask.any(0)):mask[bursts(T,0.07/30,lambda:max(1,int(rng.exponential(30))),rng),c]=False
                F,B=assemble(e['feat'][sl],e['pxy'][sl],mask,e['ball'][sl],vis)
                if rng.random()<0.5:F[...,[0,1,3,7]]*=-1;B[:,0]*=-1;lab_=np.where(lab_==2,2,1-lab_)
                if rng.random()<0.5:F[...,[2,4,8]]*=-1;B[:,1]*=-1
                Fs.append(F);Ms.append(mask);Bs.append(B);Ls.append(lab_);Bt.append(aux[i][0][sl]);Nr.append(aux[i][1][sl])
            L_=torch.from_numpy(np.stack(Ls)).to(dev);o,bo=model(torch.from_numpy(np.stack(Fs)).to(dev),torch.from_numpy(np.stack(Ms)).to(dev),torch.from_numpy(np.stack(Bs)).to(dev))
            al=L_!=2;w=torch.from_numpy(np.where(np.stack(Nr),3.0,1.0).astype(np.float32)).to(dev)[al]
            ce=Fn.cross_entropy(o[...,:2][al],L_[al],reduction='none')
            loss=(ce*w).sum()/w.sum()+0.5*Fn.binary_cross_entropy_with_logits(bo,torch.from_numpy(np.stack(Bt)).to(dev),pos_weight=torch.tensor(5.0,device=dev))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();tot+=loss.item();nb+=1
        sched.step();vl=[];va=[]
        if FIT_ALL:
            vl=tot/nb; va=float('nan')
            best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep+1)
        else:
            val_ad=np.load(OUTPUT_ROOT/'ad_probs'/f'{VAL}.npz')['aux_alive']
            for e in PARTS[VAL]:
                vis=val_ad[e['ix']] & np.isfinite(e['ball']).all(axis=1) & ~D[VAL]['nd_mask'][e['ix']]
                lp,_=e4_predict(model,e,vis)
                a=e['alive'];y=e['lab_'+lab][a]
                vl.append(-lp[a,y]);va.append(lp[a].argmax(1)==y)
            vl=float(np.mean(np.concatenate(vl)));va=float(np.mean(np.concatenate(va)))
            if vl<best[0]:best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep+1)
        print(f'{tag} epoch {ep+1:02d} loss {tot/nb:.4f} val_ce {vl:.4f} val_team_acc {va:.4f} | {(time.time()-T0)/60:.0f} min',flush=True)
    model.load_state_dict(best[1]);torch.save(best[1],f'{OUT}/e4ha_{lab}.pt');print(tag,'best epoch',best[2],flush=True);return model,best[2]
print(f'==== E4HA training, label {LAB} ====',flush=True)
_,selected_epoch=e4_train(LAB,f'[E4HA {LAB}]',args.epochs)
(OUTPUT_ROOT/'models'/('fit-all' if FIT_ALL else 'validate')/'e4_best_epoch.txt').write_text(str(selected_epoch))
print(f'saved {OUT}/e4ha_{LAB}.pt  {(time.time()-T0)/60:.1f} min')
