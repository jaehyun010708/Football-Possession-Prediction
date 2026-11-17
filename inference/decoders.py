"""Original sub_13 3-state Viterbi + a clearly labelled diagnostic spell proxy."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def vit3(pH,pA,pD,cha=2.,cad=2.5):
    l=np.log(np.clip(np.c_[pH,pA,pD],1e-7,1));n=len(l)
    if n==0:return np.array([],np.int8)
    dp=l[0].copy();bk=np.zeros((n,3),np.int8)
    pen=np.array([[0,cha,cad],[cha,0,cad],[cad,cad,0]])
    for t in range(1,n):
        v=dp[:,None]-pen;bk[t]=v.argmax(0);dp=l[t]+v.max(0)
    seq=np.zeros(n,np.int8);seq[-1]=dp.argmax()
    for t in range(n-1,0,-1):seq[t-1]=bk[t,seq[t]]
    return seq


def spells(y):
    y=np.asarray(y);ix=np.flatnonzero(np.r_[True,y[1:]!=y[:-1],True])
    return [(int(ix[i]),int(ix[i+1]),int(y[ix[i]])) for i in range(len(ix)-1)]


def spell_proxy(gt,pred,period):
    """Per-team 1:1 IoU matching; mean F1 over thresholds 0.3..0.7.
    Not a replacement for the competition's official evaluator.
    """
    vals=[];detail={}
    for threshold in (.3,.4,.5,.6,.7):
        tp=ng=np_=0
        for p in np.unique(period):
            ix=np.flatnonzero(period==p)
            for cls in (0,1):
                g=[(a,b) for a,b,c in spells(gt[ix]) if c==cls]
                q=[(a,b) for a,b,c in spells(pred[ix]) if c==cls]
                ng+=len(g);np_+=len(q)
                if not g or not q:continue
                iou=np.zeros((len(g),len(q)),float)
                for i,(a,b) in enumerate(g):
                    for j,(x,y) in enumerate(q):
                        inter=max(0,min(b,y)-max(a,x));union=max(b,y)-min(a,x)
                        iou[i,j]=inter/union if union else 0
                score=np.where(iou>=threshold,1000+iou,0)
                rr,cc=linear_sum_assignment(-score)
                tp+=sum(iou[r,c]>=threshold for r,c in zip(rr,cc))
        f1=2*tp/(ng+np_) if ng+np_ else 1.
        vals.append(f1);detail[str(threshold)]={'TP':int(tp),'GT_spells':ng,'Pred_spells':np_,'F1':round(float(f1),4)}
    return {'spell_f1_proxy':round(float(np.mean(vals)),5),'by_IoU':detail}
