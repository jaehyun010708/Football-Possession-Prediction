import torch
from torch.nn import functional as F


def boundary_targets(labels):
    """[B,T] labels H=0 A=1 D=2; target at first frame of new run.
    Unknown and padding labels cannot form a target.
    """
    good=labels!=-100
    prev=labels[:,:-1];curr=labels[:,1:]
    ok=good[:,1:]&good[:,:-1]&(curr!=prev)
    ad=ok&((curr==2)|(prev==2))
    ha=ok&(curr!=2)&(prev!=2)
    z_ad=torch.zeros_like(labels,dtype=torch.float32)
    z_ha=torch.zeros_like(labels,dtype=torch.float32)
    z_ad[:,1:]=ad.float();z_ha[:,1:]=ha.float()
    return z_ad,z_ha


def multitask_loss(logits,y,cfg,return_terms=False):
    good=(y!=-100)
    if not torch.any(good):return logits.sum()*0
    alive=(y!=2).float()
    la=F.binary_cross_entropy_with_logits(logits[...,0][good],alive[good])
    ha=good&(y!=2)
    lh=F.binary_cross_entropy_with_logits(logits[...,1][ha],(y[ha]==0).float()) if ha.any() else la*0
    b_ad,b_ha=boundary_targets(y)
    bvalid=good.clone();bvalid[:,0]=False
    bvalid[:,1:] &= good[:,:-1]
    terms=[]
    for col,target in ((2,b_ad),(3,b_ha)):
        if bvalid.any():
            pos=target[bvalid].sum().item();neg=bvalid.sum().item()-pos
            cap=float(cfg['training']['max_boundary_pos_weight'])
            pos_weight=min(cap,max(1.,neg/max(pos,1.)))
            lb=F.binary_cross_entropy_with_logits(logits[...,col][bvalid],target[bvalid],pos_weight=torch.tensor(pos_weight,device=logits.device))
        else:lb=la*0
        terms.append(lb)
    total=la+lh+float(cfg['training']['boundary_weight'])*(terms[0]+terms[1])
    return (total,dict(AD=float(la.detach()),HA=float(lh.detach()),B_AD=float(terms[0].detach()),B_HA=float(terms[1].detach()))) if return_terms else total
