"""Raw argmax, standard Viterbi and explicit-duration semi-Markov decoding.
All optimization is done separately inside each period; no labels/GT are read by decoders.
"""
import numpy as np


def duration_prior(paths):
    from .data import load_match
    lengths={k:[] for k in range(3)}
    for path in paths:
        m=load_match(path);y=m['labels'];per=m['period']
        if not len(y):continue
        start=0
        for t in range(1,len(y)+1):
            if t==len(y) or y[t]!=y[start] or per[t]!=per[start]:
                label=int(y[start]);run=t-start
                if label in lengths:lengths[label].append(run)
                start=t
    out={}
    for k,ls in lengths.items():
        a=np.log(np.maximum(ls,1.))
        out[str(k)]={'mean_log':float(np.mean(a)) if len(a) else np.log(25.),
                     'std_log':float(max(np.std(a),0.65)) if len(a)>1 else 1.2,
                     'count':len(ls)}
    return out


def _duration_score(lengths,k,prior,clip):
    pr=prior[str(k)]
    mu,sig=float(pr['mean_log']),max(0.5,float(pr['std_log']))
    # Soft prior, normalized to zero at the mode; capped to avoid killing short possessions.
    z=(np.log(np.maximum(lengths,1))-mu)/sig
    return -np.minimum(clip,0.5*z*z)


def viterbi(probs,cfg):
    probs=np.clip(np.asarray(probs,float),1e-7,1.)
    if len(probs)==0:return np.zeros(0,np.int64)
    T=len(probs);logs=np.log(probs)
    cost_ha=float(cfg['viterbi_switch_cost_ha']);cost_ad=float(cfg['viterbi_switch_cost_ad'])
    penalties=np.array([[0,cost_ha,cost_ad],[cost_ha,0,cost_ad],[cost_ad,cost_ad,0]])
    dp=logs[0].copy();back=np.zeros((T,3),np.int8)
    for t in range(1,T):
        values=dp[:,None]-penalties
        back[t]=np.argmax(values,axis=0)
        dp=logs[t]+np.max(values,axis=0)
    path=np.zeros(T,np.int8);path[-1]=np.argmax(dp)
    for t in range(T-1,0,-1):path[t-1]=back[t,path[t]]
    return path


def semi_markov(probs,boundaries,cfg,prior,fps=5):
    """Exact max-score explicit-duration dynamic program over bounded lengths.
    Segment score=sum(log class probs)+duration prior+boundary log odds at its start
    (for every segment except the first)+state transition penalty.
    Complexity O(T * 3 * max_duration * 3); vectorized over start candidates.
    """
    probs=np.clip(np.asarray(probs,float),1e-7,1.)
    boundaries=np.clip(np.asarray(boundaries,float),1e-5,1-1e-5)
    T=len(probs)
    if not T:return np.zeros(0,np.int8)
    if probs.shape!=(T,3) or boundaries.shape!=(T,2):raise ValueError('probs=[T,3], boundaries=[T,2]')
    maxlen=max(1,int(float(cfg['max_segment_seconds'])*fps))
    pre=np.concatenate((np.zeros((1,3)),np.cumsum(np.log(probs),axis=0)))
    best=np.full((T+1,3),-np.inf,dtype=np.float64)
    backstart=np.zeros((T+1,3),np.int32);backclass=np.full((T+1,3),-1,np.int8)
    cost_ha=float(cfg['switch_cost_ha']);cost_ad=float(cfg['switch_cost_ad'])
    cost=np.array([[np.inf,cost_ha,cost_ad],[cost_ha,np.inf,cost_ad],[cost_ad,cost_ad,np.inf]])
    bw=float(cfg['boundary_weight']);dw=float(cfg['duration_weight']);clip=float(cfg['duration_clip'])
    logits=np.clip(np.log(boundaries/(1-boundaries)),-5.,5.)
    for e in range(1,T+1):
        starts=np.arange(max(0,e-maxlen),e,dtype=np.int32)
        durations=e-starts
        for k in range(3):
            emission=pre[e,k]-pre[starts,k]
            duration=dw*_duration_score(durations,k,prior,clip)
            candidates=emission+duration
            # A new segment starts at s, with prev segment ending at s. It cannot keep same class.
            interior=starts>0
            prevclass=np.full(len(starts),-1,np.int8)
            if np.any(interior):
                ss=starts[interior]
                scores=best[ss]-cost[:,k][None,:]
                pc=np.argmax(scores,axis=1).astype(np.int8)
                prevclass[interior]=pc
                btype=0 if k==2 else np.where(pc==2,0,1)
                bonuses=bw*logits[ss,btype]
                candidates[interior]+=scores[np.arange(len(ss)),pc]+bonuses
            i=int(np.argmax(candidates))
            best[e,k]=candidates[i]
            backstart[e,k]=starts[i]
            backclass[e,k]=prevclass[i]
    result=np.empty(T,np.int8);k=int(np.argmax(best[T]));e=T
    while e>0:
        s=int(backstart[e,k]);result[s:e]=k
        kprev=int(backclass[e,k]);e=s;k=kprev
        if e>0 and k<0:raise RuntimeError('semi-Markov traceback invalid')
    return result
