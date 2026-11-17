"""ATD-like contiguous 5FPS ball-missing augmentation.
Adapted from the mask algorithm provided in the conversation.
The bins are approximate statistics of ATD predicted-Alive × ball-NaN,
not per-frame competition labels. A training GT Alive reference can be used
for quota; a validation predicted-Alive reference prevents GT leakage.
"""
from __future__ import annotations
import numpy as np
BINS = (
    ('<1s',1,4,48,14.0),('1-3s',5,14,29,50.0),('3-5s',15,24,8,31.0),
    ('5-10s',25,49,25,170.0),('10s+',50,70,7,79.0),
)


def _run_lengths(mask,frames,step=5):
    lengths=[];start=None
    for i in range(len(mask)+1):
        is_valid=i<len(mask) and bool(mask[i]);gap= i>0 and i<len(mask) and int(frames[i]-frames[i-1])!=step
        if start is not None and (not is_valid or gap):
            lengths.append(i-start);start=None
        if is_valid and start is None:start=i
    return lengths


def bin_summary(mask,frames,fps=5):
    lengths=_run_lengths(np.asarray(mask,bool),np.asarray(frames),25//fps)
    counts={};seconds={}
    for name,low,high,_,_ in BINS:
        xs=[x for x in lengths if low<=x<=(high if high!=70 else 10**9)]
        counts[name]=len(xs);seconds[name]=round(sum(xs)/fps,2)
    tot=sum(lengths)
    return {'run_count':len(lengths),'total_missing_seconds':round(tot/fps,2),
        'long_3s_time_fraction':round(sum(x for x in lengths if x>=3*fps)/tot,4) if tot else None,
        'longest_seconds':round(max(lengths)/fps,2) if lengths else 0.,
        'bin_runs':counts,'bin_seconds':seconds}


def _sample_run_length(rng,fps=5):
    weights=np.array([b[3] for b in BINS],float);weights/=weights.sum()
    i=int(rng.choice(len(BINS),p=weights))
    _,lo,hi,count,seconds=BINS[i]
    avg=seconds/count*fps;std=max(.6,(hi-lo)/4.5)
    return int(np.clip(round(rng.normal(avg,std)),lo,hi))


def _draw_once(alive_reference,observed,frames,target_rate,seed,fps=5,tolerance=.01):
    """Uniform starts on whole period, not selected at GT Alive positions."""
    alive=np.asarray(alive_reference,bool)
    missing=~np.asarray(observed,bool)
    total_alive=int(alive.sum())
    native=int((missing&alive).sum());goal=int(round(target_rate*total_alive))
    if native>goal+max(1,int(round(tolerance*total_alive))):
        raise ValueError(f'Already-missing rate {native/max(total_alive,1):.2%} exceeds target {target_rate:.2%}')
    rng=np.random.default_rng(seed);syn=np.zeros(len(frames),bool)
    allowed=max(3,int(round(min(tolerance,.002)*total_alive)))
    breaks=np.flatnonzero(np.diff(frames)!=25//fps)+1
    starts=np.r_[0,breaks];ends=np.r_[breaks,len(frames)]
    spans=[(int(s),int(e)) for s,e in zip(starts,ends) if e-s>2]
    if not spans:raise ValueError('No continuous 5FPS segments')
    weights=np.array([b-a for a,b in spans],float);weights/=weights.sum()
    filled=native;count=0
    for attempt in range(15000):
        remaining=goal-filled
        if remaining<=allowed:break
        i=int(rng.choice(len(spans),p=weights));s,e=spans[i]
        length=min(_sample_run_length(rng,fps),e-s)
        left=int(rng.integers(s,e-length+1));right=left+length
        # Avoid joining into existing missing runs, which changes the desired length distribution.
        if np.any(missing[max(s,left-1):min(e,right+1)]|syn[max(s,left-1):min(e,right+1)]):continue
        newly=int(alive[left:right].sum())
        if newly==0 or newly>remaining+allowed:continue
        syn[left:right]=True;filled+=newly;count+=1
    return syn,{'attempts':attempt+1,'accepted_runs':count}


def generate_mask(alive_reference,observed,frames,*,target_rate=.106,fps=5,seed=42,trials=8,tolerance=.01):
    """Return a *synthetic-only* mask and diagnostics.
    observed: True iff both original ball coordinates are finite.
    alive_reference: GT Alive for training; predicted AD Alive for validation.
    """
    alive=np.asarray(alive_reference,bool);obs=np.asarray(observed,bool);frames=np.asarray(frames,np.int64)
    if alive.ndim!=1 or alive.shape!=obs.shape or alive.shape!=frames.shape:raise ValueError('array shapes mismatch')
    if not alive.any():raise ValueError('No Alive reference frames')
    if np.any(np.diff(frames)<=0):raise ValueError('Frames must increase within each period')
    if not 0<target_rate<1 or not 0<tolerance<.1 or trials<1:raise ValueError('invalid augmentation configuration')
    best=None;score=float('inf')
    for trial in range(trials):
        syn,diag=_draw_once(alive,obs,frames,target_rate,seed+1009*trial,fps,tolerance)
        miss=(~obs|syn)&alive;actual=miss.sum()/alive.sum()
        st=bin_summary(miss,frames,fps)
        share=st['long_3s_time_fraction'] or 0
        actual_counts=np.array([st['bin_runs'][b[0]] for b in BINS],float)
        target_counts=np.array([b[3] for b in BINS],float)
        hist=(float(np.abs(actual_counts/actual_counts.sum()-target_counts/target_counts.sum()).sum())
              if actual_counts.sum() else 2.)
        sc=abs(actual-target_rate)*6+abs(share-.81)*.25+hist*.06
        if sc<score:score=sc;best=(syn,float(actual),st,diag)
    syn,actual,st,diag=best
    report={'alive_reference_frames':int(alive.sum()),'original_missing_alive':int((~obs&alive).sum()),
            'synthetic_missing_alive':int((syn&alive).sum()),'final_alive_missing_rate':round(actual,6),
            'target_alive_missing_rate':target_rate,'within_tolerance':abs(actual-target_rate)<=tolerance,
            'mask_run_statistics':st,'selection_score':round(score,6),**diag}
    if not report['within_tolerance']:
        raise ValueError(f'Mask cannot reach target without changing input observations: {report}')
    return syn,report
