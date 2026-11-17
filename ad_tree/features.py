"""Keep the source 347-dim features: 48 frame + 299 context (5 Hz)."""
import numpy as np
from ad_tree.features_original import frame_features, _fill, context_features


def make_tree_features(xy:np.ndarray,teams:np.ndarray,period:np.ndarray)->np.ndarray:
    blocks=[]; result=None
    active=teams!=0
    t=(teams[active]==1).astype(int)
    for p in np.unique(period):
        ix=np.flatnonzero(period==p)
        X=xy[ix][:,active].astype(np.float64).copy()
        X[...,0]-=52.5; X[...,1]-=34.0
        v=np.full(X.shape[:2],np.nan)
        if len(X)>1:
            d=np.linalg.norm(np.diff(X,axis=0),axis=2)*5
            d[d>12]=np.nan
            v[1:]=d
        a=np.full_like(v,np.nan)
        if len(X)>1: a[1:]=np.abs(np.diff(v,axis=0))*5
        F0,names=frame_features(X[...,0],X[...,1],v,a,t)
        f0=_fill(F0)
        fc,_=context_features(f0,names)
        F=np.hstack([f0,fc]).astype(np.float32)
        if result is None:result=np.zeros((len(xy),F.shape[1]),np.float32)
        result[ix]=F
    if result is None or result.shape[1]!=347:raise ValueError(f'Expected 347-d tree features, got {None if result is None else result.shape}')
    if not np.isfinite(result).all():raise ValueError('Non-finite tree features')
    return result
