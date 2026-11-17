"""Player-only features; no ball observations are read anywhere in this module."""
import numpy as np

NAMES = ('team','x','y','vx','vy','speed','vel_valid','nearest_opponent','nearest_teammate','opponent_density','teammate_density')


def make_features(xy, teams, fps=5, pitch=(105.,68.), vmax=12., radius=5.):
    """xy metres [T,N,2], team per-slot -1 home, +1 away. Missing xy = NaN.
    Output [T,N,11] and valid mask; respects variable observation and slot dropout.
    """
    xy=np.asarray(xy,dtype=np.float32)
    teams=np.asarray(teams,dtype=np.float32)
    if xy.ndim!=3 or xy.shape[-1]!=2 or len(teams)!=xy.shape[1]:
        raise ValueError('expected xy=[T,N,2] and teams=[N]')
    T,N,_=xy.shape
    valid=np.isfinite(xy).all(-1) & (np.abs(teams)[None,:]==1)
    z=np.zeros((T,N,11),dtype=np.float32)
    z[:,:,0]=teams[None,:]
    z[:,:,1]=xy[:,:,0]/(pitch[0]/2)-1
    z[:,:,2]=xy[:,:,1]/(pitch[1]/2)-1
    dv=np.zeros_like(xy)
    kv=np.zeros((T,N),bool)
    if T>1:
        delta=(xy[1:]-xy[:-1])*fps
        speed=np.linalg.norm(delta,axis=-1)
        good=valid[1:]&valid[:-1]&np.isfinite(delta).all(-1)&(speed<=vmax)
        dv[1:]=np.where(good[...,None],delta,0)
        kv[1:]=good
    z[:,:,3:5]=dv/8.
    z[:,:,5]=np.linalg.norm(dv,axis=-1)/8.
    z[:,:,6]=kv.astype(np.float32)
    # Compute per-frame pairwise distances in small batches to cap peak RAM.
    for a in range(0,T,128):
        b=min(a+128,T)
        v=valid[a:b]
        clean=np.nan_to_num(xy[a:b],nan=0.,posinf=0.,neginf=0.)
        dist=np.linalg.norm(clean[:,:,None,:]-clean[:,None,:,:],axis=-1)
        pair=v[:,:,None]&v[:,None,:]
        own=(teams[None,:,None]==teams[None,None,:])
        eye=np.eye(N,dtype=bool)[None,:,:]
        same=pair&own&~eye
        opp=pair&~own
        z[a:b,:,7]=np.min(np.where(opp,dist,50.),axis=2)/20.
        z[a:b,:,8]=np.min(np.where(same,dist,50.),axis=2)/20.
        z[a:b,:,9]=np.sum(opp&(dist<radius),axis=2)/11.
        z[a:b,:,10]=np.sum(same&(dist<radius),axis=2)/11.
    z=np.nan_to_num(z,nan=0.,posinf=0.,neginf=0.)
    z[~valid]=0.
    return z,valid
