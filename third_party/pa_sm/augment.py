"""Train-only augmentations, applied to positions before recomputing relational features."""
import numpy as np


def augment(xy, teams, rng, cfg, pitch):
    xy=xy.copy(); teams=teams.copy()
    if cfg.get('flip_x',False) and rng.random()<0.5:
        xy[:,:,0]=pitch[0]-xy[:,:,0]
    if cfg.get('flip_y',False) and rng.random()<0.5:
        xy[:,:,1]=pitch[1]-xy[:,:,1]
    sigma=float(cfg.get('position_noise_m',0))
    if sigma>0:
        xy+=rng.normal(0,sigma,xy.shape).astype(np.float32)
    drop=float(cfg.get('dropout_probability',0))
    if drop>0:
        # Contiguous dropout: one randomly chosen slot per window, brief interval.
        for _ in range(max(1,round(xy.shape[1]*drop))):
            j=int(rng.integers(xy.shape[1])); s=int(rng.integers(xy.shape[0]))
            e=min(xy.shape[0],s+int(rng.integers(2,11)))
            xy[s:e,j,:]=np.nan
    dup=float(cfg.get('duplicate_probability',0))
    if dup>0 and rng.random()<dup:
        # Duplicate into an empty slot only; never replace a valid player.
        free=np.flatnonzero(~np.isfinite(xy).any(axis=(0,2)))
        occupied=np.flatnonzero(np.isfinite(xy).any(axis=(0,2)))
        if len(free) and len(occupied):
            dst=int(rng.choice(free)); src=int(rng.choice(occupied))
            xy[:,dst]=xy[:,src]+rng.normal(0,0.5,xy[:,src].shape)
            teams[dst]=teams[src]
    if cfg.get('shuffle_tokens',False):
        order=rng.permutation(xy.shape[1]); xy=xy[:,order]; teams=teams[order]
    return xy,teams
