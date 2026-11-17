"""AD probability fusion (sub_13): mean of V1 D and HGB p_dead."""
from pathlib import Path
import numpy as np
from settings import OUT
from inference.decoders import vit3


def save_ad_probs(match,v1_probs,tree_dead):
    v1=np.asarray(v1_probs,dtype=np.float32)
    d=(v1[:,2]+np.asarray(tree_dead))/2
    h=v1[:,0]/np.clip(v1[:,0]+v1[:,1],1e-9,None)
    aux=vit3((1-d)*h,(1-d)*(1-h),d)
    path=OUT/'ad_probs'/f'{match}.npz'
    path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,p_v1=v1,p_tree_dead=tree_dead,p_dead=d,
                        aux_state=aux,aux_alive=(aux!=2))
    print(f'[AD] {match}: saved ensemble probs (mean p_dead={d.mean():.3f})',flush=True)
    return path
