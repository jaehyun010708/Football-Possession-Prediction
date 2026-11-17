"""Train the same 347-feature HGB specification as preprocess/step2_tree_pD_lomo.py.
This is NOT the separate noise-simulated 25FPS deadball/ final checkpoint.
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from ad_tree.features import make_tree_features
from settings import OUT


def load_data(m):
    with np.load(OUT/'base'/f'{m}.npz') as z:
        return {k:z[k] for k in z.files}


def get_features(m):
    cache=OUT/'cache'/f'{m}_tree347.npz'
    if cache.exists():
        with np.load(cache) as z:return z['F']
    d=load_data(m)
    F=make_tree_features(d['xy'],d['teams'],d['period'])
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,F=F)
    print(f'[AD-Tree] features {m}: {F.shape}',flush=True)
    return F


def train(matches,out):
    X=np.concatenate([get_features(m) for m in matches],axis=0)
    y=np.concatenate([(load_data(m)['gt_orig']==2) for m in matches]).astype(np.int8)
    clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=.08,max_leaf_nodes=31,random_state=0)
    clf.fit(X,y)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(clf,out)
    print(f'[AD-Tree] model saved: {out}',flush=True)
    return clf


def predict(clf,match):
    return clf.predict_proba(get_features(match))[:,1].astype(np.float32)
