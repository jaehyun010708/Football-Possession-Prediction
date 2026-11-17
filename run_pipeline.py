#!/usr/bin/env python3
"""IDSSE football possession learning and validation pipeline.

python run_pipeline.py --mode prepare [--download-missing]
python run_pipeline.py --mode validate
python run_pipeline.py --mode fit-all
python run_pipeline.py --mode all
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from settings import ROOT,OUT,DATA,MATCHES,DEFAULT_VAL
from preprocessing.load_idsse import download_missing,prepare,required_paths


def _ensure_data(directory,download=False):
    if download:download_missing(directory)
    for match in MATCHES:required_paths(directory,match)


def _e4_train(args,validation,fit_all=False,epochs=None):
    env=os.environ.copy();env['VAL_MATCH']=validation
    env['FIT_ALL']='1' if fit_all else '0'
    cmd=[sys.executable,'-m','e4ha.train', '--epochs',str(epochs or args.epochs_e4),
         '--mask-mode',args.mask_mode,'--windows-per-epoch',str(args.windows_per_epoch)]
    subprocess.run(cmd,cwd=str(ROOT),env=env,check=True)


def _train_ad(train_matches,validation,mode,epochs):
    from ad_v1.train import train,predict
    from ad_tree.train import train as train_tree,predict as tree_predict
    from ad_ensemble.combine import save_ad_probs
    stage=OUT/'models'/mode;stage.mkdir(parents=True,exist_ok=True)
    ckpt=stage/'v1_mid.pt';tree_path=stage/'tree.joblib'
    vep=train(train_matches,validation,ckpt,epochs,full=mode=='fit-all')
    (stage/'v1_best_epoch.txt').write_text(str(vep))
    clf=train_tree(train_matches,tree_path)
    if mode=='validate':
        # The held-out match is NEVER included in AD-V1 or Tree training.
        v=predict(ckpt,validation)
        t=tree_predict(clf,validation)
        save_ad_probs(validation,v,t)
    return stage


def _epochs_for_all(args):
    stage=OUT/'models'/'validate'
    out=[]
    for key,default in [('v1_best_epoch.txt',args.epochs_v1),('e4_best_epoch.txt',args.epochs_e4)]:
        f=stage/key
        out.append(int(f.read_text().strip()) if f.exists() else default)
    return out


def execute(args):
    if sys.version_info[:2] != (3,13):
        print(f'WARNING: Python {sys.version.split()[0]}; source ZIP targets 3.13.14',file=sys.stderr,flush=True)
    raw=Path(args.idsse_dir).expanduser().resolve()
    _ensure_data(raw,args.download_missing)
    prepare(raw,OUT/'base',MATCHES)
    if args.mode=='prepare':return
    validation=args.val_match
    train_matches=[m for m in MATCHES if m!=validation]
    from ball_missing.generate_masks import create,assemble_dataset
    if args.mode in ('validate','all'):
        _train_ad(train_matches,validation,'validate',args.epochs_v1)
        create(train_matches,validation)
        assemble_dataset(MATCHES)
        _e4_train(args,validation,fit_all=False)
        from inference.validate import validate
        validate(validation)
    if args.mode in ('fit-all','all'):
        vep,eep=_epochs_for_all(args)
        print(f'[fit-all] V1 epochs={vep}, E4 epochs={eep}; match-level validation is finished',flush=True)
        _train_ad(MATCHES,None,'fit-all',vep)
        create(MATCHES)
        assemble_dataset(MATCHES)
        _e4_train(args,validation,fit_all=True,epochs=eep)
    print(f'\nDone: {OUT}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode',choices=['prepare','validate','fit-all','all'],default='validate')
    ap.add_argument('--idsse-dir',default=str(DATA),help='folder with 7 raw Tracking/Metadata/Event XML triplets')
    ap.add_argument('--download-missing',action='store_true',help='explicitly download up to ~2.63GB from Hugging Face')
    ap.add_argument('--val-match',choices=MATCHES,default=DEFAULT_VAL)
    ap.add_argument('--epochs-v1',type=int,default=30)
    ap.add_argument('--epochs-e4',type=int,default=20)
    ap.add_argument('--windows-per-epoch',type=int,default=6000)
    ap.add_argument('--mask-mode',choices=['fixed','dynamic'],default='fixed')
    args=ap.parse_args()
    if args.epochs_v1<1 or args.epochs_e4<1 or args.windows_per_epoch<1:ap.error('epochs/windows must be >=1')
    execute(args)

if __name__=='__main__':main()
