"""Load prepared data using a strict match-level split."""
import sys,os
import numpy as np
import torch
from settings import ROOT,OUT,MATCHES,DEFAULT_VAL
sys.path.insert(0,str(ROOT/'third_party'))
VAL=os.environ.get('VAL_MATCH',DEFAULT_VAL)
FIT_ALL=os.environ.get('FIT_ALL','0')=='1'
M=list(MATCHES)
TRAIN=M if FIT_ALL else [m for m in M if m!=VAL]
LAB='mid'
D={m:dict(np.load(OUT/'dataset'/f'{m}.npz')) for m in M}
dev='cuda' if torch.cuda.is_available() else 'cpu'
OUT=str(OUT/'models'/('fit-all' if FIT_ALL else 'validate'))
os.makedirs(OUT,exist_ok=True)
