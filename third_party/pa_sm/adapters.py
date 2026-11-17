"""Adapters for PathCRF processed Sportec parquet and tactical Metrica ATD raw/XML."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .data import save_match


def _pad_xy(xy,teams,max_slots):
    T,N,_=xy.shape
    if N>max_slots:raise ValueError(f'{N} valid source slots > max_slots={max_slots}; increase max_slots, do not truncate players')
    out=np.full((T,max_slots,2),np.nan,np.float32);out[:,:N]=xy
    tm=np.zeros(max_slots,np.int8);tm[:N]=teams
    return out,tm


def prep_sportec(parquet,out_dir,cfg):
    """Read outputs of PathCRF preprocessing: tracking_processed/{match}.parquet.
    ball_x/ball_y are NEVER features. ball_state/owner are used solely as labels.
    """
    p=Path(parquet);df=pd.read_parquet(p)
    required={'period_id','frame_id','ball_state','ball_owning_team_id','player_id'}
    if not required.issubset(df.columns):raise ValueError(f'{p.name}: missing columns {required-set(df.columns)}')
    m=re.findall(r'^(home|away)_\d+_x$', ' '.join(df.columns))
    out=[]
    for per,g in df.groupby('period_id',sort=True):
        g=g.sort_values('frame_id').reset_index(drop=True)
        ids={k:sorted(c[:-2] for c in df.columns if re.match(rf'{k}_\d+_x$',c)) for k in ('home','away')}
        if not all(ids.values()):raise ValueError(f'Player columns absent in {p.name}')
        slots=ids['home']+ids['away']
        # Learn owner-team mapping from PathCRF player_id and ball_owning_team_id, same source as E4.
        player=g['player_id'].astype('string')
        known=player.str.startswith(('home_','away_'),na=False)&g['ball_owning_team_id'].notna()
        tab=pd.crosstab(g.loc[known,'ball_owning_team_id'],player[known].str.split('_').str[0])
        if 'home' not in tab.columns or not len(tab):
            raise ValueError(f'{p.name} period {per}: cannot map owning_team_id to home; check input parquet')
        home_id=tab['home'].idxmax()
        nstep=max(1,round(float(cfg['preprocessing']['source_fps'])/float(cfg['preprocessing']['fps'])))
        ix=np.arange(0,len(g),nstep)
        xy=np.stack([g[[f'{s}_x' for s in slots]].to_numpy(np.float32),
                     g[[f'{s}_y' for s in slots]].to_numpy(np.float32)],-1)[ix]
        team=np.array([-1]*len(ids['home'])+[1]*len(ids['away']),np.int8)
        xy,team=_pad_xy(xy,team,int(cfg['preprocessing']['max_slots']))
        own=g['ball_owning_team_id'].to_numpy()[ix]
        st=g['ball_state'].astype(str).str.lower().to_numpy()[ix]
        y=np.full(len(ix),-100,np.int64)
        y[st=='dead']=2
        alive=st=='alive'
        y[alive & pd.notna(own)]=np.where(own[alive & pd.notna(own)]==home_id,0,1)
        # Inconsistent labels are ignored, not guessed.
        path=Path(out_dir)/f'{p.stem}_p{per}.npz'
        save_match(path,xy,team,g['frame_id'].to_numpy()[ix],y,np.full(len(ix),int(per)))
        out.append(str(path))
    return out


def parse_atd_meta(meta):
    root=ET.parse(meta).getroot()
    team_ids=[t.get('id') for t in root.iter('Team')]
    if len(team_ids)<2:raise ValueError('Expected two Team elements in ATD XML')
    players=list(root.iter('Player'))
    if not players:raise ValueError('No Player slots in ATD metadata')
    team=np.array([-1 if p.get('teamId')==team_ids[0] else 1 if p.get('teamId')==team_ids[1] else 0 for p in players],np.int8)
    if np.any(team==0):raise ValueError('Found ATD slot with unknown teamId')
    return team


def prep_atd(meta,raw,out_dir,cfg,n_frames=199319,halves=((11185,90651),(110808,190197))):
    """Use exact tactical frame 0-based convention; no labels or ball input."""
    team=parse_atd_meta(meta);N=len(team)
    if N>int(cfg['preprocessing']['max_slots']):raise ValueError('ATD slots exceed configured maximum')
    # Read only 5FPS frames inside the graded halves; avoid storing all 25FPS objects.
    target={}
    src=int(cfg['preprocessing']['source_fps']);fps=int(cfg['preprocessing']['fps'])
    if src%fps:raise ValueError('ATD adapter requires integer downsampling ratio')
    step=src//fps
    for pi,(start,end) in enumerate(halves,1):
        idx=np.arange(start,end+1,step,dtype=np.int64)
        xy=np.full((len(idx),N,2),np.nan,np.float32)
        target.update({int(i):(pi,k) for k,i in enumerate(idx)})
        yield_obj=(idx,xy)
        if pi==1: h1=yield_obj
        else: h2=yield_obj
    arr={1:h1[1],2:h2[1]}
    with open(raw,encoding='utf-8') as f:
        for line in f:
            parts=line.rstrip().split(':')
            if len(parts)<2:continue
            try:frame=int(parts[0])-1
            except ValueError:continue
            pos=target.get(frame)
            if pos is None:continue
            pi,k=pos
            for j,g in enumerate(parts[1].split(';')[:N]):
                if not g or g[0].upper()=='N':continue
                v=g.split(',')
                if len(v)<2:continue
                try:arr[pi][k,j,0]=float(v[0])*float(cfg['preprocessing']['pitch_x']);arr[pi][k,j,1]=float(v[1])*float(cfg['preprocessing']['pitch_y'])
                except ValueError:pass
    output=[]
    for pi,(idx,xy) in enumerate((h1,h2),1):
        xy,tm=_pad_xy(xy,team,int(cfg['preprocessing']['max_slots']))
        path=Path(out_dir)/f'BAR-ATH_p{pi}.npz'
        save_match(path,xy,tm,idx,period=np.full(len(idx),pi))
        output.append(str(path))
    return output
