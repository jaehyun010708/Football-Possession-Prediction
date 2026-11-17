"""Evaluate cached 5FPS predictions with the organiser's original metrics code."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from .data import load_match
from .metrics import report,report_periods,concise
from .pipeline import ROOT

DECODERS = ('raw','viterbi','semi')


def score_cached(run_name, source_paths=None, fps=5):
    """Use compare-created NPZs without re-running Attention or Bi-GRU.

    Existing v3 caches contain `prob`, `boundary`, `raw`, `viterbi`, `semi`
    but no reference labels or frame ids; these come from data/processed.
    """
    run_dir=ROOT/'outputs'/run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Comparison output not found: {run_dir}')
    if source_paths is None:
        source_paths=sorted(ROOT.joinpath('data','processed').glob('J03*.npz'))
    sources={Path(p).stem:Path(p) for p in source_paths}
    cached=sorted(p for p in run_dir.glob('*.npz') if p.stem in sources)
    if not cached:
        raise FileNotFoundError(
            f'No cached compare NPZs in {run_dir} matching processed NPZs. '
            'Run compare once, or pass --files data/processed/J03WPY_*.npz')
    out={}
    all_periods={name:[] for name in DECODERS}
    for path in cached:
        src=sources[path.stem]
        m=load_match(src)
        with np.load(path,allow_pickle=False) as d:
            missing=set(DECODERS)-set(d.files)
            if missing:
                raise KeyError(f'{path} missing predictions: {sorted(missing)}')
            predictions={name:d[name].copy() for name in DECODERS}
        sub={}
        for name,pred in predictions.items():
            if len(pred)!=len(m['labels']):
                raise ValueError(f'{path}:{name} shape mismatch with {src}')
            sub[name]=report(m['labels'],pred,frames=m['frames'],fps=fps)
            all_periods[name].append((m['labels'],pred,m['frames']))
        out[path.stem]=sub
    aggregate={name:report_periods(periods,fps=fps) for name,periods in all_periods.items()}
    payload={
        'metric':'Official MTAH 2027 evaluation/possession_metrics.py; '
                 '5FPS IDSSE GT and predictions repeated to 25FPS; not the private ATD score',
        'periods':out,
        'combined':aggregate,
    }
    output=run_dir/'official_metrics.json'
    output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    csv_file=run_dir/'official_metrics_summary.csv'
    fields=['scope','decoder','accuracy','macro_f1','spell_f1_mean',
            'spell_f1_at_headline','alive_dead_f1','team_acc_gt_alive',
            'n_gt_spells','n_pred_spells','spell_count_error','n_graded']
    with csv_file.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for scope,scores in list(out.items())+[('combined',aggregate)]:
            for name in DECODERS:
                writer.writerow({'scope':scope,'decoder':name,**concise(scores[name])})
    return payload,output,csv_file
