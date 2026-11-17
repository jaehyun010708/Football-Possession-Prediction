"""Official More than a Hack 2027 evaluation adapter.

This module does NOT reimplement SpellF1. It runs Metrica Sports' own
`evaluation/possession_metrics.py` and adapts our 5 Hz IDSSE labels to 25 Hz.
The official evaluator is MIT-licensed and downloaded separately by the user.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from functools import lru_cache

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLASSES = {0, 1, 2, -100}


@lru_cache(maxsize=1)
def official_module():
    """Load the *actual* organiser scorer; never silently fall back to a proxy."""
    candidates = []
    override = os.environ.get('MTAH_METRICS_PY', '').strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend([
        ROOT / 'external' / 'official_scorer' / 'possession_metrics.py',
        ROOT.parent / 'evaluation' / 'possession_metrics.py',
        ROOT.parent / 'more-than-a-hack-2027' / 'evaluation' / 'possession_metrics.py',
        ROOT.parent.parent / 'more-than-a-hack-2027' / 'evaluation' / 'possession_metrics.py',
        Path.home() / 'projects' / 'more-than-a-hack-2027' / 'evaluation' / 'possession_metrics.py',
    ])
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location('mtah2027_official_possession_metrics', path)
            if not spec or not spec.loader:
                raise RuntimeError(f'Cannot import official metrics from {path}')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            required = ('evaluate','SPELL_IOUS','H','A','D','X')
            missing = [attr for attr in required if not hasattr(module,attr)]
            if missing:
                raise RuntimeError(f'{path} does not match official scorer; missing: {missing}')
            return module
    searched = '\n'.join('  - '+str(p) for p in candidates)
    raise FileNotFoundError(
        'Official possession_metrics.py not installed. Download the organiser\'s scorer once:\n'
        '  bash scripts/install_official_scorer.sh\n'
        'Or set MTAH_METRICS_PY to an existing checkout\'s evaluation/possession_metrics.py.\n'
        'Checked:\n'+searched)


def _check_labels(arr, name):
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f'{name} must be a 1D integer label array, got {a.shape}')
    if not np.isin(a, list(CLASSES)).all():
        raise ValueError(f'{name} contains values other than 0(H), 1(A), 2(D), -100(X)')
    return a.astype(np.int8,copy=False)


def expand_to_25fps(labels, frames=None, fps=5):
    """Time-align sampled IDSSE labels to the scorer's native 25 Hz timebase.

    5 Hz frames must be regularly sampled 1/5 from the Sportec 25 Hz original.
    Repeats labels 5 times, preserving half-open intervals, so spell IoU is
    invariant to this conversion. Transition tolerances and durations now
    have the correct 25 Hz meaning.

    This does *not* create extra ground-truth information for the missing 4/5
    frames. It is still a 5 Hz validation proxy for the official 25 Hz task.
    """
    labels = _check_labels(labels, 'labels')
    if 25 % fps:
        raise ValueError(f'FPS must divide 25 exactly; got {fps}')
    ratio = 25 // fps
    if frames is not None:
        frames = np.asarray(frames, dtype=np.int64)
        if frames.shape != labels.shape:
            raise ValueError('frames and labels length mismatch')
        if len(frames) > 1:
            steps = np.diff(frames)
            if not np.all(steps == ratio):
                values, counts = np.unique(steps,return_counts=True)
                detail = dict(zip(map(int,values[:10]),map(int,counts[:10])))
                raise ValueError(
                    f'Nonuniform source frame gaps; expected {ratio} at {fps}FPS, got {detail}. '
                    'Do not score irregular samples by naive repeat; inspect frame alignment first.')
    out = labels.copy()
    out[out == -100] = 3  # official X class
    return np.repeat(out, ratio).astype(np.int8,copy=False)


def _score_codes(gt25, pred25):
    scorer=official_module()
    result=scorer.evaluate(gt25,pred25,bootstrap_n=0)
    # Full scorer returns many diagnostics. Keep all official key names and values.
    return result


def report(gt, pred, frames=None, fps=5):
    """Official evaluate() applied to *one* IDSSE half in 25 Hz units."""
    gt25=expand_to_25fps(gt,frames,fps)
    pred25=expand_to_25fps(pred,frames,fps)
    return _score_codes(gt25,pred25)


def report_periods(periods, fps=5):
    """Official game-level aggregation across separate half runs.

    periods is a list of (gt, pred, frames) triples. A 1-frame X gap between
    consecutive halves prevents false cross-halftime possessions/transitions.
    Unlike averaging per-half F1 scores, this produces the correct counts and
    TP/FP/FN matching for the complete match.
    """
    if not periods:
        raise ValueError('Need at least one period')
    gts, preds = [], []
    for gt,pred,frames in periods:
        if gts:
            gts.append(np.array([3],dtype=np.int8))
            preds.append(np.array([3],dtype=np.int8))
        gts.append(expand_to_25fps(gt,frames,fps))
        preds.append(expand_to_25fps(pred,frames,fps))
    return _score_codes(np.concatenate(gts),np.concatenate(preds))


def concise(result):
    """Readable summary of the full official results object."""
    return {k:result[k] for k in (
        'accuracy','macro_f1','spell_f1_mean','spell_f1_at_headline',
        'alive_dead_f1','team_acc_gt_alive','n_gt_spells','n_pred_spells',
        'spell_count_error','n_graded'
    ) if k in result}
