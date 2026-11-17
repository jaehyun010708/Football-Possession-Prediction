"""IDSSE / Sportec XML -> player-only NPZ via kloppy.

Raw Sportec tracking XML contains the *label source* (ball_state / owning team),
whereas ball positions are deliberately not used in features or saved arrays.
The team IDs are read from metadata rather than inferred from ball ownership.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .data import save_match


MATCH_RE = re.compile(r"DFL-MAT-([A-Z0-9]+)")


def _match_id(path: Path) -> str:
    match = MATCH_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot identify match ID in {path.name}")
    return match.group(1)


def discover_sportec_matches(root: str | Path):
    """Find matching local tracking and metadata XML pairs; ignore .cache files."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"No IDSSE root directory: {root}")
    result = []
    for raw in sorted(root.glob('*/DFL_04_03_positions_raw_observed_*.xml')):
        match_id = _match_id(raw)
        metas = sorted(raw.parent.glob(f'DFL_02_01_matchinformation_*DFL-MAT-{match_id}.xml'))
        if len(metas) != 1:
            raise FileNotFoundError(f"Expected one matchinformation XML for {match_id}, found {len(metas)}")
        result.append((match_id, raw, metas[0]))
    if not result:
        raise FileNotFoundError(f"No IDSSE raw XML files found under {root}/*/")
    return result


def sportec_dataframe_to_matches(df: pd.DataFrame, metadata: Any, match_id: str,
                                 out_dir: str | Path, cfg: dict,
                                 reference_parquet: str | Path | None = None):
    """Convert a kloppy Sportec tracking DataFrame into per-period NPZ arrays.

    Assumes `coordinates='metrica'` from kloppy (normalized 0..1, top-left).
    A fixed player-ID slot mapping per period permits velocity computation.
    """
    required = {'period_id','frame_id','ball_state','ball_owning_team_id'}
    if not required.issubset(df.columns):
        raise ValueError(f"Sportec kloppy DF missing {sorted(required-set(df.columns))}")
    if len(metadata.teams) < 2:
        raise ValueError('Kloppy metadata must include home and away teams')
    home, away = metadata.teams[:2]
    home_id, away_id = str(home.team_id), str(away.team_id)
    if home_id == away_id:
        raise ValueError('Home and away have identical team IDs')
    # Enumerate known players from metadata, NOT column sequence or ball owning side.
    team_by_player = {}
    for player in home.players:
        team_by_player[str(player.player_id)] = -1
    for player in away.players:
        pid = str(player.player_id)
        if pid in team_by_player:
            raise ValueError(f'Player {pid} listed for both teams')
        team_by_player[pid] = 1

    # Sportec kloppy output columns use <player_id>_x, <player_id>_y.
    player_ids = [pid for pid in team_by_player if pid + '_x' in df.columns and pid + '_y' in df.columns]
    unknown = [c[:-2] for c in df.columns if c.endswith('_x') and c.startswith('DFL-OBJ-')
               and c[:-2] not in team_by_player]
    if unknown:
        raise ValueError(f'Unknown player IDs in XML positions: {unknown[:5]}')
    if not player_ids:
        raise ValueError('No player position columns matched home/away metadata')

    ref = None
    if reference_parquet is not None:
        path = Path(reference_parquet)
        if path.exists():
            ref = pd.read_parquet(path, columns=['frame_id','label'])
            if ref['frame_id'].duplicated().any():
                raise ValueError(f'{path}: frame_id not unique for reference labels')
            ref = ref.set_index('frame_id')['label']

    max_slots = int(cfg['preprocessing']['max_slots'])
    px, py = float(cfg['preprocessing']['pitch_x']), float(cfg['preprocessing']['pitch_y'])
    out = []
    for period, group in df.groupby('period_id', sort=True):
        g = group.sort_values('frame_id', kind='stable').reset_index(drop=True)
        # Players substituted into the match still get a stable, per-period slot.
        active = []
        for pid in player_ids:
            if g[[pid+'_x',pid+'_y']].notna().all(axis=1).any():
                active.append(pid)
        if len(active) > max_slots:
            raise ValueError(f'{match_id} p{period}: {len(active)} distinct active players; '
                             f'max_slots={max_slots}. Increase max_slots in YAML.')
        xy = np.full((len(g),max_slots,2), np.nan, dtype=np.float32)
        teams = np.zeros(max_slots, dtype=np.int8)
        for j, pid in enumerate(active):
            xy[:,j,0] = g[pid+'_x'].to_numpy(np.float32) * px
            xy[:,j,1] = g[pid+'_y'].to_numpy(np.float32) * py
            teams[j] = team_by_player[pid]
        # Discard coordinates only when at least one component is invalid.
        invalid = ~np.isfinite(xy).all(axis=-1)
        xy[invalid] = np.nan

        ball_state = g['ball_state'].astype('string').str.lower().str.strip()
        owner = g['ball_owning_team_id'].astype('string').str.strip()
        labels = np.full(len(g), -100, dtype=np.int64)
        dead = ball_state.eq('dead').fillna(False).to_numpy(bool)
        alive = ball_state.eq('alive').fillna(False).to_numpy(bool)
        labels[dead] = 2
        labels[alive & owner.eq(home_id).fillna(False).to_numpy(bool)] = 0
        labels[alive & owner.eq(away_id).fillna(False).to_numpy(bool)] = 1
        # Unknown owner on an alive frame gets ignore_index, not a guessed state.
        frame_ids = g['frame_id'].to_numpy(np.int64)
        output = Path(out_dir) / f'{match_id}_p{period}.npz'
        save_match(output, xy, teams, frame_ids, labels,
                   np.full(len(g), int(period),dtype=np.int16))
        counts = {key:int((labels==value).sum()) for key,value in [('H',0),('A',1),('D',2),('ignored',-100)]}
        print(f'  {output.name}: frames={len(g):,}, active IDs={len(active)}, labels={counts}')
        if ref is not None:
            reference = ref.reindex(frame_ids)
            matched = reference.notna().to_numpy()
            both = matched & (labels!=-100)
            agreement = float(np.mean(reference.to_numpy()[both].astype('int64') == labels[both])) if both.any() else float('nan')
            print(f'    Reference label overlap={int(matched.sum()):,}/{len(g):,}, '
                  f'agreement={agreement:.3%}; if low, check frame and label conventions')
        out.append(str(output))
    return out


def prep_sportec_xml(raw: str | Path, meta: str | Path, out_dir: str | Path,
                     cfg: dict, reference_parquet: str | Path | None = None):
    try:
        from kloppy import sportec
    except ImportError as exc:
        raise ImportError('Install kloppy first: pip install kloppy') from exc
    raw, meta = Path(raw), Path(meta)
    if not raw.exists() or not meta.exists():
        raise FileNotFoundError(f'raw={raw.exists()}, meta={meta.exists()}: {raw} / {meta}')
    if _match_id(raw) != _match_id(meta):
        raise ValueError('Sportec raw XML and metadata match IDs disagree')
    sample_rate = (float(cfg['preprocessing']['fps']) / float(cfg['preprocessing']['source_fps']))
    # Kloppy performs sampling: never downsample these 5FPS rows a second time.
    dataset = sportec.load_tracking(raw_data=str(raw), meta_data=str(meta),
                                    sample_rate=sample_rate, coordinates='metrica',
                                    only_alive=False)
    return sportec_dataframe_to_matches(dataset.to_df(), dataset.metadata,
                                        _match_id(raw), out_dir, cfg, reference_parquet)
