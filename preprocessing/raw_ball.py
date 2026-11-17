"""Read BALL positions from original Sportec XML; keep native frame IDs."""
from pathlib import Path
import re
import pandas as pd

ATTR = re.compile(r'(\w+)="([^"]*)"')


def read_ball_positions(raw_xml: Path):
    rows=[]; in_ball=False; period=0
    with open(raw_xml,encoding='utf-8') as f:
        for line in f:
            if '<FrameSet' in line:
                in_ball = 'TeamId="BALL"' in line
                if in_ball: period=1 if 'firstHalf' in line else 2
            elif in_ball and '<Frame ' in line:
                a=dict(ATTR.findall(line))
                # Source-centred metres -> upper-left corner in metres (same as original sub_13)
                x=pd.to_numeric(a.get('X'),errors='coerce')
                y=pd.to_numeric(a.get('Y'),errors='coerce')
                rows.append((int(a['N']),period,x+52.5,34.0-y))
    df=pd.DataFrame(rows,columns=['frame','period','ball_x','ball_y'])
    if df.duplicated(['frame','period']).any():
        raise ValueError(f'BALL raw XML has duplicate period/frame: {raw_xml}')
    return df
