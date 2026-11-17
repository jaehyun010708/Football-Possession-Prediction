"""Single source of truth for data and experiment conventions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'idsse'
OUT = ROOT / 'outputs'
MATCHES = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WQQ', 'J03WR9', 'J03WPY')
DEFAULT_VAL = 'J03WPY'
SOURCE_FPS = 25
FPS = 5
MAX_SLOTS = 46
PITCH_X = 105.0
PITCH_Y = 68.0
# The E4 encoder has 26 player tokens, as in the uploaded source.
E4_MAX_PLAYERS = 26
