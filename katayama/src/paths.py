import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = ROOT_DIR / 'log'
NOTE_DIR = ROOT_DIR / 'notebook'
SRC_DIR = ROOT_DIR / 'src'

DATA_DIR = SRC_DIR / 'data'
FEATURES_DIR = SRC_DIR / 'features'
