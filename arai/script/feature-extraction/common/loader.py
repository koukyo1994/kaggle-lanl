import pandas as pd

from pathlib import Path


def data_generator(data_dir: Path):
    for path in data_dir.iterdir():
        data = pd.read_csv(path)
        if len(data) != 150000:
            continue
        else:
            yield data
