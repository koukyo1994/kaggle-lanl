import sys

import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

from joblib import Parallel, delayed


def feature_generator(path: Path):
    df = pd.read_csv(path)
    if len(df) != 150000:
        return
    ts = df["acoustic_data"].values
    time = df["time_to_failure"].values

    duration = time[0] - time[-1]
    nrows = len(df)
    sample_rate = nrows * (1 / duration)
    ts = highpass_denoise(ts, low_cutoff=10000, sample_rate=sample_rate)
    features = comprehensive_feats_dict(ts)
    return features


if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")
    sys.path.append("../common")

    from comprehensive import comprehensive_feats_dict
    from denoising import highpass_denoise

    data_dir = Path("../../../../input/train_split")

    parser = ArgumentParser()
    parser.add_argument("--file_name", default="features.mat")

    args = parser.parse_args()

    list_of_dicts = Parallel(
        n_jobs=-1,
        verbose=1)([delayed(feature_generator)(p) for p in data_dir.iterdir()])
    feature_mat = pd.DataFrame(list_of_dicts)
