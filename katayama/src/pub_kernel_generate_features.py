import os
import json

import numpy as np
import pandas as pd

# os.chdir('./src')
from paths import *
from features.FeatureGenerator import FeatureGenerator

# %load_ext autoreload
# %autoreload 2
# Import config
with open(CONF_DIR / 'config.json', 'r') as f:
    config = json.load(f)
# Global valiables
TRAIN_DATA_LENGTH = config['data']['TRAIN_DATA_LENGTH']


def main():
    n_jobs = -1
    chunk_size = 9999
    denoising = False
    random_slide = True

    print(n_jobs, chunk_size, denoising, random_slide)

    feature_dir_name = f'lanl-features-{chunk_size}'

    training_fg = FeatureGenerator(dtype='train', n_jobs=n_jobs, chunk_size=chunk_size)
    training_data = training_fg.generate_2(denoising=denoising, random_slide=random_slide)

    test_fg = FeatureGenerator(dtype='test', n_jobs=n_jobs, chunk_size=150000)
    test_data = test_fg.generate_2(denoising=denoising)

    X = training_data.drop(['target'], axis=1)
    X_test = test_data.drop(['target'], axis=1)
    test_segs = test_data.seg_id
    y = training_data.target

    means_dict = {}
    for col in X.columns:
        if X[col].isnull().any():
            print(col)
            mean_value = X.loc[X[col] != -np.inf, col].mean()
            X.loc[X[col] == -np.inf, col] = mean_value
            X[col] = X[col].fillna(mean_value)
            means_dict[col] = mean_value

    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
            X_test[col] = X_test[col].fillna(means_dict[col])

    # Save datasets
    out_dir = FEATURES_DIR / feature_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if denoising:
        X.to_csv(out_dir / f'train_features_denoised_{chunk_size}.csv', index=False)
        X_test.to_csv(out_dir / 'test_features_denoised.csv', index=False)
        pd.DataFrame(y).to_csv(out_dir / f'y_denoised_{chunk_size}.csv', index=False)
    else:
        X.to_csv(out_dir / f'train_features_{chunk_size}.csv', index=False)
        X_test.to_csv(out_dir / 'test_features.csv', index=False)
        pd.DataFrame(y).to_csv(out_dir / f'y_{chunk_size}.csv', index=False)


if __name__ == '__main__':
    main()
