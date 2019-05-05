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
    n_jobs = 8
    chunk_size = 50000
    feature_dir_name = f'lanl-features-{chunk_size}'

    training_fg = FeatureGenerator(dtype='train', n_jobs=n_jobs, chunk_size=chunk_size)
    training_data = training_fg.generate_2()

    test_fg = FeatureGenerator(dtype='test', n_jobs=n_jobs, chunk_size=150000)
    test_data = test_fg.generate()

    X = training_data.drop(['target', 'seg_id'], axis=1)
    X_test = test_data.drop(['target', 'seg_id'], axis=1)
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
    X.to_csv(out_dir / f'train_features_{chunk_size}.csv', index=False)
    X_test.to_csv(out_dir / 'test_features.csv', index=False)
    pd.DataFrame(y).to_csv(out_dir / f'y_{chunk_size}.csv', index=False)

    # slide_size = 50000
    # start = 0
    # end = 150000
    # index_tuple_list = []
    # while end <= TRAIN_DATA_LENGTH:
    #     index_tuple_list.append((start, end))
    #     start += slide_size
    #     end += slide_size
    # index_tuple_list.append((start, TRAIN_DATA_LENGTH-1))
    #
    # for index_tuple in index_tuple_list:
    #     # index_tuple = index_tuple_list[1]
    #     train.iloc[index_tuple[0]:index_tuple[1], :]


if __name__ == '__main__':
    main()


    # chunk_size = 50000
    # iter_df = pd.read_csv(DATA_DIR / 'input/train.csv', iterator=True, chunksize=chunk_size, dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
    #
    # assert 150000/chunk_size, 'chunk size should be able to divide 150000.'
    #
    # df = pd.DataFrame()
    # for _ in range(int(150000/chunk_size)):
    #     df = df.append(iter_df.get_chunk())
    #
    # seg_id_num = 1
    # for sub_df in iter_df:
    #     # sub_df = iter_df.get_chunk()
    #     x = df.acoustic_data.values
    #     y = df.time_to_failure.values[-1]
    #     seg_id = 'train_' + str(seg_id_num)
    #
    #     # Update variables
    #     df = df.iloc[chunk_size:, :].append(sub_df)
    #     seg_id_num += 1
    #
    #     print(df.index[0], df.index[-1], seg_id_num)
    #
    #     del sub_df
    #     yield seg_id, x, y
