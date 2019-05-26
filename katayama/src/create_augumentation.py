import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# os.chdir('src')
from paths import *
import util.s3_functions as s3


DataPath = FEATURES_DIR / f'ojisan_added/'
Filename = 'train_x.csv'


def process(a: np.ndarray, idx: np.ndarray, train: np.ndarray, i: int,
            shape: tuple):
    train_aug = np.zeros(shape[1])
    feature_index = np.where(np.logical_not(np.in1d(a, idx)))[0]
    train_aug[feature_index] = train[i, feature_index]
    rand_row_idx = np.random.choice(shape[0], len(idx), replace=True)
    for n, j in enumerate(idx):
        train_aug[j] = train[rand_row_idx[n], j]
    return train_aug

def main(AUG_FEATURE_RATIO=0.4, denoise=False, n_jobs=1):
    # if denoise:
    #     Filename = 'train_features_denoised_50000.csv'
    # train = pd.read_csv(join(DataPath, Filename))
    train = s3.read_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan_added/train_x.csv')
    a = np.arange(0, train.shape[1])
    # initialise aug dataframe - remember to set dtype!
    print("data loaded")

    # to integer count
    AUG_FEATURE_COUNT = np.floor(train.shape[1] * AUG_FEATURE_RATIO).astype('int16')
    aug_feature_index = []
    for i in range(0, len(train)):
        idx = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
        idx.sort()
        aug_feature_index.append(idx)
    print("got random index")

    aug_feature_index = np.asarray(aug_feature_index)

    train_aug_values = Parallel(
        n_jobs=n_jobs, verbose=1)([
            delayed(process)(a, aug_feature_index[i], train.values, i, train.shape) for i in range(len(train))
        ])

    train_aug = pd.DataFrame(
        data=train_aug_values, columns=train.columns, dtype='float64')

    # train_aug[['seg_id', 'seg_start',
    #            'seg_end']] = train[['seg_id', 'seg_start', 'seg_end']]
    # train_aug.to_csv(
    #     './train_x_aug_{}.csv'.format(AUG_FEATURE_RATIO), index=False)
    s3.to_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan_added/train_x_aug_{AUG_FEATURE_RATIO}.csv', train_aug)


if __name__ == '__main__':
    main()
