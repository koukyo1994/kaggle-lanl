import pandas as pd
import numpy as np

from os.path import join
from joblib import delayed, Parallel

DataPath = './'
Filename = 'train_x.csv'
df = pd.read_csv(join(DataPath, Filename))


def process(a: np.ndarray, idx: np.ndarray, train: np.ndarray, i: int,
            shape: tuple):
    train_aug = np.zeros(shape[1])
    feature_index = np.where(np.logical_not(np.in1d(a, idx)))[0]
    train_aug[feature_index] = train[i, feature_index]
    rand_row_idx = np.random.choice(shape[0], len(idx), replace=True)
    for n, j in enumerate(idx):
        train_aug[j] = train[rand_row_idx[n], j]
    return train_aug


def make_aug(AUG_FEATURE_RATIO=0.5, denoise=False):
    # if denoise:
    #     Filename = 'train_features_denoised_50000.csv'
    train = pd.read_csv(join(DataPath, Filename))
    a = np.arange(0, train.shape[1])
    # initialise aug dataframe - remember to set dtype!

    # to integer count
    AUG_FEATURE_COUNT = np.floor(
        train.shape[1] * AUG_FEATURE_RATIO).astype('int16')
    aug_feature_index = []
    for i in range(0, len(train)):
        idx = np.random.choice(
            train.shape[1], AUG_FEATURE_COUNT, replace=False)
        idx.sort()
        aug_feature_index.append(idx)

    aug_feature_index = np.asarray(aug_feature_index)

    train_aug_values = Parallel(
        n_jobs=-1, verbose=1)([
            delayed(process)(a, aug_feature_index[i], train.values, i,
                             train.shape) for i in range(len(train))
        ])
    train_aug = pd.DataFrame(
        data=train_aug_values, columns=train.columns, dtype='float64')

    train_aug[['seg_id', 'seg_start',
               'seg_end']] = train[['seg_id', 'seg_start', 'seg_end']]
    train_aug.to_csv(
        './train_x_aug_{}.csv'.format(AUG_FEATURE_RATIO), index=False)


if __name__ == '__main__':
    make_aug(0.6)
