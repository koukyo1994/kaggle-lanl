import os
import time
import gc
import json
import datetime
import warnings
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

# os.chdir('./src')
from paths import *
import util.util_functions as util
import util.log_functions as log

# %load_ext autoreload
# %autoreload 2


def load_additional_featured_dataset(feature_dict, slide_size):
    train_features_add = pd.DataFrame()
    test_features_add = pd.DataFrame()
    for feature in feature_dict.keys():
        # feature = list(feature_dict.keys())[0]

        prefixs = feature_dict[feature]['prefix']
        suffixs = feature_dict[feature]['suffix']

        for prefix, suffix in product(prefixs, suffixs):
            # prefix, suffix = list(product(prefixs, suffixs))[1]

            sub_train_features_add = pd.read_feather(FEATURES_DIR/f'{slide_size}_slides'/f'{prefix}{feature}{suffix}_train.ftr')
            # targetをドロップ
            if feature == 'Tsfresh':
                sub_train_features_add = sub_train_features_add.drop([f'{prefix}target{suffix}'], axis=1)
            # seg_idのカラム名を修正
            sub_train_features_add = sub_train_features_add.rename(columns={f'{prefix}seg_id{suffix}':'seg_id'})

            if train_features_add.shape[1] == 0:
                train_features_add = sub_train_features_add
            else:
                train_features_add = pd.merge(train_features_add, sub_train_features_add, on='seg_id', how='outer')

            sub_test_features_add = pd.read_feather(FEATURES_DIR/f'{slide_size}_slides'/f'{prefix}{feature}{suffix}_test.ftr')
            # seg_idのカラム名を修正
            sub_test_features_add = sub_test_features_add.rename(columns={f'{prefix}seg_id{suffix}':'seg_id'})

            if test_features_add.shape[1] == 0:
                test_features_add = sub_test_features_add
            else:
                test_features_add = pd.merge(test_features_add, sub_test_features_add, on='seg_id', how='outer')

    return train_features_add, test_features_add

def parallel_shuffle_argumentation(X, aug_feature_ratio, n_jobs):
    # n_jobs = 4
    # df = X.iloc[:10000, :]
    th_indices = list(range(0, int(X.shape[0]), int(X.shape[0]/(n_jobs-1))))
    sub_Xs = []
    for i in range(1, len(th_indices)):
        sub_Xs.append(X.iloc[th_indices[i-1]:th_indices[i], :])
    sub_Xs.append(X.iloc[th_indices[-1]:, :])

    sub_X_augs = Parallel(n_jobs=n_jobs)([delayed(shuffle_argumentation)(id, sub_X, aug_feature_ratio) for id, sub_X in enumerate(sub_Xs)])

    X_aug = pd.DataFrame()
    for id, sub_X_aug in sub_X_augs:
        print(f'arg data id:{id}')
        X_aug = X_aug.append(sub_X_aug)

    return X_aug


def shuffle_argumentation(id, df, aug_feature_ratio):
    # aug_feature_ratio = 0.9
    # df = X.copy()

    a = np.arange(0, df.shape[1])
    #initialise aug dataframe - remember to set dtype!
    df_aug = pd.DataFrame(index=df.index, columns=df.columns, dtype='float64')

    for i in tqdm(range(0, len(df))):
        # i = 0
        #ratio of features to be randomly sampled
        #to integer count
        AUG_FEATURE_COUNT = np.floor(df.shape[1]*aug_feature_ratio).astype('int16')

        #randomly sample half of columns that will contain random values
        aug_feature_index = np.random.choice(df.shape[1], AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()

        #obtain indices for features not in aug_feature_index
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]

        #first insert real values for features in feature_index
        df_aug.iloc[i, feature_index] = df.iloc[i, feature_index]

        #random row index to randomly sampled values for each features
        rand_row_index = np.random.choice(len(df), len(aug_feature_index), replace=True)

        #for each feature being randomly sampled, extract value from random row in train
        for n, j in enumerate(aug_feature_index):
            df_aug.iloc[i, j] = df.iloc[rand_row_index[n], j]

    return id, df_aug

def define_args():
    parser = argparse.ArgumentParser()
    list_func = lambda x:list(map(str, x.split(',')))

    parser.add_argument('version', type=int)
    parser.add_argument('slide_size', type=int)
    parser.add_argument('aug_feature_ratio', type=int)
    parser.add_argument('n_jobs', type=int)

    return parser.parse_args()

def get_and_validate_args(args):
    version = args.version
    slide_size = args.slide_size
    aug_feature_ratio = args.aug_feature_ratio
    n_jobs = args.n_jobs

    return version, slide_size, aug_feature_ratio, n_jobs

def make_log_filename():
    try:
        filename = os.path.basename(__file__) # with .py file
    except:
        filename = os.path.basename('__file__') # with atom hydrogen

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_filename = 'log_{filename}_{time}.log'.format(filename=filename, time=time)
    return log_filename

def main():
    # Arguments
    # version = 2; slide_size = 50000; aug_feature_ratio = 90; n_jobs = 4
    args = define_args()
    version, slide_size, aug_feature_ratio, n_jobs = get_and_validate_args(args)

    # Define logger
    global logger
    logger = log.define_logger(make_log_filename())

    # load basic features
    train_features = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/train_features_{slide_size}.csv')
    test_features = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/test_features.csv')

    # Load denoised features
    train_features_denoised = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/train_features_denoised_{slide_size}.csv')
    test_features_denoised = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/test_features_denoised.csv')
    train_features_denoised.columns = [f'{i}_denoised' for i in train_features_denoised.columns]
    test_features_denoised.columns = [f'{i}_denoised' for i in test_features_denoised.columns]

    # Load additional features
    feature_dict = {
        'Tsfresh': {
            'prefix': ['', 'fftr_', 'ffti_'],
            'suffix': ['', '_denoised']
        },
        'Tsfresh2': {
            'prefix': [''],
            'suffix': ['', '_denoised']
        }
    }
    if len(feature_dict.keys()) != 0:
        train_features_add, test_features_add = load_additional_featured_dataset(feature_dict, slide_size)
    else:
        train_features_add, test_features_add = pd.DataFrame(index=train_features.index, columns=[]), pd.DataFrame(index=test_features.index, columns=[])

    # Concat features
    X = pd.concat([train_features, train_features_denoised, train_features_add], axis=1)[:-1]
    y = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/y_{slide_size}.csv')[:-1]
    logger.info(f'Created X & y. X shape:{X.shape}, y shape:{y.shape}')

    # Shuffed argumentation
    if aug_feature_ratio != 0:
        logger.info(f'Started Shuffle argumentation. Before shape:{X.shape}, aug_feature_ratio:{aug_feature_ratio}')
        X_arg = parallel_shuffle_argumentation(X, aug_feature_ratio/100, n_jobs)
        X = X.append(X_arg)

        y = y.append(y)
        logger.info(f'Finished Shuffle argumentation. After shape:{X.shape}')

    # Save train dataset
    train = pd.concat([X, y], axis=1)
    train.to_csv(DATA_DIR/'input'/'featured'/f'featured_train_ver{version}_{slide_size}_{aug_feature_ratio}.csv', index=False)

    # Save test dataset
    test = pd.concat([test_features, test_features_denoised, test_features_add], axis=1)
    test.to_csv(DATA_DIR/'input'/'featured'/f'featured_test_ver{version}_{slide_size}_{aug_feature_ratio}.csv', index=False)

    return


if __name__ == '__main__':
    main()
