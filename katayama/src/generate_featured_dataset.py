import os
import time
import gc
import json
import datetime
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# os.chdir('./src')
from paths import *
import util.util_functions as util
import util.log_functions as log

# %load_ext autoreload
# %autoreload 2


def shuffle_argumentation(df, aug_feature_ratio):
    # aug_feature_ratio = 0.5

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

    return df_aug

def define_args():
    parser = argparse.ArgumentParser()
    list_func = lambda x:list(map(str, x.split(',')))

    parser.add_argument('version', type=int)
    parser.add_argument('slide_size', type=int)
    parser.add_argument('aug_feature_ratio', type=int)

    return parser

def get_and_validate_args(args):
    version = args.version
    slide_size = args.slide_size
    aug_feature_ratio = args.aug_feature_ratio

    return version, slide_size, aug_feature_ratio

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
    # version = 1; slide_size = 50000; aug_feature_ratio = 90
    slide_size, n_fold, random_state = get_and_validate_args(args)

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
        }
    }

    # Concat features
    X = pd.concat([train_features, train_features_denoised], axis=1)[:-1]
    y = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/y_{slide_size}.csv')[:-1]
    logger.info(f'Created X & y. X shape:{X.shape}, y shape:{y.shape}')

    # Shuffed argumentation
    if aug_feature_ratio != 0:
        logger.info(f'Started Shuffle argumentation. Before shape:{X.shape}, aug_feature_ratio:{aug_feature_ratio}')
        X_arg = shuffle_argumentation(X, aug_feature_ratio/100)
        X = X.append(X_arg)

        y = y.append(y)
        logger.info(f'Finished Shuffle argumentation. After shape:{X.shape}')

    # Save train dataset
    train = pd.concat([X, y], axis=1)
    train.to_csv(DATA_DIR/'input'/'featured'/f'featured_train_ver{version}_{slide_size}_{aug_feature_ratio}.csv', index=False)

    # Save test dataset
    test = pd.concat([test_features, test_features_denoised], axis=1)
    test.to_csv(DATA_DIR/'input'/'featured'/f'featured_test_ver{version}_{slide_size}_{aug_feature_ratio}.csv', index=False)

    return


if __name__ == '__main__':
    main()
