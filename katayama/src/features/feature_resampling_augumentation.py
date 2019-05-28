import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# os.chdir('src')
from paths import *
import util.s3_functions as s3


def create_augumentation(train, AUG_FEATURE_RATIO):
    a = np.arange(0, train.shape[1])
    # initialise aug dataframe - remember to set dtype!
    print("data loaded")

    #initialise aug dataframe - remember to set dtype!
    train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')

    #ratio of features to be randomly sampled
    #AUG_FEATURE_RATIO = 0.5
    #to integer count
    AUG_FEATURE_COUNT = np.floor(train.shape[1]*AUG_FEATURE_RATIO).astype('int16')

    for i in tqdm(range(0, len(train))):

        #randomly sample half of columns that will contain random values
        aug_feature_index = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()

        #obtain indices for features not in aug_feature_index
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]

        #first insert real values for features in feature_index
        train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]

        #random row index to randomly sampled values for each features
        rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)

        #for each feature being randomly sampled, extract value from random row in train
        for n, j in enumerate(aug_feature_index):
            train_aug.iloc[i, j] = train.iloc[rand_row_index[n], j]

    try:
        train_aug[['seg_id', 'seg_start', 'seg_end']] = train[['seg_id', 'seg_start', 'seg_end']]
    except:
        pass
    
    return train_aug
