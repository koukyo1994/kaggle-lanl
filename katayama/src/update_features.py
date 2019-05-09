import os

import numpy as np
import pandas as pd

# os.chdir('./src')
from paths import *
import features.denoising as deno
from features.base import Feature
import utils.util_functions as util

# %load_ext autoreload
# %autoreload 2


class BasicStats(Feature):
    def calc_features(self, x, record):
        # basic stats
        record['mean'] = x.mean()
        record['std'] = x.std()
        record['max'] = x.max()
        record['min'] = x.min()

        # basic stats on absolute values
        record['mean_change_abs'] = np.mean(np.diff(x))
        record['abs_max'] = np.abs(x).max()
        record['abs_mean'] = np.abs(x).mean()
        record['abs_std'] = np.abs(x).std()

        return record

class HarminicMean(Feature):
    def calc_features(self, x, record):
        # geometric and harminic means
        record['hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
        return record

class GeometricMean(Feature):
    def calc_features(self, x, record):
        # geometric and harminic means
        record['gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))
        return record

class Kstatics(Feature):
    def calc_features(self, x, record):
        # k-statistic and moments
        for i in range(1, 5):
            record[f'kstat_{i}'] = stats.kstat(x, i)
            record[f'moment_{i}'] = stats.moment(x, i)

        for i in [1, 2]:
            record[f'kstatvar_{i}'] = stats.kstatvar(x, i)
        return record

def main():
    # Argument
    slide_size = 150000
    overwrite = False

    # Laod train data
    train = util.read_train_data(nrows=1000000)

    basicStats = BasicStats(slide_size)
    basicStats.create_features(train)
    basicStats.save()


if __name__ == '__main__':
    main()
