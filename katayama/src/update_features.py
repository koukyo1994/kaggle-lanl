import os

import numpy as np
import pandas as pd

# os.chdir('./src')
from paths import *
import features.denoising as deno
from features.base import Feature
import util.util_functions as util

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

from itertools import product
from tsfresh.feature_extraction import feature_calculators
class Tsfresh(Feature):
    def calc_features(self, x, record):
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        lags = [10, 100, 1000, 10000]

        record['abs_energy'] = feature_calculators.abs_energy(x)
        record['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        record['count_above_mean'] = feature_calculators.count_above_mean(x)
        record['count_below_mean'] = feature_calculators.count_below_mean(x)
        record['mean_abs_change'] = feature_calculators.mean_abs_change(x)
        record['mean_change'] = feature_calculators.mean_change(x)
        record['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)

        record['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        record['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        record['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        record['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        record['last_loc_max'] = feature_calculators.last_location_of_maximum(x)

        for coeff, attr in product([1, 2, 3, 4, 5], ['real', 'imag', 'angle']):
            record[f'fft_{coeff}_{attr}'] = list(feature_calculators.fft_coefficient(x, [{'coeff': coeff, 'attr': attr}]))[0][1]

        record['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        record['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        record['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        record['cid_ce_1'] = feature_calculators.cid_ce(x, 1)

        for p in percentiles:
            record[f'binned_entropy_{p}'] = feature_calculators.binned_entropy(x, p)

        record['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)

        return record

class Tsfresh2(Feature):
    def calc_features(self, x, record):
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        lags = [10, 100, 1000, 10000]

        record['abs_energy'] = feature_calculators.abs_energy(x)
        record['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        record['count_above_mean'] = feature_calculators.count_above_mean(x)
        record['count_below_mean'] = feature_calculators.count_below_mean(x)
        record['mean_abs_change'] = feature_calculators.mean_abs_change(x)
        record['mean_change'] = feature_calculators.mean_change(x)
        record['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)

        record['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        record['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        record['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        record['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        record['last_loc_max'] = feature_calculators.last_location_of_maximum(x)

        for coeff, attr in product([1, 2, 3, 4, 5], ['real', 'imag', 'angle']):
            record[f'fft_{coeff}_{attr}'] = list(feature_calculators.fft_coefficient(x, [{'coeff': coeff, 'attr': attr}]))[0][1]

        record['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        record['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        record['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        record['cid_ce_1'] = feature_calculators.cid_ce(x, 1)

        for p in percentiles:
            record[f'binned_entropy_{p}'] = feature_calculators.binned_entropy(x, p)

        record['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)

        return record

def main():
    # Argument
    slide_size = 50000
    overwrite = False

    # Laod train data
    train = util.read_train_data(nrows=150000)

    tsfresh = Tsfresh(slide_size)
    tsfresh.run(train=train)
    tsfresh.save()

    tsfresh = Tsfresh(slide_size, series_type='fftr')
    tsfresh.run(train=train)
    tsfresh.save()

    tsfresh = Tsfresh(slide_size, series_type='ffti')
    tsfresh.run(train=train)
    tsfresh.save()

    tsfresh = Tsfresh(slide_size, denoising=True)
    tsfresh.run(train=train)
    tsfresh.save()

    tsfresh = Tsfresh(slide_size, series_type='fftr', denoising=True)
    tsfresh.run(train=train)
    tsfresh.save()

    tsfresh = Tsfresh(slide_size, series_type='ffti', denoising=True)
    tsfresh.run(train=train)
    tsfresh.save()


    from tsfresh.feature_extraction import extract_features, EfficientFCParameters
    params = EfficientFCParameters()

    train_tmp = train.copy()
    train_tmp = train_tmp[['acoustic_data']]
    train_tmp['column_id'] = 1
    tmp = extract_features(train_tmp, column_id='column_id', default_fc_parameters=params)


if __name__ == '__main__':
    main()
