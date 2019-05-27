import os
import sys
import gc
import time
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from tsfresh.feature_extraction import feature_calculators
from sklearn.linear_model import LinearRegression
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats

# os.chdir('src')
from paths import *
import util.s3_functions as s3
import features.denoising as deno

# os.chdir('./src')
# %load_ext autoreload
# %autoreload 2

from paths import *
from features.FeatureGenerator import FeatureGenerator

# Import config
with open(CONF_DIR / 'config.json', 'r') as f:
    config = json.load(f)
# Global valiables
TRAIN_DATA_LENGTH = config['data']['TRAIN_DATA_LENGTH']


# Create a training file with simple derived features
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):

    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


class FeatureGeneratorOjisan(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = DATA_DIR / 'input/train.csv'
            self.total_data = int(TRAIN_DATA_LENGTH / self.chunk_size)
        else:
            submission = pd.read_csv(DATA_DIR / 'input/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, DATA_DIR / f'input/test/{seg_id}.csv'))
            self.total_data = int(len(submission))

    def slice_train_data(self, train, counter, index_tuple, denoising=False):
        df = train.iloc[index_tuple[0]:index_tuple[1], :]
        x = df.acoustic_data.values
        if denoising:
            x = deno.denoise_signal(deno.high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1)
        y = df.time_to_failure.values[-1]
        seg_id = 'train_' + str(counter)
        return list(x), y, seg_id

    def get_features_2(self, train, counter, index_tuple, denoising):
        """
        Gets three groups of features: from original data and from reald and imaginary parts of FFT.
        """
        x, y, seg_id = self.slice_train_data(train, counter, index_tuple, denoising=denoising)

        print(f'seg_id: {seg_id}, x length: {len(x)}, y: {y}')

        x = pd.Series(x)

        zc = np.fft.fft(x)
        realFFT = pd.Series(np.real(zc))
        imagFFT = pd.Series(np.imag(zc))

        main_dict = self.features(x, y, seg_id)
        r_dict = self.features(realFFT, y, seg_id)
        i_dict = self.features(imagFFT, y, seg_id)

        for k, v in r_dict.items():
            if k not in ['target', 'seg_id']:
                main_dict[f'fftr_{k}'] = v

        for k, v in i_dict.items():
            if k not in ['target', 'seg_id']:
                main_dict[f'ffti_{k}'] = v

        main_dict['seg_start'] = index_tuple[0]
        main_dict['seg_end'] = index_tuple[1]

        return main_dict

    def features(self, x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # create features here

        # lists with parameters to iterate over them
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        hann_windows = [50, 150, 1500, 15000]
        spans = [300, 3000, 30000, 50000]
        # windows = [10, 50, 100, 500, 1000, 10000]
        windows = [50, 100, 500, 1000]
        borders = list(range(-4000, 4001, 1000))
        peaks = [10, 20, 50, 100]
        coefs = [1, 5, 10, 50, 100]
        lags = [10, 100, 1000, 10000]
        # autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]
        autocorr_lags = [5, 10, 50]

        # basic stats
        # feature_dict['mean'] = x.mean()
        # feature_dict['std'] = x.std()
        # feature_dict['max'] = x.max()
        # feature_dict['min'] = x.min()

        # basic stats on absolute values
        # feature_dict['mean_change_abs'] = np.mean(np.diff(x))
        # feature_dict['abs_max'] = np.abs(x).max()
        # feature_dict['abs_mean'] = np.abs(x).mean()
        # feature_dict['abs_std'] = np.abs(x).std()

        # geometric and harminic means
        # feature_dict['hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
        feature_dict['gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))

        # k-statistic and moments
        # for i in range(1, 5):
        # for i in [3]:
        #     feature_dict[f'kstat_{i}'] = stats.kstat(x, i)
        #     feature_dict[f'moment_{i}'] = stats.moment(x, i)

        # for i in [1, 2]:
        #     feature_dict[f'kstatvar_{i}'] = stats.kstatvar(x, i)

        # aggregations on various slices of data
        # for agg_type, slice_length, direction in product(['std', 'min', 'max', 'mean'], [1000, 10000, 50000], ['first', 'last']):
        #     if direction == 'first':
        #         feature_dict[f'{agg_type}_{direction}_{slice_length}'] = x[:slice_length].agg(agg_type)
        #     elif direction == 'last':
        #         feature_dict[f'{agg_type}_{direction}_{slice_length}'] = x[-slice_length:].agg(agg_type)

        # feature_dict['max_to_min'] = x.max() / np.abs(x.min())
        # feature_dict['max_to_min_diff'] = x.max() - np.abs(x.min())
        # feature_dict['count_big'] = len(x[np.abs(x) > 500])
        # feature_dict['sum'] = x.sum()

        # feature_dict['mean_change_rate'] = calc_change_rate(x)
        # # calc_change_rate on slices of data
        # for slice_length, direction in product([1000, 10000, 50000], ['first', 'last']):
        #     if direction == 'first':
        #         feature_dict[f'mean_change_rate_{direction}_{slice_length}'] = calc_change_rate(x[:slice_length])
        #     elif direction == 'last':
        #         feature_dict[f'mean_change_rate_{direction}_{slice_length}'] = calc_change_rate(x[-slice_length:])

        # percentiles on original and absolute values
        for p in percentiles:
            feature_dict[f'percentile_{p}'] = np.percentile(x, p)
            feature_dict[f'abs_percentile_{p}'] = np.percentile(np.abs(x), p)

        # feature_dict['trend'] = add_trend_feature(x)
        # feature_dict['abs_trend'] = add_trend_feature(x, abs_values=True)

        # feature_dict['mad'] = x.mad()
        # feature_dict['kurt'] = x.kurtosis()
        # feature_dict['skew'] = x.skew()
        # feature_dict['med'] = x.median()

        # feature_dict['Hilbert_mean'] = np.abs(hilbert(x)).mean()

        # for hw in hann_windows:
        #     feature_dict[f'Hann_window_mean_{hw}'] = (convolve(x, hann(hw), mode='same') / sum(hann(hw))).mean()

        # feature_dict['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
        # feature_dict['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
        # feature_dict['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
        # feature_dict['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
        # feature_dict['classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
        # feature_dict['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
        # feature_dict['classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
        # feature_dict['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

        # exponential rolling statistics
        # ewma = pd.Series.ewm
        # for s in spans:
        #     feature_dict[f'exp_Moving_average_{s}_mean'] = (ewma(x, span=s).mean(skipna=True)).mean(skipna=True)
        #     feature_dict[f'exp_Moving_average_{s}_std'] = (ewma(x, span=s).mean(skipna=True)).std(skipna=True)
        #     feature_dict[f'exp_Moving_std_{s}_mean'] = (ewma(x, span=s).std(skipna=True)).mean(skipna=True)
        #     feature_dict[f'exp_Moving_std_{s}_std'] = (ewma(x, span=s).std(skipna=True)).std(skipna=True)

        feature_dict['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_dict['iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
        # feature_dict['ave10'] = stats.trim_mean(x, 0.1)

        for slice_length, threshold in product([50000, 100000, 150000],
                                                     [5, 10, 20, 50, 100]):
            feature_dict[f'count_big_{slice_length}_threshold_{threshold}'] = (np.abs(x[-slice_length:]) > threshold).sum()
            feature_dict[f'count_big_{slice_length}_less_threshold_{threshold}'] = (np.abs(x[-slice_length:]) < threshold).sum()

        # tfresh features take too long to calculate, so I comment them for now

        # feature_dict['abs_energy'] = feature_calculators.abs_energy(x)
        # feature_dict['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        # feature_dict['count_above_mean'] = feature_calculators.count_above_mean(x)
        # feature_dict['count_below_mean'] = feature_calculators.count_below_mean(x)
        # feature_dict['mean_abs_change'] = feature_calculators.mean_abs_change(x)
        # feature_dict['mean_change'] = feature_calculators.mean_change(x)
        # feature_dict['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)
        # feature_dict['range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
        # feature_dict['range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

        for i, j in zip(borders, borders[1:]):
            feature_dict[f'range_{i}_{j}'] = feature_calculators.range_count(x, i, j)

        # feature_dict['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        # feature_dict['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        # feature_dict['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        # feature_dict['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        # feature_dict['last_loc_max'] = feature_calculators.last_location_of_maximum(x)

        # for lag in lags:
        #     feature_dict[f'time_rev_asym_stat_{lag}'] = feature_calculators.time_reversal_asymmetry_statistic(x, lag)

        for autocorr_lag in autocorr_lags:
            feature_dict[f'autocorrelation_{autocorr_lag}'] = feature_calculators.autocorrelation(x, autocorr_lag)
            feature_dict[f'c3_{autocorr_lag}'] = feature_calculators.c3(x, autocorr_lag)

        # for coeff, attr in product([1, 2, 3, 4, 5], ['real', 'imag', 'angle']):
        #     feature_dict[f'fft_{coeff}_{attr}'] = list(feature_calculators.fft_coefficient(x, [{'coeff': coeff, 'attr': attr}]))[0][1]
        #
        # feature_dict['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        # feature_dict['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        # feature_dict['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        # feature_dict['cid_ce_1'] = feature_calculators.cid_ce(x, 1)
        #
        # for p in percentiles:
        #     feature_dict[f'binned_entropy_{p}'] = feature_calculators.binned_entropy(x, p)
        #
        # feature_dict['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)

        for peak in peaks:
            feature_dict[f'num_peaks_{peak}'] = feature_calculators.number_peaks(x, peak)

        for c in coefs:
            feature_dict[f'spkt_welch_density_{c}'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': c}]))[0][1]
            # feature_dict[f'time_rev_asym_stat_{c}'] = feature_calculators.time_reversal_asymmetry_statistic(x, c)

        # statistics on rolling windows of various sizes
        for w in windows:
            x_roll_std = x.rolling(w).std().dropna().values
            x_roll_mean = x.rolling(w).mean().dropna().values

            # feature_dict[f'ave_roll_std_{w}'] = x_roll_std.mean()
            # feature_dict[f'std_roll_std_{w}'] = x_roll_std.std()
            # feature_dict[f'max_roll_std_{w}'] = x_roll_std.max()
            # feature_dict[f'min_roll_std_{w}'] = x_roll_std.min()

            for p in percentiles:
                feature_dict[f'percentile_roll_std_{p}_window_{w}'] = np.percentile(x_roll_std, p)

            # feature_dict[f'av_change_abs_roll_std_{w}'] = np.mean(np.diff(x_roll_std))
            # feature_dict[f'av_change_rate_roll_std_{w}'] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            # feature_dict[f'abs_max_roll_std_{w}'] = np.abs(x_roll_std).max()

            # feature_dict[f'ave_roll_mean_{w}'] = x_roll_mean.mean()
            # feature_dict[f'std_roll_mean_{w}'] = x_roll_mean.std()
            # feature_dict[f'max_roll_mean_{w}'] = x_roll_mean.max()
            # feature_dict[f'min_roll_mean_{w}'] = x_roll_mean.min()

            for p in percentiles:
                feature_dict[f'percentile_roll_mean_{p}_window_{w}'] = np.percentile(x_roll_mean, p)

            # feature_dict[f'av_change_abs_roll_mean_{w}'] = np.mean(np.diff(x_roll_mean))
            # feature_dict[f'av_change_rate_roll_mean_{w}'] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            feature_dict[f'abs_max_roll_mean_{w}'] = np.abs(x_roll_mean).max()

        return feature_dict

    def generate_2(self, denoising=False):
        feature_list = []

        if self.dtype == 'train':
            train = pd.read_csv(DATA_DIR / 'input/train.csv', dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})

            start_indices = s3.read_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan36/start_indices_4k.csv', header=None)
            index_tuple_list = []
            for i in range(6):
                # i = 1
                for start_index in (start_indices[i] + i*104_832_580).values:
                    index_tuple_list.append((start_index, start_index+150000))

            res = Parallel(n_jobs=self.n_jobs, backend='threading')([delayed(self.get_features_2)(train, counter, index_tuple, denoising) for counter, index_tuple in enumerate(index_tuple_list)])

        else:
            res = Parallel(n_jobs=self.n_jobs,
                           backend='threading')(delayed(self.get_features)(x, y, s)
                                                for s, x, y in tqdm(self.read_chunks(denoising=denoising), total=self.total_data))

        for r in res:
            feature_list.append(r)

        return pd.DataFrame(feature_list)

def main():
    n_jobs = -1
    denoising = False

    feature_dir_name = f'add_ojisan36'
    # Save datasets
    out_dir = FEATURES_DIR / feature_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    training_fg = FeatureGeneratorOjisan(dtype='train', n_jobs=n_jobs, chunk_size=150000)
    training_data = training_fg.generate_2(denoising=denoising)

    X = training_data.drop(['target'], axis=1)

    means_dict = {}
    for col in X.columns:
        if X[col].isnull().any():
            print(col)
            mean_value = X.loc[X[col] != -np.inf, col].mean()
            X.loc[X[col] == -np.inf, col] = mean_value
            X[col] = X[col].fillna(mean_value)
            means_dict[col] = mean_value

    if denoising:
        X.to_csv(out_dir / f'train_features_denoised.csv', index=False)
    else:
        X.to_csv(out_dir / f'train_features.csv', index=False)

    # test_fg = FeatureGeneratorOjisan(dtype='test', n_jobs=n_jobs, chunk_size=150000)
    # test_data = test_fg.generate_2(denoising=denoising)
    #
    # X_test = test_data.drop(['target'], axis=1)
    # test_segs = test_data.seg_id
    # y = training_data.target
    #
    # for col in X_test.columns:
    #     if X_test[col].isnull().any():
    #         X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
    #         X_test[col] = X_test[col].fillna(means_dict[col])
    #
    # if denoising:
    #     X_test.to_csv(out_dir / 'test_features_denoised.csv', index=False)
    #     pd.DataFrame(y).to_csv(out_dir / f'y_denoised_{chunk_size}.csv', index=False)
    # else:
    #     X_test.to_csv(out_dir / 'test_features.csv', index=False)
    #     pd.DataFrame(y).to_csv(out_dir / f'y_{chunk_size}.csv', index=False)

    return


if __name__ == '__main__':
    main()
