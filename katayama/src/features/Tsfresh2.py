import os

import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

# os.chdir('./src')
from paths import *
import features.denoising as deno
from features.base import Feature
import util.util_functions as util

# %load_ext autoreload
# %autoreload 2


class Tsfresh2(Feature):
    def calc_features(self, x, record):
        params = {
            'has_duplicate_max': None,
            'has_duplicate_min': None,
            'has_duplicate': None,
            'sum_values': None,
            'mean_second_derivative_central': None,
            'length': None,
            'variance': None,
            'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
            'percentage_of_reoccurring_values_to_all_values': None,
            'sum_of_reoccurring_values': None,
            'sum_of_reoccurring_data_points': None,

            # 時系列の対称性を見るやつ。今回は不要と判断
            # 'symmetry_looking': [
            #     {'r': 0.0}, {'r': 0.05},
            #     {'r': 0.1}, {'r': 0.15},
            #     {'r': 0.2}, {'r': 0.25},
            #     {'r': 0.3}, {'r': 0.35},
            #     {'r': 0.4}, {'r': 0.45},
            #     {'r': 0.5}, {'r': 0.55},
            #     {'r': 0.6}, {'r': 0.65},
            #     {'r': 0.7}, {'r': 0.75},
            #     {'r': 0.8}, {'r': 0.85},
            #     {'r': 0.9}, {'r': 0.95}
            # ],

            # 1 sec
            'large_standard_deviation': [
                {'r': 0.05},
                {'r': 0.1},
                {'r': 0.2},
                {'r': 0.3},
                {'r': 0.4},
                {'r': 0.5},
                {'r': 0.6},
                {'r': 0.7},
                {'r': 0.8},
                {'r': 0.9},
            ],
            # 1 sec
            'agg_autocorrelation': [
                {'f_agg': 'mean', 'maxlag': 40},
                {'f_agg': 'median', 'maxlag': 40},
                {'f_agg': 'var', 'maxlag': 40}
            ],
            # 3 sec
            # 'partial_autocorrelation': [
            #     {'lag': 0}, {'lag': 1}, {'lag': 2}, {'lag': 3}, {'lag': 4}, {'lag': 5},
            #     {'lag': 6}, {'lag': 7}, {'lag': 8}, {'lag': 9}
            # ],
            # 2 min 43 sec
            # 'number_cwt_peaks': [
            #     {'n': 1}, {'n': 5}
            # ],
            'binned_entropy': [{'max_bins': 10}],
            # 1 sec
            'index_mass_quantile': [
                {'q': 0.1}, {'q': 0.2}, {'q': 0.3}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}, {'q': 0.8}, {'q': 0.9}
            ],
            # 1 sec
            'cwt_coefficients': [
                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2},
                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5},
                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 10},
                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 20},
                {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 2},
                {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 5},
                {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 10},
                {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 20},
                {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 2},
                {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 5},
                {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 10},
                {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 20},
                {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 2},
                {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 5},
                {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 10},
                {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 20},
                {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 2},
                {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 5},
                {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 10},
                {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 20},
                {'widths': (2, 5, 10, 20), 'coeff': 5, 'w': 2},
                {'widths': (2, 5, 10, 20), 'coeff': 5, 'w': 5},
                {'widths': (2, 5, 10, 20), 'coeff': 5, 'w': 10},
                {'widths': (2, 5, 10, 20), 'coeff': 5, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 11, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 11, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 11, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 11, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 20},
                # {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 2},
                # {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 5},
                # {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 10},
                # {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 20}
            ],
            # 1 sec
            'ar_coefficient': [
                {'coeff': 0, 'k': 10},
                {'coeff': 1, 'k': 10},
                {'coeff': 2, 'k': 10},
                {'coeff': 3, 'k': 10},
                {'coeff': 4, 'k': 10}
            ],

            # 1 min 4 sec
            # 多すぎるので一旦スキップ。
            # 'change_quantiles': [
            #     {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            #     {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            #     {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}
            # ],
            # 1 sec
            'fft_aggregated': [
                {'aggtype': 'centroid'},
                {'aggtype': 'variance'},
                {'aggtype': 'skew'},
                {'aggtype': 'kurtosis'}
            ],
            # 1 sec
            'value_count': [{'value': 0}, {'value': 1}, {'value': -1}],
            # 1 sec
            'friedrich_coefficients': [
                {'coeff': 0, 'm': 3, 'r': 30},
                {'coeff': 1, 'm': 3, 'r': 30},
                {'coeff': 2, 'm': 3, 'r': 30},
                {'coeff': 3, 'm': 3, 'r': 30}
            ],
            'max_langevin_fixed_point': [{'m': 3, 'r': 30}],
            # 1 sec
            'linear_trend': [
                {'attr': 'pvalue'},
                {'attr': 'rvalue'},
                {'attr': 'intercept'},
                {'attr': 'slope'},
                {'attr': 'stderr'}
            ],
            # 5 sec
            # 'agg_linear_trend': [
            #     {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'},
            #     {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'min'},
            #     {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'},
            #     {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'},
            #     # {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'max'},
            #     # {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'},
            #     # {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'},
            #     # {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'},
            #     # {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'},
            #     # {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'min'},
            #     # {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'mean'},
            #     # {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'var'},
            #     {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'},
            #     {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'},
            #     {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'},
            #     {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'},
            #     # {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'},
            #     # {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'},
            #     # {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'},
            #     # {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'},
            #     # {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'},
            #     # {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'},
            #     # {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'},
            #     # {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'},
            #     {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'max'},
            #     {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'min'},
            #     {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'},
            #     {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'var'},
            #     # {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'max'},
            #     # {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'min'},
            #     # {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'mean'},
            #     # {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'},
            #     # {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'},
            #     # {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'min'},
            #     # {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'},
            #     # {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'},
            #     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'},
            #     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'min'},
            #     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
            #     {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'},
            #     # {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'},
            #     # {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'},
            #     # {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'},
            #     # {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'var'},
            #     # {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'max'},
            #     # {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'},
            #     # {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'},
            #     # {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'var'}
            # ],
            # 38 sec
            # 'augmented_dickey_fuller': [
            #     {'attr': 'teststat'},
            #     {'attr': 'pvalue'},
            #     {'attr': 'usedlag'}
            # ],
            'number_crossing_m': [{'m': 0}, {'m': -1}, {'m': 1}],
            # 1 sec
            'energy_ratio_by_chunks': [
                {'num_segments': 10, 'segment_focus': 0},
                {'num_segments': 10, 'segment_focus': 1},
                {'num_segments': 10, 'segment_focus': 2},
                {'num_segments': 10, 'segment_focus': 3},
                {'num_segments': 10, 'segment_focus': 4},
                {'num_segments': 10, 'segment_focus': 5},
                {'num_segments': 10, 'segment_focus': 6},
                {'num_segments': 10, 'segment_focus': 7},
                {'num_segments': 10, 'segment_focus': 8},
                {'num_segments': 10, 'segment_focus': 9}
            ],
            'ratio_beyond_r_sigma': [
                {'r': 0.5}, {'r': 1}, {'r': 1.5}, {'r': 2}, {'r': 2.5}, {'r': 3},
                {'r': 5}, {'r': 6}, {'r': 7}, {'r': 10}
            ]
        }

        # x = pd.Series(train['acoustic_data'].values)
        input = pd.DataFrame(x)
        input['column_id'] = 99
        features = extract_features(input, column_id='column_id', default_fc_parameters=params, n_jobs=1)

        features.columns = self.cleanse_columns(features.columns)
        feature_record = {key:val for key, val in zip(features.columns, features.T.iloc[:,0].values)}

        record.update(feature_record)

        return record

    def cleanse_columns(self, columns):
        new_columns = []
        for column in columns:
            column = column.replace('0__', '')
            column = column.replace('"', '')
            new_columns.append(column)
        return new_columns

if __name__ == '__main__':
    tsfresh_data['seg_id']
    tsfresh2_data['seg_id']

    train_tmp = train.copy()
    train_tmp = train_tmp[['acoustic_data']]
    train_tmp['column_id'] = 1
    tmp = extract_features(train_tmp, column_id='column_id', default_fc_parameters=params, n_jobs=16)
    tmp.shape

    single_params = {
        'change_quantiles': [
            {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
            {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
            {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}
        ],
    }
    tmp_single = extract_features(train_tmp, column_id='column_id', default_fc_parameters=single_params, n_jobs=1)

    # 3 min 30 sec

    params = EfficientFCParameters()

    default_non_tsfresh_params = {
        'mean': None,
        'maximum': None,
        'minimum': None,
        'standard_deviation': None,
        'kurtosis': None,
        'skewness': None,
        'median': None
    }

    default_tsfresh_params = {
        'range_count': {'min': -np.inf, 'max': -4000},
        'range_count': {'min': -4000, 'max': -3000},
        'range_count': {'min': -3000, 'max': -2000},
        'range_count': {'min': -2000, 'max': -1000},
        'range_count': {'min': -1000, 'max': 0},
        'range_count': {'min': 0, 'max': 1000},
        'range_count': {'min': 1000, 'max': 2000},
        'range_count': {'min': 2000, 'max': 3000},
        'range_count': {'min': 3000, 'max': 4000},
        'range_count': {'min': 4000, 'max': np.inf},
        'autocorrelation': {
            'lag': [5, 10, 50, 100, 500, 1000, 5000, 10000]
        },
        'c3': {
            'lag': [5, 10, 50, 100, 500, 1000, 5000, 10000]
        },
        'number_peaks': {
            'n': [10, 20, 50, 100]
        },
        'spkt_welch_density': {
            'param': [{'coeff': 1}, {'coeff': 5}, {'coeff': 10}, {'coeff': 50}, {'coeff': 100}]
        },
        'time_reversal_asymmetry_statistic': {
            'lag': [1, 5, 10, 50, 100]
        }
    }

    tsfresh_params = {
        'abs_energy': None,
        'absolute_sum_of_changes': None,
        'count_above_mean': None,
        'count_below_mean': None,
        'mean_abs_change': None,
        'mean_change': None,
        'variance_larger_than_standard_deviation': None,
        'ratio_value_number_to_time_series_length': None,
        'first_location_of_minimum': None,
        'first_location_of_maximum': None,
        'last_location_of_minimum': None,
        'last_location_of_maximum': None,
        # 'fft_coefficient': {
        #     'coeff': [1,2,3,4,5],
        #     'attr': ['real', 'imag', 'angle', 'abs']
        # },
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'cid_ce': {
            'normalize': [False, True]
        }
    }

    drop_params = {
        'fft_coefficient': None
    }

    features = list(set(params.keys()) - (set(tsfresh_params.keys()) | set(default_tsfresh_params.keys()) | set(default_non_tsfresh_params.keys()) | set(drop_params.keys())))
    current_keys = list(params.keys())
    for key in current_keys:
        if not key in features:
            params.pop(key)
