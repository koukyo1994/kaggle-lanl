import os
import time
import gc
import json
import builtins
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import eli5
import shap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats

# os.chdir('./katayama/src')
from paths import *


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
pd.options.display.precision = 15
warnings.filterwarnings('ignore')


def plot_feature_importance_lgb(model):
    # feature importance
    fold_importance = pd.DataFrame()
    fold_importance['feature'] = columns
    fold_importance['importance'] = model.feature_importances_
    fold_importance['fold'] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None):
    """
    Note:
        A function to train a variety of regression models.
        Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    Args:
        X: training data, can be pd.DataFrame or np.ndarray (after normalizing)
        X_test: test data, can be pd.DataFrame or np.ndarray (after normalizing)
        y: target
        folds: folds to split data
        model_type: type of model to use
        eval_metric: metric to use
        columns: columns to use. If None - use all columns
        plot_feature_importance: whether to plot feature importance of LGB
        model: sklearn model, works only for "sklearn" model type
    """
    columns = X.columns if columns == None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {
                        'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': mean_absolute_error
                        }
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=10000, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        elif model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1,)

        elif model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            plot_feature_importance_lgb()

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance['importance'] /= folds.n_splits
            cols = feature_importance[['feature', 'importance']].groupby('feature').mean().sort_values(
                by='importance', ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x='importance', y='feature', data=best_features.sort_values(by='importance', ascending=False));
            plt.title('LGB Features (avg over folds)');

            result_dict['feature_importance'] = feature_importance

    return result_dict

def scale_dataset(X, X_test):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_scaled, X_test_scaled, scaler

def create_dataset_nn(X, X_test, n=10):
    # n = 10;

    # Execute NearestNeighbors
    neigh = NearestNeighbors(n, n_jobs=-1)
    neigh.fit(X)

    dists, _ = neigh.kneighbors(X, n_neighbors=n)
    mean_dist = dists.mean(axis=1)
    max_dist = dists.max(axis=1)
    min_dist = dists.min(axis=1)

    X['mean_dist'] = mean_dist
    X['max_dist'] = max_dist
    X['min_dist'] = min_dist

    test_dists, _ = neigh.kneighbors(X_test, n_neighbors=n)

    test_mean_dist = test_dists.mean(axis=1)
    test_max_dist = test_dists.max(axis=1)
    test_min_dist = test_dists.min(axis=1)

    X_test['mean_dist'] = test_mean_dist
    X_test['max_dist'] = test_max_dist
    X_test['min_dist'] = test_min_dist

    return X_scaled, X_test

def create_submission_file(y_pred):
    submission = pd.read_csv(DATA_DIR / 'input/sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = y_pred
    return submission

def main():
    # load featureed datasets
    train_features = pd.read_csv(FEATURES_DIR / 'lanl-features/train_features.csv')
    test_features = pd.read_csv(FEATURES_DIR / 'lanl-features/test_features.csv')

    train_features_denoised = pd.read_csv(FEATURES_DIR / 'lanl-features/train_features_denoised.csv')
    test_features_denoised = pd.read_csv(FEATURES_DIR / 'lanl-features/test_features_denoised.csv')
    train_features_denoised.columns = [f'{i}_denoised' for i in train_features_denoised.columns]
    test_features_denoised.columns = [f'{i}_denoised' for i in test_features_denoised.columns]

    y = pd.read_csv(FEATURES_DIR / 'lanl-features/y.csv')


    # shape data for model
    X = pd.concat([train_features, train_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)
    X_test = pd.concat([test_features, test_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)
    X = X[:-1]
    y = y[:-1]


    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

    params = {'num_leaves': 128,
              'min_child_samples': 79,
              'objective': 'gamma',
              'max_depth': -1,
              'learning_rate': 0.01,
              'boosting_type': 'gbdt',
              'subsample_freq': 5,
              'subsample': 0.9,
              'bagging_seed': 11,
              'metric': 'mae',
              'verbosity': -1,
              'reg_alpha': 0.1302650970728192,
              'reg_lambda': 0.3603427518866501,
              'colsample_bytree': 0.1
              }

    result_dict_lgb = train_model_regression(X=X,
                                             X_test=X_test,
                                             y=y,
                                             params=params,
                                             folds=folds,
                                             plot_feature_importance=False
                                             )

    print(np.mean(result_dict_lgb['scores']))

    submission = create_submission_file(result_dict_lgb['prediction'])
    submission.to_csv(DATA_DIR / 'output/best_kernel/submission.csv')

    sub1 = pd.read_csv(FEATURES_DIR / 'lanl-features/submission_1.csv')
    sub1.to_csv('submission_1.csv', index=False)


    # NN features
    # Scale data
    X_scaled, X_test_scaled, scaler = scale_dataset(X, X_test)

    # Execute NearestNeighbors
    X_nn, X_test_nn = create_dataset_nn(X_scaled, X_test_scaled)

    params = {'num_leaves': 32,
              'min_data_in_leaf': 79,
              'objective': 'gamma',
              'max_depth': -1,
              'learning_rate': 0.01,
              'boosting': 'gbdt',
              'bagging_freq': 5,
              'bagging_fraction': 0.8126672064208567,
              'bagging_seed': 11,
              'metric': 'mae',
              'verbosity': -1,
              'reg_alpha': 0.1302650970728192,
              'reg_lambda': 0.3603427518866501,
              'feature_fraction': 0.1
              }
    result_dict_lgb_nn = train_model_regression(X=X_nn,
                                                X_test=X_test_nn,
                                                y=y,
                                                params=params,
                                                folds=folds,
                                                plot_feature_importance=False
                                                )

    print(np.mean(result_dict_lgb_nn['scores']))

    submission_nn = create_submission_file(result_dict_lgb_nn['prediction'])
    submission_nn.to_csv(DATA_DIR / 'output/best_kernel/submission_nn.csv')


    # Model interpretation
    top_columns = ['iqr1_denoised', 'percentile_5_denoised', 'abs_percentile_90_denoised', 'percentile_95_denoised', 'ave_roll_std_10', 'num_peaks_10', 'percentile_roll_std_20',
                   'ratio_unique_values_denoised', 'fftr_percentile_roll_std_75_denoised', 'num_crossing_0_denoised', 'percentile_95', 'ffti_percentile_roll_std_75_denoised',
                   'min_roll_std_10000', 'percentile_roll_std_1', 'percentile_roll_std_10', 'fftr_percentile_roll_std_70_denoised', 'ave_roll_std_50', 'ffti_percentile_roll_std_70_denoised',
                   'exp_Moving_std_300_mean_denoised', 'ffti_percentile_roll_std_30_denoised', 'mean_change_rate', 'percentile_roll_std_5', 'range_-1000_0', 'mad',
                   'fftr_range_1000_2000_denoised', 'percentile_10_denoised', 'ffti_percentile_roll_std_80', 'percentile_roll_std_25', 'fftr_percentile_10_denoised',
                   'ffti_range_-2000_-1000_denoised', 'autocorrelation_5', 'min_roll_std_100', 'fftr_percentile_roll_std_80', 'min_roll_std_500', 'min_roll_std_50', 'min_roll_std_1000',
                   'ffti_percentile_20_denoised', 'iqr1', 'classic_sta_lta5_mean_denoised', 'classic_sta_lta6_mean_denoised', 'percentile_roll_std_10_denoised',
                   'fftr_percentile_70_denoised', 'ffti_c3_50_denoised', 'ffti_percentile_roll_std_75', 'abs_percentile_90', 'range_0_1000', 'spkt_welch_density_50_denoised',
                   'ffti_percentile_roll_std_40_denoised', 'ffti_range_-4000_-3000', 'mean_change_rate_last_50000']


    X_train, X_valid, y_train, y_valid = train_test_split(X[top_columns], y, test_size=0.1)
    model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1, verbose=-1)
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
            verbose=10000, early_stopping_rounds=200)

    perm = eli5.sklearn.PermutationImportance(model, random_state=1).fit(X_train, y_train)
