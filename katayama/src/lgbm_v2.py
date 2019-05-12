import os
import time
import gc
import json
import builtins
import datetime
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# os.chdir('./src')
from paths import *
import util.log_functions as log

# %load_ext autoreload
# %autoreload 2
pd.options.display.precision = 15
warnings.filterwarnings('ignore')


def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None, early_stopping_rounds=50):
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
            model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=10000, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=500, params=params)
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
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance['feature'] = columns
            fold_importance['importance'] = model.feature_importances_
            fold_importance['fold'] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

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

    return result_dict, feature_importance

def drop_high_cc_features(df, th):
    df_corr = df.corr()

    retained_features = []
    removed_features = []
    features = df_corr.index.tolist()
    for feature1 in features:
        if feature1 in removed_features:
            continue
        else:
            retained_features.append(feature1)

        for feature2 in features:
            if feature1 == feature2:
                continue

            if df_corr.loc[feature1, feature2] > th:
                removed_features.append(feature2)

    return retained_features

def create_submission_file(y_pred):
    submission = pd.read_csv(DATA_DIR / 'input/sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = y_pred
    return submission

def make_log_filename():
    try:
        filename = os.path.basename(__file__) # with .py file
    except:
        filename = os.path.basename('lgbm') # with atom hydrogen

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_filename = 'log_{filename}_{time}.log'.format(filename=filename, time=time)
    return log_filename


def define_args():
    parser = argparse.ArgumentParser()
    list_func = lambda x:list(map(str, x.split(',')))

    parser.add_argument('slide_size', type=int)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--random_state', default=11, type=int)

    return parser

def get_and_validate_args(args):
    slide_size = args.slide_size
    n_fold = args.n_fold
    random_state = args.random_state

    return slide_size, n_fold, random_state

def main():
    # Arguments
    # data_ver = 1; slide_size = 50000; aug_feature_ratio = 90; n_fold = 5; random_state = 11; early_stopping_rounds = 5
    slide_size, n_fold, random_state = get_and_validate_args(args)

    model_ver = 2

    # Define logger
    global logger
    logger = log.define_logger(make_log_filename())

    # load featureed datasets
    train = pd.read_csv(DATA_DIR/'input'/'featured'/f'featured_train_ver{data_ver}_{slide_size}_{aug_feature_ratio}.csv')
    test = pd.read_csv(DATA_DIR/'input'/'featured'/f'featured_test_ver{data_ver}_{slide_size}_{aug_feature_ratio}.csv')

    y_train = train[['target']]
    X_train = train.drop('target', axis=1)
    X_test = test.copy()

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)

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

    result_dict_lgb, importances = train_model_regression(X=X_train,
                                                          X_test=X_test,
                                                          y=y_train,
                                                          params=params,
                                                          folds=folds,
                                                          plot_feature_importance=True,
                                                          early_stopping_rounds=early_stopping_rounds
                                                          )

    importances = importances[['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()


    train_features = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/train_features_{slide_size}.csv')
    train_features_denoised = pd.read_csv(FEATURES_DIR / f'lanl-features-{slide_size}/train_features_denoised_{slide_size}.csv')
    train_features_denoised.columns = [f'{i}_denoised' for i in train_features_denoised.columns]
    th = 0.975
    retained_features = drop_high_cc_features(pd.concat([train_features, train_features_denoised], axis=1)[:-1], th)
    X_train_red = X_train[retained_features]
    X_test_red = X_test[retained_features]

    result_dict_lgb_red, importances_red = train_model_regression(X=X_train_red,
                                                          X_test=X_test_red,
                                                          y=y_train,
                                                          params=params,
                                                          folds=folds,
                                                          plot_feature_importance=True,
                                                          early_stopping_rounds=early_stopping_rounds
                                                          )

    importances_red = importances_red[['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

    submission = create_submission_file(result_dict_lgb_red['prediction'])
    submission.to_csv(DATA_DIR / f'output/best_kernel/submission_lgbm_dver{data_ver}_{slide_size}slide_topall_corrred1_esr{early_stopping_rounds}_sharg{aug_feature_ratio}.csv')

    # 100: 2.3828563944616645
    # 200: 2.3679517421261935
    # 300: 2.3789489122758023
    # 400: 2.3918688865133584
    # 500: 2.4054866885002766
    # 600: 2.4136703579330736
    # 700: 2.4161196546293704
    # 800: 2.4199258041886225
    # 900: 2.422044245630842
    # 1000: 2.4314562691205506
    n_tops = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    cv_means_dict = {}
    for n_top in n_tops:
        # n_top = n_tops[0]
        top_features = importances_red.iloc[:n_top, :]['feature'].tolist()

        X_train_selected = X_train_red[top_features]
        X_test_selected = X_test_red[top_features]

        result_dict_lgb_selected, importances_selected = train_model_regression(X=X_train_selected,
                                                                                X_test=X_test_selected,
                                                                                y=y_train,
                                                                                params=params,
                                                                                folds=folds,
                                                                                plot_feature_importance=True,
                                                                                early_stopping_rounds=early_stopping_rounds
                                                                                )
        cv_means_dict[n_top] = np.mean(result_dict_lgb_selected['scores'])
        print(f'n_top:{n_top}, cv mean:{np.mean(result_dict_lgb_selected["scores"])}')


    n_top = 600 # slide_sizeが50000の場合
    # n_top = 1000 # slide_sizeが30000の場合
    top_features = importances_red.iloc[:n_top, :]['feature'].tolist()

    X_train_selected = X_train_red[top_features]
    X_test_selected = X_test_red[top_features]

    result_dict_lgb_selected, importances_selected = train_model_regression(X=X_train_selected,
                                                                            X_test=X_test_selected,
                                                                            y=y_train,
                                                                            params=params,
                                                                            folds=folds,
                                                                            plot_feature_importance=True
                                                                            )

    submission = create_submission_file(result_dict_lgb_selected['prediction'])
    submission.to_csv(DATA_DIR / f'output/best_kernel/submission_lgbm_dver{data_ver}_{slide_size}slide_top{n_top}_corrred1_esr{early_stopping_rounds}_sharg{aug_feature_ratio}.csv')


if __name__ == '__main__':
    main()
