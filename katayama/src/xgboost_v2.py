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
from scipy.stats import pearsonr
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


def train_model_regression(X, X_test, y, params, folds, eval_metric='mae', columns=None, plot_feature_importance=False, model=None, early_stopping_rounds=50):
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
    # X = X_train.copy(); y = y_train.copy();
    # columns=None; params={}; eval_metric='mae'

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

        train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=500, params=params)
        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        prediction += y_pred

        # feature importance
        if plot_feature_importance:
            fold_importance = pd.DataFrame({'feature': list(model.get_score().keys()), 'importance':list(model.get_score().values())})
            fold_importance['fold'] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores


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

def drop_pearson(X_train, y_train, th=0.01):
    pcol = []
    pcor = []
    pval = []
    for col in X_train.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(X_train[col].values, y_train['target'].values)[0]))
        pval.append(abs(pearsonr(X_train[col].values, y_train['target'].values)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df['pval'] <= th]

    drop_cols = []

    for col in X_train.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    return drop_cols

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
    # data_ver = 1; slide_size = 50000; aug_feature_ratio = 90; n_fold = 6; random_state = 11; early_stopping_rounds = 20
    slide_size, n_fold, random_state = get_and_validate_args(args)

    model_ver = 2

    # Define logger
    global logger
    logger = log.define_logger(make_log_filename())

    # load featureed datasets
    train = pd.read_csv(DATA_DIR/'input'/'featured'/f'featured_train_ver{data_ver}_{slide_size}_{aug_feature_ratio}.csv')
    test = pd.read_csv(DATA_DIR/'input'/'featured'/f'featured_test_ver{data_ver}_{slide_size}_{aug_feature_ratio}.csv')

    y_train = train[['target']]
    # X_train = train.drop(['target', 'seg_id'], axis=1)
    X_train = train.drop(['target'], axis=1)
    X_test = test.copy()

    drop_columns = list(filter(lambda x:x.find('target') != -1, X_test.columns))
    # X_test = X_test.drop(drop_columns+['seg_id'], axis=1)
    X_test = X_test.drop(drop_columns, axis=1)

    # 教師データと相関しない特徴量を落とす
    drop_cols = drop_pearson(X_train, y_train, th=0.001)
    X_train = X_train.drop(drop_cols, axis=1)
    X_test = X_test.drop(drop_cols, axis=1)

    logger.info(f'X_train:{X_train.shape}, y_train:{y_train.shape}, X_test:{X_test.shape}')

    folds = KFold(n_splits=n_fold, shuffle=False, random_state=random_state)

    # パラメーターチューニング
    params = {
        'max_depth': [3, 5, 10, 15],
        'learning_rate': [0.05, 0.1],
        'gamma': [0.5, 1, 2, 5],
        'min_child_weight': [1.0, 2.0, 5.0],
    }

    model = xgb.XGBRegressor(early_stopping_rounds=early_stopping_rounds, eval_metric='mae', seed=random_state)

    os.environ['JOBLIB_TEMP_FOLDER'] = '/home/katay'
    cv = GridSearchCV(model, params, cv=folds, verbose=10, scoring='mean_absolute_error', n_jobs=-1)
    cv.fit(X_train, y_train)

    cv.best_params_
    cv_result = pd.DataFrame(cv.cv_results_)
    cv_result.to_csv(SRC_DIR/'models'/'xgboost_grid_v2.csv')

    params = {'gamma': 0.5, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 2.0}
    result_dict_lgb, importances = train_model_regression(X=X_train,
                                                          X_test=X_test,
                                                          y=y_train,
                                                          params={},
                                                          folds=folds,
                                                          plot_feature_importance=True,
                                                          early_stopping_rounds=early_stopping_rounds
                                                          )

    importances = importances[['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

    submission = create_submission_file(result_dict_lgb['prediction'])
    submission.to_csv(DATA_DIR / f'output/best_kernel/submission_xgb_dver{data_ver}_{slide_size}slide_topall_pearson1_esr{early_stopping_rounds}_sharg{aug_feature_ratio}_gscv1.csv')



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


    n_top = 1000 # slide_sizeが50000の場合
    top_features = importances.iloc[:n_top, :]['feature'].tolist()
    # top_features = importances_red.iloc[:n_top, :]['feature'].tolist()

    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    # X_train_selected = X_train_red[top_features]
    # X_test_selected = X_test_red[top_features]

    result_dict_lgb_selected, importances_selected = train_model_regression(X=X_train_selected,
                                                                            X_test=X_test_selected,
                                                                            y=y_train,
                                                                            params={},
                                                                            folds=folds,
                                                                            plot_feature_importance=True,
                                                                            early_stopping_rounds=early_stopping_rounds
                                                                            )

    submission = create_submission_file(result_dict_lgb_selected['prediction'])
    submission.to_csv(DATA_DIR / f'output/best_kernel/submission_xgb_dver{data_ver}_{slide_size}slide_top{n_top}_pearson1_esr{early_stopping_rounds}_sharg{aug_feature_ratio}.csv')


if __name__ == '__main__':
    main()
