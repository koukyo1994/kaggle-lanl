import os
import time
import gc
import json
import datetime
import warnings
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from scipy.stats import pearsonr

# os.chdir('./src')

import util.s3_functions as s3


pd.options.display.precision = 15
warnings.filterwarnings('ignore')


def drop_pearson(X_train, y_train, th=0.01):
    pcol = []
    pcor = []
    pval = []
    for col in X_train.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(X_train[col].values, y_train['time_to_failure'].values)[0]))
        pval.append(abs(pearsonr(X_train[col].values, y_train['time_to_failure'].values)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df['pval'] <= th]

    drop_cols = []

    for col in X_train.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    return drop_cols

def lgbm_oof(X, y, X_test, folds, params, early_stopping_rounds):
    columns = X.columns

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    models = []
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

        model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                verbose=10000, early_stopping_rounds=early_stopping_rounds)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance['feature'] = columns
        fold_importance['importance'] = model.feature_importances_
        fold_importance['fold'] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        models.append(model)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict = {}
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    result_dict['models'] = models

    feature_importance['importance'] /= folds.n_splits
    cols = feature_importance[['feature', 'importance']].groupby('feature').mean().sort_values(
        by='importance', ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(16, 12));
    sns.barplot(x='importance', y='feature', data=best_features.sort_values(by='importance', ascending=False));
    plt.title('LGB Features (avg over folds)');

    result_dict['feature_importance'] = feature_importance

    return result_dict, feature_importance

def main():
    params_dict = {
        'lgbm1': {
            'X_train_path': 's3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan/train_x_sorted.csv',
            'y_train_path': 's3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan/train_y_sorted.csv',
            'X_test_path': 's3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan/test_x.csv',
            'th_peason': 0.0001,
            'n_feature_top': 300,
            'feature_selection_ear': 200,
            'feature_selection_params': {
                'num_leaves': 128,
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
            },
            'feature_resampling_rate': 0.5,
            'prediction_ear': 5,
            'n_cv': 6,
            'shuffle': False,
            'prediction_params': {
                'num_leaves': 128,
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
            },
            'random_state': 11
        }
    }

    # 1層目の学習
    first_layer_models = {}
    first_layer_oofs = {}
    for model_name in params_dict.keys():
        # model_name = 'lgbm1'
        params = params_dict[model_name]

        # データをダウンロード
        X_train = s3.read_csv_in_s3(params['X_train_path'])
        y_train = s3.read_csv_in_s3(params['y_train_path'])
        X_test = s3.read_csv_in_s3(params['X_test_path'])

        # 不要なカラムを削除
        drops = ['seg_id', 'seg_start', 'seg_end', 'Unnamed: 0']
        for drop in drops:
            if drop in X_train.columns:
                X_train = X_train.drop(drop, axis=1)
            if drop in y_train.columns:
                y_train = y_train.drop(drop, axis=1)

        # ピアソンの相関係数を用いた特徴量選択
        drop_cols = drop_pearson(X_train, y_train, th=params['th_peason'])
        X_train = X_train.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)

        # モデルの寄与度で特徴量選択
        folds = KFold(n_splits=params['n_cv'], shuffle=params['shuffle'], random_state=params['random_state'])
        result_dict, importances = lgbm_oof(X_train, y_train, X_test, folds, params['feature_selection_params'], params['feature_selection_ear'])
        importances = importances[['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
        drop_cols = importances['feature'].iloc[params['n_feature_top']:].tolist()
        X_train = X_train.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)

        # Feature resamplingでデータ拡張
        X_train_aug_path = params['X_train_path'].replace('x_sorted', f'x_aug_{params["feature_resampling_rate"]}')
        X_train_aug = s3.read_csv_in_s3(X_train_aug_path)
        y_train_aug = s3.read_csv_in_s3('kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan/train_y.csv')

        X_train_aug = X_train_aug[X_train.columns]
        X_train = X_train.append(X_train_aug)
        y_train = y_train.append(y_train_aug)

        # 予測の実行
        result_dict, importances = lgbm_oof(X_train, y_train, X_test, folds, params['prediction_params'], params['prediction_ear'])

        # 1層の学習済みのモデルを保存
        first_layer_models[model_name] = result_dict['models']
        first_layer_oofs[model_name] = result_dict['oof']

    # 2層目の学習
    X_train_2 = pd.DataFrame(first_layer_oofs)
    model_2 = lgb.LGBMRegressor(n_estimators=50000, n_jobs=-1)
    model_2.fit(X_train_2, y_train, eval_metric='mae', verbose=10000)

    # テストデータに対する1層目の予測を実施
    y_pred_1 = {}
    for model_name in params_dict.keys():
        # model_name = 'lgbm1'
        y_preds = []
        for model in first_layer_models[model_name]:
            # model = first_layer_models[model_name][0]
            y_preds.append(model.predict(X_test))
        y_pred_1[model_name] = np.array(y_preds).mean(axis=0)
    y_pred_1 = pd.DataFrame(y_pred_1)

    # テストデータに対する2層目の予測を実施
    y_pred = model_2.predict(y_pred_1)

    return


if __name__ == '__main__':
    main()
