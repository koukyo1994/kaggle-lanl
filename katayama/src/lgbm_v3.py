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
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from scipy.stats import pearsonr

# os.chdir('./src')

from paths import *
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

def lgbm_oof(X, y, X_test, folds, params, early_stopping_rounds, random_seed=None):
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

        model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1, random_seed=random_seed)
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

    return result_dict

def create_submission_file(y_pred):
    submission = pd.read_csv(DATA_DIR / 'input/sample_submission.csv', index_col='seg_id')
    submission['time_to_failure'] = y_pred
    return submission

def main():
    with open(SRC_DIR/'models'/'stacking_params'/f'lgbm6.json', 'r') as f:
        params = json.load(f)
        params['shuffle'] = params['shuffle'] == 'True'

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
    result_dict = lgbm_oof(X_train, y_train, X_test, folds, params['feature_selection_params'], params['feature_selection_ear'])
    importances = result_dict['feature_importance'][['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
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

    random_seeds = [1,2,3,4,5,6,7,8,9,10]
    result_dict_dict = {}
    for random_seed in random_seeds:
        # random_seed = random_seeds[0]
        # 予測の実行
        result_dict = lgbm_oof(X_train, y_train, X_test, folds, params['prediction_params'], params['prediction_ear'], random_seed=random_seed)
        result_dict_dict[random_seed] = result_dict

    prediction = {}
    oof = {}
    for random_seed in result_dict_dict.keys():
        prediction[random_seed] = result_dict_dict[random_seed]['prediction']
        oof[random_seed] = result_dict_dict[random_seed]['oof']
    prediction = pd.DataFrame(prediction)
    oof = pd.DataFrame(oof)

    submission = create_submission_file(prediction.mean(axis=1).values)

    s3.to_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_best_seedave_oof.csv', pd.DataFrame({'oof':oof.mean(axis=1).values}), index=False)
    s3.to_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_best_seedave.csv', submission)

    return


if __name__ == '__main__':
    main()
