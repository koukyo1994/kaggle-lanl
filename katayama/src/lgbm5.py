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
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# os.chdir('./src')
# %load_ext autoreload
# %autoreload 2

from paths import *
import util.s3_functions as s3
import features.feature_resampling_augumentation as fraug


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
    json_name = 'lgbm74.json'
    with open(SRC_DIR/'models'/'stacking_params'/f'{json_name}', 'r') as f:
        params = json.load(f)
        params['shuffle'] = params['shuffle'] == 'True'

    # データをダウンロード
    X_train = s3.read_csv_in_s3(params['X_train_path'])
    y_train = s3.read_csv_in_s3(params['y_train_path'])
    X_test = s3.read_csv_in_s3(params['X_test_path'])

    # For debug
    # ojisan_columns = s3.read_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/input/featured/ojisan/train_x_sorted.csv').columns
    # added_columns = set(X_train) - set(ojisan_columns)

    # 不要なカラムを削除
    drops = ['seg_id', 'seg_id_denoised', 'seg_start', 'seg_start_denoised', 'seg_end', 'seg_end_denoised', 'Unnamed: 0']
    for drop in drops:
        if drop in X_train.columns:
            X_train = X_train.drop(drop, axis=1)
        if drop in X_test.columns:
            X_test = X_test.drop(drop, axis=1)
        if drop in y_train.columns:
            y_train = y_train.drop(drop, axis=1)

    print('input data sahpes: ', X_train.shape, y_train.shape, X_test.shape)

    # ピアソンの相関係数を用いた特徴量選択
    pear_drop_cols = drop_pearson(X_train, y_train, th=params['th_peason'])
    X_train = X_train.drop(pear_drop_cols, axis=1)
    X_test = X_test.drop(pear_drop_cols, axis=1)

    # drop_info = pd.DataFrame({'pear_drop_cols':pear_drop_cols})
    # drop_info['type'] = 'added'
    # drop_info['type'] = drop_info['type'].where(~drop_info['pear_drop_cols'].isin(ojisan_columns), 'original')
    # drop_info['type'].value_counts()/drop_info.shape[0]

    # 前回のモデルで落とした特徴量をドロップ
    drop_columns_list = pd.read_csv(SRC_DIR/'models'/'drop_columns'/f'drop_lgbm64.csv')
    best_drop_cols = set(X_train.columns) & set(drop_columns_list['drop_columns'].tolist())
    X_train = X_train.drop(best_drop_cols, axis=1)
    X_test = X_test.drop(best_drop_cols, axis=1)

    print('input data sahpes: ', X_train.shape, y_train.shape, X_test.shape)

    # モデルの寄与度で特徴量選択
    folds = KFold(n_splits=params['n_cv'], shuffle=params['shuffle'], random_state=params['random_state'])
    result_dict = lgbm_oof(X_train, y_train, X_test, folds, params['feature_selection_params'], params['feature_selection_ear'])
    importances = result_dict['feature_importance'][['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    lgbm_drop_cols = importances['feature'].iloc[params['n_feature_top']:].tolist()
    X_train = X_train.drop(lgbm_drop_cols, axis=1)
    X_test = X_test.drop(lgbm_drop_cols, axis=1)

    # drop_info = pd.DataFrame({'lgbm_drop_cols':lgbm_drop_cols})
    # drop_info['type'] = 'added'
    # drop_info['type'] = drop_info['type'].where(~drop_info['lgbm_drop_cols'].isin(ojisan_columns), 'original')
    # drop_info['type'].value_counts()/drop_info.shape[0]
    # importances['type'] = 'added'
    # importances['type'] = importances['type'].where(~importances['feature'].isin(ojisan_columns), 'original')
    # importances.groupby('type')['importance'].sum()/importances['importance'].sum()

    # 削除するカラムを保存
    use_columns = pd.Series(X_train.columns)
    use_columns.to_csv(SRC_DIR/'models'/'use_columns'/f'{json_name.split(".")[0]}.csv')
    drop_columns = pd.DataFrame({'drop_columns':pear_drop_cols+lgbm_drop_cols})
    drop_columns.to_csv(SRC_DIR/'models'/'drop_columns'/f'drop_{json_name.split(".")[0]}.csv', index=False)

    # PCAの特徴量を追加
    n_pca = 30

    pca = PCA()
    pca.fit(X_train)

    X_train_pca = pd.DataFrame(pca.transform(X_train)).iloc[:, :n_pca]
    X_train_pca.columns = [f'PCA_{x+1}' for x in X_train_pca.columns]
    X_train = pd.concat([X_train, X_train_pca], axis=1)

    X_test_pca = pd.DataFrame(pca.transform(X_test)).iloc[:, :n_pca]
    X_test_pca.columns = [f'PCA_{x+1}' for x in X_test_pca.columns]
    X_test = pd.concat([X_test, X_test_pca], axis=1)

    # Feature resamplingでデータ拡張
    dir_name = 'ojisan36_added12'

    X_train_aug_path = f's3://kaggle-nowcast/kaggle_lanl/data/input/featured/{dir_name}/train_x_aug_{params["feature_resampling_rate"]}.csv'
    X_train_aug = s3.read_csv_in_s3(X_train_aug_path)

    if pca != 0:
        filename = f'train_x_aug_{params["feature_resampling_rate"]}_pca{n_pca}.csv'
        if s3.exists(f's3://kaggle-nowcast/kaggle_lanl/data/input/featured/{dir_name}/{filename}'):
            X_train_pca_aug = fraug.create_augumentation(X_train_pca, params["feature_resampling_rate"])
            s3.to_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/input/featured/{dir_name}/{filename}', X_train_pca_aug, index=False)
        else:
            X_train_pca_aug = s3.read_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/input/featured/{dir_name}/{filename}')

        X_train_aug = pd.concat([X_train_aug, X_train_pca_aug], axis=1)

    y_train_aug = s3.read_csv_in_s3(params['y_train_path'])

    X_train_aug = X_train_aug[X_train.columns]
    X_train = X_train.append(X_train_aug)
    y_train = y_train.append(y_train_aug)

    print(X_train.shape, y_train.shape, X_test.shape)

    random_seeds = [1,2,3,4,5,6,7,8,9,10]
    # random_seeds = [None]
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

    s3.to_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_ojisan36_ours1_aug06_pca{n_pca}_sav10_oof.csv', pd.DataFrame({'oof':oof.mean(axis=1).values}), index=False)
    s3.to_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_ojisan36_ours1_aug06_pca{n_pca}_sav10.csv', submission)
    # s3.to_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_best_seedave_oof.csv', pd.DataFrame({'oof':oof.mean(axis=1).values}), index=False)
    # s3.to_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_best_seedave.csv', submission)
    # s3.to_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_added_04_pca{n_pca}_oof.csv', pd.DataFrame({'oof':oof.mean(axis=1).values}), index=False)
    # s3.to_csv_in_s3(f's3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_added_04_pca{n_pca}.csv', submission)
    # s3.to_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_07_oof.csv', pd.DataFrame({'oof':oof.mean(axis=1).values}), index=False)
    # s3.to_csv_in_s3('s3://kaggle-nowcast/kaggle_lanl/data/output/lgbm_07.csv', submission)

    # result_dict = result_dict_dict[random_seed]
    # importances = result_dict['feature_importance'][['feature', 'importance']].groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    # importances['type'] = 'added'
    # importances['type'] = importances['type'].where(~importances['feature'].isin(ojisan_columns), 'original')
    # importances.groupby('type')['importance'].sum()/importances['importance'].sum()

    return


if __name__ == '__main__':
    main()
