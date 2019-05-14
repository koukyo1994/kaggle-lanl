import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

from itertools import product

from sklearn.linear_model import LinearRegression


def linear_trend(ts, is_abs=False):
    idx = np.array(range(len(ts)))
    if is_abs:
        ts = np.abs(ts)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), ts)
    return lr.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.require(np.cumsum(x**2), dtype=np.float)
    lta = sta.copy()

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    sta[:length_lta - 1] = 0
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta


def change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[~(change != -np.inf)]
    change = change[change != np.inf]
    return {
        "mean_change": np.mean(change),
        "std_change": np.std(change),
        "median_change": np.median(change)
    }


def mean(x):
    return {"mean": x.mean()}


def std(x):
    return {"std": x.std()}


def ts_max(x):
    return {"max": x.max()}


def ts_min(x):
    return {"min": x.min()}


def mean_change_abs(x):
    return {"mean_change_abs": np.mean(np.diff(x))}


def abs_max(x):
    return {"abs_max": np.abs(x).max()}


def abs_std(x):
    return {"abs_std": np.abs(x).std()}


def abs_mean(x):
    return {"abs_mean": np.abs(x).mean()}


def hmean(x):
    return {"hmean": stats.hmean(np.abs(x[np.nonzero(x)[0]]))}


def gmean(x):
    return {"gmean": stats.gmean(np.abs(x[np.nonzero(x)[0]]))}


def kstat(x):
    return {f"kstat_{i}": stats.kstat(x, i) for i in range(1, 5)}


def moment(x):
    return {f"moment_{i}": stats.moment(x, i) for i in range(1, 5)}


def kstatvar(x):
    return {"kstatvar_{i}": stats.kstatvar(x, i) for i in range(1, 2)}


def slice_agg(x):
    agg = ["std", "min", "max", "mean", "median"]
    slice_ = [1000, 10000, 50000]
    feats = {}
    for agg_type, slice_len in product(agg, slice_):
        feats[f"first_{agg_type}_{slice_len}"] = x[:slice_len].agg(agg_type)
        feats[f"last_{agg_type}_{slice_len}"] = x[-slice_len:].agg(agg_type)
    return feats


def max_to_min(x):
    return {"max_to_min": x.max() / np.abs(x.min())}


def max_to_min_diff(x):
    return {"max_to_min_diff": x.max() - np.abs(x.min())}


def count_big(x):
    return {"count_big": len(x[np.abs(x) > 500])}


def ts_sum(x):
    return {"ts_sum": x.sum()}


def slice_change(x):
    feats = {}
    for slice_len in [1000, 10000, 50000]:
        first_change = change_rate(x[:slice_len])
        last_change = change_rate(x[-slice_len:])

        first_change = {
            f"first_{slice_len}_" + k: v
            for k, v in first_change.items()
        }
        last_change = {
            f"last_{slice_len}_" + k: v
            for k, v in last_change.items()
        }

        feats.update(first_change)
        feats.update(last_change)
    return feats


def percentile(x):
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 95, 99]
    return {f"{p}_percentiles": np.percentile(x, p) for p in percentiles}


def abs_percentile(x):
    percentils = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 95, 99]
    return {
        f"{p}_abs_percentiles": np.percentile(np.abs(x), p)
        for p in percentils
    }


def trend(x):
    return {"linear_trend": linear_trend(x)}


def abs_trend(x):
    return {"abs_linear_trend": linear_trend(x, is_abs=True)}


def mad(x):
    return {"mad": x.mad()}


def kurt(x):
    return {"kurtosis": x.kurtosis()}


def skew(x):
    return {"skewness": x.skew()}


def median(x):
    return {"median": x.median()}


def hilbert_mean(x):
    return {"hilbert_mean": np.abs(signal.hilbert(x)).mean()}


def hann_mean(x):
    hann_windows = [50, 150, 1500, 15000]
    feats = {}
    for hw in hann_windows:
        feats[f"hann_{hw}_mean"] = (signal.convolve(
            x, signal.hann(hw), mode="same") / sum(signal.hann(hw))).mean()
    return feats


def classic_sta_lta_mean(x):
    sta = [500, 5000, 3333, 10000, 50, 100, 333, 40000]
    lta = [10000, 100000, 6666, 25000, 1000, 666, 10000]
    return {
        f"classic_sta_{s}_lta_{l}": classic_sta_lta(x, s, l).mean()
        for s, l in zip(sta, lta)
    }


def exp_moving_agg(x):
    spans = [300, 3000, 30000, 50000]
    ewma = pd.Series.ewm
    feats = {}
    for s in spans:
        feats[f"exp_moving_average_{s}_mean"] = (ewma(
            x, span=s).mean(skipna=True)).mean(skipna=True)
        feats[f"exp_moving_average_{s}_std"] = (ewma(
            x, span=s).mean(skipna=True)).std(skipna=True)
        feats[f"exp_moving_std_{s}_mean"] = (ewma(
            x, span=s).std(skipna=True)).mean(skipna=True)
        feats[f"exp_moving_std_{s}_std"] = (ewma(
            x, span=s).std(skipna=True)).std(skipna=True)
    return feats


def iqr(x):
    feats = {}
    feats["iqr"] = np.subtract(*np.percentile(x, [75, 25]))
    feats["iqr1"] = np.subtract(*np.percentile(x, [95, 5]))
    return feats


def ave10(x):
    return {"ave10": stats.trim_mean(x, 0.1)}


def count_bigger(x):
    slices = [50000, 100000, 150000]
    threshold = [5, 10, 20, 50, 100]
    feats = {}
    for sl, th in product(slices, threshold):
        feats[f"count_big_{sl}_thres_{th}"] = (np.abs(x[-sl:]) > th).sum()
        feats[f"count_big_{sl}_thres_{th}"] = (np.abs(x[-sl:] < th)).sum()
    return feats


def roll_features(x):
    windows = [10, 50, 100, 500, 1000, 10000]
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 95, 99]
    feats = {}
    for w in windows:
        x_roll_std = x.rolling(w).std().dropna().values
        x_roll_mean = x.rolling(w).mean().dropna().values
        feats[f"avg_roll_std_{w}"] = x_roll_std.mean()
        feats[f"std_roll_std_{w}"] = x_roll_std.std()
        feats[f"max_roll_std_{w}"] = x_roll_std.max()
        feats[f"min_roll_std_{w}"] = x_roll_std.min()

        feats[f"avg_roll_mean_{w}"] = x_roll_mean.mean()
        feats[f"std_roll_mean_{w}"] = x_roll_mean.std()
        feats[f"max_roll_mean_{w}"] = x_roll_mean.max()
        feats[f"min_roll_mean_{w}"] = x_roll_mean.min()

        for p in percentiles:
            feats[f"percentile_roll_std_{p}_window_{w}"] = np.percentile(
                x_roll_std, p)
            feats[f"percentile_roll_mean_{p}_window_{w}"] = np.percentile(
                x_roll_mean, p)
        feats[f"avg_change_abs_roll_std_{w}"] = np.mean(np.diff(x_roll_std))
        feats[f"avg_change_abs_roll_mean_{w}"] = np.mean(np.diff(x_roll_mean))

        feats[f"avg_change_rate_roll_std_{w}"] = np.mean(
            np.nonzero(np.diff(x_roll_std) / x_roll_std[:-1])[0])
        feats[f"avg_change_rate_roll_mean_{w}"] = np.mean(
            np.nonzero(np.diff(x_roll_mean) / x_roll_mean[:-1])[0])

        feats[f"abs_max_roll_std_{w}"] = np.abs(x_roll_std).max()
        feats[f"abs_max_roll_mean_{w}"] = np.abs(x_roll_mean).max()
    return feats


def comprehensive_feats_dict(ts):
    ts = pd.Series(ts)
    zc = np.fft.fft(ts)
    real_fft = pd.Series(np.real(zc))
    imag_fft = pd.Series(np.imag(zc))

    funcs = [
        mean, std, ts_max, ts_min, ts_sum, slice_agg, slice_change,
        mean_change_abs, abs_max, abs_mean, abs_percentile, abs_std, abs_trend,
        hmean, gmean, kstat, kstatvar, moment, percentile, trend, mad, median,
        kurt, skew, hilbert_mean, hann_mean, max_to_min, max_to_min_diff,
        count_big, count_bigger, classic_sta_lta_mean, exp_moving_agg, iqr,
        ave10, roll_features
    ]
    feats = {}
    for f in funcs:
        feature = f(ts)
        feats.update(feature)

        rfft = f(real_fft)
        ifft = f(imag_fft)

        rfft = {"real_" + k: v for k, v in rfft.items()}
        ifft = {"imag_" + k: v for k, v in ifft.items()}
        feats.update(rfft)
        feats.update(ifft)
    return feats
