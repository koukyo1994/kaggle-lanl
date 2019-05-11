import numpy as np
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
    change = change[~change != -np.inf]
    change = change[change != np.inf]
    return [np.mean(change), np.std(change), np.median(change)]


def mean(x):
    return x.mean()


def std(x):
    return x.std()


def ts_max(x):
    return x.max()


def ts_min(x):
    return x.min()


def mean_change_abs(x):
    return np.mean(np.diff(x))


def abs_max(x):
    return np.abs(x).max()


def abs_std(x):
    return np.abs(x).std()


def abs_mean(x):
    return np.abs(x).mean()


def hmean(x):
    return stats.hmean(np.abs(x[np.nonzero(x)[0]]))


def gmean(x):
    return stats.gmean(np.abs(x[np.nonzero(x)[0]]))


def kstat(x):
    return [stats.kstat(x, i) for i in range(1, 5)]


def moment(x):
    return [stats.moment(x, i) for i in range(1, 5)]


def kstatvar(x):
    return [stats.kstatvar(x, i) for i in range(1, 2)]


def slice_agg(x):
    agg = ["std", "min", "max", "mean", "median"]
    slice_ = [1000, 10000, 50000]
    feats = []
    for agg_type, slice_len in product(agg, slice_):
        feats.append(x[:slice_len].agg(agg_type))
        feats.append(x[-slice_len:].agg(agg_type))
    return feats


def max_to_min(x):
    return x.max() / np.abs(x.min())


def max_to_min_diff(x):
    return x.max() - np.abs(x.min())


def count_big(x):
    return len(x[np.abs(x) > 500])


def ts_sum(x):
    return x.sum()


def slice_change(x):
    feats = []
    for slice_len in [1000, 10000, 50000]:
        feats.append(change_rate(x[:slice_len]))
        feats.append(change_rate(x[-slice_len:]))
    return feats


def percentile(x):
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 95, 99]
    return [np.percentile(x, p) for p in percentiles]


def abs_percentile(x):
    percentils = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 95, 99]
    return [np.percentile(np.abs(x), p) for p in percentils]


def trend(x):
    return linear_trend(x)


def abs_trend(x):
    return linear_trend(x, is_abs=True)


def mad(x):
    return x.mad()


def kurt(x):
    return x.kurtosis()


def skew(x):
    return x.skew()


def median(x):
    return x.median()


def hilbert_mean(x):
    return np.abs(signal.hilbert(x)).mean()


def hann_mean(x):
    hann_windows = [50, 150, 1500, 15000]
    feats = []
    for hw in hann_windows:
        feats.append((signal.convolve(x, signal.hann(hw), mode="same") / sum(
            signal.hann(hw))).mean())
    return feats


def classic_sta_lta_mean(x):
    sta = [500, 5000, 3333, 10000, 50, 100, 333, 40000]
    lta = [10000, 100000, 6666, 25000, 1000, 666, 10000]
    return [classic_sta_lta(x, s, l).mean() for s, l in zip(sta, lta)]
