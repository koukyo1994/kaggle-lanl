import pywt
import numpy as np

from scipy import signal
from scipy import stats


def get_peak_features(x):
    peaks, properties = signal.find_peaks(x)
    widths = signal.peak_widths(x, peaks)[0]
    prominences = signal.peak_prominences(x, peaks)[0]
    return {
        "count": peaks.size,
        "width_mean": widths.mean() if widths.size else -1,
        "width_max": widths.max() if widths.size else -1,
        "width_min": widths.min() if widths.size else -1,
        "prominence_mean": prominences.mean() if prominences.size else -1,
        "prominence_max": prominences.max() if prominences.size else -1,
        "prominence_min": prominences.min() if prominences.size else -1
    }


def signal_entropy(x):
    max_pos = x.argmax()
    for i in range(3):
        max_pos = x.argmax()
        x[max_pos - 1000:max_pos + 1000] = 0.0

    return stats.entropy(np.histogram(x, 15)[0])


def detailed_coeffs_entropy(x, wavelet="db1"):
    _, c_d = pywt.dwt(x, wavelet)
    return stats.entropy(np.histogram(c_d, 15)[0])


def bucketed_entropy(x):

    return {
        f'bucket_{i}': stats.entropy(np.histogram(bucket, 10)[0])
        for i, bucket in enumerate(np.split(x, 10))
    }


def calculate_signal_features(ts):
    features = []
    funcs = [
        np.mean, np.std, stats.kurtosis, get_peak_features, signal_entropy,
        detailed_coeffs_entropy, bucketed_entropy
    ]
    for f in funcs:
        feature = f(ts)
        if isinstance(feature, dict):
            feature = list(feature.values())
        if isinstance(feature, list):
            features.extend(feature)
        else:
            features.append(feature)
    return features
