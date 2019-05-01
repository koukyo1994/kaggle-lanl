import numpy as np

import pywt

from scipy import signal
from scipy.signal import butter


def _maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def _high_pass_filter(x, low_cutoff, sample_rate):
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

    sos = butter(10, Wn=[norm_low_cutoff], btype="highpass", output="sos")
    filtered_sig = signal.sosfilt(sos, x)
    return filtered_sig


def _denoise_signal(x, wavelet="db4", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * _maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard")
                 for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")


def highpass_denoise(x, low_cutoff, sample_rate):
    x = _high_pass_filter(x, low_cutoff, sample_rate)
    x = _denoise_signal(x, wavelet="haar", level=1)
    return x
