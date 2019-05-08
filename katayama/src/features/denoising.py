import os
import gc
import warnings

import numpy as np
from numpy.fft import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter, deconvolve

# os.chdir('./src')
from paths import *

# %load_ext autoreload
# %autoreload 2
warnings.filterwarnings('ignore')

SIGNAL_LEN = 150000
SAMPLE_RATE = 4000


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, SAMPLE_RATE=SAMPLE_RATE):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """

    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * SAMPLE_RATE
    norm_low_cutoff = low_cutoff / nyquist

    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

def main():
    # Load data
    seismic_signals = pd.read_csv(DATA_DIR / 'input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    acoustic_data = seismic_signals.acoustic_data
    time_to_failure = seismic_signals.time_to_failure
    data_len = len(seismic_signals)
    # Clear memory
    del seismic_signals
    gc.collect()

    # Slaice data
    signals = []
    targets = []
    for i in range(data_len//SIGNAL_LEN):
        min_lim = SIGNAL_LEN * i
        max_lim = min([SIGNAL_LEN * (i + 1), data_len])

        signals.append(list(acoustic_data[min_lim : max_lim]))
        targets.append(time_to_failure[max_lim])

    del acoustic_data
    del time_to_failure
    gc.collect()

    signals = np.array(signals)
    targets = np.array(targets)

    denoise_signal(high_pass_filter(signals[2], low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1)

if __name__ == '__main__':
    pass
