import os

import numpy as np
import pandas as pd

# os.chdir('./src')
from paths import *


def read_train_data(**kwargs):
    pd.options.display.precision = 15
    return pd.read_csv(str(DATA_DIR/'input'/'train.csv'), dtype={'acoustic_data':np.float64, 'time_to_failure':np.float64}, **kwargs)
