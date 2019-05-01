import numpy as np

from fastprogress import progress_bar


def _min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    ts = np.clip(ts, a_max=max_data, a_min=min_data)
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data + abs(min_data))
    if range_needed[0] < 0:
        return ts_std * (
            range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def transform_ts(ts, n_dim=150, min_max=(-1, 1)):
    ts_std = _min_max_transf(ts, min_data=-1000, max_data=999)
    bucket_size = int(150000 / n_dim)
    new_ts = []
    for i in progress_bar(range(0, 150000, bucket_size)):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]
        relative_percentile = percentil_calc - mean
        new_ts.append(
            np.concatenate([
                np.asarray([mean, std, std_top, std_bot, max_range]),
                percentil_calc, relative_percentile
            ]))
    return np.asarray(new_ts)
