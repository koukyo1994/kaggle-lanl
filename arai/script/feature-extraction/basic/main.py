import sys

import numpy as np

from pathlib import Path

from scipy.io import savemat

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")
    sys.path.append("../common")

    from basic import transform_ts
    from denoising import highpass_denoise
    from loader import data_generator

    data_dir = Path("../../../../input/train_split")
    generator = data_generator(data_dir)

    feature_list = []
    target_list = []
    cnt = 0
    while True:
        try:
            data = next(generator)
            ts = data["acoustic_data"].values
            time = data["time_to_failure"].values

            duration = time[0] - time[-1]
            nrows = len(data)
            sample_rate = nrows * (1 / duration)
            ts = highpass_denoise(
                ts, low_cutoff=10000, sample_rate=sample_rate)
            features = transform_ts(ts)
            feature_list.append(features)
            target_list.append(time[-1])
            print(cnt)
            cnt += 1
        except StopIteration:
            break

    features = np.asarray(feature_list)
    target = np.array(target_list)

    save_dir = Path("../../../../features/basic")
    save_dir.mkdir(exist_ok=True, parents=True)

    mat = {"features": features, "target": target}
    savemat(save_dir / "features.mat", mat)
