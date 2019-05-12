import sys

import numpy as np

from argparse import ArgumentParser
from pathlib import Path

from scipy.io import savemat

if __name__ == '__main__':
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")
    sys.path.append("../common")

    from signals import calculate_signal_features
    from denoising import highpass_denoise
    from denoising import denoise, _high_pass_filter
    from loader import data_generator

    data_dir = Path("../../../../input/train_split")
    generator = data_generator(data_dir)

    parser = ArgumentParser()
    parser.add_argument("--file_name", default="features.mat")
    parser.add_argument("--highpass", default=1000, type=int)
    parser.add_argument("--denoise", action="store_true")

    args = parser.parse_args()

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
            if args.highpass > 0 and args.denoise:
                ts = highpass_denoise(
                    ts, low_cutoff=args.highpass, sample_rate=sample_rate)
            elif args.highpass > 0 and not args.denoise:
                ts = _high_pass_filter(
                    ts, low_cutoff=args.highpass, sample_rate=sample_rate)
            elif args.highpass == 0 and args.denoise:
                ts = denoise(ts)
            else:
                pass
            features = calculate_signal_features(ts)
            feature_list.append(np.array(features))
            target_list.append(time[-1])
            print(cnt)
            cnt += 1
        except StopIteration:
            break

    features = np.asarray(feature_list)
    target = np.array(target_list)

    save_dir = Path("../../../../features/signals")
    save_dir.mkdir(exist_ok=True, parents=True)

    mat = {"features": features, "target": target}
    savemat(save_dir / args.file_name, mat)
