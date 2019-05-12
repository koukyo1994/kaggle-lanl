import sys

import numpy as np
import pandas as pd

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
    from loader import test_generator

    data_dir = Path("../../../../input/test")
    generator = test_generator(data_dir)

    parser = ArgumentParser()
    parser.add_argument("--file_name", default="test_features.mat")
    parser.add_argument("--highpass", default=1000, type=int)
    parser.add_argument("--denoise", action="store_true")

    args = parser.parse_args()

    feature_list = []
    fname_list = []
    cnt = 0
    while True:
        try:
            data, fname = next(generator)
            fname_list.append(fname.name.replace(".csv", ""))
            ts = data["acoustic_data"].values

            sample_rate = 3916163.063108701
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
            print(cnt)
            cnt += 1
        except StopIteration:
            break

    features = np.asarray(feature_list)

    save_dir = Path("../../../../features/signals")
    save_dir.mkdir(exist_ok=True, parents=True)

    mat = {"features": features}
    savemat(save_dir / args.file_name, mat)

    sample_submission = pd.DataFrame(columns=["seg_id", "time_to_failure"])
    sample_submission["seg_id"] = fname_list
    sample_submission["time_to_failure"] = 0

    sample_submission.to_csv(save_dir / "sample_submission.csv", index=False)
