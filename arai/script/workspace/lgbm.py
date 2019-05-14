import sys

import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

from scipy.io import loadmat

if __name__ == '__main__':
    sys.path.append("../../../..")
    sys.path.append("../../..")
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")

    from arai.script.trainer.lgbm_trainer import LGBMTrainer
    from arai.script.utils.logging import get_logger

    parser = ArgumentParser()
    parser.add_argument(
        "--features", default="../../../features/signals/features.mat")
    parser.add_argument(
        "--test", default="../../../features/signals/test_features.mat")
    parser.add_argument(
        "--sample_submission",
        default="../../../features/signals/sample_submission.csv")
    parser.add_argument("--num_leaves", default=31, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--min_child_weight", default=1e-3, type=float)
    parser.add_argument("--subsample", default=0.8, type=float)
    parser.add_argument("--subsample_freq", default=5, type=int)
    parser.add_argument("--colsample_bytree", default=0.8, type=float)
    parser.add_argument("--reg_alpha", default=0.01, type=float)
    parser.add_argument("--reg_lambda", default=0.01, type=float)
    parser.add_argument("--n_jobs", default=4, type=int)

    args = parser.parse_args()

    logger = get_logger("Main", tag="lightgbm")

    args_dir = dir(args)
    for d in args_dir:
        if d[0] != "_":
            logger.info(f"{d}: {eval('args.' + d)}")

    features_dict = loadmat(args.features)
    test_feat_dict = loadmat(args.test)
    sample_submission = pd.read_csv(args.sample_submission)

    X = features_dict["features"]
    y = features_dict["target"].reshape(-1)

    X_test = test_feat_dict["features"]

    trainer = LGBMTrainer(
        logger,
        n_splits=5,
        kwargs={
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "subsample_freq": args.subsample_freq,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "n_jobs": args.n_jobs,
            "objective": "regression_l1"
        })

    trainer.fit(X, y, n_epochs=5000)
    prediction = trainer.predict(X_test)

    prediction_dir = Path("../../../prediction") / ("lgbm" + trainer.tag)
    prediction_dir.mkdir(exist_ok=True, parents=True)

    sample_submission["time_to_failure"] = prediction
    sample_submission.to_csv(prediction_dir / "submission.csv", index=False)
