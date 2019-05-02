import sys

from argparse import ArgumentParser

from scipy.io import loadmat

if __name__ == '__main__':
    sys.path.append("../../../..")
    sys.path.append("../../..")
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")

    from arai.script.models.lstm_attention import LSTMAttentionNet
    from arai.script.trainer.nn_trainer import NNTrainer
    from arai.script.utils.logging import get_logger
    # from arai.script.utils.timer import timer

    parser = ArgumentParser()
    parser.add_argument(
        "--features", default="../../../features/basic/features.mat")
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--linear_size", default=64, type=int)
    parser.add_argument("--n_attention", default=20, type=int)

    args = parser.parse_args()

    logger = get_logger("Main", tag="lstm-attention")

    features_dict = loadmat(args.features)

    X = features_dict["features"]
    y = features_dict["target"].reshape(-1)

    trainer = NNTrainer(
        LSTMAttentionNet,
        logger,
        lr=0.001,
        train_batch=32,
        kwargs={
            "hidden_size": args.hidden_size,
            "linear_size": args.linear_size,
            "n_attention": args.n_attention,
            "input_shape": X.shape
        })

    trainer.fit(X, y, n_epochs=5)
