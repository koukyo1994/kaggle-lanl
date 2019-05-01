import sys
import logging

import datetime as dt

from pathlib import Path


def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
