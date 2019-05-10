import os
import logging
import datetime
import codecs
import configparser
from logging import getLogger, StreamHandler, FileHandler, shutdown


loggers = {} # type: Dict[str, logging.Logger]
def set_logger(name: str, log_path:str)->logging.Logger:
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
        slackFormatter = logging.Formatter('@channel\n%(asctime)s - %(levelname)s : %(message)s')

        fhandler = FileHandler(filename = log_path)
        fhandler.setLevel(logging.INFO)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

        shandler = StreamHandler()
        shandler.setLevel(logging.DEBUG)
        shandler.setFormatter(formatter)
        logger.addHandler(shandler)

        # loggers.update(dict(name=logger))
        loggers[name] = logger

        return logger

def define_logger(log_filename):
    logger_name = config['log']['logger_name']
    log_path = os.path.join(LOG_DIR, log_filename)
    logger = set_logger(logger_name, log_path)
    return logger

def killLoggers():
    for l in loggers:
        logger = loggers.get(l)
        for h in logger.handlers:
            logger.removeHandler(h)
    shutdown()
    return
