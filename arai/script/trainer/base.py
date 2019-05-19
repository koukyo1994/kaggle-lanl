import abc

import numpy as np

from datetime import datetime as dt

from sklearn.model_selection import KFold


class AbstractTrainer(metaclass=abc.ABCMeta):
    def __init__(self, logger, n_splits=5, seed=42, objective="regression"):
        self.logger = logger
        self.n_splits = n_splits
        self.seed = seed
        self.objective = objective

        self.fold = KFold(n_splits, shuffle=True, random_state=seed)
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

        if (self.objective != "classification") and (self.objective !=
                                                     "regression"):
            raise NotImplementedError

    def __str__(self):
        return f"Trainer_{self.tag}"

    def fit(self, X, y, n_epochs=10):
        if self.objective == "classification":
            self.n_classes = 1
            self.train_preds = np.zeros((len(X), ))
        elif self.objective == "regression":
            self.train_preds = np.zeros((len(X), ))

        idx = np.arange(len(X))

        for i, (trn_idx, val_idx) in enumerate(self.fold.split(idx)):
            self.fold_num = i + 1
            self.logger.info("=" * 20)
            self.logger.info(f"Fold {self.fold_num}")
            self.logger.info("=" * 20)

            X_train, X_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_train, y_val = y[trn_idx], y[val_idx]

            valid_preds = self._fit(X_train, y_train, X_val, y_val, n_epochs)
            self.train_preds[val_idx] = valid_preds

        self.logger.info(f"End Training")
        self._save_config()

    @abc.abstractmethod
    def _fit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _val(self):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _save_config(self):
        raise NotImplementedError
