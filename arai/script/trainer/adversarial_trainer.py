import pickle

import numpy as np

import lightgbm as lgb

from pathlib import Path

from arai.script.trainer.base import AbstractTrainer


class AdversarialTrainer(AbstractTrainer):
    def __init__(self,
                 logger,
                 n_splits=5,
                 seed=42,
                 objective='classification',
                 kwargs={}):
        super(AdversarialTrainer, self).__init__(
            logger, n_splits=n_splits, seed=seed, objective=objective)
        self.kwargs = kwargs

        path = Path(f"bin/lgb_{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

        self.trees = []

    def _log_evaluation(self, period=1, show_stdv=True):
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (
                    env.iteration + 1) % period == 0:
                result = '\t'.join([
                    lgb.callback._format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ])
                self.logger.info('[{}]\t{}'.format(env.iteration + 1, result))

        _callback.order = 10
        return _callback

    def _save_model(self, model):
        with open(self.path / f"model{self.fold_num}.pkl", "wb") as f:
            pickle.dump(model, f)

    def _fit(self, X_train, y_train, X_val, y_val, n_epochs):
        model = lgb.LGBMClassifier(n_estimators=n_epochs, **self.kwargs)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            callbacks=[self._log_evaluation(period=50)],
            verbose=False)
        self.trees.append(model)
        valid_preds = self._val(X_val, y_val, model)

        self._save_model(model)
        return valid_preds

    def _val(self, X_val, y_val, model):
        valid_preds = model.predict_proba(X_val)[:, 0]
        return valid_preds

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for m in self.trees:
            pred += m.predict_proba(X)[:, 0] / 5
        return pred

    def _save_config(self):
        path = self.path
        del self.logger, self.fold, self.path

        save_dir = path / "trainer"
        save_dir.mkdir(exist_ok=True, parents=True)

        with open(save_dir / f"lgbm_{self.tag}.pkl", "wb") as f:
            pickle.dump(self, f)
