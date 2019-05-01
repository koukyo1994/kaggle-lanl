import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from torch.optim.lr_scheduler import (CosineAnnealingLR, MultiStepLR, StepLR,
                                      ReduceLROnPlateau)

from pathlib import Path
from fastprogress import master_bar, progress_bar

from arai.script.trainer.base import AbstractTrainer
from arai.script.utils.seed_fixing import seed_torch


class NNTrainer(AbstractTrainer):
    def __init__(self,
                 model,
                 logger,
                 n_splits=5,
                 seed=42,
                 device="cpu",
                 lr=0.001,
                 scheduler="CosineAnnealingLR",
                 train_batch=128,
                 valid_batch=512,
                 kwargs={}):
        super(NNTrainer, self).__init__(
            logger, n_splits, seed, objective="regression")

        self.model = model
        self.device = device
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.kwargs = kwargs

        self.lr = lr
        self.scheduler = scheduler

        self.loss_fn = nn.L1Loss().to(self.device)

        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

    def _set_scheduler(self, optimizer, n_epochs):
        if isinstance(self.scheduler,
                      str) and self.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=1e-6)
        elif isinstance(self.scheduler, int):
            scheduler = StepLR(optimizer, step_size=self.scheduler)
        elif isinstance(self.scheduler, tuple):
            scheduler = MultiStepLR(
                optimizer, milestones=self.scheduler, gamma=0.5)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-7)

        return scheduler

    def _prepare_loader(self, X, y=None, mode="train"):
        if y is not None:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            y = torch.tensor(
                y[:, np.newaxis], dtype=torch.float32, device=self.device)
            dataset = torchdata.TensorDataset(X, y)
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            dataset = torchdata.TensorDataset(X)

        if mode == "train":
            loader = torchdata.DataLoader(
                dataset, batch_size=self.train_batch, shuffle=True)
        elif mode == "val":
            loader = torchdata.DataLoader(
                dataset, batch_size=self.valid_batch, shuffle=False)
        return loader

    def _checkpoint(self, model, epoch, val_score, higher_is_better=False):
        if higher_is_better:
            if val_score > self.best_score:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold_num}.pth")
                self.logger.info(f"Save weight on epoch {epoch}")
                self.best_score = val_score
        else:
            if self.best_score > val_score:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold_num}.pth")
                self.logger.info(f"Save weight on epoch {epoch}")
                self.best_score = val_score

    def _fit(self, X_train, y_train, X_val, y_val, n_epochs):
        seed_torch(self.seed)
        train_loader = self._prepare_loader(X_train, y_train, mode="train")
        valid_loader = self._prepare_loader(X_val, y_val, mode="val")

        model = self.model(**self.kwargs)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = self._set_scheduler(optimizer, n_epochs)

        self.best_score = np.inf
        mb = master_bar(range(n_epochs))

        for epoch in mb:
            model.train()
            avg_loss = 0.0
            for x_batch, y_batch in progress_bar(train_loader, parent=mb):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

            valid_preds, avg_val_loss = self._val(valid_loader, model)

            self.logger.info("-" * 20)
            self.logger.info(f"Epoch {epoch + 1} / {n_epochs}")
            self.logger.info("-" * 20)
            self.logger.info(f"Avg Loss: {avg_loss:.4f}")
            self.logger.info(f"Avg Val Loss: {avg_val_loss:.4f}")

            self._checkpoint(
                model, epoch + 1, avg_val_loss, higher_is_better=False)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        model.load_state_dict(
            torch.load(self.path / f"best{self.fold_num}.pth"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        self.logger.info(f"Best Validation Loss: {avg_val_loss:.4f}")
        return valid_preds

    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros((len(loader.dataset.tensors[0], )))
        avg_val_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                y_pred = model(x_batch).detach()
                avg_val_loss += self.loss_fn(y_pred,
                                             y_batch).item() / len(loader)
                valid_preds[i * self.valid_batch:(i + 1) * self.valid_batch] =\
                    y_pred.cpu().numpy().reshape(-1)
        return valid_preds, avg_val_loss

    def predict(self, X):
        loader = self._prepare_loader(X, mode="val")
        model = self.model(**self.kwargs)
        bin_path = self.path / "bin"
        preds = np.zeros((len(X), ))
        for path in bin_path.iterdir():
            model.load_state_dict(torch.load(path))
            model.to(self.device)
            model.eval()

            temp = np.zeros_like(preds)
            for i, (x_batch, ) in enumerate(loader):
                x_batch = x_batch.to(self.device)
                with torch.no_grad():
                    y_pred = model(x_batch).detach()
                    temp[i * self.valid_batch:(i + 1) * self.valid_batch] = \
                        y_pred.cpu().numpy().reshape(-1)
            preds += temp / self.n_splits
        return preds

    def _save_config(self):
        path = self.path
        del self.model, self.logger, self.fold, self.loss_fn, self.path

        save_dir = path / "trainer"
        save_dir.mkdir(exist_ok=True, parents=True)

        with open(save_dir / f"{self.tag}.pkl", "wb") as f:
            pickle.dump(self, f)
