import numpy as np

import torch
import torch.utils.data as torchdata

from torch.optim.lr_scheduler import (CosineAnnealingLR, MultiStepLR, StepLR,
                                      ReduceLROnPlateau)

from arai.script.trainer.base import AbstractTrainer


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

    def _prepare_loader(self, X, y, mode="train"):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(
            y[:, np.newaxis], dtype=torch.float32, device=self.device)
        dataset = torchdata.TensorDataset(X, y)

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
