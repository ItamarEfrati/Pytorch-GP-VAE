import os
import pytorch_lightning as pl

from typing import Optional

import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms

from hydra.utils import to_absolute_path


# self.mnist_test.test_data[0]

def black_white(x):
    return (x > 0).float()


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, download_dir, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transformers = transforms.Compose([transforms.ToTensor(), transforms.Lambda(black_white), ])
        self.mnist_test = None
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_predict = None

    def prepare_data(self):
        MNIST(to_absolute_path(self.hparams.download_dir), train=True, download=True)
        MNIST(to_absolute_path(self.hparams.download_dir), train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        mnist_full = MNIST(to_absolute_path(self.hparams.download_dir), train=True, transform=self.transformers)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(to_absolute_path(self.hparams.download_dir), transform=self.transformers, train=False)
        self.mnist_predict, _ = random_split(self.mnist_test, [20, 9980])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_predict, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)
