import os

import pytorch_lightning as pl
import numpy as np
import requests
import torch

from tqdm import tqdm
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

from hydra.utils import to_absolute_path

from typing import Optional


# https://github.com/NikitaChizhov/deep_kalman_filter_for_BM


class Spirits(pl.LightningDataModule):

    def __init__(self, download_dir, batch_size, num_workers, download_url, file_name, validation_split):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(to_absolute_path(self.hparams.download_dir)):
            os.makedirs(to_absolute_path(self.hparams.download_dir))
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        if not os.path.exists(file_path):
            headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
            response = requests.get(self.hparams.download_url, stream=True, headers=headers)
            content_length = int(response.headers['Content-Length'])
            pbar = tqdm(total=content_length)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=20_000_000):
                    if chunk:
                        f.write(chunk)
                    pbar.update(len(chunk))

    def setup(self, stage: Optional[str] = None):
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        data = np.load(file_path)

        # full data
        train_tensors = [tensor(data['x_train_full'])]
        val_tensors = [tensor(data['x_train_full'][self.hparams.validation_split:])]
        test_tensors = [tensor(data['x_test_full'])]

        # missing and mask, 1 is an evidence for missing data

        train_tensors += [tensor(data['x_train_miss']), (tensor(data['m_train_miss']) == 0)]

        val_tensors += [tensor(data['x_train_miss'][self.hparams.validation_split:]),
                        (tensor(data['m_train_miss'][self.hparams.validation_split:]) == 0)]

        test_tensors += [tensor(data['x_test_miss']), (tensor(data['m_test_miss']) == 0)]

        train_tensors.append(torch.zeros(train_tensors[0].shape[0]))
        val_tensors.append(torch.zeros(val_tensors[0].shape[0]))
        test_tensors.append(torch.zeros(test_tensors[0].shape[0]))

        self.train_dataset = TensorDataset(*train_tensors)
        self.val_dataset = TensorDataset(*val_tensors)
        self.test_dataset = TensorDataset(*test_tensors)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)
