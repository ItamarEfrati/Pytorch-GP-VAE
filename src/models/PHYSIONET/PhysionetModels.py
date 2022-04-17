from typing import Optional

import torch
from torch.distributions import MultivariateNormal

from src.metrices.MaskMeanSquaredError import MaskMeanSquaredError
from src.models.abstracts.VAE import AbstractImputationVAE
from src.utils.evaluation import calculate_auroc
from src.utils.model_utils import get_gp_prior


class PhysionetHIVAE(AbstractImputationVAE):
    def __init__(self, is_mask=True, **kwargs):
        super().__init__(**kwargs)

        self.train_mse = MaskMeanSquaredError()
        self.val_mse = MaskMeanSquaredError()
        self.test_mse = MaskMeanSquaredError()

    def _get_prior(self):
        if self.prior is None:
            self.prior = MultivariateNormal(loc=torch.zeros(self.hparams.z_dim, device=self.device),
                                            scale_tril=torch.diag(torch.ones(self.hparams.z_dim, device=self.device)))
        return self.prior

    # region Pytorch lightning overwrites

    def training_step(self, batch, batch_idx):
        x_full, x_miss, x_mask, y = batch
        loss, kl_mean, nll_loss, nll_missing_mean, x_hat = self.step(batch)

        self.train_mse(x_hat, x_full, ~x_mask)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_kl_mean', kl_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_nll_mean_loss', nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_nll_missing', nll_missing_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mse_missing', self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_full, x_miss, x_mask, y = batch
        loss, kl_mean, nll_loss, nll_missing_mean, x_hat = self.step(batch)

        self.val_mse(x_hat, x_full, ~x_mask)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_kl_mean', kl_mean, on_step=False, on_epoch=True)
        self.log('val_nll_mean_loss', nll_loss, on_step=False, on_epoch=True)
        self.log('val_nll_missing', nll_missing_mean, on_step=False, on_epoch=True)
        self.log('val_mse_missing', self.val_mse, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x_full, x_miss, x_mask, y = batch
        loss, kl_mean, nll_loss, nll_missing_mean, x_hat = self.step(batch)

        self.test_mse(x_hat, x_full, ~x_mask)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_kl_mean', kl_mean, on_step=False, on_epoch=True)
        self.log('test_nll_mean_loss', nll_loss, on_step=False, on_epoch=True)
        self.log('test_nll_missing', nll_missing_mean, on_step=False, on_epoch=True)
        self.log('test_mse_missing', self.test_mse, on_step=False, on_epoch=True)

        x_hat[x_mask] = x_miss[x_mask]

        return x_hat, y

    def test_epoch_end(self, results) -> None:
        X, y = list(zip(*results))
        X, y = torch.concat(X), torch.concat(y)
        X = X.reshape(X.shape[0], -1)
        auprc, auroc = calculate_auroc(X, y, self.classifier, is_binary=True)

        self.log('test auroc', auroc)
        self.log('test auprc', auprc)

    # endregion


class PhysionetGPVAE(PhysionetHIVAE):

    def __init__(self, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, time_length=10, **kwargs):
        super().__init__(**kwargs)

    def _get_prior(self):
        if self.prior is None:
            self.prior = get_gp_prior(kernel=self.hparams.kernel, kernel_scales=self.hparams.kernel_scales,
                                      time_length=self.hparams.time_length, sigma=self.hparams.sigma,
                                      length_scale=self.hparams.length_scale, z_dim=self.hparams.z_dim,
                                      device=self.device)
        return self.prior

    def encode(self, x):
        return self.encoder(x.permute(0, 2, 1))

    def decode(self, z):
        z = z.permute(0, 2, 1)
        return self.decoder(z)
