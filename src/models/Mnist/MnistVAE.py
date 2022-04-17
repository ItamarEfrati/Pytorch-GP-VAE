import torch
import torchmetrics
import torchvision
from torch.distributions import MultivariateNormal

from src.models.abstracts.VAE import AbstractVAE


class MnistVAE(AbstractVAE):
    """
    VAE working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def _get_prior(self):
        if self.prior is None:
            self.prior = MultivariateNormal(loc=torch.zeros(self.hparams.z_dim, device=self.device),
                                            covariance_matrix=torch.eye(self.hparams.z_dim, device=self.device))
        return self.prior

    def step(self, batch):
        x, _ = batch
        x, x_original_shape = self.get_network_input_shape(x)
        q_z, px_z, x_hat = self.forward(x)

        log_likelihood = self.compute_log_likelihood(px_z, x)

        kl = self.compute_kl_divergence(q_z)
        kl = torch.where(torch.torch.isfinite(kl), kl, torch.zeros_like(kl))

        elbo = log_likelihood - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        return loss, kl.mean(), (-log_likelihood).mean(), x_hat.reshape(x_original_shape), x.reshape(x_original_shape)

    def get_network_input_shape(self, x):
        x_shape = x.shape
        batch_size = x_shape[0]
        x = x.reshape(batch_size, -1)
        return x, x_shape

    # region Loss computations

    def compute_log_likelihood(self, px_z, x):
        log_likelihood = px_z.log_prob(x)
        log_likelihood = torch.where(torch.isfinite(log_likelihood), log_likelihood, torch.zeros_like(log_likelihood))
        return log_likelihood.sum(1)

    def compute_kl_divergence(self, q):
        return torch.distributions.kl.kl_divergence(q, self._get_prior())

    # endregion

    # region Pytorch lightning overwrites

    def training_step(self, batch, batch_idx):
        loss, kl_mean, negative_log_likelihood, x_hat, x = self.step(batch)

        self.train_mse(x_hat, x)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_kl_mean', kl_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mean_negative_log_likelihood', negative_log_likelihood, on_step=False, on_epoch=True,
                 prog_bar=True)
        self.log('train_MSE', self.train_mse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, kl_mean, negative_log_likelihood, x_hat, x = self.step(batch)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(torch.concat([x[:20], x_hat[:20]]), nrow=20)
            list(map(lambda l: l.add_image('val reconstruction images', grid, self.current_epoch),
                     self.logger.experiment))

        self.val_mse(x_hat, x)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_kl_mean', kl_mean, on_step=False, on_epoch=True)
        self.log('val_mean_negative_log_likelihood', negative_log_likelihood, on_step=False, on_epoch=True)
        self.log('val_MSE', self.val_mse, on_step=False, on_epoch=True)

        return x_hat, loss

    def test_step(self, batch, batch_idx):
        loss, kl_mean, negative_log_likelihood, x_hat, x = self.step(batch)

        self.test_mse(x_hat, x)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_kl_mean', kl_mean, on_step=False, on_epoch=True)
        self.log('test_mean_negative_log_likelihood', negative_log_likelihood, on_step=False, on_epoch=True)
        self.log('test_MSE', self.test_mse, on_step=False, on_epoch=True)

        return x_hat, loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None):
        loss, kl_mean, negative_log_likelihood, x_hat, x = self.step(batch)
        grid = torchvision.utils.make_grid(torch.concat([x, x_hat]), nrow=20)
        list(map(lambda l: l.add_image('prediction reconstruction images', grid, 0),
                 self.logger.experiment))

    # endregion
