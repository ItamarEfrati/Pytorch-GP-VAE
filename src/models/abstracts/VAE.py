import torch
import pytorch_lightning as pl

from abc import ABC, abstractmethod


class AbstractVAE(ABC, pl.LightningModule):
    """
    VAE working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 image_preprocessor,
                 encoder,
                 decoder,
                 classifier,
                 z_dim=256,
                 lr=1e-3,
                 weight_decay=0.005,
                 beta=1.0):
        super(AbstractVAE, self).__init__()
        self.save_hyperparameters(ignore=['image_preprocessor', 'encoder', 'decoder'])

        self.image_preprocessor = image_preprocessor
        self.encoder = encoder
        self.decoder = decoder

        self.classifier = classifier

        self.prior = None
        self.original_shape = None

    @abstractmethod
    def _get_prior(self):
        pass

    @abstractmethod
    def step(self, batch):
        pass

    @abstractmethod
    def compute_log_likelihood(self, px_z, x):
        pass

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def compute_kl_divergence(self, q):
        kl = torch.distributions.kl.kl_divergence(q, self._get_prior())
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl

    # region Pytorch lightning overwrites

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def forward(self, x):
        if self.image_preprocessor:
            x = self.image_preprocessor(x)
        q_z = self.encode(x)
        z = q_z.rsample()
        z_mean = q_z.mean
        px_z = self.decode(z)
        x_hat = self.decode(z_mean).mean
        return q_z, px_z, x_hat

    # endregion

    def get_latent_vectors(self, data_loader):
        labels = []
        latent_vectors = []
        for batch in data_loader:
            x, y = batch
            q_z = self.encoder(x)
            z = q_z.rsample()
            latent_vectors.append(z)
            labels.append(y)
        return torch.concat(latent_vectors), torch.concat(labels)


class AbstractImputationVAE(AbstractVAE):

    def step(self, batch):
        x_full, x_miss, x_mask, y = batch
        q_z, px_z, x_hat = self.forward(x_miss)

        if self.hparams.is_mask:
            log_likelihood = self.compute_log_likelihood(px_z, x_miss, x_mask)
        else:
            log_likelihood = self.compute_log_likelihood(px_z, x_miss)

        mean_nll = -self.compute_log_likelihood(px_z, x_full, ~x_mask)
        mean_nll = mean_nll.sum() / (~x_mask).sum()

        log_likelihood_per_series = log_likelihood.sum((1, 2))
        kl = self.compute_kl_divergence(q_z)

        elbo = log_likelihood_per_series - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo

        return loss, kl.mean(), (-log_likelihood_per_series).mean(), mean_nll, x_hat

    def compute_log_likelihood(self, px_z, x, x_mask=None):
        log_likelihood = px_z.log_prob(x)
        log_likelihood = torch.where(torch.isfinite(log_likelihood), log_likelihood,
                                     torch.zeros_like(log_likelihood))
        if x_mask is not None:
            log_likelihood = log_likelihood * x_mask
        return log_likelihood

    def compute_kl_divergence(self, qz_x):
        kl = torch.distributions.kl.kl_divergence(qz_x, self._get_prior())
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl.sum(-1)
