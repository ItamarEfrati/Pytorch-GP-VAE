import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from src.utils.model_utils import make_nn, make_2d_cnn, make_cnn


class ImagePreprocessor(nn.Module):
    def __init__(self, image_shape, hidden_sizes: tuple, kernel_size: tuple):
        """ Encoder parent class without specified output distribution. This is an image preprocess.
         This layer suppose to run over an single image
            :param image_shape: input image size
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param kernel_size: kernel/filter width and height
        """
        super(ImagePreprocessor, self).__init__()
        self.image_shape = tuple(image_shape)
        self.net = make_2d_cnn(hidden_sizes, kernel_size)

    def __call__(self, x):
        # single image is of shape (1, height, width, channel) where channel is pixel and can be RGB (3) of grayscale(1)
        # we need to change to the shape a 2d-cnn excepts
        x_shape = x.shape
        shaped_images = x.reshape(self.image_shape).permute(0, 3, 1, 2)
        preprocessed_images = self.net(shaped_images)
        return preprocessed_images.permute(0, 2, 3, 1).reshape(x_shape)


class DiagonalEncoder(nn.Module):
    def __init__(self, hidden_sizes: tuple, z_dim):
        """ Encoder with factorized Normal posterior over temporal dimension
            Used by disjoint VAE and HI-VAE with Standard Normal prior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(DiagonalEncoder, self).__init__()
        self.z_size = z_dim
        self.net = make_nn(2 * self.z_size, hidden_sizes)

    def __call__(self, x):
        statistics = self.net(x)
        mu, log_var = statistics[..., :self.z_size], statistics[..., self.z_size:]
        std = F.softplus(log_var)
        return MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))


class BandedJointEncoder(nn.Module):
    def __init__(self, hidden_sizes, z_size, kernel_size, precision_activation):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            Used by GP-VAE with proposed banded covariance matrix
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param kernel_size: kernel size for Conv1D layer
        """
        super(BandedJointEncoder, self).__init__()
        self.z_size = z_size
        self.net = make_cnn(hidden_sizes + [3 * z_size], kernel_size=kernel_size)
        self.precision_activation = precision_activation

    def _get_sparse_matrix_indices(self, batch_size, sequence_length):
        num_variational_parameters = (2 * sequence_length - 1)
        first_dim = np.repeat(np.arange(batch_size), self.z_size * num_variational_parameters)
        second_dim = np.tile(np.repeat(np.arange(self.z_size), num_variational_parameters), batch_size)
        third_dim = np.tile(np.concatenate([np.arange(sequence_length), np.arange(sequence_length - 1)]),
                            batch_size * self.z_size)
        forth_dim = np.tile(np.concatenate([np.arange(sequence_length), np.arange(1, sequence_length)]),
                            batch_size * self.z_size)

        return np.stack([first_dim, second_dim, third_dim, forth_dim])

    def _get_covariance_matrix(self, precision_parameters, batch_size, sequence_length):
        sparse_matrix_indices = self._get_sparse_matrix_indices(batch_size, sequence_length)

        # There are 2T parameters for each sequence. Taking only 2T -1
        precision_parameters = precision_parameters[:, :, :-1].reshape(-1)
        sparse_matrix = torch.sparse_coo_tensor(sparse_matrix_indices, precision_parameters,
                                                (batch_size, self.z_size, sequence_length,
                                                 sequence_length))
        precision_tridiagonal = sparse_matrix.to_dense()

        batch_eye_matrix = torch.eye(sequence_length).reshape(1, 1, sequence_length, sequence_length).to(
            precision_parameters.device)
        batch_eye_matrix = batch_eye_matrix.repeat(batch_size, self.z_size, 1, 1)

        precision_tridiagonal += batch_eye_matrix
        # inverse of precision in covariance, precision_tridiagonal is upper tridiagonal
        covariance_upper_tridiagonal = torch.triangular_solve(batch_eye_matrix, precision_tridiagonal).solution
        covariance_upper_tridiagonal = torch.where(torch.isfinite(covariance_upper_tridiagonal),
                                                   covariance_upper_tridiagonal,
                                                   torch.zeros_like(covariance_upper_tridiagonal))

        return covariance_upper_tridiagonal

    def __call__(self, x):
        batch_size, sequence_length = x.shape[0], x.shape[-1]
        statistics = self.net(x)

        statistics = torch.permute(statistics, dims=(0, 2, 1))
        mu = statistics[:, :self.z_size]
        precision_parameters = statistics[:, self.z_size:]

        if self.precision_activation:
            precision_parameters = self.precision_activation(precision_parameters)
        precision_parameters = precision_parameters.reshape(batch_size, self.z_size, 2 * sequence_length)

        covariance_upper_tridiagonal = self._get_covariance_matrix(precision_parameters, batch_size, sequence_length)
        covariance_lower_tridiagonal = torch.permute(covariance_upper_tridiagonal, dims=(0, 1, 3, 2))

        return MultivariateNormal(loc=mu, scale_tril=covariance_lower_tridiagonal)
