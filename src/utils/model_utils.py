import numpy as np
import torch
import torch.nn as nn

from src.utils.gp_kernels import rbf_kernel, diffusion_kernel, matern_kernel, cauchy_kernel


class Permute(nn.Module):
    def __init__(self, perm):
        super(Permute, self).__init__()
        self.perm = perm

    def forward(self, x):
        return x.permute(self.perm)


def make_nn(output_size, hidden_sizes):
    """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
    """
    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], dtype=torch.float32))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_sizes[-1], output_size, dtype=torch.float32))
    return nn.Sequential(*layers)


def make_cnn(hidden_sizes, kernel_size=(3,)):
    """ Construct neural network consisting of one 1d-convolutional layer that utilizes temporal dependences,
        fully connected network
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layer
    """

    layers = [nn.Conv1d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=kernel_size,
                        dtype=torch.float32, padding="same"),
              nn.ReLU(),
              Permute((0, 2, 1))]
    for i in range(1, len(hidden_sizes) - 2):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], dtype=torch.float32))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1], dtype=torch.float32))
    return nn.Sequential(*layers)


def make_2d_cnn(hidden_sizes=(1, 256, 1), kernel_size=(3, 3)):
    """ Creates fully convolutional neural network.
        Used as CNN preprocessor for image data (HMNIST, SPRITES)

        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layers
    """

    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Conv2d(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i + 1], kernel_size=kernel_size,
                                padding="same", dtype=torch.float32))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def get_gp_prior(kernel, kernel_scales, time_length, sigma, length_scale, z_dim, device):
    # Compute kernel matrices for each latent dimension
    kernel_matrices = []
    for i in range(kernel_scales):
        if kernel == "rbf":
            kernel_matrices.append(rbf_kernel(time_length, length_scale / 2 ** i))
        elif kernel == "diffusion":
            kernel_matrices.append(diffusion_kernel(time_length, length_scale / 2 ** i))
        elif kernel == "matern":
            kernel_matrices.append(matern_kernel(time_length, length_scale / 2 ** i))
        elif kernel == "cauchy":
            kernel_matrices.append(cauchy_kernel(time_length, sigma, length_scale / 2 ** i))

    # Combine kernel matrices for each latent dimension
    tiled_matrices = []
    total = 0
    for i in range(kernel_scales):
        if i == kernel_scales - 1:
            multiplier = z_dim - total
        else:
            multiplier = int(np.ceil(z_dim / kernel_scales))
            total += multiplier
        tiled_matrices.append(torch.tile(torch.unsqueeze(kernel_matrices[i], 0), [multiplier, 1, 1]))
    kernel_matrix_tiled = np.concatenate(tiled_matrices)
    assert len(kernel_matrix_tiled) == z_dim
    kernel_matrix_tiled = torch.tensor(kernel_matrix_tiled, device=device)

    return torch.distributions.MultivariateNormal(
        loc=torch.zeros([z_dim, time_length], dtype=torch.float32, device=device),
        covariance_matrix=kernel_matrix_tiled)
