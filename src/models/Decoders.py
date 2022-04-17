import torch
import torch.nn as nn

from src.utils.model_utils import make_nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli


class Decoder(nn.Module):
    def __init__(self, hidden_sizes):
        """ Decoder parent class with no specified output distribution
            :param hidden_sizes: tuple of hidden layer sizes. The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        self.net = make_nn(hidden_sizes[-1], hidden_sizes[:-1])

    def __call__(self, x):
        pass


class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """

    def __call__(self, x):
        logits = self.net(x)
        return Bernoulli(logits=logits)


class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """

    def __call__(self, x):
        mean = self.net(x)
        return Normal(loc=mean, scale=torch.ones(mean.shape, device=mean.device))
