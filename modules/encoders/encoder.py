import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..utils import log_sum_exp

class GaussianEncoderBase(nn.Module):
    """docstring for EncoderBase"""
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        raise NotImplementedError

    def sa_forward(self, x, meta_optimizer):
        mean, logvar = self.forward(x)
        mean_svi = Variable(mean.data, requires_grad=True)
        logvar_svi = Variable(logvar.data, requires_grad=True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], x)
        mean_svi_final, logvar_svi_final = var_params_svi

        return mean_svi_final, logvar_svi_final

    def sample_no_meta(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        # (batch_size, nz)
        mean, logvar = self.forward(input)
        mu = Variable(mean.data, requires_grad=True)
        logvar = Variable(logvar.data, requires_grad=True)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def sample(self, input, meta_optimizer, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        # (batch_size, nz)
        mu, logvar = self.sa_forward(input, meta_optimizer)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL, (mu, logvar)

    def calc_kl(self, mu, logvar):
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

    def reparameterize(self, mu, logvar, nsamples=1, z=None):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        if z is None:
            eps = torch.zeros_like(std_expd).normal_()
        else:
            eps = z.unsqueeze(1)

        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        """this function samples from q(Z | X), for the Gaussian family we use
        mode locations as samples

        Returns: Tensor
            Tensor: the mode locations, shape [batch_size, nsamples, nz]

        """
        # (batch_size, nz)
        mu, logvar = self.forward(x)
        # std = logvar.mul(0.5).exp()

        # batch_size = mu.size(0)
        # zrange = zrange.unsqueeze(1).expand(zrange.size(0), batch_size, self.nz)

        # infer_dist = torch.distributions.normal.Normal(mu, std)

        # # (batch_size, k^2)
        # log_prob = infer_dist.log_prob(zrange).sum(dim=-1).permute(1, 0)


        # # (K^2)
        # log_prob = log_prob.sum(dim=0)
        batch_size, nz = mu.size()

        return mu.unsqueeze(1).expand(batch_size, nsamples, nz)

    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)

        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density

    def calc_mi(self, x, meta_optimizer):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        # [x_batch, nz]
        mu, logvar = self.sa_forward(x, meta_optimizer)

        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = self.reparameterize(mu, logvar, 1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
