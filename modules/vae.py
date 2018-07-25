import torch
import torch.nn as nn

from .utils import log_sum_exp


class VAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.nz = args.nz

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)

        self.prior = torch.distributions.normal.Normal(loc, scale)

    def sample_from_posterior(self, x, nsamples=1):
        """sample from posterior
        Args:
            x: Tensor 
                shape (batch, seq_len, ni)
            nsamples: int. 
                Number of samples for each data instance

        Returns: Tensor
                shape (batch, nsamples, nz)
        """

        return self.encoder.sample_from_posterior(x, nsamples)

    def encode(self, x, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples)

    def decode(self, z):
        """generate samples from z (perhaps beam search ?)
        """
        

    def loss(self, x, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains 
                the data tensor and length list

        Returns: Tensor1, Tensor2
            Tensor1: reconstruction loss shape [batch]
            Tensor2: KL loss shape [batch]
        """
        z, KL = self.encode(x, nsamples)

        # (batch, nsamples)
        reconstruct_err = self.decoder.reconstruct_error(x, z)

        return reconstruct_err.mean(dim=1), KL

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with 
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=1)



    def eval_true_posterior_dist(self, x, zrange, log_prior):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with 
                shape (k^2, nz), where k=(zmax - zmin)/space
            log_prior: tenor
                the prior log density with shape (k^2) 

        Returns: Tensor
            Tensor: the posterior density tensor with 
                shape (batch_size, k^2)
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        zrange = zrange.repeat(batch_size, 1, 1)

        # (batch_size, k^2)
        log_gen = self.decoder.log_probability(x, zrange)


        # (batch_size, k^2)
        log_post = log_gen + log_prior

        # (batch_size, k^2)
        return (log_post - log_sum_exp(log_post, dim=1, keepdim=True)).exp()


    def eval_inference_dist(self, x, zrange):
        """
        Returns: Tensor
            Tensor: the posterior density tensor with 
                shape (batch_size, k^2)
        """
        return self.encoder.eval_inference_dist(x, zrange)

        