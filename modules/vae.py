import torch
import torch.nn as nn


class VAE(nn.Module):
    """docstring for VAE"""
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.nz = args.nz

        loc = torch.zeros(nz)
        scale = torch.ones(nz)

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


    def true_posterior_dist(self, x, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with 
                shape (nsample, nz)
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, nsample, nz)
        zrange = zrange.repeat(batch_size)

        # (batch_size, nsample)
        log_gen = self.decoder.log_probability(x, zrange)

    def inference_dist(self, x, zrange)
        return self.encoder.inference_dist(self, x, zrange)

        