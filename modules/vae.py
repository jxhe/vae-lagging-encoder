import torch
import torch.nn as nn


class VAE(nn.Module):
    """docstring for VAE"""
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

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
        Returns: Tensor1, Tensor2
            Tensor1: reconstruction loss shape [batch]
            Tensor2: KL loss shape [batch]
        """
        z, KL = self.encode(x, nsamples)

        # (batch, nsamples)
        reconstruct_err = self.decoder.reconstruct_error(x, z)

        return reconstruct_err.mean(dim=1), KL


    def true_posterior_dist(self, x, z_range):
        """perform grid search to calculate the true posterior
        Args:


        """
        pass

    def plot_true_posterior(self, x, z_range):
        pass

    def plot_approx_posterior(self, x, z_range):
        pass

        