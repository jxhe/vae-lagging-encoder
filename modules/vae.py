import torch
import torch.nn as nn

from .utils import log_sum_exp
from .lm import LSTM_LM


class VAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.args = args

        self.nz = args.nz

        if args.enc_type == 'mix':
            self.baseline = torch.load(args.baseline_path)

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


    def loss(self, x, kl_weight, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """
        if self.args.enc_type == 'mix':
            # reinforce loss

            # z: (batch, nsamples, nz)
            # KL: (batch, nsamples)
            # log_posterior: (batch, nsamples)
            # mix_prob: (batch, nsamples)
            z, (KL, log_posterior, mix_prob) = self.encode(x, nsamples)

            # (batch, nsamples)
            reconstruct_err = self.decoder.reconstruct_error(x, z)

            # this is actually the negative learning signal
            learning_signal = (reconstruct_err + kl_weight * KL +
                               self.baseline.log_probability(x).unsqueeze(1)).detach()

            encoder_loss = (learning_signal * log_posterior).mean(dim=1)
            decoder_loss = reconstruct_err.mean(dim=1)

            return encoder_loss + decoder_loss, reconstruct_err.mean(dim=1), KL.mean(dim=1), mix_prob

        else:
            z, KL = self.encode(x, nsamples)

            # (batch)
            reconstruct_err = self.decoder.reconstruct_error(x, z).mean(dim=1)


            return reconstruct_err + kl_weight * KL, reconstruct_err, KL, None

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
         (actually the complete likelihood), this function computes
         the complete likelihood over a popultation, i.e. P(Z, X)
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
            log_prior: tenor
                the prior log density with shape (k^2)

        Returns: Tensor
            Tensor: the posterior density tensor with
                shape (k^2)
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
        log_comp = log_gen + log_prior

        # (k^2)
        log_comp = log_comp.sum(dim=0)

        # (k^2)
        return (log_comp - log_sum_exp(log_comp)).exp()


    def eval_inference_dist(self, x, zrange):
        """
        Returns: Tensor
            Tensor: the posterior density tensor with
                shape (batch_size, k^2)
        """
        return self.encoder.eval_inference_dist(x, zrange)

    # def eval_inference_mode(self, x):
    #     """compute the mode points in the inference distribution
    #     (in Gaussian case)
    #     Returns: Tensor
    #         Tensor: the posterior mode points with shape (*, nz)
    #     """
    #     return self.encoder.eval_inference_mode(x)


