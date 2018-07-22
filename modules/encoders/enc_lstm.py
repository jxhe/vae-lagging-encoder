from itertools import chain
import torch
import torch.nn as nn
from modules import utils

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTMEncoder(nn.Module):
    """Gaussian LSTM Encoder"""
    def __init__(self, args, vocab_size, rnn_initializer, emb_initializer):
        super(LSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.nh
        self.nz = args.nz

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)

        # dimension transformation to z (mean and logvar)
        self.linear = nn.Linear(args.nh, 2 * args.nz, bias=False)

        self.reset_parameters(rnn_initializer, emb_initializer)

    def reset_parameters(self):
        for param in self.parameters():
            # self.initializer(param)
            self.rnn_initializer(param)

        self.emb_initializer(self.embed.weight)

    def reparameterize(self, mu, logvar, nsamples=1):
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

        return torch.normal(mu_expd, std_expd)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)

        return mean.squeeze(0), logvar.squeeze(0)

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

        return z, KL



