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

        self.rnn_initializer = rnn_initializer
        self.emb_initializer = emb_initializer

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)

        # dimension transformation to z (mean and logvar)
        self.linear = nn.Linear(args.nh, 2 * args.nz, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.rnn_initializer is not None:
            for param in self.parameters():
                # self.initializer(param)
                self.rnn_initializer(param)

        if self.emb_initializer is not None:
            self.emb_initializer(self.embed.weight.data)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)

        return mean.squeeze(0), logvar.squeeze(0)

