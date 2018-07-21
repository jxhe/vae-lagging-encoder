import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

class LM(nn.Module):
    """docstring for LSTMDecoder"""
    def __init__(self, args, model_init, mlp_init):
        super(LSTMDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.nh

        self.nz = args.nz
        self.device = args.device

        self.rnn_initializer = rnn_initializer
        self.emb_initializer = emb_initializer
        self.vocab_size = args.vocab_size

        self.embed = nn.Embedding(self.vocab_size, args.ni)

        self.h_init = torch.zeros()

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.nh,
                            num_layers=1,
                            batch_first=True)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.nh, bias=False)

        # prediction layer
        self.out_linear1 = nn.Linear(args.nh, self.vocab_size, bias=False)
        self.out_linear2 = nn.Linear(args.nz, self.vocab_size, bias=False)

        self.reset_parameters(model_init, mlp_init)

    def reset_parameters(self, model_init, mlp_init):
        for param in self.parameters():
            # self.initializer(param)
            model_init(param)

        mlp_init(self.out_linear2.weight)


    def forward(self, input, z):
        """
        Args:
            input: (batch_size, 1), the randomly generated start token
            z: (batch_size, nz), the sampled latent variable from

            length: the required sentence length
        """

        # (batch_size, 1, ni)
        word_embed = self.embed(input)


        z = z.unsqueeze(1)
        # (batch_size, 1, ni + nz)
        word_embed = torch.cat((word_embed, z), -1)

        # (batch_size, 1, nh)
        c_init = self.trans_linear(z)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = Variable(torch.zeros(1, batch_size * n_sample, self.nh).type_as(z.data), requires_grad=False)

        # (batch_size, 1, nh)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        # (batch_size, 1, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def sample(self, nsample, length):
        """Sample synthetic data from this language model
        Args:
            nsample: number of samples
            length: length of each sentence

        Returns:
            output: (nsample, length), the sampled output ids

        """
        input = torch.rand(nsample, 1, device=self.device) \
                     .mul(self.vocab_size).long()

        z = torch.normal(mean=torch.zeros(nsample, self.nz), 
                         std=torch.ones(nsample, self.nz)).to(self.device)

        samples = []
        for _ in range(length):
            output_logits = self.forward(input, z)
            sample_weights = output_logits.squeeze().exp()

            # (nsample, 1)
            idx = torch.multinomial(sample_weights, 1)
            samples.append(idx)


        return samples
        

