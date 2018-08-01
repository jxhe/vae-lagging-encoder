import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

class LM(nn.Module):
    """docstring for LSTMDecoder"""
    def __init__(self, args, model_init, mlp_init):
        super(LM, self).__init__()
        self.ni = args.ni
        self.nh = args.nh

        self.nz = args.nz
        self.device = args.device

        self.vocab_size = args.vocab_size

        self.embed = nn.Embedding(self.vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni + args.nz,
                            hidden_size=args.nh,
                            num_layers=1,
                            batch_first=True)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.nh, bias=False)

        # prediction layer
        self.out_linear1 = nn.Linear(args.nh, self.vocab_size, bias=False)
        self.out_linear2 = nn.Linear(args.nz, self.vocab_size, bias=False)


        self.cat_dist = dist.categorical.Categorical(
            probs=torch.ones(4, device=args.device))

        scale = torch.ones(2, device=args.device)

        self.gauss_dist = [dist.normal.Normal(torch.tensor([-2.0, -2.0], device=args.device), scale), 
                           dist.normal.Normal(torch.tensor([2.0, 2.0], device=args.device), scale),
                           dist.normal.Normal(torch.tensor([-2.0, 2.0], device=args.device), scale),
                           dist.normal.Normal(torch.tensor([2.0, -2.0], device=args.device), scale)]

        self.reset_parameters(model_init, mlp_init)

    def reset_parameters(self, model_init, mlp_init):
        for param in self.parameters():
            # self.initializer(param)
            model_init(param)

        mlp_init(self.out_linear2.weight)


    def forward(self, input, z, h_c_init):
        """
        Args:
            input: (batch_size, 1), the randomly generated start token
            z: (batch_size, nz), the sampled latent variable from

            length: the required sentence length
        """

        # (batch_size, 1, ni)
        word_embed = self.embed(input)


        z_ = z.unsqueeze(1)
        # (batch_size, 1, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        # (batch_size, 1, nh)
        output, h_c_last = self.lstm(word_embed, h_c_init)

        # (batch_size, 1, vocab_size)
        output_logits = self.out_linear1(output) + self.out_linear2(z_)

        return output_logits, h_c_last

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

        z_cat = self.cat_dist.sample((nsample, 1)).squeeze()
        z = []
        for index in z_cat:
            z.append(self.gauss_dist[index.item()].sample().unsqueeze(0))

        # (batch_size, nz)
        z = torch.cat(z, dim=0)

        # (1, batch_size, nh)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        h_c_init = (h_init, c_init)

        samples = [input.squeeze().tolist()]
        for _ in range(length-1):
            output_logits, h_c_last = self.forward(input, z, h_c_init)
            sample_weights = output_logits.squeeze().exp()

            # (nsample, 1)
            idx = torch.multinomial(sample_weights, 1, replacement=True)
            input = idx
            h_c_init = h_c_last

            samples.append(idx.squeeze().tolist())

        # (length, nsample)
        samples = torch.tensor(samples)
        return samples.permute(1, 0)
        