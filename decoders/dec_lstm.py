# import torch

import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

class LSTMDecoder(object):
    """docstring for LSTMDecoder"""
    def __init__(self, args):
        super(LSTMDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.nh

        self.rnn_initializer = rnn_initializer
        self.emb_initializer = emb_initializer
        self.vocab = vocab

        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=vocab['<pad>'])

        self.dropout_in = nn.Dropout(args.dec_dropout_in)
        self.dropout_out = nn.Dropout(args.dec_dropout_out)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.nh, bias=False)

        # concatenate z with input
        self.lstm = nn.LSTM(input_size=args.ni + args.nz,
                            hidden_size=args.nh,
                            num_layers=1,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(args.nh, len(vocab), bias=False)

        vocab_mask = torch.ones(len(vocab))
        vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.rnn_initializer is not None:
            for param in self.parameters():
                # self.initializer(param)
                self.rnn_initializer(param)

        if self.emb_initializer is not None:
            self.emb_initializer(self.embed.weight.data)


    def decode(self, input_z_sents_len):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
            sents_len: list of sequence lengths
        """

        input, z, sents_len = input_z_sents_len

        # not predicting start symbol
        sents_len -= 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)
        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        sents_len = sents_len.unsqueeze(1).expand(batch_size, n_sample).contiguous().view(-1)

        packed_embed = pack_padded_sequence(word_embed, sents_len.tolist(), batch_first=True)

        z = z.view(batch_size * n_sample, self.nz)
        # c_init = self.trans_linear(z).unsqueeze(0)
        # h_init = torch.tanh(c_init)
        h_init = self.trans_linear(z).unsqueeze(0)
        c_init = Variable(torch.zeros(1, batch_size * n_sample, self.nh).type_as(z.data), requires_grad=False)
        output, _ = self.lstm(packed_embed, (h_init, c_init))
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x_sents_len, z):
        """Cross Entropy in the language case
        Args:
            x_sents_len: (data, sents_len)
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: sum of loss over minibatch
        """

        pass

    def log_probability(self, x_sents_len, z):
        pass
        

