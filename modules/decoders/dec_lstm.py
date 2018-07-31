# import torch

import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

class LSTMDecoder(nn.Module):
    """LSTM decoder with constant-length data"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(LSTMDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.nh
        self.nz = args.nz

        self.vocab = vocab

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=-1)

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
        # vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.trans_linear.weight)
        # model_init(self.pred_linear.weight)
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)


    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

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

        z = z.view(batch_size * n_sample, self.nz)
        # c_init = self.trans_linear(z).unsqueeze(0)
        # h_init = torch.tanh(c_init)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
            log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)


class VarLSTMDecoder(LSTMDecoder):
    """LSTM decoder with constant-length data"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(VarLSTMDecoder, self).__init__(args, vocab, model_init, emb_init)

        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=vocab['<pad>'])
        vocab_mask = torch.ones(len(vocab))
        vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters(model_init, emb_init)

    def decode(self, input, z):
        """
        Args:
            input: tuple which contains x and sents_len
                    x: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        """

        input, sents_len = input

        # not predicting start symbol
        sents_len = sents_len - 1

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
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(packed_embed, (h_init, c_init))
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: tuple which contains x_ and sents_len
                    x_: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        x, sents_len = x

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode((src, sents_len), z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: tuple which contains x_ and sents_len
                    x_: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        Returns:
            log_p(x|z): (batch_size, n_sample).
        """

        return -self.reconstruct_error(x, z)

