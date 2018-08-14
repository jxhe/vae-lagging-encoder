import math

from itertools import chain
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ..utils import log_sum_exp

class CNNClassifier(nn.Module):
    """CNNClassifier from Yoon Kim's paper"""
    def __init__(self, args):
        super(CNNClassifier, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (K, args.ni)) \
                                    for K in args.kernel_sizes])

        self.dropout = nn.Dropout(args.cnn_dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.mix_num)

    def forward(self, x):
        """
        Args:
            x: Tensor
                the embedding of input, with shape (batch_size, seq_length, ni)


        Returns: Tensor1
            Tensor1: the logits for the mixture prob, shape (batch_size, mix_num)
        """

        # TODO: support static vectors

        x = x.unsqueeze(1)

        # [(batch_size, kernel_num, seq_length)] * len(args.kernel_sizes)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # [(batch_size, kernel_num)] * len(args.kernel_sizes)
        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in x]

        x = torch.cat(x, 1)

        x = self.dropout(x)

        # return the logit 
        return self.fc1(x)

        


class MixLSTMEncoder(nn.Module):
    """Mixture of Gaussian LSTM Encoder with constant-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(MixLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.classifier = CNNClassifier(args)

        self.lstm_lists = nn.ModuleList([nn.LSTM(input_size=args.ni,
            hidden_size=self.nh, num_layers=1, 
            batch_first=True, dropout=0) for _ in range(args.mix_num)])

        # dimension transformation to z (mean and logvar)
        self.linear_lists = nn.ModuleList([nn.Linear(self.nh, 
            2 * args.nz, bias=False) for _ in range(args.mix_num)])

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.linear.weight)
        # emb_init(self.embed.weight)
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def sample(self, mu, logvar, mix_prob, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean tensors of mixed gaussian distribution, 
                with shape (batch_size, mix_num, nz)

            logvar: Tensor
                logvar tensors of mixed gaussian distibution,
                 with shape (batch_size, mix_num, nz)

            mix_prob: Tensor
                the mixture probability weights, 
                with shape (batch_size, mix_num)

        Returns: Tensor
            Sampled z with shape (batch_size, nsamples, nz)
        """

        batch_size, = mix_prob.size(0)
        # (batch_size, nsamples, nz)
        classes = torch.multinomial(mix_prob, nsamples, replacement=True) \
                       .unsqueeze(2).expand(batch_size, nsamples, self.nz)

        # (batch_size, nsamples, nz)
        mu_ = torch.gather(mu, dim=1, index=classes)
        logvar_ = torch.gather(logvar, dim=1, index=classes)

        std = (0.5 * logvar_).exp()

        return torch.normal(mu_, std)

    def forward(self, input):
        """
        Args:
            input: (batch_size, seq_len, ni)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor of different inference nets, 
                shape (batch, mix_num, nz)
            Tensor2: the logvar tensor of different inference nets, 
                shape (batch, mix_num, nz)
        """

        mean_list = []
        logvar_list = []

        for lstm, linear in zip(self.lstm_lists, self.linear_lists):
            _, (last_state, last_cell) = lstm(input)

            mean, logvar = linear(last_state).unsqueeze(2).chunk(2, -1)
            mean_list.append(mean)
            logvar_list.append(logvar)

        return torch.cat(mean_list, dim=2).squeeze(0), \
               torch.cat(logvar_list, dim=2).squeeze(0)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term
        Args:
            input: (batch_size, seq_len)

        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: containes two tensors:
                1. the tensor of KL for each x with shape [batch, nsamples]
                2. the tensor of log q(z | x) with shape [batch, nsamples]

        """

        # (batch_size, seq_length, ni)
        embed = self.embed(input)

        # the logit, (batch_size, mix_num)
        log_mix_weights = self.classifier(embed)

        mix_prob = (log_mix_weights - 
                    log_sum_exp(log_mix_weights, dim=1, keepdim=True)).exp()

        # (batch_size, mix_num, nz)
        mu, logvar = self.forward(embed)

        # (batch, nsamples, nz)
        z = self.sample(mu, logvar, mix_prob, nsamples)

        # compute KL with MC, (batch_size, nsamples)
        log_posterior = self.log_posterior(z, mu, logvar, log_mix_weights)
        KL = (log_posterior - self.log_prior(z))

        return z, (KL, log_posterior)

    def log_prior(self, z):
        """evaluate the log density of prior at z
        Args:
            z: Tensor
                the points to be evaluated, with shape 
                (batch_size, nsamples, nz)

        Returns: Tensor1
            Tensor1: the log density of shape (batch_size, nsamples)     
        """
        
        return -0.5 * (z ** 2).sum(-1) - 0.5 * self.nz * math.log(2 * math.pi)

    def log_posterior(self, z, mu, logvar, log_mix_weights):
        """evaluate the log density of approximate 
        posterior at z

        Args:
            z: Tensor
                the points to be evaluated, with shape 
                (batch_size, nsamples, nz)

            mu: Tensor
                Mean tensors of mixed gaussian distribution, 
                with shape (batch_size, mix_num, nz)

            logvar: Tensor
                logvar tensors of mixed gaussian distibution,
                 with shape (batch_size, mix_num, nz)

            log_mix_weights: Tensor
                the mixture weights (the logits), 
                with shape (batch_size, mix_num)

        Returns: Tensor1
            Tensor1: the log density of shape (batch_size, nsamples)
        """

        
        z = z.unsqueeze(1)
        mu, logvar = mu.unsqueeze(2), logvar.unsqueeze(2)
        var = logvar.exp()

        # (batch_size, mix_num, nsamples, nz)
        dev = z - mu

        # (batch_size, mix_num, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (self.nz * math.log(2 * math.pi) + logvar.sum(-1))

        # (batch_size, mix_num, nsamples)
        log_density = log_density + log_mix_weights.unsqueeze(2)

        return log_sum_exp(log_density, dim=1)

    # def eval_inference_dist(self, x, zrange):
    #     """this function computes the inference posterior 
    #     over a popultation, i.e. P(Z | X)
    #     Args:
    #         zrange: tensor
    #             different z points that will be evaluated, with
    #             shape (k^2, nz), where k=(zmax - zmin)/space
    #     """
    #     # (batch_size, nz)
    #     mu, logvar = self.forward(x)
    #     std = logvar.mul(0.5).exp()

    #     batch_size = mu.size(0)
    #     zrange = zrange.unsqueeze(1).expand(zrange.size(0), batch_size, self.nz)

    #     infer_dist = torch.distributions.normal.Normal(mu, std)

    #     # (batch_size, k^2)
    #     log_prob = infer_dist.log_prob(zrange).sum(dim=-1).permute(1, 0)


    #     # (K^2)
    #     log_prob = log_prob.sum(dim=0)

    #     return (log_prob - log_sum_exp(log_prob)).exp()
