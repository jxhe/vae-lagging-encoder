import torch
import torch.nn as nn


class DecoderBase(nn.Module):
    """docstring for Decoder"""
    def __init__(self):
        super(DecoderBase, self).__init__()
    
    def decode(self, x, z):

        raise NotImplementedError

    def reconstruct_error(self, x, z):
        """reconstruction loss
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        raise NotImplementedError

    def log_probability(self, x, z):
        """
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        raise NotImplementedError



        