import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .decoder import DecoderBase

def he_init(m):
    s = np.sqrt(2./ m.in_features)
    m.weight.data.normal_(0, s)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, in_dim, out_dim=None, kernel_size = 3, mask = 'B'):
        super(GatedMaskedConv2d, self).__init__()
        if out_dim is None:
            out_dim = in_dim    
        self.dim = out_dim
        self.size = kernel_size
        self.mask = mask
        pad = self.size // 2

        #vertical stack    
        self.v_conv = nn.Conv2d(in_dim, 2*self.dim, kernel_size=(pad+1, self.size))
        self.v_pad1 = nn.ConstantPad2d((pad, pad, pad, 0), 0)
        self.v_pad2 = nn.ConstantPad2d((0, 0, 1, 0), 0)
        self.vh_conv = nn.Conv2d(2*self.dim, 2*self.dim, kernel_size = 1)

        #horizontal stack
        self.h_conv = nn.Conv2d(in_dim, 2*self.dim, kernel_size=(1, pad+1))
        self.h_pad1 = nn.ConstantPad2d((self.size // 2, 0, 0, 0), 0)
        self.h_pad2 = nn.ConstantPad2d((1, 0, 0, 0), 0)
        self.h_conv_res = nn.Conv2d(self.dim, self.dim, 1)

    def forward(self, v_map, h_map):
        v_out = self.v_pad2(self.v_conv(self.v_pad1(v_map)))[:, :, :-1, :]
        v_map_out = F.tanh(v_out[:, :self.dim])*F.sigmoid(v_out[:, self.dim:])
        vh = self.vh_conv(v_out)
        
        h_out = self.h_conv(self.h_pad1(h_map))
        if self.mask == 'A':
            h_out = self.h_pad2(h_out)[:, :, :, :-1]
        h_out = h_out + vh    
        h_out = F.tanh(h_out[:, :self.dim])*F.sigmoid(h_out[:, self.dim:])
        h_map_out = self.h_conv_res(h_out)
        if self.mask == 'B':
            h_map_out = h_map_out + h_map
        return v_map_out, h_map_out

class StackedGatedMaskedConv2d(nn.Module):
    def __init__(self, 
                 img_size = [1, 28, 28], layers = [64,64,64],
                 kernel_size = [7,7,7], latent_dim=64, latent_feature_map = 1):
        super(StackedGatedMaskedConv2d, self).__init__()
        input_dim = img_size[0]
        self.conv_layers = []
        if latent_feature_map > 0:
            self.latent_feature_map = latent_feature_map
            self.z_linear = nn.Linear(latent_dim, latent_feature_map*28*28)    
        for i in range(len(kernel_size)):
            if i == 0:
                self.conv_layers.append(GatedMaskedConv2d(input_dim+latent_feature_map,
                                                      layers[i],  kernel_size[i], 'A'))
            else:
                self.conv_layers.append(GatedMaskedConv2d(layers[i-1], layers[i],  kernel_size[i]))
            
        self.modules = nn.ModuleList(self.conv_layers)
    
    def forward(self, img, q_z=None):
        """
        Args:
            img: (batch, nc, H, W)
            q_z: (batch, nsamples, nz)
        """

        batch_size, nsamples, _ = q_z.size()
        if q_z is not None:
            z_img = self.z_linear(q_z) 
            z_img = z_img.view(img.size(0), nsamples, self.latent_feature_map, img.size(2), img.size(3))

            # (batch, nsamples, nc, H, W)
            img = img.unsqueeze(1).expand(batch_size, nsamples, *img.size()[1:])

        for i in range(len(self.conv_layers)):
            if i == 0:
                if q_z is not None:
                    # (batch, nsamples, nc + fm, H, W) --> (batch * nsamples, nc + fm, H, W)
                    v_map = torch.cat([img, z_img], 2)
                    v_map = v_map.view(-1, *v_map.size()[2:])
                else:
                    v_map = img
                h_map = v_map
            v_map, h_map = self.conv_layers[i](v_map, h_map)
        return h_map

class PixelCNNDecoder(DecoderBase):
    """docstring for PixelCNNDecoder"""
    def __init__(self, args):
        super(PixelCNNDecoder, self).__init__()
        self.dec_cnn = StackedGatedMaskedConv2d(img_size=args.img_size, layers = args.dec_layers,
                                                latent_dim= args.nz, kernel_size = args.dec_kernel_size,
                                                latent_feature_map = args.latent_feature_map)

        self.dec_linear = nn.Conv2d(args.dec_layers[-1], args.img_size[0], kernel_size = 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def decode(self, img, q_z):
        dec_cnn_output = self.dec_cnn(img, q_z)
        pred = F.sigmoid(self.dec_linear(dec_cnn_output))
        return pred

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, nc, H, W)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        batch_size, nsamples, _ = z.size()
        # (batch * nsamples, nc, H, W)
        pred = self.decode(x, z)
        prob = torch.clamp(pred.view(pred.size(0), -1), min=1e-5, max=1.-1e-5)

        # (batch, nsamples, nc, H, W) --> (batch * nsamples, nc, H, W)
        x = x.unsqueeze(1).expand(batch_size, nsamples, *x.size()[1:])
        tgt_vec = x.view(-1, *x.size()[2:])

        # (batch * nsamples, *)
        tgt_vec = tgt_vec.view(tgt_vec.size(0), -1)

        log_bernoulli = tgt_vec * torch.log(prob) + (1. - tgt_vec)*torch.log(1. - prob)

        log_bernoulli = log_bernoulli.view(batch_size, nsamples, -1)
        
        return -torch.sum(log_bernoulli, 2)


    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, nc, H, W)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)
        
        