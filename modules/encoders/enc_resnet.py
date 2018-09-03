import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def he_init(m):
    s = np.sqrt(2./ m.in_features)
    m.weight.data.normal_(0, s)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, include_center=False, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (include_center == True):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask.cuda()
        return super(MaskedConv2d, self).forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True, mask=None,
               kernel_size = 3, padding = 1):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        if mask is None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding)
        else:
            self.conv1 = MaskedConv2d(mask, in_dim, out_dim, kernel_size=kernel_size, padding=padding)
            self.conv2 = MaskedConv2d(mask, out_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        return out



class ResNetEncoder(nn.Module):
    """docstring for ResNetEncoder"""
    def __init__(self, args):
        super(ResNetEncoder, self).__init__()
        
        enc_modules = []
        img_h = args.img_size[1]
        img_w = args.img_size[2] 
        for i in range(len(args.enc_layers)):
            if i == 0:
                input_dim = args.img_size[0]
            else:
                input_dim = args.enc_layers[i-1]
            enc_modules.append(ResidualBlock(input_dim, args.enc_layers[i]))
            enc_modules.append(nn.Conv2d(args.enc_layers[i], args.enc_layers[i], kernel_size=2, stride=2))

            img_h //= 2
            img_w //= 2

        latent_in_dim = img_h*img_w*args.enc_layers[-1]
        self.enc_cnn = nn.Sequential(*enc_modules)
        self.latent_linear_mean = nn.Linear(latent_in_dim, args.nz)
        self.latent_linear_logvar = nn.Linear(latent_in_dim, args.nz)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def forward(self, img):
        img_code = self.enc_cnn(img)
        img_code = img_code.view(img.size(0), -1)
        self.img_code = img_code
        mean = self.latent_linear_mean(img_code)
        logvar = self.latent_linear_logvar(img_code)
        return mean, logvar

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

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)
        
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
