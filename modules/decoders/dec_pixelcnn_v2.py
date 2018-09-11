import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .decoder import DecoderBase

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, masked_channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :masked_channels, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :masked_channels, kH // 2 + 1:] = 0

    def reset_parameters(self):
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super(MaskedConv2d, self).forward(x)

class PixelCNNBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(PixelCNNBlock, self).__init__()
        self.mask_type = 'B'
        padding = kernel_size // 2
        out_channels = in_channels // 2

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            MaskedConv2d(self.mask_type, out_channels, out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        return self.activation(self.main(input) + input)


class MaskABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, masked_channels):
        super(MaskABlock, self).__init__()
        self.mask_type = 'A'
        padding = kernel_size // 2

        self.main = nn.Sequential(
            MaskedConv2d(self.mask_type, masked_channels, in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[1]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    def forward(self, input):
        return self.main(input)


class PixelCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_sizes, masked_channels):
        super(PixelCNN, self).__init__()
        assert num_blocks == len(kernel_sizes)
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = MaskABlock(in_channels, out_channels, kernel_sizes[i], masked_channels)
            else:
                block = PixelCNNBlock(out_channels, kernel_sizes[i])
            self.blocks.append(block)

        self.main = nn.ModuleList(self.blocks)

        self.direct_connects = []
        for i in range(1, num_blocks - 1):
            self.direct_connects.append(PixelCNNBlock(out_channels, kernel_sizes[i]))

        self.direct_connects = nn.ModuleList(self.direct_connects)

    def forward(self, input):
        # [batch, out_channels, H, W]
        direct_inputs = []
        for i, layer in enumerate(self.main):
            if i > 2:
                direct_input = direct_inputs.pop(0)
                direct_conncet = self.direct_connects[i - 3]
                input = input + direct_conncet(direct_input)

            input = layer(input)
            direct_inputs.append(input)
        assert len(direct_inputs) == 3, 'architecture error: %d' % len(direct_inputs)
        direct_conncet = self.direct_connects[-1]
        return input + direct_conncet(direct_inputs.pop(0))

class PixelCNNDecoderV2(DecoderBase):
    def __init__(self, args, mode='large'):
        super(PixelCNNDecoderV2, self).__init__()
        nz = args.nz
        self.ngpu = args.ngpu
        self.gpu_ids = args.gpu_ids
        self.nc = 1
        self.fm_latent = 4
        self.img_latent = 28 * 28 * self.fm_latent
        self.z_transform = nn.Sequential(
            nn.Linear(nz, self.img_latent),
        )
        if mode == 'small':
            kernal_sizes = [7, 7, 7, 5, 5, 3, 3]
        elif mode == 'large':
            kernal_sizes = [7, 7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3]
        else:
            raise ValueError('unknown mode: %s' % mode)

        hidden_channels = 64
        self.main = nn.Sequential(
            PixelCNN(self.nc + self.fm_latent, hidden_channels, len(kernal_sizes), kernal_sizes, self.nc),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, self.nc, 1, bias=False),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.z_transform[0].weight)
        nn.init.constant_(self.z_transform[0].bias, 0)

        m = self.main[2]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)
        return output

    def reconstruct_error(self, x, z):
        eps = 1e-12
        batch_size, nsampels, nz = z.size()

        # [batch, nsamples, -1] --> [batch, nsamples, fm, H, W]
        z = self.z_transform(z).view(batch_size, nsampels, self.fm_latent, 28, 28)

        # [batch, nc, H, W] --> [batch, 1, nc, H, W] --> [batch, nsample, nc, H, W]
        img = x.unsqueeze(1).expand(batch_size, nsampels, *x.size()[1:])
        # [batch, nsample, nc+fm, H, W] --> [batch * nsamples, nc+fm, H, W]
        img = torch.cat([img, z], dim=2)
        img = img.view(-1, *img.size()[2:])

        # [batch * nsamples, *] --> [batch, nsamples, -1]
        recon_x = self.forward(img).view(batch_size, nsampels, -1)
        # [batch, -1]
        x_flat = x.view(batch_size, -1)
        BCE = (recon_x + eps).log() * x_flat.unsqueeze(1) + (1.0 - recon_x + eps).log() * (1. - x_flat).unsqueeze(1)
        # [batch, nsamples]
        return BCE.sum(dim=2) * -1.0

    def log_probability(self, x, z):
        bce = self.reconstruct_error(x, z)
        return bce * -1.

    # def decode(self, z, deterministic):
    #     '''

    #     Args:
    #         z: Tensor
    #             the tensor of latent z shape=[batch, nz]
    #         deterministic: boolean
    #             randomly sample of decode via argmaximizing probability

    #     Returns: Tensor
    #         the tensor of decoded x shape=[batch, *]

    #     '''
    #     H = W = 28
    #     batch_size, nz = z.size()

    #     # [batch, -1] --> [batch, fm, H, W]
    #     z = self.z_transform(z).view(batch_size, self.fm_latent, H, W)
    #     img = Variable(z.data.new(batch_size, self.nc, H, W).zero_(), volatile=True)
    #     # [batch, nc+fm, H, W]
    #     img = torch.cat([img, z], dim=1)
    #     for i in range(H):
    #         for j in range(W):
    #             # [batch, nc, H, W]
    #             recon_img = self.forward(img)
    #             # [batch, nc]
    #             img[:, :self.nc, i, j] = torch.ge(recon_img[:, :, i, j], 0.5).float() if deterministic else torch.bernoulli(recon_img[:, :, i, j])
    #             # img[:, :self.nc, i, j] = torch.bernoulli(recon_img[:, :, i, j])

    #     # [batch, nc, H, W]
    #     img_probs = self.forward(img)
    #     return img[:, :self.nc], img_probs