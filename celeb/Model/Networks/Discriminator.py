import torch
import torch.nn as nn
import functools
from utils.Logging import Config
from torch.nn.utils import spectral_norm

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, config):
        super(NLayerDiscriminator, self).__init__()
        # norm_layer = nn.InstanceNorm2d
        input_nc = config.netD.feature.input_dim
        ndf = config.netD.feature.ndf
        n_layers = config.netD.feature.n_layers
        use_sigmoid = config.netD.feature.use_sigmoid

        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, config):
        super(PixelDiscriminator, self).__init__()

        norm_layer = nn.InstanceNorm2d
        input_nc = config.netD.pixel.input_dim
        ndf = config.netD.pixel.ndf
        use_sigmoid = config.netD.pixel.use_sigmoid

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

if __name__ == '__main__':
    config = Config('/home/la_belva/PycharmProjects/LABInpainting/Configs/training_config.yaml')
    x = torch.rand(4,3,256,256).float().cuda()
    pd = PixelDiscriminator(config).cuda()
    fd = NLayerDiscriminator(config).cuda()
    o1 = pd(x)
    o2 = fd(x)
    print(o1.shape)
    print(o2.shape)