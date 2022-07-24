import torch.nn as nn
import torch
from Layers.ConvLayers import generator_conv
import torch.nn.functional as F


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.activation(out)
        return out


class DilationBlock(nn.Module):
    def __init__(self, dim):
        super(DilationBlock, self).__init__()
        self.model = nn.Sequential(
            generator_conv(dim, dim, 3, 1, 2, 2),
            generator_conv(dim, dim, 3, 1, 4, 4),
            generator_conv(dim, dim, 3, 1, 8, 8),
            generator_conv(dim, dim, 3, 1, 16, 16))

    def forward(self, x):
        out = self.model(x)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            generator_conv(in_channels=dim, out_channels=256, kernel_size=3, stride=1, padding=0, dilation=dilation),

            nn.ReflectionPad2d(1),
            generator_conv(in_channels=256, out_channels=dim, kernel_size=3, stride=1, padding=0, dilation=1,
                           activation='none', norm='none')
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
