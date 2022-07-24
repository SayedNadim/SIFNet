import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn



def generator_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',
                   norm='none'):
    return Conv2dBlock(input_channels=in_channels, output_dim=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, activation=activation, norm=norm)


def generator_transposed_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                              activation='relu',
                              norm='none', scale_factor=2):
    return TransposeConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride,
                                padding=padding, dilation=dilation, activation=activation, norm=norm,
                                scale_factor=scale_factor)


def generator_pixel_shuffle(in_channels, scale, out_channels):
    return PixelShuffleLayer(in_channels, scale, out_channels)


# -----------------------------------------------
#                Normal ConvBlock
# -----------------------------------------------
class Conv2dBlock(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='elu', pad_type='reflect'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_channels, output_dim, kernel_size, stride,
                              padding=conv_padding, dilation=dilation,
                              bias=self.use_bias, padding_mode=pad_type)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, xin):
        if self.pad:
            x = self.conv(self.pad(xin))
        else:
            x = self.conv(xin)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='lrelu', norm='in', scale_factor=2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                  pad_type=pad_type,
                                  activation=activation, norm=norm)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv2d(x)
        return x


class PixelShuffleLayer(nn.Module):
    def __init__(self, in_channels, scale_factor, out_channels):
        super(PixelShuffleLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.initial_conv = nn.Conv2d(in_channels, scale_factor ** 2, 3, 1, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.conv2d = generator_conv(1, out_channels, 1, 1, 0, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.pixel_shuffle(x)
        x = self.conv2d(x)
        return x
