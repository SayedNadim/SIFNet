import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers.ConvLayers import generator_conv, generator_transposed_conv
from Layers.RecurrentContextualAttention import RecurrentContextualAttention
from Layers.SpatialChannelContextPropagation import SpatialChannelContext
from Layers.Modules import ResnetBlock
from utils.Logging import Config


class G_Net(nn.Module):
    def __init__(self, config):
        super(G_Net, self).__init__()
        self.config = config
        self.l_generator = LGenerator(input_dim=self.config.netG.net_l.input_dim, cnum=self.config.netG.net_l.cnum,
                                      output_dim=self.config.netG.net_l.output_dim)
        self.ab_generator = ABGenerator(input_dim=self.config.netG.net_ab.input_dim, cnum=self.config.netG.net_ab.cnum,
                                        output_dim=self.config.netG.net_ab.output_dim)
        self.fusion_generator = FusionGenerator(input_dim=self.config.netG.net_fusion.input_dim,
                                                cnum=self.config.netG.net_fusion.cnum,
                                                output_dim=self.config.netG.net_fusion.output_dim)

    def forward(self, x_l, x_ab, xin, mask):
        l_predict, flow = self.l_generator(x_l, mask)
        ab_predict = self.ab_generator(x_ab, mask)
        coarse = torch.cat((l_predict, ab_predict), dim=1)
        fused = self.fusion_generator(xin, coarse, mask)
        return fused, coarse, l_predict, ab_predict, flow


class LGenerator(nn.Module):
    def __init__(self, input_dim, cnum, output_dim):
        super(LGenerator, self).__init__()

        self.conv1 = generator_conv(input_dim + 1, cnum, 5, 1, 2)
        self.conv2_downsample = generator_conv(cnum, cnum, 3, 2, 1)
        self.conv3 = generator_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = generator_conv(cnum * 2, cnum * 2, 3, 2, 1)
        self.conv5 = generator_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = generator_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.rca = RecurrentContextualAttention(cnum * 4)
        self.conv7_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 2, dilation=2)
        self.conv8_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 4, dilation=4)
        self.conv9_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 8, dilation=8)
        self.conv10_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 16, dilation=16)
        self.allconv11 = generator_conv(cnum * 8, cnum * 4, 3, 1, 1, 1)
        self.allconv12 = generator_conv(cnum * 4, cnum * 4, 3, 1, 1, 1)
        self.allconv13_ps = generator_transposed_conv(cnum * 4, cnum * 2, 3, 1, 1, 1, scale_factor=2)
        self.allconv14_ps = generator_transposed_conv(cnum * 2, cnum * 2, 3, 1, 1, 1, scale_factor=2)
        self.allconv15 = generator_conv(cnum * 2, cnum, 3, 1, 1, 1)
        self.out_conv = generator_conv(cnum, output_dim, 3, 1, 1, 1, activation='none')

    def forward(self, l_in, mask):
        mask = mask.cuda()
        l_in = l_in.cuda()
        x_corrupted_l = l_in * (1. - mask)
        xnow = torch.cat([x_corrupted_l, mask], dim=1)

        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x_rca, flow = self.rca(x, x, mask)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x_atr = self.conv10_atrous(x)
        x = torch.cat((x_atr, x_rca), dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_ps(x)
        x = self.allconv14_ps(x)
        x = self.allconv15(x)
        x = self.out_conv(x)
        x = (F.tanh(x) + 1) / 2
        return x, flow


class ABGenerator(nn.Module):
    def __init__(self, input_dim, cnum, output_dim):
        super(ABGenerator, self).__init__()

        self.conv1 = generator_conv(input_dim + 1, cnum, 5, 1, 2)
        self.conv2_downsample = generator_conv(cnum, cnum, 3, 2, 1)
        self.conv3 = generator_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = generator_conv(cnum * 2, cnum * 2, 3, 2, 1)
        self.conv5 = generator_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = generator_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.scsa = SpatialChannelContext(cnum * 4)
        self.conv7_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 2, dilation=2)
        self.conv8_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 4, dilation=4)
        self.conv9_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 8, dilation=8)
        self.conv10_atrous = generator_conv(cnum * 4, cnum * 4, 3, 1, 16, dilation=16)
        self.allconv11 = generator_conv(cnum * 8, cnum * 4, 3, 1, 1, 1)
        self.allconv12 = generator_conv(cnum * 4, cnum * 4, 3, 1, 1, 1)
        self.allconv13_ps = generator_transposed_conv(cnum * 4, cnum * 2, 3, 1, 1, 1, scale_factor=2)
        self.allconv14_ps = generator_transposed_conv(cnum * 2, cnum * 2, 3, 1, 1, 1, scale_factor=2)
        self.allconv15 = generator_conv(cnum * 2, cnum, 3, 1, 1, 1)
        self.out_conv = generator_conv(cnum, output_dim, 3, 1, 1, 1, activation='none')

    def forward(self, ab_in, mask):
        mask = mask.cuda()
        ab_in = ab_in.cuda()
        x_corrupted_ab = ab_in * (1. - mask)
        xnow = torch.cat([x_corrupted_ab, mask], dim=1)

        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x_scsa = self.scsa(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_atr = x
        x = torch.cat((x_atr, x_scsa), dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_ps(x)
        x = self.allconv14_ps(x)
        x = self.allconv15(x)
        x = self.out_conv(x)
        x = (F.tanh(x) + 1) / 2
        return x


class FusionGenerator(nn.Module):
    def __init__(self, input_dim, cnum, output_dim):
        super(FusionGenerator, self).__init__()

        self.input_dim = input_dim
        self.cnum = cnum
        self.output_dim = output_dim

        filters = [self.cnum, self.cnum * 2, self.cnum * 4, self.cnum * 8, self.cnum * 16]

        self.Downsample1 = conv_block(filters[0], filters[0], down=True)
        self.Downsample2 = conv_block(filters[1], filters[1], down=True)
        self.Downsample3 = conv_block(filters[2], filters[2], down=True)
        self.Downsample4 = conv_block(filters[3], filters[3], down=True)

        self.Conv1 = conv_block(self.input_dim, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], self.output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, xin, coarse, mask):
        mask = mask.cuda()
        xin = xin.cuda()
        coarse = coarse.cuda()
        x1_inpaint = coarse * mask + xin * (1. - mask)
        xnow = torch.cat([x1_inpaint, mask], dim=1)

        e1 = self.Conv1(xnow)

        e2 = self.Downsample1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Downsample2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Downsample3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Downsample4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, F.interpolate(d5, size=e4.shape[2:])), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = (F.tanh(out) + 1) / 2

        return out


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = generator_transposed_conv(in_ch, out_ch, 3, 1, 1, 1)

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, down=False):
        super(conv_block, self).__init__()

        if down:
            self.conv = generator_conv(in_ch, out_ch, 3, 2, 1, 1)
        else:
            self.conv = generator_conv(in_ch, out_ch, 3, 1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    if __name__ == '__main__':
        config = Config('/home/la_belva/PycharmProjects/LABInpainting/Configs/training_config.yaml')
        net = G_Net(config).cuda()
        dummy_input_l = torch.rand(config.batch_size, 1, config.input_size[0], config.input_size[1]).float().cuda()
        dummy_input_ab = torch.rand(config.batch_size, 2, config.input_size[0], config.input_size[1]).float().cuda()
        dummy_input = torch.rand(config.batch_size, 3, config.input_size[0], config.input_size[1]).float().cuda()
        dummy_mask = torch.rand(config.batch_size, 1, config.input_size[0], config.input_size[1]).float().cuda()
        out, _, _, _, _ = net(dummy_input_l, dummy_input_ab, dummy_input, dummy_mask)
        print(out.shape)
