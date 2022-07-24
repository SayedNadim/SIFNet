import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelContext(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super(SpatialChannelContext, self).__init__()
        self.size_reduction_pool = nn.AdaptiveAvgPool2d(64)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

        self.channel_excitation_x = nn.Sequential(nn.Linear(in_dim, in_dim // reduction),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(in_dim // reduction, in_dim),
                                                  nn.Sigmoid())
        self.avg_pool_x = nn.AdaptiveAvgPool2d(1)

    def forward(self, xin):
        if xin.is_cuda:
            self.gamma_1 = self.gamma_1.cuda()
            self.gamma_2 = self.gamma_2.cuda()
        _, _, width_in, height_in = xin.size()
        x = self.size_reduction_pool(xin)
        batch_size, channel, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B X C X N

        s_out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        s_out = s_out.view(batch_size, channel, width, height)
        s_out = x + self.gamma_1 * s_out

        chn_se_feature = self.avg_pool_x(s_out).view(batch_size, channel)
        chn_se_x = self.channel_excitation_x(chn_se_feature).view(batch_size, channel, 1, 1)
        x_ch = torch.mul(s_out, chn_se_x)
        feature = x + self.gamma_2 * x_ch
        feature = F.interpolate(feature, size=(height_in, width_in), scale_factor=None, mode='bilinear',
                                align_corners=False)
        return feature


if __name__ == '__main__':
    x_y = torch.rand(4, 512, 256, 256).float().cuda()
    scsa = SpatialChannelContext(512).cuda()
    out = scsa(x_y)
    print(out.shape)
