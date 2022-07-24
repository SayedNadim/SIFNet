import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedFeatureReweigh(nn.Module):
    def __init__(self, channel, expansion=16):
        super(MaskGuidedFeatureReweigh, self).__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation_x = nn.Sequential(nn.Linear(channel, int(channel * expansion)),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(int(channel * expansion), channel),
                                                  nn.Sigmoid())
        self.spatial_se_x = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.ReLU())
        self.avg_pool_y = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation_y = nn.Sequential(nn.Linear(channel, int(channel * expansion)),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(int(channel * expansion), channel),
                                                  nn.Sigmoid())
        self.spatial_se_y = nn.Conv2d(channel, 1, kernel_size=1,
                                                    stride=1, padding=0, bias=False)

    def forward(self, x, mask):
        b, c, h, w = x.shape
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        y = x * (1. - mask)
        bahs, chs, _, _ = x.size()
        chn_se_x = self.avg_pool_x(x).view(bahs, chs)
        chn_se_x = self.channel_excitation_x(chn_se_x).view(bahs, chs, 1, 1)
        spa_se_x = self.spatial_se_x(x)

        bahy, chy, _, _ = y.size()
        chn_se_y = self.avg_pool_y(y).view(bahy, chy)
        chn_se_y = self.channel_excitation_y(chn_se_y).view(bahy, chy, 1, 1)
        out = torch.mul(torch.mul(torch.mul(x, chn_se_x), spa_se_x), chn_se_y)
        return out