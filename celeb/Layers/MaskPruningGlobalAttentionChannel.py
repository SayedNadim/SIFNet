import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskPruningGlobalAttentionChannel(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(MaskPruningGlobalAttentionChannel, self).__init__()
        self.chanel_in = in_dim

        self.query_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax_channel = nn.Softmax(dim=-1)
        self.gamma = torch.tensor([1.0], requires_grad=True).cuda()

    def forward(self, foreground, background, mask=None):
        feature_size = list(foreground.size())
        if mask is not None:
            mask = F.interpolate(mask, size=feature_size[2:])
        else:
            mask = torch.ones(size=(feature_size[0], 1, feature_size[2], feature_size[3]))
        if foreground.is_cuda:
            mask = mask.cuda()
        query_channel = self.query_channel(foreground).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        key_channel = self.key_channel(background).view(feature_size[0], -1, feature_size[2] * feature_size[3]).permute(0,
                                                                                                                        2,
                                                                                                                        1)
        channel_correlation = torch.bmm(query_channel, key_channel)
        m_r = mask.view(feature_size[0], -1, feature_size[2] * feature_size[3])
        channel_correlation = torch.bmm(channel_correlation, m_r)
        energy_channel = self.softmax_channel(channel_correlation)
        value_channel = self.value_channel(foreground).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        attented_channel = (energy_channel * value_channel).view(feature_size[0], feature_size[1],
                                                                 feature_size[2],
                                                                 feature_size[3])
        out = foreground * mask + self.gamma * (1.0 - mask) * attented_channel
        return out
