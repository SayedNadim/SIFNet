import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskPruningGlobalAttentionSpatial(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(MaskPruningGlobalAttentionSpatial, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #
        self.rate = 1
        self.gamma = nn.Parameter(torch.tensor([0.0])).cuda()

    def forward(self, foreground, background, mask=None):
        b, c, h, w = foreground.size()
        if mask is not None:
            mask = F.interpolate(mask, size=(h, w), mode='nearest')
        else:
            mask = torch.ones(size=(b,1,h,w), dtype=torch.float32)

        if foreground.is_cuda:
            mask =mask.cuda()

        proj_query = self.query_conv(foreground).view(b, -1, w * h).permute(0, 2,1)  # B, C, N -> B N C
        proj_key = self.key_conv(background).view(b, -1, w * h)  # B, C, N
        feature_similarity = torch.bmm(proj_query, proj_key)  # B, N, N

        mask_view = mask.view(b, -1, w * h)  # B, C, N

        feature_pruning = feature_similarity * mask_view
        attention = self.softmax(feature_pruning)  # B, N, C
        feature_pruning = torch.bmm(self.value_conv(foreground).view(b, -1, w * h),
                                    attention.permute(0, 2, 1))  # -. B, C, N
        out = feature_pruning.view(b, c, w, h)  # B, C, H, W
        out = foreground * mask + self.gamma * (1.0 - mask) * out
        return out
