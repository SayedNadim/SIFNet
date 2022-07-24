import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers.MaskPruningGlobalAttentionSpatial import MaskPruningGlobalAttentionSpatial
from Layers.MaskPruningGlobalAttentionChannel import MaskPruningGlobalAttentionChannel
from utils.ImageRelated import *


class GlobalLocalAttention(nn.Module):
    def __init__(self, in_dim, patch_size=3, propagation_size=3, stride=1):
        super(GlobalLocalAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagation_size
        self.stride = stride
        self.prop_kernels = None
        self.in_dim = in_dim
        self.feature_attention = MaskPruningGlobalAttentionSpatial(in_dim)

    def forward(self, foreground, background, mask=None):
        b, nc, w, h = foreground.size()

        if mask is None:
            mask = torch.ones(b, 1, h, w).float()
        else:
            mask = F.interpolate(mask, size=(h, w), mode='nearest')

        if foreground.is_cuda:
            mask = mask.cuda()

        background = background * (1 - mask)
        foreground = self.feature_attention(foreground, background, mask)
        background = F.pad(background,
                           [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
                                                                                     self.stride).contiguous().view(b,
                                                                                                                    nc,
                                                                                                                    -1,
                                                                                                                    self.patch_size,
                                                                                                                    self.patch_size)

        mask_resized = mask.repeat(1, self.in_dim, 1, 1)
        mask_resized = F.pad(mask_resized,
                             [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        mask_kernels_all = mask_resized.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
                                                                                       self.stride).contiguous().view(
            b,
            nc,
            -1,
            self.patch_size,
            self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        mask_kernels_all = mask_kernels_all.transpose(2, 1)
        offsets = []
        output_tensor = []

        for i in range(b):
            m = mask_kernels_all[i]  # m shape: [L, C, k, k]
            mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
            mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]
            feature_map = foreground[i:i + 1]
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch_size // 2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))
            conv_result = conv_result * mm
            attention_scores = F.softmax(conv_result * 10, dim=1)
            attention_scores = attention_scores * mm
            offset = torch.argmax(attention_scores, dim=1, keepdim=True)  # 1*1*H*W

            if foreground.shape != background.shape:
                # Normalize the offset value to match foreground dimension
                times = float(foreground.shape[2] * foreground.shape[3]) / float(
                    background.shape[2] * background.shape[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat(
                [offset // foreground.shape[2], offset % foreground.shape[3]],
                dim=1)  # 1*3*H*W

            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride=1,
                                                      padding=self.patch_size // 2)
            # # average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * mask[i:i + 1]) / (self.patch_size ** 2)
            # recover the image
            final_output = recovered_foreground * mask[i:i + 1] + feature_map * (1 - mask[i:i + 1])
            output_tensor.append(final_output)
            offsets.append(offset)
        attended = torch.cat(output_tensor, dim=0)

        offsets = torch.cat(offsets, dim=0).float()
        offsets = offsets.view(foreground.size()[0], 2, *foreground.size()[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(h).view([1, 1, h, 1]).expand(b, -1, -1, w)
        w_add = torch.arange(w).view([1, 1, 1, w]).expand(b, -1, h, -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        flow = flow.cuda()
        return attended, flow


class ContextualModule(nn.Module):
    def __init__(self, patch=3, propagation=3, stride=1):
        super(ContextualModule, self).__init__()
        self.patch = patch
        self.propagation = propagation
        self.stride = stride
        self.prop_kernels = None

    def forward(self, foreground, background, mask=None):
        b, c, h, w = foreground.shape
        if mask is None:
            mask = torch.ones(b, 1, h, w).float()
        else:
            mask = F.interpolate(mask, size=(h, w), mode='nearest')
        if foreground.is_cuda:
            mask = mask.cuda()
        background = background * (1 - mask)
        background = F.pad(background, [self.patch // 2, self.patch // 2, self.patch // 2, self.patch // 2])
        mask_padded = F.pad(mask, [self.patch // 2, self.patch // 2, self.patch // 2, self.patch // 2])
        conv_kernel = background.unfold(2, self.patch, self.stride).unfold(3, self.patch,
                                                                           self.stride).contiguous().view(b, c, -1,
                                                                                                          self.patch,
                                                                                                          self.patch)
        conv_kernel = conv_kernel.transpose(2, 1)

        mask_kernel = mask_padded.unfold(2, self.patch, self.stride).unfold(3, self.patch,
                                                                            self.stride).contiguous().view(b, 1, -1,
                                                                                                           self.patch,
                                                                                                           self.patch)
        mask_kernel = mask_kernel.transpose(2, 1)

        offsets = []
        output_tensor = []

        for i in range(b):
            m = mask_kernel[i]  # m shape: [L, C, k, k]
            mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
            mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]
            feature_map = foreground[i:i + 1]
            conv_kernels = conv_kernel[i] + torch.finfo(torch.float32).eps
            conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch // 2)
            if self.propagation != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagation, self.propagation])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))
            conv_result = conv_result * mm
            attention_scores = F.softmax(conv_result * 10, dim=1)
            attention_scores = attention_scores * mm
            offset = torch.argmax(attention_scores, dim=1, keepdim=True)  # 1*1*H*W

            if foreground.shape != background.shape:
                # Normalize the offset value to match foreground dimension
                times = float(foreground.shape[2] * foreground.shape[3]) / float(
                    background.shape[2] * background.shape[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat(
                [offset // foreground.shape[2], offset % foreground.shape[3]],
                dim=1)  # 1*3*H*W

            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride=1,
                                                      padding=self.patch // 2)
            # # average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * mask[i:i + 1]) / (self.patch ** 2)
            # recover the image
            final_output = recovered_foreground * mask[i:i + 1] + feature_map * (1 - mask[i:i + 1])
            output_tensor.append(final_output)
            offsets.append(offset)
        attended = torch.cat(output_tensor, dim=0)

        offsets = torch.cat(offsets, dim=0).float()
        offsets = offsets.view(foreground.size()[0], 2, *foreground.size()[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(h).view([1, 1, h, 1]).expand(b, -1, -1, w)
        w_add = torch.arange(w).view([1, 1, 1, w]).expand(b, -1, h, -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().Data.numpy()))
        flow = F.interpolate(flow, size=foreground.size()[2:])

        return attended, flow


class RecurrentContextualAttention(nn.Module):

    def __init__(self, channel, patch_size_list=[3, 3], propagate_size_list=[3, 3],
                 stride_list=[1, 1]):
        assert isinstance(patch_size_list,
                          list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(
            stride_list), "the input_lists should have same lengths"
        super(RecurrentContextualAttention, self).__init__()
        self.sca_1 = ContextualModule(patch_size_list[0], propagate_size_list[0],
                                          stride_list[0])
        self.sca_2 = ContextualModule(patch_size_list[1], propagate_size_list[1],
                                      stride_list[1])
        self.sca_conv = nn.Conv2d(channel * 2, channel, 3,1,1,1)


    def forward(self, foreground, background, mask=None):
        b, c, h, w = foreground.shape
        foreground_2 = F.interpolate(foreground, size=(h//2, w//2), mode='nearest')
        background_2 = F.interpolate(background, size=(h//2, w//2), mode='nearest')
        out_1, flow_1 = self.sca_1(foreground_2, background_2, mask)

        out_1_reshaped = F.interpolate(out_1, size=(h, w), mode='bilinear')
        out_2, flow_2 = self.sca_2(out_1_reshaped, background, mask)
        out = self.sca_conv(torch.cat((out_1_reshaped, out_2), dim=1))

        flow1 = F.interpolate(flow_1, (h, w), mode='bilinear')
        flow2 = F.interpolate(flow_2, (h, w), mode='bilinear')

        return out, [flow1, flow2]


if __name__ == '__main__':
    x = torch.rand(1, 128, 256, 256).float().cuda()
    mask = torch.rand(1, 1, 256, 256).float().cuda()
    net = RecurrentContextualAttention(1280).cuda()
    out, flow = net(x, x, mask)
    print(out.shape, flow[0].shape)
