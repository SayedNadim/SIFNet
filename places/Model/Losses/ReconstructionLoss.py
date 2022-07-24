import torch
import torch.nn as nn
from Layers.GaussianSmoothing import GaussianSmoothing
from torchvision import models


class SmoothL1ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super(SmoothL1ReconstructionLoss, self).__init__()
        self.config = config
        self.smoothl1_valid = nn.SmoothL1Loss()
        self.smoothl1_hole = nn.SmoothL1Loss()

    def forward(self, gt, image, mask):
        hole_loss = self.smoothl1_hole(gt * mask, image * mask)
        valid_loss = self.smoothl1_valid(gt * (1. - mask), image * (1. - mask))
        return (self.config.hole_loss_weight * hole_loss + self.config.valid_loss_weight * valid_loss)


class EdgeReconstructionLoss(nn.Module):
    def __init__(self, config):
        super(EdgeReconstructionLoss, self).__init__()
        self.config = config
        self.l1_valid = nn.L1Loss()
        self.l1_hole = nn.L1Loss()

    def forward(self, gt, image, mask):
        hole_loss = self.l1_hole(gt * mask, image * mask)
        valid_loss = self.l1_valid(gt * (1. - mask), image * (1. - mask))
        return (self.config.hole_loss_weight * hole_loss + self.config.valid_loss_weight * valid_loss)


class L1ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super(L1ReconstructionLoss, self).__init__()
        self.config = config
        self.l1_valid = nn.L1Loss()
        self.l1_hole = nn.L1Loss()

    def forward(self, gt, image, mask):
        hole_loss = self.l1_hole(gt * mask, image * mask)
        valid_loss = self.l1_valid(gt * (1. - mask), image * (1. - mask))
        return (self.config.hole_loss_weight * hole_loss + self.config.valid_loss_weight * valid_loss)


class L1Loss(nn.Module):
    def __init__(self, config):
        super(L1Loss, self).__init__()
        self.config = config
        self.l1_hole = nn.L1Loss()
        self.l1_valid = nn.L1Loss()

    def forward(self, image, gt, mask):
        assert image.shape == gt.shape
        hole_loss = self.l1_hole(gt * mask, image * mask)
        valid_loss = self.l1_valid(gt * (1. - mask), image * (1. - mask))
        return (self.config.hole_loss_weight * hole_loss + self.config.valid_loss_weight * valid_loss)





def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class RefineReconstructionLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, output, gt, mask):
        output_comp = (1. - mask) * gt + mask * output

        valid_loss = self.l1((1 - mask) * output, (1 - mask) * gt)
        hole_loss = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        prc_loss = 0.0
        for i in range(3):
            prc_loss += self.l1(feat_output[i], feat_gt[i])
            prc_loss += self.l1(feat_output_comp[i], feat_gt[i])

        style_loss = 0.0
        for i in range(3):
            style_loss += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            style_loss += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        tv_loss = total_variation_loss(output_comp)

        return hole_loss, valid_loss, prc_loss, style_loss, tv_loss
