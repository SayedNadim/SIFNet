import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .Losses.ReconstructionLoss import L1ReconstructionLoss, SmoothL1ReconstructionLoss, L1Loss, \
    RefineReconstructionLoss


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class G_reconstruction_loss(nn.Module):
    def __init__(self, config):
        super(G_reconstruction_loss, self).__init__()
        extractor = VGG16FeatureExtractor()
        self.coarse_l_reconstruction = L1Loss(config)
        self.coarse_ab_reconstruction = SmoothL1ReconstructionLoss(config)
        self.coarse_reconstruction = L1ReconstructionLoss(config)
        self.refine_reconstruction = RefineReconstructionLoss(extractor)

    def __call__(self, pred_l, gt_l, pred_ab, gt_ab, coarse, prediction, refined, gt, mask):
        l_loss = self.coarse_l_reconstruction(gt_l, pred_l, mask)
        ab_loss = self.coarse_ab_reconstruction(gt_ab, pred_ab, mask)
        coarse_loss = self.coarse_reconstruction(gt, coarse, mask)
        hole_loss, valid_loss, prc_loss, style_loss, tv_loss = self.refine_reconstruction(prediction, gt, mask)
        return (l_loss + ab_loss + coarse_loss + hole_loss + valid_loss + prc_loss + style_loss + tv_loss)
