import torch
import torch.nn as nn
import torch.optim as optim

from .Networks.Generator import G_Net
from .Networks.Discriminator import PixelDiscriminator, NLayerDiscriminator
from .Loss import G_reconstruction_loss
from .Losses.AdversarialLoss import RelativisticLeastSquareGANLoss


class InpaintingModel(nn.Module):
    def __init__(self, config, iter=0):
        super(InpaintingModel, self).__init__()

        self.generator = G_Net(config)
        self.pixel_discriminator = PixelDiscriminator(config)
        self.feature_discriminator = NLayerDiscriminator(config)

        self.g_reconstruction_loss = G_reconstruction_loss(config)

        self.g_gan_loss_feature = RelativisticLeastSquareGANLoss(mode='gen')
        self.g_gan_loss_pixel = RelativisticLeastSquareGANLoss(mode='gen')

        self.d_gan_loss_feature = RelativisticLeastSquareGANLoss(mode='dis')
        self.d_gan_loss_pixel = RelativisticLeastSquareGANLoss(mode='dis')

        self.g_lr = config.g_lr
        self.d_pixel_lr = config.d_pixel_lr
        self.d_feature_lr = config.d_feature_lr
        self.gan_weight = config.gan_weight
        self.l1_weight = config.l1_weight

        self.global_iter = iter

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(self.g_lr),
            betas=(0.5, 0.9)
        )

        self.dis_pixel_optimizer = optim.Adam(
            params=self.pixel_discriminator.parameters(),
            lr=float(self.d_pixel_lr),
            betas=(0.5, 0.9)
        )

        self.dis_feature_optimizer = optim.Adam(
            params=self.feature_discriminator.parameters(),
            lr=float(self.d_feature_lr),
            betas=(0.5, 0.9)
        )
