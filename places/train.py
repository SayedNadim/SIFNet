from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import os
import torch.backends.cudnn as cudnn

from utils.PrintNetwork import print_network
from utils.ImageRelated import SizeAdapter, lab2rgb_tensor

from Data.Dataloader import build_dataloader
import time
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from skimage.measure import compare_ssim

from Model.Model import InpaintingModel
from utils.Logging import Config
from utils.SaveResume import checkpoint, resume
from utils.LearningRate import get_scheduler

from tensorboardX import SummaryWriter

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Training settings
parser = argparse.ArgumentParser(description='DeBlurNet')
parser.add_argument('--config', type=str,
                    default='/home/cvip/PycharmProjects/LAB_Inpainting_Places/Configs/training_config.yaml',
                    help='path to config file')

config = parser.parse_args()
# gpus_list = list(range(config.gpus))  # the list of gpu

config = Config(config.config)
config.checkpoint_dir += '/' + config.expname
cudnn.benchmark = config.cudnn_benchmark
if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)


def train(epoch, step):
    iteration, avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss = step, 0, 0, 0, 0
    last_l1_loss, last_gan_loss, cur_l1_loss, cur_gan_loss = 0, 0, 0, 0
    model.train()
    t0 = time.time()
    t_io1 = time.time()
    for batch in training_data_loader:
        l_image, ab_image, gt, mask = batch
        t_io2 = time.time()
        if cuda:
            l_image = l_image.cuda()
            ab_image = ab_image.cuda()
            gt = gt.cuda()
            mask = mask.cuda()

        prediction, coarse, l_predict, ab_predict, flow = model.generator(l_image, ab_image, gt, mask)
        merged_result = prediction * mask + gt * (1 - mask)
        c_merged_result = coarse * mask + gt * (1 - mask)
        l_merged_result = l_predict * mask + l_image * (1 - mask)

        # Compute Loss
        g_loss, d_loss = 0, 0

        ## Reconstruction
        recon_loss = model.g_reconstruction_loss(l_predict, l_image, ab_predict, ab_image, c_merged_result, prediction,
                                                 merged_result, gt, mask)

        # Feature discriminator for G
        g_real_f = model.feature_discriminator(gt)
        g_fake_f = model.feature_discriminator(prediction)
        g_gan_loss_feature = model.g_gan_loss_feature(g_fake_f, g_real_f)

        ## Pixel discriminator for G
        g_real_p = model.pixel_discriminator(gt)
        g_fake_p = model.pixel_discriminator(prediction)
        g_gan_loss_pixel = model.g_gan_loss_pixel(g_fake_p, g_real_p)

        ## Total Loss
        g_gan_loss = (model.gan_weight * g_gan_loss_pixel + model.gan_weight * g_gan_loss_feature) / 2.
        g_recon_loss = model.l1_weight * recon_loss
        g_loss += g_gan_loss + g_recon_loss

        # Record
        cur_l1_loss += g_recon_loss.data.item()
        cur_gan_loss += g_gan_loss.data.item()
        avg_l1_loss += g_recon_loss.data.item()
        avg_gan_loss += g_gan_loss.data.item()
        avg_g_loss += g_loss.data.item()

        # Backward
        model.gen_optimizer.zero_grad()
        g_loss.backward()
        model.gen_optimizer.step()

        ## Feature Discriminator for D
        d_real_f = model.feature_discriminator(gt)
        d_fake_f = model.feature_discriminator(prediction.detach())
        d_loss_feature = model.d_gan_loss_feature(d_fake_f, d_real_f)

        ## Pixel Discriminator for D
        d_real_p = model.pixel_discriminator(gt)
        d_fake_p = model.pixel_discriminator(prediction.detach())
        d_loss_pixel = model.d_gan_loss_pixel(d_fake_p, d_real_p)

        ## Total D loss
        d_loss += (d_loss_feature + d_loss_pixel)/2.

        avg_d_loss += d_loss.data.item()

        model.dis_feature_optimizer.zero_grad()
        model.dis_pixel_optimizer.zero_grad()
        d_loss.backward()
        model.dis_feature_optimizer.step()
        model.dis_pixel_optimizer.step()

        model.global_iter += 1
        iteration += 1
        t1 = time.time()
        td, t0 = t1 - t0, t1

        if iteration % config.print_interval == 0:
            print(
                "=> Epoch[{}/{}]({}/{}): Avg L1 loss: {:.6f} | G loss: {:.6f} | Avg D loss: {:.6f} || Timer: {:.4f} sec. | IO: {:.4f}".format(
                    epoch, config.epoch, iteration, len(training_data_loader), avg_l1_loss / config.print_interval,
                                                                               avg_g_loss / config.print_interval,
                                                                               avg_d_loss / config.print_interval, td,
                                                                               t_io2 - t_io1),
                flush=True)

            if config.tensorboard:
                gt = lab2rgb_tensor(gt)
                merged_result = lab2rgb_tensor(merged_result)
                c_merged_result = lab2rgb_tensor(c_merged_result)
                input_image = gt * (1. - mask)
                input_l_image = l_image * (1. - mask)
                flow1 = F.interpolate(flow[0], size=gt.size()[2:])
                flow2 = F.interpolate(flow[1], size=gt.size()[2:])
                # l_predict = l_predict.repeat(1, 3, 1, 1)
                # l_image = l_image.repeat(1, 3, 1, 1)
                # l_merged_result = lab2rgb_tensor(l_merged_result)
                # l_image = lab2rgb_tensor(l_image)
                ims_images = torch.cat([input_image, c_merged_result, merged_result, gt, flow1, flow2], dim=3)
                ims_l = torch.cat([input_l_image, l_merged_result, l_image], dim=3)
                writer.add_images('Training/Input____Coarse____Final____GroundTruth___Flow', ims_images,
                                  model.global_iter)
                writer.add_images('Training/Input_L_Image___Reconstructed_L_Image____GT_l_Image', ims_l,
                                  model.global_iter)
                writer.add_scalar('scalar/G_loss', avg_g_loss / config.print_interval, model.global_iter)
                writer.add_scalar('scalar/G_l1_loss', avg_l1_loss / config.print_interval, model.global_iter)
                writer.add_scalar('scalar/G_gan_loss', avg_gan_loss / config.print_interval, model.global_iter)
                writer.add_scalar('scalar/D_loss', avg_d_loss / config.print_interval, model.global_iter)

            avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss = 0, 0, 0, 0
            if epoch % config.save_epoch == 0 and iteration % config.save_step == 0:
                checkpoint(config, model, epoch, iteration)
        t_io1 = time.time()


def test(gen, dataloader):
    model = gen.eval()
    psnr = 0
    mse = 0
    ssim = 0
    l1 = 0
    count = 0
    flag = False
    for batch in dataloader:
        l_image_batch, ab_image_batch, gt_batch, mask_batch = batch
        if gt_batch.shape[2] % 256 != 0 or gt_batch.shape[3] % 256 != 0:
            gt_batch, mask_batch = SizeAdapter().pad(gt_batch), SizeAdapter().pad(mask_batch)
            flag = True
        if cuda:
            gt_batch = gt_batch.cuda()
            mask_batch = mask_batch.cuda()
        with torch.no_grad():
            pred_batch, _, _, _, _ = model.generator(l_image_batch, ab_image_batch, gt_batch, mask_batch)
            pred_batch = lab2rgb_tensor(pred_batch)
            gt_batch = lab2rgb_tensor(gt_batch)
            if flag:
                pred_batch = SizeAdapter().unpad(pred_batch)
                gt_batch = SizeAdapter().unpad(gt_batch)
                mask_batch = SizeAdapter().unpad(mask_batch)
        for i in range(gt_batch.size(0)):
            gt, pred = gt_batch[i], pred_batch[i]
            psnr += compare_psnr(pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy(), data_range=1)
            mse += compare_mse(pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy(), )
            ssim += compare_ssim(pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy(),
                                 multichannel=True)
            count += 1
        if config.tensorboard:
            input_val = gt_batch * (1. - mask_batch)
            pred_merged = pred_batch * mask_batch + gt_batch * (1. - mask_batch)
            ims_val = torch.cat([input_val, pred_merged, gt_batch], dim=3)
            writer.add_images('Validation', ims_val, model.global_iter)
        print('tested {}th image'.format(count))
    avg_psnr = psnr / count
    avg_mse = mse / count
    avg_ssim = ssim / count
    return avg_psnr, avg_mse, avg_ssim


if __name__ == '__main__':
    if config.tensorboard:
        writer = SummaryWriter()

    # Set the GPU mode
    cuda = config.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Set the random seed
    torch.manual_seed(config.seed)
    if cuda:
        torch.cuda.manual_seed_all(config.seed)

    # Model
    model = InpaintingModel(config)
    print('---------- Networks architecture -------------')
    print("Generator:")
    print_network(model.generator)
    print("Feature Discriminator:")
    print_network(model.feature_discriminator)
    print("Pixel Discriminator:")
    print_network(model.pixel_discriminator)
    print('----------------------------------------------')

    if cuda:
        model = model.cuda()

    # if cuda:
    #     model = model.cuda()
    # if config.gpus > 1:
    #     model.generator = torch.nn.DataParallel(model.generator, device_ids=gpus_list)
    #     model.local_discriminator = torch.nn.DataParallel(model.local_discriminator, device_ids=gpus_list)

    # Load the pretrain model.
    start_epoch = config.start_epoch
    end_epoch = config.epoch
    start_step = 0

    if config.resume:
        model, _, start_epoch = resume(config, model)

    # Datasets
    print('===> Loading datasets')
    training_data_loader = build_dataloader(
        data_path=config.train_data_root,
        mask_path= None,
        augment=config.augment,
        batch_size=config.batch_size,
        num_workers=config.threads,
        shuffle=config.shuffle,
        input_size=config.input_size,
        resize=config.resize,
        random_crop=config.random_crop,
        with_subfolder=config.with_subfolder,
        training=True
    )
    print('===> Loaded datasets')

    if config.validation:
        test_data_loader = build_dataloader(
            data_path=config.val_data_root,
            mask_path= None,
            augment=False,
            batch_size=config.batch_size,
            num_workers=config.threads,
            random_crop= False,
            shuffle=False,
            input_size=config.input_size,
            resize= True,
            with_subfolder= False,
            training=False
        )
        print('===> Loaded test datasets')

    # Start training
    for epoch in range(start_epoch, end_epoch + 1):

        train(epoch, start_step)

        if config.with_test:
            print("Testing images...")
            test_psnr, test_mse, test_ssim = test(model, test_data_loader)
            if config.tensorboard:
                writer.add_scalar('scalar/test_PSNR', test_psnr, model.global_iter)
                writer.add_scalar('scalar/test_MSE', test_mse, model.global_iter)
                writer.add_scalar('scalar/test_SSIM', test_ssim, model.global_iter)
                print("PSNR: {}, MSE: {}, SSIM: {}".format(test_psnr, test_mse, test_ssim))
        count = (epoch - 1)
        if epoch % config.reduce_lr_epoch == 0:
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            for param_group in model.gen_optimizer.param_groups:
                param_group['lr'] = model.g_lr * (0.8 ** count)
                print('===> Current G learning rate: ', param_group['lr'])
            for param_group in model.dis_pixel_optimizer.param_groups:
                param_group['lr'] = model.d_pixel_lr * (0.8 ** count)
                print('===> Current D learning rate: ', param_group['lr'])
            for param_group in model.dis_feature_optimizer.param_groups:
                param_group['lr'] = model.d_feature_lr * (0.8 ** count)
                print('===> Current D learning rate: ', param_group['lr'])

    if config.tensorboard:
        writer.close()
    os._exit(0)
