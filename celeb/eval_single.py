from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import fnmatch
import os
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import skimage.io as io
from PIL import Image
from pathlib import Path
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

from utils.PrintNetwork import print_network
from utils.ImageRelated import SizeAdapter, lab2rgb_tensor
from Layers.MaskGeneration import Masks

from Data.Dataloader import build_dataloader
import time
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from skimage.measure import compare_ssim
from skimage import img_as_ubyte
import re
import cv2

from Model.Model import InpaintingModel
from utils.Logging import Config
from utils.ImageRelated import save_img, SizeAdapter

import torchvision.utils as vutils
import utils.ColorSpaceConversion as colors

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def dataset_files(rootdir, pattern):
    """Returns a list of all image files in the given directory"""

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    matches.sort(key=natural_keys)

    return matches


def sort(filename):
    num_array = sorted([str(y) for y in [x.split('.')[0] for x in filename]])
    return np.array([str(x) + '.jpg' for x in num_array])


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def _size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = transforms.ToTensor()(img)
    return img_t


def eval(config, image_file, mask_file):
    model.eval()
    model.generator.eval()
    count = 1
    du = 0
    avg_du = 0
    avg_psnr, avg_ssim, avg_l1 = 0., 0., 0.

    ## The test or ensemble test

    t0 = time.time()
    with torch.no_grad():
        print(image_file)
        filename_full, ext = image_file.split('.')
        filename = filename_full.split('/')[-1]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        img = transforms.Resize(config.input_size)(img)
        mask = transforms.Resize(config.input_size)(mask)
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        # input()
        lab_image = img * (1. - mask)
        lab_image = colors.rgb_to_lab(lab_image)
        lab_image[:, 0, :, :] = lab_image[:, 0, :, :] / 100.
        lab_image[:, 1, :, :] = (lab_image[:, 1, :, :] + 128.) / 255.
        lab_image[:, 2, :, :] = (lab_image[:, 2, :, :] + 128.) / 255.
        l_img, ab_image = _size_splits(lab_image, [1, 2], 1)
        if cuda:
            l_img = l_img.cuda()
            ab_image = ab_image.cuda()
            lab_image = lab_image.cuda()
            mask = mask.cuda()
        prediction, coarse, l_predict, ab_predict, flow = model.generator(l_img, ab_image, lab_image, mask)
        prediction = lab2rgb_tensor(prediction)  # * 255.
        gt = lab2rgb_tensor(lab_image)  # * 255.
        if prediction.shape[2] != gt.shape[2] or prediction.shape[3] != gt.shape[3]:
            prediction = F.interpolate(prediction, size=(gt.shape[2], gt.shape[3]))

        prediction = prediction * mask + gt * (1 - mask)

        if config.save_test_images:
            ######################
            output_save_path = 'output/'
            input_save_path = 'input/'
            L_output_save_path = 'L_output/'
            L_input_save_path = 'L_input/'
            AB_output_save_path = 'AB_output/'
            AB_input_save_path = 'AB_input/'
            FLOW_1_output_save_path = 'Flow_1_output/'
            FLOW_2_output_save_path = 'Flow_2_output/'
            coarse_output_save_path = 'coarse_output/'
            gt_save_path = 'gt_resized/'
            #######################
            ############################
            if not os.path.exists(config.save_results_path + '/' + input_save_path):
                os.makedirs(config.save_results_path + '/' + input_save_path)
            if not os.path.exists(config.save_results_path + '/' + output_save_path):
                os.makedirs(config.save_results_path + '/' + output_save_path)
            if not os.path.exists(config.save_results_path + '/' + L_output_save_path):
                os.makedirs(config.save_results_path + '/' + L_output_save_path)
            if not os.path.exists(config.save_results_path + '/' + L_input_save_path):
                os.makedirs(config.save_results_path + '/' + L_input_save_path)
            if not os.path.exists(config.save_results_path + '/' + AB_output_save_path):
                os.makedirs(config.save_results_path + '/' + AB_output_save_path)
            if not os.path.exists(config.save_results_path + '/' + AB_input_save_path):
                os.makedirs(config.save_results_path + '/' + AB_input_save_path)
            if not os.path.exists(config.save_results_path + '/' + FLOW_1_output_save_path):
                os.makedirs(config.save_results_path + '/' + FLOW_1_output_save_path)
            if not os.path.exists(config.save_results_path + '/' + FLOW_2_output_save_path):
                os.makedirs(config.save_results_path + '/' + FLOW_2_output_save_path)
            if not os.path.exists(config.save_results_path + '/' + coarse_output_save_path):
                os.makedirs(config.save_results_path + '/' + coarse_output_save_path)
            if not os.path.exists(config.save_results_path + '/' + gt_save_path):
                os.makedirs(config.save_results_path + '/' + gt_save_path)
            ############################
            ###################

            input_batch = (gt * (1 - mask))
            l_input_batch = l_img * (1. - mask)
            l_output_batch = l_predict * mask + l_input_batch

            ab_input_batch = ab_image * (1. - mask)
            ab_output_batch = ab_predict * mask + ab_input_batch

            dummy_channel = torch.zeros_like(l_img)
            ab_input_batch = torch.cat((dummy_channel, ab_input_batch), dim=1)
            ab_output_batch = torch.cat((dummy_channel, ab_output_batch), dim=1)

            coarse_output_batch = torch.cat((l_predict, ab_predict), dim=1)
            coarse_output_batch = lab2rgb_tensor(coarse_output_batch)


            l_input_batch = l_input_batch.repeat(1, 3, 1, 1)
            l_output_batch = l_output_batch.repeat(1, 3, 1, 1)

            l_input_batch = (l_input_batch.detach().permute(0, 2, 3, 1).cpu().numpy())
            l_output_batch = (l_output_batch.detach().permute(0, 2, 3, 1).cpu().numpy())
            ab_input_batch = (ab_input_batch.detach().permute(0, 2, 3, 1).cpu().numpy())
            ab_output_batch = (ab_output_batch.detach().permute(0, 2, 3, 1).cpu().numpy())
            flow_1_batch = (flow[0].detach().permute(0, 2, 3, 1).cpu().numpy())
            flow_2_batch = (flow[1].detach().permute(0, 2, 3, 1).cpu().numpy())
            coarse_output_batch = (coarse_output_batch.detach().permute(0, 2, 3, 1).cpu().numpy())
            gt_resize_batch = (img.detach().permute(0, 2, 3, 1).cpu().numpy())

            ####################
            input_batch = (input_batch.detach().permute(0, 2, 3, 1).cpu().numpy())

            mask_batch = (mask.detach().permute(0, 2, 3, 1).cpu().numpy()[:, :, :, 0])
            gt_batch = (gt.detach().permute(0, 2, 3, 1).cpu().numpy())
            pred_batch = (prediction.detach().permute(0, 2, 3, 1).cpu().numpy())
            #####################
            #####################
            plt.imsave(config.save_results_path + '/' + L_output_save_path + '/' + 'l_output_{}.{}'.format(filename, ext), (l_output_batch[0]))
            plt.imsave(
                config.save_results_path + '/' + L_input_save_path + '/' + 'l_input_{}.{}'.format(filename, ext), (l_input_batch[0]))
            plt.imsave(
                config.save_results_path + '/' + AB_output_save_path + '/' + 'ab_output_{}.{}'.format(filename, ext), (ab_output_batch[0]))
            plt.imsave(
                config.save_results_path + '/' + AB_input_save_path + '/' + 'ab_input_{}.{}'.format(filename, ext), (ab_input_batch[0]))
            plt.imsave(
                config.save_results_path + '/' + coarse_output_save_path + '/' + 'coarse_{}.{}'.format(filename, ext) + '.png', (coarse_output_batch[0]))
            io.imsave(
                config.save_results_path + '/' + FLOW_1_output_save_path + '/' + 'flow_1_{}.{}'.format(filename, ext), img_as_ubyte(cv2.resize(flow_1_batch[0], (256, 256))))
            io.imsave(
                config.save_results_path + '/' + FLOW_2_output_save_path + '/' + 'flow_2_{}.{}'.format(filename, ext), img_as_ubyte(cv2.resize(flow_2_batch[0], (256, 256))))
            io.imsave(
                config.save_results_path + '/' + gt_save_path + '/' + 'gt_resized_{}.{}'.format(filename, ext), img_as_ubyte(cv2.resize(gt_resize_batch[0], (256, 256))))
            #######################
            io.imsave(
                config.save_results_path + '/' + output_save_path + '/' + 'output_{}.{}'.format(filename, ext), img_as_ubyte(pred_batch[0]))
            io.imsave(
                config.save_results_path + '/' + input_save_path + '/' + 'input_{}.{}'.format(filename, ext) + '.png', img_as_ubyte(input_batch[0]))

        t1 = time.time()
        du = t1 - t0
        print("===> Processed in: %.4f sec." % (du))

if __name__ == '__main__':
    image_path = '/home/la-belva/code_ground/INPAINTING/SIFNet/celeb/sample_image/image.jpg'
    mask_path = '/home/la-belva/code_ground/INPAINTING/SIFNet/celeb/sample_image/mask_20_30.png'
    # Evaluation settings
    parser = argparse.ArgumentParser(description='LAB')
    parser.add_argument('--config', type=str,
                        default='Configs/testing_config.yaml',
                        help='path to config file')

    config = parser.parse_args()

    config = Config(config.config)
    config.checkpoint_dir += '/' + config.expname
    cudnn.benchmark = config.cudnn_benchmark
    if not config.gpu_mode:
        print("===== Using CPU to Test! =====")
    else:
        print("===== Using GPU to Test! =====")

    ## Set the GPU mode
    gpus_list = range(config.gpus)
    cuda = config.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

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

    pretained_model = torch.load(config.checkpoint, map_location=lambda storage, loc: storage)

    if cuda:
        model = model.cuda()
        model.load_state_dict(pretained_model)
    else:
        new_state_dict = model.state_dict()
        for k, v in pretained_model.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    print('Pre-trained G model is loaded.')
    ## Eval Start!!!!
    eval(config, image_path, mask_path)
