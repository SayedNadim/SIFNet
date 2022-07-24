import os
from os import listdir
import torchvision.transforms as transforms
from skimage.color import gray2rgb
from torch.utils.data import DataLoader
import cv2
import skimage.io as io
import glob
import numpy as np
import random
import torch
import fnmatch
import sys
from PIL import Image
import utils.ColorSpaceConversion as colors
from Layers.MaskGeneration import Masks


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mask_path=None, augment=False, resize=False, random_crop=False, training=False,
                 input_size=(256, 256),
                 with_subfolder=False, with_name=False, random_sample=False, random_sample_size=4):
        super(Dataset, self).__init__()
        if not random_sample:
            if with_subfolder:
                self.samples = self._find_samples_in_subfolders(data_path)
            else:
                self.samples = [os.path.join(data_path, x) for x in listdir(data_path) if self.is_image_file(x)]
        else:
            self.samples = [os.path.join(data_path, x) for x in random.sample(listdir(data_path), random_sample_size) if
                            self.is_image_file(x)]
        self.augment = augment
        self.training = training
        self.data_path = data_path
        self.mask_data = mask_path
        self.input_size = input_size
        self.with_name = with_name
        self.resize_image = resize
        self.random_crop = random_crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            raise ValueError('loading error: ' + self.samples[index])
        return item

    def is_image_file(self, filename):
        img_extension = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        filename_lower = filename.lower()
        if not fnmatch.fnmatch(filename, '*seg*') and not fnmatch.fnmatch(filename, '*parts*'):
            return any(filename_lower.endswith(extension) for extension in img_extension)

    def load_name(self, index):
        name = self.samples[index]
        return os.path.basename(name)

    def load_item(self, index):
        # load image
        img = cv2.imread(self.samples[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if self.training:
            h, w = self.input_size
            if self.resize_image:
                img = self.resize(img, h, w)
            elif self.random_crop:
                img = self.get_random_crop(img, h, w)
            else:
                img = img
        else:
            if self.input_size is None:
                img = img
            else:
                h, w = self.input_size
                if self.resize_image:
                    img = self.resize(img, h, w)
                elif self.random_crop:
                    img = self.get_random_crop(img, h, w)

        mask = self.load_mask(img, index)

        # augment Data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        lab_image = colors.rgb_to_lab(img)
        lab_image[0, :, :] = lab_image[0, :, :] / 100.
        lab_image[1, :, :] = (lab_image[1, :, :] + 128.) / 255.
        lab_image[2, :, :] = (lab_image[2, :, :] + 128.) / 255.
        l_img, ab_image = self._size_splits(lab_image, [1, 2], 0)

        if self.with_name:
            return l_img, ab_image, lab_image, mask, index
        else:
            return l_img, ab_image, lab_image, mask

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # external
        if self.training:
            if self.mask_data is None:
                mask = Masks.get_random_mask(imgh, imgw)
            else:
                mask = io.imread(self.mask_data[index], as_gray=True)
        else:  # in test mode, there's a one-to-one relationship between mask and image; masks are loaded non random
            if self.mask_data is None:
                mask = Masks.get_random_mask(imgh, imgw)
            else:
                mask = io.imread(self.mask_data, as_gray=True)
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = transforms.ToTensor()(img)
        return img_t

    def resize(self, img, height=256, width=256, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, (height, width))
        return img

    def get_random_crop(self, image, crop_height, crop_width):
        if image.shape[0] <= crop_height or image.shape[1] <= crop_width:
            image = cv2.resize(image, (crop_width * 2, crop_height * 2))

        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        image_crop = image[y: y + crop_height, x: x + crop_width]

        image_crop = cv2.resize(image_crop, (crop_width, crop_height))

        return image_crop

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                # print(np.genfromtxt(flist, dtype=np.str))
                # return np.genfromtxt(flist, dtype=np.str)
                try:
                    return np.genfromtxt(flist, dtype=np.str)
                except:
                    return [flist]
        return []

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def _size_splits(self, tensor, split_sizes, dim=0):
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


def build_dataloader(data_path, mask_path, augment, training, resize, random_crop, input_size, batch_size,
                     num_workers, shuffle, with_subfolder=False, with_name=False, random_sample=False):
    dataset = Dataset(
        data_path=data_path,
        mask_path=mask_path,
        augment=augment,
        training=training,
        resize=resize,
        random_crop=random_crop,
        input_size=input_size,
        with_subfolder=with_subfolder,
        with_name=with_name,
        random_sample=random_sample
    )

    print('Total instance number:', dataset.__len__())

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=shuffle,
        pin_memory=False
    )

    return dataloader


