import cv2
import numpy as np
import random
import scipy.ndimage as ndimage
import math
from PIL import Image, ImageDraw
import torchvision.utils as vutils
import torchvision.transforms as transforms


## https://arxiv.org/pdf/2010.01110.pdf
class Masks():
    @staticmethod
    def get_ff_mask(H, W):
        # Source: Generative Inpainting https://github.com/JiahuiYu/generative_inpainting
        min_num_vertex = 3
        max_num_vertex = 8
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 10
        max_width = 50

        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 20)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        return np.asarray(mask, np.float32)

    @staticmethod
    def get_box_mask(h, w):
        height, width = h, w

        mask = np.zeros((height, width))

        mask_width = random.randint(int(0.10 * width), int(0.50 * width))
        mask_height = random.randint(int(0.10 * height), int(0.50 * height))

        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)

        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return np.asarray(mask, np.float32)

    @staticmethod
    def get_ca_mask(h, w, scale=None, r=None):

        if scale is None:
            scale = random.choice([1, 2, 4, 8])
        if r is None:
            r = random.randint(2, 6)  # repeat median filter r times

        height = h
        width = w
        mask = np.random.randint(2, size=(height // scale, width // scale))

        # for _ in range(r):
        #     mask = ndimage.median_filter(mask, size=3, mode='constant')

        mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)
        if scale > 1:
            struct = ndimage.generate_binary_structure(2, 1)
            mask = ndimage.morphology.binary_dilation(mask, struct)
        # elif scale > 3:
        #     struct = np.array([[0., 0., 1., 0., 0.],
        #                        [0., 1., 1., 1., 0.],
        #                        [1., 1., 1., 1., 1.],
        #                        [0., 1., 1., 1., 0.],
        #                        [0., 0., 1., 0., 0.]])

        return np.asarray(mask, np.float32)

    @staticmethod
    def get_random_mask(h, w):
        f = random.choice([Masks.get_ff_mask])
        return f(h, w)


if __name__ == '__main__':
    totalNumOfMask = 100
    for i in range(totalNumOfMask):
        mask = Masks.get_random_mask(256, 256)
        mask = transforms.ToTensor()(mask)
        mask = mask.unsqueeze(0)
        print(mask)
        vutils.save_image(mask, '../masks/mask_{}.png'.format(i),
                          padding=0, normalize=True)
