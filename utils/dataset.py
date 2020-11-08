from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageFilter
import cv2 as cv
from matplotlib import pyplot as plt


def gray2color(img):
    gray_image = np.array(img, dtype=np.uint8)
    rgb_img = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))
    rgb_img[:, :, 0] = gray_image
    rgb_img[:, :, 1] = gray_image
    rgb_img[:, :, 2] = gray_image
    rgb_img = rgb_img.astype(np.uint8)
    rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)
    PIL_img = Image.fromarray(rgb_img)
    return PIL_img


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', augmentation=None, preprocessing=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    # @classmethod
    # def preprocess(cls, pil_img, scale, is_mask=False):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small'
    #     # pil_img = pil_img.resize((newW, newH))
    #     pil_img = pil_img.resize((546, 546))
    #
    #     img_nd = np.array(pil_img)
    #
    #     if is_mask:
    #         kernel = np.ones((6, 6), np.uint8)
    #         img_nd = cv.dilate(img_nd, kernel)
    #         ret, img_nd = cv.threshold(img_nd, 1, 255, cv.THRESH_BINARY)
    #         # cv.imshow("img", img_nd)
    #         # cv.waitKey(0)
    #
    #     if len(img_nd.shape) == 2:
    #         img_nd = np.expand_dims(img_nd, axis=2)
    #
    #
    #     # HWC to CHW
    #     img_trans = img_nd.transpose((2, 0, 1))
    #     if img_trans.max() > 1:
    #         img_trans = img_trans / 255
    #
    #     return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # read data
        img = cv.imread(img_file[0], flags=1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (512, 512))

        mask = cv.imread(mask_file[0], flags=-1)
        mask = cv.resize(mask, (512, 512))
        kernel = np.ones((13, 13), np.uint8)  # 膨張処理のためのカーネルサイズを選ぶ
        mask = cv.dilate(mask, kernel)  # 膨張処理
        ret, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

        # 画像の縁が白で縁取りされているときがあるので黒で埋める
        margin = 5
        h = img.shape[0]
        w = img.shape[1]
        img[:margin, :, :] = 0
        img[h-margin:, :, :] = 0
        img[:, :margin, :] = 0
        img[:, w-margin:, :] = 0

        # mask = gray2color(mask)
        # img = gray2color(img)

        # mask = mask.filter(ImageFilter.MaxFilter())  # 膨張処理
        # plt.imshow(mask)
        # plt.show()

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess(img, self.scale, is_mask=False)
        # mask = self.preprocess(mask, self.scale, is_mask=True)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return {
            'image': img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
