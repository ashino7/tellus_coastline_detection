import argparse
import logging
import os
import glob
import copy

import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from augmentation.augmentation import img_to_tensor


def post_process(_mask):
    _mask = _mask.astype(np.uint8)
    ret, _mask = cv.threshold(_mask, 0.5, 255, cv.THRESH_BINARY)
    orig_mask = copy.deepcopy(_mask)
    h = _mask.shape[0]
    w = _mask.shape[1]

    kernel = np.ones((3, 3), np.uint8)
    _mask = cv.dilate(_mask, kernel)  # 膨張処理
    # mask = cv.erode(mask, kernel)  # 収縮処理
    # cv.imshow('orig_mask', orig_mask)
    # cv.imshow('mask', mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(_mask)
    idx = []
    for n in range(1, n_labels):
        label_w = stats[n, 2]
        label_h = stats[n, 3]
        if label_w > int(w / 4) or label_h > int(h / 4):  # 除去するラベルの条件式
            idx.append(n)

    removal_img = np.zeros((h, w))
    for n in idx:
        removal_img[labels == n] = 255

    removal_img = cv.erode(removal_img, kernel)  # 収縮処理
    # cv.imshow('orig_mask', orig_mask)
    # cv.imshow('removal_img', removal_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return removal_img


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    # img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    full_h = full_img.shape[0]
    full_w = full_img.shape[1]

    img = cv.resize(full_img, (512, 512))
    img = img_to_tensor(img)

    # img = img.unsqueeze(0)
    # img = img.to(device=device, dtype=torch.float32)
    x_tensor = torch.from_numpy(img).to(device).unsqueeze(0)

    with torch.no_grad():
        output = net(x_tensor)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((full_h, full_w)),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./runs/Sep22_13-38-01_ashidaLR_0.0001_BS_1_SCALE_0.5/CP_epoch10.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)
    parser.add_argument('--input_dir', '-i', default='./input/test_images_png/',
                        help='input dir')
    parser.add_argument('--output_dir', '-o', default='./output/',
                        help='output dir')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    input_file_list = glob.glob(input_dir + "*.png")

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(input_file_list):
        logging.info("\nPredicting image {} ...".format(fn))
        base_name = os.path.basename(fn)

        # img = Image.open(fn)
        img = cv.imread(fn, flags=1)
        mono_img = cv.imread(fn, flags=-1)
        # 画像の縁が白で縁取りされているときがあるので黒で埋める
        margin = 5
        h = img.shape[0]
        w = img.shape[1]
        img[:margin, :, :] = 0
        img[h-margin:, :, :] = 0
        img[:, :margin, :] = 0
        img[:, w-margin:, :] = 0

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # 画像内でデータが欠損している箇所の予測値は無視する
        mask = np.where(cv.resize(mono_img, (512, 512)) < 1, 0, mask)
        mask = post_process(mask)
        mask = cv.resize(mask, (w, h))

        if not args.no_save:
            # result = mask_to_image(mask)
            # result.save(output_dir + base_name)
            cv.imwrite(output_dir + base_name, mask)

            logging.info("Mask saved to {}".format(output_dir + base_name))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
