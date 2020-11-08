import numpy as np
import cv2 as cv
import tifffile
import glob
import os
import json


def main():
    input_tiff_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/input/train_images/'
    input_image_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/input/train_images_png_kouten/'
    output_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/input/train_images_png_kouten_ignore_blank/'

    tif_list = sorted(glob.glob(input_tiff_dir + "*.tif"))
    image_list = sorted(glob.glob(input_image_dir + "*.png"))

    for tif_path, img_path in zip(tif_list, image_list):
        tif_data = tifffile.imread(tif_path)
        tif_data = tif_data.astype(np.float)

        points = np.where(tif_data == 0)
        points_len = len(points[0])

        png_image = cv.imread(img_path, flags=-1)

        points_list = []
        for i in range(points_len):
            y = int(points[0][i])
            x = int(points[1][i])
            # points_list.append(list([x, y]))
            png_image[y, x] = 0
        cv.imwrite(output_dir + os.path.basename(img_path), png_image)


if __name__ == '__main__':
    main()
