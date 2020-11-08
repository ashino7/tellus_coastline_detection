import os
import glob
import cv2 as cv
import numpy as np
import copy


def main():
    """
    2値化後の画像を元画像にオーバーレイする
    (2値化後の画像で上書きする)
    :return:
    """
    input_ori_dir = '/mnt/d_drive/home/ashida/work/Tellus_v2/input/test_images_png/'
    input_bin_dir = '/mnt/d_drive/home/ashida/work/Tellus_v2/runs/Sep18_16-27-08_ashidaLR_0.0001_BS_1_SCALE_0.5/ver0.6/'
    output_dir = '/mnt/d_drive/home/ashida/work/Tellus_v2/runs/Sep18_16-27-08_ashidaLR_0.0001_BS_1_SCALE_0.5/ver0.6//over_lay/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    orig_img_list = glob.glob(input_ori_dir + '*.png')
    for orig_img_name in orig_img_list:
        orig_base_name = os.path.basename(orig_img_name)
        bin_img_name = input_bin_dir + orig_base_name
        if not os.path.isfile:
            print('{}が存在しません'.format(bin_img_name))
            continue

        orig_img = cv.imread(orig_img_name, flags=1)
        bin_img = cv.imread(bin_img_name, flags=-1)
        overlay_img = copy.deepcopy(orig_img)
        points = np.where(bin_img > 1)

        for idx in range(len(points[0])):
            y = points[0][idx]
            x = points[1][idx]
            overlay_img[y, x] = [0, 0, 255]
        # cv.imshow("img", orig_img)
        # cv.waitKey(0)
        cv.imwrite(output_dir + "overlay_" + orig_base_name, overlay_img)


if __name__ == '__main__':
    main()