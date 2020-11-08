import os
import glob
import cv2 as cv
import numpy as np

def main():
    orig_img_dir = '/mnt/d_drive/home/ashida/work/Tellus_v4/input/test_images_png/'
    predict_img_dir = '/mnt/d_drive/home/ashida/work/Tellus_v4/postprocess/ensemble_v2/output_line/'
    output_dir = './output/'

    slide_pix = 10

    file_list = sorted(glob.glob(orig_img_dir + '*.png'))
    for i, file_name in enumerate(file_list):
        base_name = os.path.basename(file_name)
        orig_img = cv.imread(file_name, flags=-1)
        predict_img = cv.imread(predict_img_dir + base_name, flags=-1)

        h, w = orig_img.shape

        # 長い方を横にする
        if w > h:
            rotate_flag = True
            orig_img = cv.rotate(orig_img, cv.ROTATE_90_CLOCKWISE)
            predict_img = cv.rotate(predict_img, cv.ROTATE_90_CLOCKWISE)
            rotated_img_h, rotated_img_w = orig_img.shape
        else:
            rotate_flag = False
            rotated_img_h, rotated_img_w = orig_img.shape

        center_h = rotated_img_h // 2
        top_img = orig_img[:center_h, :]
        bot_img = orig_img[center_h:, ]

        # 平均輝度が低い方を海とする
        top_mean = top_img.mean()
        bot_mean = bot_img.mean()

        padding_img = np.zeros((slide_pix, rotated_img_w))

        if top_mean > bot_mean:
            # 下側が海
            predict_img = np.vstack([padding_img, predict_img[:-slide_pix, :]])
        else:
            # 上側が海
            predict_img = np.vstack([predict_img[slide_pix:, :], padding_img])

        # if top_mean < bot_mean:
        #     # 下側が山
        #     predict_img = np.vstack([padding_img, predict_img[:-slide_pix, :]])
        # else:
        #     # 上側が山
        #     predict_img = np.vstack([predict_img[slide_pix:, :], padding_img])

        if rotate_flag:
            predict_img = cv.rotate(predict_img, cv.ROTATE_90_COUNTERCLOCKWISE)

        cv.imwrite(output_dir + base_name, predict_img)
        c=0


if __name__ == '__main__':
    main()
