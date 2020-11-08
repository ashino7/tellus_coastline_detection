import numpy as np
import cv2
import tifffile
import glob
import os
import json


def main():
    input_json_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/input/train_annotations/'
    input_image_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/input/train_images_png/'

    json_list = sorted(glob.glob(input_json_dir + "*.json"))
    image_list = sorted(glob.glob(input_image_dir + "*.png"))

    for json_path, img_path in zip(json_list, image_list):
        with open(json_path, "r")as f:
            ano_data = json.load(f)
        img = cv2.imread(img_path)
        ano_data["validate_lines"] = np.asarray(ano_data["validate_lines"], dtype=int)
        ano_data["coastline_points"] = np.asarray(ano_data["coastline_points"], dtype=int)

        ans_list = []
        sortkey_list=[]
        for point1, point2 in ano_data["validate_lines"]:
            min_list = []
            for c_point in ano_data["coastline_points"]:
                min_list.append(np.linalg.norm(c_point - point1) + np.linalg.norm(c_point - point2))
            index = np.argmin(np.asarray(min_list, dtype=np.int))
            ans_list.append(ano_data["coastline_points"][index])
            if img.shape[0]<img.shape[1]:
                sortkey_list.append(ano_data["coastline_points"][index][0])
            else:
                sortkey_list.append(ano_data["coastline_points"][index][1])

        ans_list=np.asarray(ans_list,dtype=np.int)[np.argsort(sortkey_list)]

        line_list = []
        for i in range(len(ans_list) - 2):
            line_list.append([ans_list[i], ans_list[i + 1]])
        line_list = np.asarray(line_list, dtype=np.int)

        h = img.shape[0]
        w = img.shape[1]
        zero_img = np.zeros((h, w), dtype=np.uint8)
        zero_img = cv2.polylines(zero_img, line_list, False, 255, 1)

        cv2.imwrite("results/{}".format(os.path.basename(img_path)), zero_img)


if __name__ == '__main__':
    main()
