import numpy as np
import cv2
import tifffile
import glob
import os


def main():
    img_folders_path = "/home/ashida/work/Tellus/input/test_images/"

    file_list = glob.glob(img_folders_path + "*.tif")
    for tif_path in file_list:
        data = tifffile.imread(tif_path)
        data = data.astype(np.float)
        data /= data.max()
        data *= 255
        data = data.astype(np.uint8)
        tif_name = os.path.basename(tif_path).rsplit(".", 1)[0]
        cv2.imwrite("{}/{}/{}.png".format(img_folders_path, "png", tif_name), data)


if __name__ == '__main__':
    main()
