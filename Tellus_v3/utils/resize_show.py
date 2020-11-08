import cv2 as cv
import numpy as np


def resize_show(window_name, img, ratio=1):
    h = img.shape[0]
    w = img.shape[1]
    resize_h = int(h*ratio)
    resize_w = int(w*ratio)
    resize_img = cv.resize(img,(resize_w, resize_h))
    cv.imshow(window_name, resize_img)
    # cv.waitKey()
    # cv.destroyAllWindows()
