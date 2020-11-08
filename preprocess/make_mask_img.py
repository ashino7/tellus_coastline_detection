import cv2 as cv
import json


path_imgs = './input/train_images_png/'
path_json = './input/train_annotations/'
path_out = './train_images_inpainted_labels/'

for k in range(0, 25):
    k_str = str(k)
    if k < 10: k_str = '0' + str(k)
    img = cv.imread(path_imgs + 'train_' + str(k_str) + '.png', 0)
    img[...] = 0

    data_all = json.load(open(path_json + 'train_' + str(k_str) + '.json'))

    data = data_all['coastline_points']
    for i in range(0, len(data)):
        img[data[i][1], data[i][0]] = 255

    cv.imwrite(path_out + 'label_' + k_str + '.png', img)
