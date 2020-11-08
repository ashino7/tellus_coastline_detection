import os
import glob
import cv2 as cv
import numpy as np
import random
import copy
from utils.resize_show import resize_show

import numpy as np

# 与えられた引数全てについての論理積を返すメソッドです。
def multi_logical_and(*args):
    result = np.copy(args[0])
    for arg in args:
        result = np.logical_and(result, arg)
    return result

# 2値画像について、周囲1ピクセルをFalseで埋めるメソッドです
def padding(binary_image):
    row, col = np.shape(binary_image)
    result = np.zeros((row+2,col+2))
    result[1:-1, 1:-1] = binary_image[:, :]
    return result

# paddingの逆です
def unpadding(image):
    return image[1:-1, 1:-1]

# そのピクセルの周囲のピクセルの情報を格納したarrayを返します。
def generate_mask(image):
    row, col = np.shape(image)
    p2 = np.zeros((row, col)).astype(bool)
    p3 = np.zeros((row, col)).astype(bool)
    p4 = np.zeros((row, col)).astype(bool)
    p5 = np.zeros((row, col)).astype(bool)
    p6 = np.zeros((row, col)).astype(bool)
    p7 = np.zeros((row, col)).astype(bool)
    p8 = np.zeros((row, col)).astype(bool)
    p9 = np.zeros((row, col)).astype(bool)
    #上
    p2[1:row-1, 1:col-1] = image[0:row-2, 1:col-1]
    #右上
    p3[1:row-1, 1:col-1] = image[0:row-2, 2:col]
    #右
    p4[1:row-1, 1:col-1] = image[1:row-1, 2:col]
    #右下
    p5[1:row-1, 1:col-1] = image[2:row, 2:col]
    #下
    p6[1:row-1, 1:col-1] = image[2:row, 1:col-1]
    #左下
    p7[1:row-1, 1:col-1] = image[2:row, 0:col-2]
    #左
    p8[1:row-1, 1:col-1] = image[1:row-1, 0:col-2]
    #左上
    p9[1:row-1, 1:col-1] = image[0:row-2, 0:col-2]
    return (p2, p3, p4, p5, p6, p7, p8, p9)

# 周囲のピクセルを順番に並べたときに白→黒がちょうど1箇所だけあるかどうかを判定するメソッドです。
def is_once_change(p_tuple):
    number_change = np.zeros_like(p_tuple[0])
    # P2~P9,P2について、隣接する要素の排他的論理和を取った場合のTrueの個数を数えます。
    for i in range(len(p_tuple) - 1):
        number_change = np.add(number_change, np.logical_xor(p_tuple[i], p_tuple[i+1]).astype(int))
    number_change = np.add(number_change, np.logical_xor(p_tuple[7], p_tuple[0]).astype(int))
    array_two = np.ones_like(p_tuple[0]) * 2

    return np.equal(number_change, array_two)

# 周囲の黒ピクセルの数を数え、2以上6以下となっているかを判定するメソッドです。
def is_black_pixels_appropriate(p_tuple):
    number_of_black_pxels = np.zeros_like(p_tuple[0])
    array_two = np.ones_like(p_tuple[0]) * 2
    array_six = np.ones_like(p_tuple[0]) * 6
    for p in p_tuple:
        number_of_black_pxels = np.add(number_of_black_pxels, p.astype(int))
    greater_two = np.greater_equal(number_of_black_pxels, array_two)
    less_six = np.less_equal(number_of_black_pxels, array_six)
    return np.logical_and(greater_two, less_six)

def step1(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(multi_logical_and(p_tuple[0], p_tuple[2], p_tuple[4]))

    #条件5
    condition5 = np.logical_not(multi_logical_and(p_tuple[2], p_tuple[4], p_tuple[6]))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

def step2(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[2], p_tuple[6])))

    #条件5
    condition5 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[4], p_tuple[6])))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

# 2値化画像を細線化して返すメソッドです。
def ZhangSuen(image):

    image = padding(image)

    while True:
        old_image = np.copy(image)

        p_tuple = generate_mask(image)
        image = step1(image, p_tuple)
        p_tuple = generate_mask(image)
        image = step2(image, p_tuple)

        if (np.array_equal(old_image, image)):
            break

    return unpadding(image)

# # Zhang-Suenのアルゴリズムを用いて2値化画像を細線化します
# def Zhang_Suen_thinning(binary_image):
#     # オリジナルの画像をコピー
#     image_thinned = binary_image.copy()
#     # 初期化します。この値は次のwhile文の中で除かれます。
#     changing_1 = changing_2 = [1]
#     while changing_1 or changing_2:
#         # ステップ1
#         changing_1 = []
#         rows, columns = image_thinned.shape
#         for x in range(1, rows - 1):
#             for y in range(1, columns -1):
#                 p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
#                 if (image_thinned[x][y] == 1 and
#                     2 <= sum(neighbour_points) <= 6 and # 条件2
#                     count_transition(neighbour_points) == 1 and # 条件3
#                     p2 * p4 * p6 == 0 and # 条件4
#                     p4 * p6 * p8 == 0): # 条件5
#                     changing_1.append((x,y))
#         for x, y in changing_1:
#             image_thinned[x][y] = 0
#         # ステップ2
#         changing_2 = []
#         for x in range(1, rows - 1):
#             for y in range(1, columns -1):
#                 p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
#                 if (image_thinned[x][y] == 1 and
#                     2 <= sum(neighbour_points) <= 6 and # 条件2
#                     count_transition(neighbour_points) == 1 and # 条件3
#                     p2 * p4 * p8 == 0 and # 条件4
#                     p2 * p6 * p8 == 0): # 条件5
#                     changing_2.append((x,y))
#         for x, y in changing_2:
#             image_thinned[x][y] = 0
#
#     return image_thinned
#
# # 2値画像の黒を1、白を0とするように変換するメソッドです
# def black_one(binary):
#     bool_image = binary.astype(bool)
#     inv_bool_image = ~bool_image
#     return inv_bool_image.astype(int)
#
# # 画像の外周を0で埋めるメソッドです
# def padding_zeros(image):
#     import numpy as np
#     m,n = np.shape(image)
#     padded_image = np.zeros((m+2,n+2))
#     padded_image[1:-1,1:-1] = image
#     return padded_image
#
# # 外周1行1列を除くメソッドです。
# def unpadding(image):
#     return image[1:-1, 1:-1]
#
# # 指定されたピクセルの周囲のピクセルを取得するメソッドです
# def neighbours(x, y, image):
#     return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1], # 2, 3, 4, 5
#              image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]] # 6, 7, 8, 9
#
# # 0→1の変化の回数を数えるメソッドです
# def count_transition(neighbours):
#     neighbours += neighbours[:1]
#     return sum( (n1, n2) == (0, 1) for n1, n2 in zip(neighbours, neighbours[1:]) )
#
# # 黒を1、白を0とする画像を、2値画像に戻すメソッドです
# def inv_black_one(inv_bool_image):
#     bool_image = ~inv_bool_image.astype(bool)
#     return bool_image.astype(int) * 255


def show_labels_img(labels, n_labels):
    """
    入力されたラベル情報に色を付けて表示させる
    :param labels:
    :param n_labels:
    :return:
    """
    h = labels.shape[0]
    w = labels.shape[1]
    cols = []
    label_img = np.zeros((h, w, 3))
    for i in range(1, n_labels):
        cols.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for i in range(1, n_labels):
        label_img[labels == i, ] = cols[i - 1]
    label_img = label_img.astype(np.uint8)
    cv.imshow('label_img', label_img)
    cv.waitKey(0)


def main():
    """
    指定した条件のラベルを除去する
    :return:
    """
    input_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/runs_v3/20201009_ver2/output/'
    output_dir = '/mnt/d_drive/home/ashida/work/Tellus_v3/output/'
    file_list = glob.glob(input_dir + '*.png')

    for file_name in file_list:
        base_name = os.path.basename(file_name)
        img = cv.imread(file_name, flags=-1)
        h = img.shape[0]
        w = img.shape[1]
        # img = cv.resize(img, (512, 512))
        resize_h = img.shape[0]
        resize_w = img.shape[1]

        orig_img = copy.deepcopy(img)

        # 膨張収縮処理
        kernel = np.ones((13, 13), np.uint8)
        # img = cv.dilate(img, kernel)  # 膨張処理

        # img = cv.erode(img, kernel, 2)  # 収縮処理
        # resize_show('orig_img', orig_img, 1)
        # resize_show('img', img, 1)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        ret, threshold_img = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
        # cv.imshow('img', img)
        # cv.waitKey(0)

        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold_img)
        # show_labels_img(labels, n_labels)

        idx = []
        for i in range(1, n_labels):
            label_w = stats[i, 2]
            label_h = stats[i, 3]
            if label_w > int(resize_w/100) or label_h > int(resize_h/100):  # 除去するラベルの条件式
                idx.append(i)

        removal_img = np.zeros((resize_h, resize_w))
        for i in idx:
            removal_img[labels == i] = 255

        kernel_2 = np.ones((5, 5), np.uint8)
        # removal_img = cv.erode(removal_img, kernel_2)  # 収縮処理

        # cv.imshow('orig_img', orig_img)
        # cv.imshow('removal_img', removal_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        removal_img = cv.resize(removal_img, (w, h), cv.INTER_LINEAR)
        # removal_img = cv.resize(removal_img, (int(w/2), int(h/2)), cv.INTER_LINEAR)
        # removal_img = cv.erode(removal_img, kernel_2)  # 収縮処理
        ret, removal_img = cv.threshold(removal_img, 1, 255, cv.THRESH_BINARY)
        removal_img = removal_img.astype(np.uint8)
        removal_img = cv.bitwise_not(removal_img)

        # # 2値化画像の黒を1、白を0に変換します。外周を0で埋めておきます。

        # th2 = padding_zeros(removal_img)
        # new_image = black_one(th2)
        # # Zhang Suenアルゴリズムによる細線化を行います
        # result_image = inv_black_one(Zhang_Suen_thinning(new_image)).astype(np.uint8)

        bool_img = np.where(removal_img < 1, True, False)
        bool_img = ZhangSuen(bool_img)
        result_image = np.where(bool_img == False, 0, 255).astype(np.uint8)
        # result_image = cv.bitwise_not(removal_img)
        # result_image = cv.resize(result_image, (w, h), cv.INTER_LINEAR)

        # resize_show('result_image', result_image, 1)
        # result_image = cv.resize(result_image, (w, h), cv.INTER_LINEAR)
        cv.imwrite(output_dir + base_name, result_image)
        print(base_name)



if __name__ == '__main__':
    main()
