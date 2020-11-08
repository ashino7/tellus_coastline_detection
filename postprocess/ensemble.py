import os
import glob
import cv2 as cv
import numpy as np

def main():
    dir_list = ['/mnt/d_drive/home/ashida/work/Tellus_v3/runs_v3/20201009_ver2/output_heatmap/',
                '/mnt/d_drive/home/ashida/work/Tellus_v4/runs/Oct25_07-36-05_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
                '/mnt/d_drive/home/ashida/work/Tellus_v4/runs/Nov01_14-42-50_ashida_ENCODER_densenet121_LR_0.0001/output_heatmap/',
                '/mnt/d_drive/home/ashida/work/Tellus_v2/runs/Sep18_16-27-08_ashidaLR_0.0001_BS_1_SCALE_0.5/ver0.3/',
                '/mnt/d_drive/home/ashida/work/Tellus_v3/runs/Nov02_18-43-34_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
                '/mnt/d_drive/home/ashida/work/Tellus_v3/runs/Nov02_19-10-45_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
                '/mnt/d_drive/home/ashida/work/Tellus_v3/runs/Nov02_19-46-14_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
                ]
    # dir_list = ['/mnt/d_drive/home/ashida/work/Tellus_v3/runs/Nov02_18-43-34_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
    #             '/mnt/d_drive/home/ashida/work/Tellus_v3/runs/Nov02_19-10-45_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
    #             '/mnt/d_drive/home/ashida/work/Tellus_v3/runs/Nov02_19-46-14_ashida_ENCODER_inceptionv4_LR_0.0001/output_heatmap/',
    #             ]
    threshold = 20
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_list = glob.glob(dir_list[0] + '*.png')
    for n, file_name in enumerate(sorted(file_list)):
        base_name = os.path.basename(file_name)
        tmp_img = cv.imread(file_name, flags=-1)
        h = tmp_img.shape[0]
        w = tmp_img.shape[1]
        images = np.zeros([len(dir_list), h, w])
        for i, dir_name in enumerate(dir_list):
            image = cv.imread(dir_name + base_name, flags=-1)
            # image = image.reshape(1, h, w)
            images[i, :, :] = image
            c=0

        # 平均を取る
        images_mean = np.mean(images, axis=0)
        # images_mean = np.clip(images_mean, None, 1)
        # images_mean = (images_mean*255).astype(np.uint8)
        ret, predict_image = cv.threshold(images_mean, threshold, 255, cv.THRESH_BINARY)
        cv.imwrite(output_dir + base_name, predict_image)
        print(str(n) + '/' + str(len(file_list)))
        c=0



if __name__ == '__main__':
    main()