import os
import glob
import random
import copy
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import albumentations as albu
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['unlabelled', 'coastline'] #変更

    def __init__(
            self,
            images_dir,
            pre_detection_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.pre_detection_fps = [os.path.join(pre_detection_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # train読み込み
        image = cv.imread(self.images_fps[i])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        # image = cv.resize(image, (586, 586))

        # mask読み込み
        mask = cv.imread(self.masks_fps[i], 0)

        # 事前検出読み込み
        pre_detection_image = cv.imread(self.pre_detection_fps[i], flags=-1)

        if h > 2000 and w > 2000:
            image = cv.resize(image, (int(w // 2), int(h // 2)))
            mask = cv.resize(mask, (int(w // 2), int(h // 2)))
            pre_detection_image = cv.resize(pre_detection_image, (int(w // 2), int(h // 2)))

        # roi_size = 1156
        roi_size = 786
        roi_half_size = int(roi_size / 2)

        if image.shape[0] < roi_size or image.shape[1] < roi_size:
            image = cv.resize(image, (roi_size, roi_size))
            mask = cv.resize(mask, (roi_size, roi_size))
            pre_detection_image = cv.resize(pre_detection_image, (roi_size, roi_size))

        pre_detection_image[:roi_half_size, :] = 0
        pre_detection_image[-roi_half_size:, :] = 0
        pre_detection_image[:, :roi_half_size] = 0
        pre_detection_image[:, -roi_half_size:] = 0
        pre_detection_points = np.where(pre_detection_image > 1)

        points_len = len(pre_detection_points[0])
        if points_len == 0:
            center_y = pre_detection_image.shape[0] // 2
            center_x = pre_detection_image.shape[1] // 2
        else:
            idx = random.randint(0, points_len - 1)
            center_y = int(pre_detection_points[0][idx])
            center_x = int(pre_detection_points[1][idx])

        image = image[center_y - roi_half_size:center_y + roi_half_size, center_x - roi_half_size:center_x + roi_half_size, :]
        mask = mask[center_y - roi_half_size:center_y + roi_half_size, center_x - roi_half_size:center_x + roi_half_size]

        # mask = cv.resize(mask, (586, 586))
        kernel = np.ones((5, 5), np.uint8)  # 膨張処理のためのカーネルサイズを選ぶ
        mask = cv.dilate(mask, kernel)  # 膨張処理
        ret, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

        mask = np.where(mask > 1, 1, 0).astype('float')
        mask = mask.reshape((roi_size, roi_size, 1))

        # cv.imshow('image', image)
        # cv.imshow('mask', mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        # albu.Resize(height=1248, width=1248, p=1),
        # albu.HorizontalFlip(p=0.5),
        # albu.RandomScale(0.1, p=0.5),
        albu.Flip(p=0.5),
        albu.Transpose(p=0.5),
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.RandomCrop(height=512, width=512, always_apply=True),
        # albu.RandomGridShuffle(p=0.5),
        albu.RandomSnow(brightness_coeff=1.0, p=0.5),
        # albu.RandomSunFlare(p=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.Resize(height=512, width=512, p=1),
        albu.CenterCrop(height=512, width=512),
        # albu.PadIfNeeded(384, 480),
        # albu.RandomCrop(height=320, width=320, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def main():
    data_dir = './input/another_cv_1/'
    epoch = 100
    batch_size = 1
    lr = 0.0001

    # 分割した画像データのディレクトリを指定する．
    x_train_dir = os.path.join(data_dir, 'train_images_png')
    y_train_dir = os.path.join(data_dir, 'train_images_inpainted_labels')
    x_train_pre_detection_dir = os.path.join(data_dir, 'train_imaes_predetection')

    x_valid_dir = os.path.join(data_dir, 'val_images_png')
    y_valid_dir = os.path.join(data_dir, 'val_images_inpainted_labels')
    x_valid_pre_detection_dir = os.path.join(data_dir, 'val_imaes_predetection')

    # 分割した画像データのファイルリストを作成する．
    x_train_files = glob.glob(x_train_dir + '/*')
    y_train_files = glob.glob(y_train_dir + '/*')

    # ENCODER = 'resnet18'
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['coastline']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    # model = smp.Unet(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=len(CLASSES),
    #     activation=ACTIVATION,
    # )
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    # tensorboardの設定
    writer = SummaryWriter(comment=f'_ENCODER_{ENCODER}_LR_{lr}')
    log_dir = writer.log_dir

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        x_train_pre_detection_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        x_valid_pre_detection_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # lossの設定
    loss = smp.utils.losses.DiceLoss()
    # loss = smp.utils.losses.BCELoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # optimizerの設定
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=lr),
    ])

    # 学習ループの設定
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    # train accurascy, train loss, val_accuracy, val_loss をグラフ化できるように設定．
    x_epoch_data = []
    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []

    for i in range(epoch):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        x_epoch_data.append(i)
        train_dice_loss.append(train_logs['dice_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_dice_loss.append(valid_logs['dice_loss'])
        valid_iou_score.append(valid_logs['iou_score'])

        writer.add_scalar('Loss/train', train_logs['dice_loss'], i)
        writer.add_scalar('iou/train', train_logs['iou_score'], i)
        writer.add_scalar('Loss/valid', valid_logs['dice_loss'], i)
        writer.add_scalar('iou/valid', valid_logs['iou_score'], i)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, log_dir + '/best_model.pth')
            print('Model saved!')

        if i == 50:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    main()
