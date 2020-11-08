import os
import glob
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import albumentations as albu
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam

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
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv.imread(self.images_fps[i])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (586, 586))
        # mean = cv.mean(image)[0]
        # offset_image = (np.ones((586, 586, 3)) * mean * 1.2).astype(np.uint8)
        # image = cv.subtract(image, offset_image)

        mask = cv.imread(self.masks_fps[i], 0)
        mask = cv.resize(mask, (586, 586))
        kernel = np.ones((5, 5), np.uint8)  # 膨張処理のためのカーネルサイズを選ぶ
        mask = cv.dilate(mask, kernel)  # 膨張処理
        ret, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

        mask = np.where(mask > 1, 1, 0).astype('float')
        mask = mask.reshape((586, 586, 1))

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
        albu.Resize(height=512, width=512, p=1),
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
    data_dir = './input/another_cv_3/'
    epoch = 100
    batch_size = 1
    lr = 0.0001

    # 分割した画像データのディレクトリを指定する．
    x_train_dir = os.path.join(data_dir, 'train_images_png')
    y_train_dir = os.path.join(data_dir, 'train_images_inpainted_labels')

    x_valid_dir = os.path.join(data_dir, 'val_images_png')
    y_valid_dir = os.path.join(data_dir, 'val_images_inpainted_labels')

    # 分割した画像データのファイルリストを作成する．
    x_train_files = glob.glob(x_train_dir + '/*')
    y_train_files = glob.glob(y_train_dir + '/*')

    # ENCODER = 'resnet18'
    ENCODER = 'inceptionv4'
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
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
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
    # optimizer = torch.optim.Adam([
    #     dict(params=model.parameters(), lr=lr),
    # ])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)

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

        if i < 30:

            if i != 0:
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
        elif i > 29 and i < 32:
            optimizer.param_groups[0]['lr'] = 1e-5
        else:
            optimizer.param_groups[0]['lr'] = 5e-6

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

        # if i == 50:
        #     optimizer.param_groups[0]['lr'] = 1e-5
        #     print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    main()
