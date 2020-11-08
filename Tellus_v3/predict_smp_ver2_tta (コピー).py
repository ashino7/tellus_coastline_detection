import os
import glob
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import albumentations as albu
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import ttach as tta
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


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

    CLASSES = ['unlabelled', 'building', 'coastline'] #変更

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
        image = cv.resize(image, (512, 512))

        mask = cv.imread(self.masks_fps[i], 0)
        mask = cv.resize(mask, (512, 512))
        kernel = np.ones((2, 2), np.uint8)  # 膨張処理のためのカーネルサイズを選ぶ
        mask = cv.dilate(mask, kernel)  # 膨張処理
        ret, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
        mask = np.where(mask > 1, 1, 0).astype('float')
        mask = mask.reshape((512, 512, 1))


        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

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
    DATA_DIR = './input/test_images_png/'
    output_dir = './output/'
    x_valid_dir = DATA_DIR
    y_valid_dir = DATA_DIR

    ENCODER = 'inceptionv4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['coastline']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # load best saved checkpoint
    best_model = torch.load('./runs/Nov02_19-46-14_ashida_ENCODER_inceptionv4_LR_0.0001/best_model.pth')
    # tta_model = tta.SegmentationTTAWrapper(best_model, tta.aliases.d4_transform(), merge_mode='mean')
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180])
        ]
    )
    tta_model = tta.SegmentationTTAWrapper(best_model, transforms)

    # create test dataset
    test_dataset = Dataset(
        x_valid_dir,  # x_test_dir
        y_valid_dir,  # y_test_dir
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # test_dataloader = DataLoader(test_dataset)
    #
    # # evaluate model on test set
    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=best_model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    # )
    #
    # logs = test_epoch.run(test_dataloader)

    # test dataset without transformations for image visualization
    test_dataset_vis2 = Dataset(
        x_valid_dir, y_valid_dir,  # x_test_dir, y_test_dir,
        augmentation=get_validation_augmentation(),
        classes=CLASSES,
    )

    for i in range(len(test_dataset)):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis2[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        file_name = test_dataset.images_fps[i]
        base_name = os.path.basename(file_name)
        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # pr_mask = best_model.predict(x_tensor)
        pr_mask = tta_model.forward(x_tensor)
        # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = pr_mask.squeeze().to('cpu').detach().numpy().copy()
        pr_mask = (pr_mask*255).astype(np.uint8)
        ret, pr_mask = cv.threshold(pr_mask, 1, 255, cv.THRESH_BINARY)

        # visualize(
        #     image=image_vis,
        #     ground_truth_mask=gt_mask,
        #     predicted_mask=pr_mask
        # )

        org_image = cv.imread(file_name)
        h = org_image.shape[0]
        w = org_image.shape[1]
        pr_mask = cv.resize(pr_mask, (w, h))
        cv.imwrite(output_dir + base_name, pr_mask)


if __name__ == '__main__':
    main()