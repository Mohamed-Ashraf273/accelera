import random

import numpy as np
import torch
from PIL import Image
from accelera.src.automl.core.image_dataset import ImageDataset


class SegmentationImageDataset(ImageDataset):
    def __init__(
        self,
        image_paths,
        masks=None,
        image_size=(224, 224),
        mask_type="binary",
        mask_classes=0,
        augment=True,
        augmentation_probability=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        rotation=True,
        rotation_angle=30,
        brightness=True,
        brightness_factors=(0.8, 1.2),
        contrast=True,
        contrast_factors=(0.8, 1.2),
    ):
        super().__init__(
            image_paths,
            masks,
            image_size,
            augment,
            augmentation_probability,
            horizontal_flip,
            vertical_flip,
            rotation,
            rotation_angle,
            brightness,
            brightness_factors,
            contrast,
            contrast_factors,
        )
        self.mask_type=mask_type
        self.mask_classes=mask_classes
    def load_mask(self,index):
        mask_path = self.labels[index]
        mask = Image.open(mask_path)
        mask=mask.convert("L")
        mask_array = np.array(mask, dtype=np.int64)
        if self.mask_type=="binary":
            mask_array = (mask_array>0).astype(np.int64)
        elif self.mask_type=="multi_class":
            mask_array[mask_array>=self.mask_classes]=0
        return Image.fromarray(mask_array.astype(np.uint8))
            

    def load_image_masks(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        img = img.resize(self.image_size)
        mask = None
        if self.labels is not None:
            mask=self.load_mask(index)
            mask = mask.resize(self.image_size, resample=Image.NEAREST)
        if self.augment:
            img, mask = self.augmentation(img, mask)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_transpose = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.tensor(img_transpose, dtype=torch.float32)
        mask_tensor = None
        if mask is not None:
            if self.mask_type=="grayscale_intensity":
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                mask_tensor = torch.tensor(mask_array, dtype=torch.float32)
            else:
                mask_array = np.array(mask, dtype=np.int64)
                mask_tensor = torch.tensor(mask_array, dtype=torch.long)
        return img_tensor, mask_tensor

    def random_horizontal_flip(self, img, mask):
        transposed_img = img
        transposed_mask = mask
        if self.horizontal_flip and random.random() < self.augmentation_probability:
            transposed_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                transposed_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return transposed_img, transposed_mask

    def random_vertical_flip(self, img, mask):
        transposed_img = img
        transposed_mask = mask
        if self.vertical_flip and random.random() < self.augmentation_probability:
            transposed_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if mask is not None:
                transposed_mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return transposed_img, transposed_mask

    def random_rotation(self, img, mask):
        rotated_img = img
        rotated_mask = mask
        if self.rotation and random.random() < self.augmentation_probability:
            random_angle = random.uniform(-1 * self.rotation_angle, self.rotation_angle)
            rotated_img = img.rotate(random_angle)
            if mask is not None:
                rotated_mask = mask.rotate(random_angle, resample=Image.NEAREST)
        return rotated_img, rotated_mask

    def augmentation(self, img, mask):
        img, mask = self.random_horizontal_flip(img, mask)
        img, mask = self.random_vertical_flip(img, mask)
        img, mask = self.random_rotation(img, mask)
        img = self.random_brightness(img)
        img = self.random_contrast(img)
        return img, mask

    def __getitem__(self, index):
        img_tensor, mask_tensor = self.load_image_masks(index)
        return img_tensor, mask_tensor
