import random

import numpy as np
import torch
from PIL import Image

from accelera.src.automl.core.image_dataset import ImageDataset


class ClassificationImageDataset(ImageDataset):
    def __init__(
        self,
        image_paths,
        labels=None,
        image_size=(224, 224),
        augment=False,
        augmentation_probability=0.5,
        horizontal_flip=False,
        vertical_flip=False,
        rotation=False,
        rotation_angle=30,
        brightness=False,
        brightness_factors=(0.8, 1.2),
        contrast=False,
        contrast_factors=(0.8, 1.2),
    ):
        super().__init__(
            image_paths,
            labels,
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

    def load_image(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        img = img.resize(self.image_size)
        if self.augment:
            img = self.augmentation(img)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_transpose = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.tensor(img_transpose, dtype=torch.float32)
        return img_tensor

    def random_horizontal_flip(self, img):
        if (
            self.horizontal_flip
            and random.random() < self.augmentation_probability
        ):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def random_vertical_flip(self, img):
        if (
            self.vertical_flip
            and random.random() < self.augmentation_probability
        ):
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def random_rotation(self, img):
        if self.rotation and random.random() < self.augmentation_probability:
            random_angle = random.uniform(
                -1 * self.rotation_angle, self.rotation_angle
            )
            return img.rotate(random_angle)
        return img

    def augmentation(self, img):
        img = self.random_horizontal_flip(img)
        img = self.random_vertical_flip(img)
        img = self.random_rotation(img)
        img = self.random_brightness(img)
        img = self.random_contrast(img)
        return img

    def __getitem__(self, index):
        img_tensor = self.load_image(index)
        label_tensor = None
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return img_tensor, label_tensor
