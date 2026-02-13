import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import numpy as np
import random


class ClassificationImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels=None,
        image_size=(224, 224),
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
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.augment = augment
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.augmentation_probability = augmentation_probability
        self.rotation = rotation
        self.rotation_angle = rotation_angle
        self.brightness = brightness
        self.brightness_factors = brightness_factors
        self.contrast = contrast
        self.contrast_factors = contrast_factors

    def __len__(self):
        return len(self.image_paths)

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
        if self.horizontal_flip and random.random() < self.augmentation_probability:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def random_vertical_flip(self, img):
        if self.vertical_flip and random.random() < self.augmentation_probability:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def random_rotation(self, img):
        if self.rotation and random.random() < self.augmentation_probability:
            random_angle = random.uniform(-1 * self.rotation_angle, self.rotation_angle)
            return img.rotate(random_angle)
        return img

    def random_brightness(self, img):
        if self.brightness and random.random() < self.augmentation_probability:
            random_factor = random.uniform(*self.brightness_factors)
            return ImageEnhance.Brightness(img).enhance(random_factor)
        return img

    def random_contrast(self, img):
        if self.contrast and random.random() < self.augmentation_probability:
            random_factor = random.uniform(*self.contrast_factors)
            return ImageEnhance.Contrast(img).enhance(random_factor)
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
