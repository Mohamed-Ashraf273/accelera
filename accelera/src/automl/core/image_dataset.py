import random

from PIL import ImageEnhance
from torch.utils.data import Dataset


class ImageDataset(Dataset):
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



    
