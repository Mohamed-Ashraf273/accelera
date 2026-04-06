import os

import pandas as pd
from sklearn.model_selection import train_test_split

from accelera.src.automl.core.preprocessing_base import PreprocessingBase
from accelera.src.utils.preprocessing import check_path_exists


class ImageTrainingPreprocessing(PreprocessingBase):
    def __init__(
        self,
        training_folder,
        folder_path,
        validation_folder,
        split_training,
        val_size,
        random_state,
        images_size,
        augment=True,
        batch_size=16,
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
        super().__init__(folder_path)
        self.training_folder = training_folder
        self.validation_folder = validation_folder
        self.split_training = split_training
        self.val_size = val_size
        self.random_state = random_state
        self.image_size = images_size
        self.augment = augment
        self.batch_size = batch_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.augmentation_probability = augmentation_probability
        self.rotation = rotation
        self.rotation_angle = rotation_angle
        self.brightness = brightness
        self.brightness_factors = brightness_factors
        self.contrast = contrast
        self.contrast_factors = contrast_factors
        self.report_data = {}
        if self.training_folder is None:
            raise ValueError("Training folder must be not null")
        check_path_exists(self.training_folder, "")
        if self.validation_folder is not None:
            check_path_exists(self.validation_folder, "")
            if self.split_training:
                raise ValueError(
                    "The validation folder not none so there is no need "
                    "for splitting data again to training and validation"
                )
            if self.training_folder == self.validation_folder:
                raise ValueError(
                    "The validation folder and training folder must be "
                    "different"
                )

        if (not (isinstance(self.val_size, (float)))) or (
            not (0 < self.val_size <= 0.5)
        ):
            raise ValueError("Test size is invalid it must be less than 0.5")
        if (self.random_state is not None) and not (
            isinstance(self.random_state, int)
        ):
            raise ValueError(
                "Random state is invalid it must be integer or None"
            )
        if not isinstance(self.image_size, tuple):
            raise ValueError("Image size must be tuple")
        if not isinstance(self.image_size[0], int) or not isinstance(
            self.image_size[1], int
        ):
            raise ValueError("Image size is not integer")
        if not (32 <= self.image_size[0] <= 1024) or not (
            32 <= self.image_size[1] <= 1024
        ):
            raise ValueError(
                "Image size is must be greater than or equal 32 and "
                "less than or equal 1024"
            )
        if not isinstance(self.image_size, tuple):
            raise ValueError("Image size must be tuple")
        if not isinstance(self.image_size[0], int) or not isinstance(
            self.image_size[1], int
        ):
            raise ValueError("Image size is not integer")
        if not (32 <= self.image_size[0] <= 1024) or not (
            32 <= self.image_size[1] <= 1024
        ):
            raise ValueError(
                "Image size is must be greater than or equal 32 and "
                "less than or equal 1024"
            )
        if not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be a integer")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive value")

        if not isinstance(self.augment, bool):
            raise ValueError("augment must be a boolean")

        if not isinstance(self.horizontal_flip, bool):
            raise ValueError("horizontal_flip must be a boolean")

        if not isinstance(self.vertical_flip, bool):
            raise ValueError("vertical_flip must be a boolean")

        if not isinstance(self.rotation, bool):
            raise ValueError("rotation must be a boolean")

        if not isinstance(self.brightness, bool):
            raise ValueError("brightness must be a boolean")

        if not isinstance(self.contrast, bool):
            raise ValueError("contrast must be a boolean")
        if not isinstance(self.augmentation_probability, (int, float)) or not (
            0 <= self.augmentation_probability <= 1
        ):
            raise ValueError("augmentation_probability must be between [0,1]")
        if (
            not isinstance(self.rotation_angle, (int, float))
            or self.rotation_angle < 0
        ):
            raise ValueError("rotation_angle must be positive integer or float")

        if (
            not isinstance(self.brightness_factors, tuple)
            or len(self.brightness_factors) != 2
            or not all(
                isinstance(x, (int, float)) for x in self.brightness_factors
            )
        ):
            raise ValueError(
                "brightness_factors must be tuple of two items float "
                "or integers"
            )
        if (
            not isinstance(self.contrast_factors, tuple)
            or len(self.contrast_factors) != 2
            or not all(
                isinstance(x, (int, float)) for x in self.contrast_factors
            )
        ):
            raise ValueError(
                "contrast_factors must be tuple of two items float or integers"
            )
        os.makedirs(self.folder_path, exist_ok=True)

    def get_sample_random(self, data_type, images_path, labels, num_samples=5):
        df = (
            pd.DataFrame(
                {
                    f"{data_type}_paths": images_path,
                    f"{data_type}_labels": labels,
                }
            )
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        ).head(num_samples)
        return df

    def splitting(
        self,
        training_folder_images_paths,
        validation_folder_images_paths,
        training_folder_images_labels,
        validation_folder_images_labels,
    ):
        if self.split_training:
            (
                self.training_paths,
                self.validation_paths,
                self.training_labels,
                self.validation_labels,
            ) = train_test_split(
                training_folder_images_paths,
                training_folder_images_labels,
                test_size=self.val_size,
                random_state=self.random_state,
            )
            self.report_data["split_data"] = {
                "validation_size": self.val_size,
                "training_data_size": len(self.training_paths),
                "random_training_sample": self.get_sample_random(
                    "training", self.training_paths, self.training_labels
                ),
                "validation_data_size": len(self.validation_paths),
                "random_validation_sample": self.get_sample_random(
                    "validation", self.validation_paths, self.validation_labels
                ),
            }
        else:
            (
                self.training_paths,
                self.validation_paths,
                self.training_labels,
                self.validation_labels,
            ) = (
                training_folder_images_paths,
                validation_folder_images_paths,
                training_folder_images_labels,
                validation_folder_images_labels,
            )
