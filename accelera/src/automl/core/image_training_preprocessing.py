from accelera.src.automl.core.preprocessing_base import PreprocessingBase
from accelera.src.automl.utils.preprocessing import check_path_exists
import os
import pandas as pd
from PIL import Image


class ImageTrainingPreprocessing(PreprocessingBase):
    def __init__(
        self,
        training_folder_images,
        folder_path,
        validation_folder_images,
        split_training,
        val_size,
        random_state,
        images_size,
    ):
        super().__init__(folder_path)
        self.training_folder_images = training_folder_images
        self.validation_folder_images = validation_folder_images
        self.split_training = split_training
        self.val_size = val_size
        self.random_state = random_state
        self.image_size = images_size
        self.report_data = {}
        self.valid_extension = (".jpg", ".png", ".jpeg")
        if self.training_folder_images is None:
            raise ValueError("Training folder must be not null")
        check_path_exists(self.training_folder_images, "")
        if self.validation_folder_images is not None:
            check_path_exists(self.validation_folder_images, "")
            if self.split_training == True:
                raise ValueError(
                    "The validation folder not none so there is no need for splitting data again to training and validation"
                )
            if self.training_folder_images == self.validation_folder_images:
                raise ValueError(
                    "The validation folder and training folder must be different"
                )

        if self.split_training:
            if (not (isinstance(self.val_size, (int, float)))) or (
                not (0 < self.val_size < 0.5)
            ):
                raise ValueError("Test size is invalid it must be less than 0.5")
            if (self.random_state is not None) and not (
                isinstance(self.random_state, int)
            ):
                raise ValueError("Random state is invalid it must be integer or None")
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
                "Image size is must be greater than or equal 32 and less than or equal 1024"
            )
        os.makedirs(self.folder_path, exist_ok=True)

    def get_sample_random(self, data_type, images_path, labels, num_samples=5):
        df = (
            pd.DataFrame(
                {
                    f"{data_type}_path": images_path,
                    f"{data_type}_labels": labels,
                }
            )
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        ).head(num_samples)
        return df

    def is_valid_image(self, image_path):
        if image_path.endswith(self.valid_extension):
            try:
                with Image.open(image_path) as img:
                    img.load()
                return True
            except Exception:
                return False
        return False
