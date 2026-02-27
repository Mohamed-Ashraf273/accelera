import os

from torch.utils.data import DataLoader

from accelera.src.automl.core.image_training_preprocessing import (
    ImageTrainingPreprocessing,
)
from accelera.src.automl.core.segmentation_image_dataset import (
    SegmentationImageDataset,
)
from accelera.src.automl.utils.preprocessing import check_path_exists
from accelera.src.automl.utils.preprocessing import is_valid_image
from accelera.src.automl.utils.preprocessing import save_pickle
from accelera.src.automl.wrappers.display_sample_images_segmentation import (
    DisplaySampleImagesSegmentation,
)
from accelera.src.automl.wrappers.image_preprocessing_report import (
    ImagePreprocessingReport,
)
from accelera.src.automl.wrappers.segmentation_data_summary import (
    Segmentation_data_summary,
)
from accelera.src.automl.wrappers.segmentation_images_after_loader import (
    SegmentationImagesAfterLoader,
)


class SegmentationImageTrainingPreprocessing(ImageTrainingPreprocessing):
    def __init__(
        self,
        training_folder_images,
        training_folder_masks,
        folder_path,
        binary_mask_threshold=128,
        validation_folder_images=None,
        validation_folder_masks=None,
        split_training=False,
        val_size=0.2,
        random_state=42,
        images_size=(224, 224),
        augment=False,
        batch_size=16,
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
            training_folder_images,
            folder_path,
            validation_folder_images,
            split_training,
            val_size,
            random_state,
            images_size,
            augment,
            batch_size,
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

        self.training_folder_masks = training_folder_masks
        self.validation_folder_masks = validation_folder_masks
        self.binary_mask_threshold = binary_mask_threshold
        (
            self.validation_folder_images_paths,
            self.validation_folder_masks_paths,
        ) = (
            None,
            None,
        )
        self.training_invalid_images_paths = []
        self.training_invalid_masks_paths = []
        self.validation_invalid_images_paths = []
        self.validation_invalid_masks_paths = []
        if self.training_folder_masks is None:
            raise ValueError("Training folder masks must be not null")
        check_path_exists(self.training_folder_masks, "")
        if self.training_folder == self.training_folder_masks:
            raise ValueError(
                "training folder images and training folder masks must be "
                "different"
            )

        if self.binary_mask_threshold is None:
            raise ValueError(
                "you must add binary_mask_threshold value if pixel value >= "
                "binary_mask_threshold it will be 1 and 0 else"
            )
        if not (
            isinstance(self.binary_mask_threshold, int)
            and (0 <= self.binary_mask_threshold <= 255)
        ):
            raise ValueError(
                "binary_mask_threshold must be integer between 0 and 255"
            )

        if self.validation_folder is not None:
            if self.validation_folder_masks is None:
                raise ValueError("Validation folder masks must be not null")
            check_path_exists(self.validation_folder_masks, "")
            if self.validation_folder == self.validation_folder_masks:
                raise ValueError(
                    "validation folder images and validation folder masks "
                    "must be different"
                )
        data_info = {
            "image_size": self.image_size,
            "binary_mask_threshold": self.binary_mask_threshold,
        }
        save_pickle(self.folder_path, data_info, "data_info.pkl")

    def data_preparing(
        self,
        images_folder_path,
        masks_folder_path,
        invalid_images_paths,
        invalid_masks_paths,
    ):
        images_paths = []
        masks_paths = []
        images_dict = {
            os.path.splitext(image_path)[0]: image_path
            for image_path in os.listdir(images_folder_path)
        }
        masks_dict = {
            os.path.splitext(mask_path)[0]: mask_path
            for mask_path in os.listdir(masks_folder_path)
        }
        matches = set(images_dict.keys()) & set(masks_dict.keys())
        if len(matches) == 0:
            raise ValueError("no matches between masks and images names")

        for key in sorted(matches):
            image_path = os.path.join(images_folder_path, images_dict[key])
            mask_path = os.path.join(masks_folder_path, masks_dict[key])
            if is_valid_image(image_path) and is_valid_image(mask_path):
                images_paths.append(image_path)
                masks_paths.append(mask_path)
            else:
                invalid_images_paths.append(image_path)
                invalid_masks_paths.append(mask_path)
        if len(images_paths) == 0:
            raise ValueError("There is no valid path")
        return images_paths, masks_paths

    def data_overview(self):
        train_df = self.get_sample_random(
            "training_folder",
            self.training_folder_images_paths,
            self.training_folder_masks_paths,
        )
        self.report_data["data_overview"] = {}
        self.report_data["data_overview"]["training_folder"] = {
            "images_len": len(self.training_folder_images_paths),
            "random_sample": train_df.head(),
            "invalid_len": len(self.training_invalid_images_paths),
            "invalid_images": self.training_invalid_images_paths,
            "invalid_masks": self.training_invalid_masks_paths,
        }

        if self.validation_folder is not None:
            val_df = self.get_sample_random(
                "validation_folder",
                self.validation_folder_images_paths,
                self.validation_folder_masks_paths,
            )
            self.report_data["data_overview"]["validation_folder"] = {
                "images_len": len(self.validation_folder_images_paths),
                "random_sample": val_df.head(),
                "invalid_len": len(self.validation_invalid_images_paths),
                "invalid_images": self.validation_invalid_images_paths,
                "invalid_masks": self.validation_invalid_masks_paths,
            }

    def make_graphs_data_summary(self):
        Segmentation_data_summary(
            self.training_paths,
            self.training_invalid_images_paths,
            folder_path=self.folder_path,
            title="Training Folder Summary",
            file_name="training_folder_summary",
        ).build_graph()
        self.report_data["graphs"]["images_name"].append(
            "training_folder_summary"
        )
        if self.validation_folder is not None:
            Segmentation_data_summary(
                self.validation_paths,
                self.validation_invalid_images_paths,
                folder_path=self.folder_path,
                title="Validation Folder Summary",
                file_name="validation_folder_summary",
            ).build_graph()
        self.report_data["graphs"]["images_name"].append(
            "validation_folder_summary"
        )

    def make_garphs_sample(self):
        DisplaySampleImagesSegmentation(
            self.training_folder_images_paths,
            self.training_folder_masks_paths,
            self.folder_path,
            title="Random Samples of Training Folder",
            file_name="training_folder_random_samples",
        ).build_graph()
        self.report_data["graphs"]["images_name"].append(
            "training_folder_random_samples"
        )
        if self.validation_folder is not None:
            DisplaySampleImagesSegmentation(
                self.validation_folder_images_paths,
                self.validation_folder_masks_paths,
                self.folder_path,
                title="Random Samples of Validation Folder",
                file_name="validation_folder_random_samples",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "validation_folder_random_samples"
            )
        if self.split_training:
            DisplaySampleImagesSegmentation(
                self.training_paths,
                self.training_labels,
                self.folder_path,
                title="Samples of Training Data After Splitting",
                file_name="training_after_splitting_random_samples",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "training_after_splitting_random_samples"
            )
            DisplaySampleImagesSegmentation(
                self.validation_paths,
                self.validation_labels,
                self.folder_path,
                title="Samples of Validation Data After Splitting",
                file_name="validation_after_splitting_random_samples",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "validation_after_splitting_random_samples"
            )

    def make_graphs_loader(self):
        training_images, training_labels = next(iter(self.training_loader))
        n_samples = min(5, len(training_images))

        training_images, training_labels = (
            training_images[:n_samples],
            training_labels[:n_samples],
        )
        SegmentationImagesAfterLoader(
            training_images,
            training_labels,
            self.folder_path,
            title="Samples of Training Data After Data Loader",
            file_name="training_after_data_loader_samples",
        ).build_graph()
        self.report_data["graphs"]["images_name"].append(
            "training_after_data_loader_samples"
        )
        if self.validation_loader is not None:
            validation_images, validation_labels = next(
                iter(self.validation_loader)
            )
            n_samples = min(5, len(validation_images))
            validation_images, validation_labels = (
                validation_images[:n_samples],
                validation_labels[:n_samples],
            )
            SegmentationImagesAfterLoader(
                validation_images,
                validation_labels,
                self.folder_path,
                title="Samples of Validation Data After Data Loader",
                file_name="validation_after_data_loader_samples",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "validation_after_data_loader_samples"
            )

    def make_graphs(self):
        self.report_data["graphs"] = {
            "folder_path": self.folder_path,
            "images_name": [],
        }
        self.make_graphs_data_summary()
        self.make_garphs_sample()
        self.make_graphs_loader()

    def get_loaders(self):
        training_dataset = SegmentationImageDataset(
            self.training_paths,
            self.training_labels,
            self.image_size,
            self.binary_mask_threshold,
            self.augment,
            self.augmentation_probability,
            self.horizontal_flip,
            self.vertical_flip,
            self.rotation,
            self.rotation_angle,
            self.brightness,
            self.brightness_factors,
            self.contrast,
            self.contrast_factors,
        )
        self.training_loader = DataLoader(
            training_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.validation_loader = None
        if self.validation_paths is not None:
            validation_dataset = SegmentationImageDataset(
                self.validation_paths,
                self.validation_labels,
                self.image_size,
                self.binary_mask_threshold,
                False,
                self.augmentation_probability,
                self.horizontal_flip,
                self.vertical_flip,
                self.rotation,
                self.rotation_angle,
                self.brightness,
                self.brightness_factors,
                self.contrast,
                self.contrast_factors,
            )
            self.validation_loader = DataLoader(
                validation_dataset, batch_size=self.batch_size, shuffle=False
            )

    def common_preprocessing(self):
        (
            self.training_folder_images_paths,
            self.training_folder_masks_paths,
        ) = self.data_preparing(
            self.training_folder,
            self.training_folder_masks,
            self.training_invalid_images_paths,
            self.training_invalid_masks_paths,
        )
        if self.validation_folder is not None:
            (
                self.validation_folder_images_paths,
                self.validation_folder_masks_paths,
            ) = self.data_preparing(
                self.validation_folder,
                self.validation_folder_masks,
                self.validation_invalid_images_paths,
                self.validation_invalid_masks_paths,
            )

        self.data_overview()
        self.splitting(
            self.training_folder_images_paths,
            self.validation_folder_images_paths,
            self.training_folder_masks_paths,
            self.validation_folder_masks_paths,
        )
        self.get_loaders()
        self.make_graphs()

        report = ImagePreprocessingReport(self.folder_path, self.report_data)
        report.execute()
        return self.training_loader, self.validation_loader
