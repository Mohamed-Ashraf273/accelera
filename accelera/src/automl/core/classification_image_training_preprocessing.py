import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from accelera.src.automl.core.classification_image_dataset import (
    ClassificationImageDataset,
)
from accelera.src.automl.core.image_training_preprocessing import (
    ImageTrainingPreprocessing,
)
from accelera.src.automl.utils.preprocessing import get_sub_folders_names
from accelera.src.automl.utils.preprocessing import is_valid_image
from accelera.src.automl.utils.preprocessing import save_pickle
from accelera.src.automl.wrappers.classification_images_after_loader import (
    ClassificationImagesAfterLoader,
)
from accelera.src.automl.wrappers.display_sample_images_classification import (
    DisplaySampleImagesClassification,
)
from accelera.src.automl.wrappers.image_label_classification import (
    ImageLabelClassification,
)
from accelera.src.automl.wrappers.image_preprocessing_report import (
    ImagePreprocessingReport,
)


class ClassificationImageTrainingPreprocessing(ImageTrainingPreprocessing):
    def __init__(
        self,
        training_folder_images,
        folder_path,
        validation_folder_images=None,
        split_training=False,
        val_size=0.2,
        random_state=42,
        images_size=(224, 224),
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
        self.training_class = get_sub_folders_names(self.training_folder_images)
        self.validation_class = None
        self.training_folder_invalid_images = []
        self.training_folder_invalid_images_labels = []
        self.validation_folder_invalid_images = []
        self.validation_folder_invalid_images_labels = []
        if self.validation_folder_images is not None:
            self.validation_class = get_sub_folders_names(
                self.validation_folder_images
            )
            for class_name in self.validation_class:
                if class_name not in self.training_class:
                    raise ValueError(
                        f"This category {class_name} not in the training "
                        f"categories which are {self.training_class}"
                    )
        (
            self.validation_folder_images_paths,
            self.validation_folder_images_labels,
        ) = (
            None,
            None,
        )
        data_info = {"image_size": self.image_size}
        save_pickle(self.folder_path, data_info, "data_info.pkl")

    def get_classes_mapping(self):
        self.class2label_mapping = {}
        self.label2class_mapping = {}
        for idx, class_name in enumerate(sorted(self.training_class)):
            self.class2label_mapping[class_name] = idx
            self.label2class_mapping[idx] = class_name
        save_pickle(
            self.folder_path,
            self.class2label_mapping,
            "class2label_mapping.pkl",
        )
        save_pickle(
            self.folder_path,
            self.label2class_mapping,
            "label2class_mapping.pkl",
        )

    def data_preparing(
        self, folder_path, invalid_list_paths, invalid_list_labels, classes_name
    ):
        paths = []
        labels = []
        for class_name in classes_name:
            sub_folder_path = os.path.join(folder_path, class_name)
            for path in os.listdir(sub_folder_path):
                path = os.path.join(sub_folder_path, path)
                mapping = self.class2label_mapping[class_name]
                if is_valid_image(path):
                    paths.append(path)
                    labels.append(mapping)
                else:
                    invalid_list_paths.append(path)
                    invalid_list_labels.append(mapping)
        return paths, labels

    def data_overview(self):
        train_df = self.get_sample_random(
            "training_folder",
            self.training_folder_images_paths,
            self.training_folder_images_labels,
        )
        self.report_data["data_overview"] = {}
        self.report_data["data_overview"]["training_folder"] = {
            "classes": self.training_class,
            "images_len": len(self.training_folder_images_paths),
            "random_sample": train_df.head(),
            "invalid_len": len(self.training_folder_invalid_images),
            "invalid_images": self.training_folder_invalid_images,
            "mapping": self.class2label_mapping,
        }

        if self.validation_folder_images is not None:
            val_df = self.get_sample_random(
                "validation_folder",
                self.validation_folder_images_paths,
                self.validation_folder_images_labels,
            )
            self.report_data["data_overview"]["validation_folder"] = {
                "classes": self.validation_class,
                "images_len": len(self.validation_folder_images_paths),
                "random_sample": val_df.head(),
                "invalid_len": len(self.validation_folder_invalid_images),
                "invalid_images": self.validation_folder_invalid_images,
            }

    def splitting(self):
        if self.split_training:
            (
                self.training_paths,
                self.validation_paths,
                self.training_labels,
                self.validation_labels,
            ) = train_test_split(
                self.training_folder_images_paths,
                self.training_folder_images_labels,
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
                self.training_folder_images_paths,
                self.validation_folder_images_paths,
                self.training_folder_images_labels,
                self.validation_folder_images_labels,
            )

    def make_graphs_label_summary(self):
        ImageLabelClassification(
            self.training_folder_images_labels,
            self.label2class_mapping,
            self.training_folder_invalid_images_labels,
            self.folder_path,
            "Training Folder Labels Summary ",
            "training_folder_labels_summary",
        ).build_graph()
        self.report_data["graphs"]["images_name"].append(
            "training_folder_labels_summary"
        )
        if self.validation_folder_images is not None:
            ImageLabelClassification(
                self.validation_folder_images_labels,
                self.label2class_mapping,
                self.validation_folder_invalid_images_labels,
                self.folder_path,
                "Validation Folder Labels Summary ",
                "validation_folder_labels_summary",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "validation_folder_labels_summary"
            )
        if self.split_training:
            ImageLabelClassification(
                self.training_labels,
                self.label2class_mapping,
                None,
                self.folder_path,
                "Training Data After Splitting Labels Summary ",
                "training_after_splitting_labels_summary",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "training_after_splitting_labels_summary"
            )
            ImageLabelClassification(
                self.validation_labels,
                self.label2class_mapping,
                None,
                self.folder_path,
                "Validation Data After Splitting Labels Summary ",
                "validation_after_splitting_labels_summary",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "validation_after_splitting_labels_summary"
            )

    def make_garphs_sample(self):
        DisplaySampleImagesClassification(
            self.training_folder_images_paths,
            self.training_folder_images_labels,
            self.label2class_mapping,
            self.folder_path,
            title=" Random Samples of Training Folder",
            file_name="training_folder_random_samples",
        ).build_graph()
        self.report_data["graphs"]["images_name"].append(
            "training_folder_random_samples"
        )
        if self.validation_folder_images is not None:
            DisplaySampleImagesClassification(
                self.validation_folder_images_paths,
                self.validation_folder_images_labels,
                self.label2class_mapping,
                self.folder_path,
                title="Random Samples of Validation Folder",
                file_name="validation_folder_random_samples",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "validation_folder_random_samples"
            )
        if self.split_training:
            DisplaySampleImagesClassification(
                self.training_paths,
                self.training_labels,
                self.label2class_mapping,
                self.folder_path,
                title="Samples of Training Data After Splitting",
                file_name="training_after_splitting_random_samples",
            ).build_graph()
            self.report_data["graphs"]["images_name"].append(
                "training_after_splitting_random_samples"
            )
            DisplaySampleImagesClassification(
                self.validation_paths,
                self.validation_labels,
                self.label2class_mapping,
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
        ClassificationImagesAfterLoader(
            training_images,
            training_labels,
            self.label2class_mapping,
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
            ClassificationImagesAfterLoader(
                validation_images,
                validation_labels,
                self.label2class_mapping,
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
        self.make_graphs_label_summary()
        self.make_garphs_sample()
        self.make_graphs_loader()

    def get_loaders(self):
        training_dataset = ClassificationImageDataset(
            self.training_paths,
            self.training_labels,
            self.image_size,
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
            validation_dataset = ClassificationImageDataset(
                self.validation_paths,
                self.validation_labels,
                self.image_size,
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
        self.get_classes_mapping()
        (
            self.training_folder_images_paths,
            self.training_folder_images_labels,
        ) = self.data_preparing(
            self.training_folder_images,
            self.training_folder_invalid_images,
            self.training_folder_invalid_images_labels,
            self.training_class,
        )
        if self.validation_folder_images is not None:
            (
                self.validation_folder_images_paths,
                self.validation_folder_images_labels,
            ) = self.data_preparing(
                self.validation_folder_images,
                self.validation_folder_invalid_images,
                self.validation_folder_invalid_images_labels,
                self.validation_class,
            )
        self.data_overview()
        self.splitting()
        self.get_loaders()
        self.make_graphs()

        report = ImagePreprocessingReport(self.folder_path, self.report_data)
        report.execute()
        return self.training_loader, self.validation_loader
